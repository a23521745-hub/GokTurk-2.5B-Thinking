#!/usr/bin/env python3
"""GökTürk-2.5B-Thinking — LoRA → merged 16-bit → GGUF (Q4_K_M / Q8_0) → Hugging Face.

    python push_to_hf.py --lora outputs/gokturk-2.5b-thinking/lora_adapter --quants q4_k_m q8_0 --push

Adımlar:
  1. Merge   : LoRA + taban model → 16-bit safetensors
               --merge peft    (varsayılan) CPU'da, GPU/Unsloth sürüm sorunlarından bağımsız
               --merge unsloth GPU'da, Unsloth save_pretrained_merged
  2. GGUF    : llama.cpp convert_hf_to_gguf.py → F16 → llama-quantize (Q4_K_M, Q8_0)
               (--gguf-backend unsloth ile model.save_pretrained_gguf kullanılabilir)
  3. RAM     : her dosya için bağlam uzunluğuna göre tahmini RAM tablosu (2.5 GB hedefi)
  4. Push    : <kullanıcı>/GokTurk-2.5B-Thinking (merged) ve <...>-GGUF (GGUF dosyaları) depoları

GGUF'a GökTürk sohbet şablonu gömülür: sistem mesajı verilmezse 6 adımlı CoT sistem promptu
otomatik eklenir → PocketPal AI / LM Studio / llama.cpp'de ek ayar gerekmez.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gokturk_env  # noqa: E402
from gokturk_cot import CHAT_TEMPLATE, MODEL_NAME, SYSTEM_PROMPT  # noqa: E402

SLUG = "GokTurk-2.5B-Thinking"   # HF depo adlarında Türkçe karakter kullanılamaz
RAM_BUDGET_GB = 2.5


def run(cmd, **kw):
    print("$", " ".join(map(str, cmd)))
    subprocess.check_call([str(c) for c in cmd], **kw)


# ---------------------------------------------------------------------------
# 1) MERGE
# ---------------------------------------------------------------------------
def resolve_base(lora_dir: Path) -> str:
    base = json.loads((lora_dir / "adapter_config.json").read_text())["base_model_name_or_path"]
    # 4-bit Unsloth kopyası → aynı modelin 16-bit sürümü (merge 16-bit ağırlıkla yapılmalı)
    return re.sub(r"-(bnb-4bit|unsloth-bnb-4bit)$", "", base).replace("unsloth/", "Qwen/")


def patch_tokenizer_config(model_dir: Path):
    """Sohbet şablonunu tokenizer_config.json'a yazar (GGUF dönüştürücü buradan okur)."""
    cfg_path = model_dir / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["chat_template"] = CHAT_TEMPLATE
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (model_dir / "chat_template.jinja").unlink(missing_ok=True)   # yeni transformers ayrı dosya kullanabilir


def merge_peft(lora_dir: Path, out_dir: Path, base: str | None = None) -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = base or resolve_base(lora_dir)
    print(f"🔗 PEFT merge (CPU): {base} + {lora_dir}")
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu",
                                                 low_cpu_mem_usage=True, token=os.environ.get("HF_TOKEN"))
    model = PeftModel.from_pretrained(model, str(lora_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(str(lora_dir)).save_pretrained(str(out_dir))
    del model
    return out_dir


def merge_unsloth(lora_dir: Path, out_dir: Path) -> Path:
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(str(lora_dir), max_seq_length=2048, load_in_4bit=True)
    model.save_pretrained_merged(str(out_dir), tok, save_method="merged_16bit")
    return out_dir


# ---------------------------------------------------------------------------
# 2) GGUF (llama.cpp)
# ---------------------------------------------------------------------------
def ensure_llamacpp(dir_: Path, ref: str = "master") -> Path:
    quant = dir_ / "build/bin/llama-quantize"
    if quant.exists():
        return dir_
    if not dir_.exists():
        run(["git", "clone", "--depth", "1", "--branch", ref, "https://github.com/ggml-org/llama.cpp", dir_])
    # NOT: requirements-convert_hf_to_gguf.txt CPU torch kurar ve Kaggle/Colab GPU torch'unu bozar →
    # yalnızca eksik hafif bağımlılıkları kur; gguf-py betik tarafından depodan otomatik yüklenir.
    gokturk_env.pip_install("sentencepiece", "protobuf", "numpy")
    run(["cmake", "-S", dir_, "-B", dir_ / "build", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF",
         "-DGGML_CUDA=OFF", "-DGGML_NATIVE=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
         "-DLLAMA_BUILD_SERVER=OFF"])
    run(["cmake", "--build", dir_ / "build", "--target", "llama-quantize", "-j", str(os.cpu_count() or 2)])
    return dir_


def gguf_llamacpp(merged: Path, out_dir: Path, quants: list[str], llama_dir: Path, keep_f16=False) -> list[Path]:
    ensure_llamacpp(llama_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f16 = out_dir / f"{SLUG}-F16.gguf"
    run([sys.executable, llama_dir / "convert_hf_to_gguf.py", merged, "--outtype", "f16", "--outfile", f16])
    outs = []
    for q in quants:
        dst = out_dir / f"{SLUG}-{q.upper()}.gguf"
        run([llama_dir / "build/bin/llama-quantize", f16, dst, q.upper()])
        outs.append(dst)
    if keep_f16:
        outs.append(f16)
    else:
        f16.unlink()
    return outs


def gguf_unsloth(lora_dir: Path, out_dir: Path, quants: list[str]) -> list[Path]:
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(str(lora_dir), max_seq_length=2048, load_in_4bit=True)
    tok.chat_template = CHAT_TEMPLATE
    build = out_dir / "_unsloth_gguf"
    model.save_pretrained_gguf(str(build), tok, quantization_method=[q.lower() for q in quants])
    outs = []
    for q in quants:
        hits = [p for p in glob.glob(f"{build}*/**/*.gguf", recursive=True)
                if q.lower().replace("_", "") in Path(p).name.lower().replace("_", "").replace("-", "")]
        if not hits:
            raise FileNotFoundError(f"{q} GGUF bulunamadı ({build})")
        dst = out_dir / f"{SLUG}-{q.upper()}.gguf"
        shutil.move(max(hits, key=os.path.getmtime), dst)
        outs.append(dst)
    shutil.rmtree(build, ignore_errors=True)
    return outs


# ---------------------------------------------------------------------------
# 3) RAM tahmini
# ---------------------------------------------------------------------------
def ram_table(gguf_files: list[Path], config_path: Path | None, ctxs=(2048, 4096, 8192)) -> str:
    kv_per_tok = None
    if config_path and config_path.exists():
        c = json.loads(config_path.read_text())
        hd = c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"]
        kv_per_tok = 2 * c["num_hidden_layers"] * c.get("num_key_value_heads", c["num_attention_heads"]) * hd * 2
    lines = ["| Dosya | Boyut | " + " | ".join(f"RAM @ ctx {c}" for c in ctxs) + " |",
             "|---|---|" + "---|" * len(ctxs)]
    for f in gguf_files:
        size = f.stat().st_size / 1024**3
        cells = []
        for c in ctxs:
            if kv_per_tok is None:
                cells.append("?")
                continue
            total = size + kv_per_tok * c / 1024**3 + 0.25   # + hesaplama tamponları / çalışma zamanı
            cells.append(f"{total:.2f} GB {'✅' if total <= RAM_BUDGET_GB else '⚠️'}")
        lines.append(f"| `{f.name}` | {size:.2f} GB | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4) Hugging Face
# ---------------------------------------------------------------------------
def model_card(base: str, repo_merged: str, gguf_files: list[Path], table: str, gguf=True) -> str:
    files = "\n".join(f"| `{f.name}` | {f.stem.split('-')[-1]} | {f.stat().st_size / 1024**3:.2f} GB |"
                      for f in gguf_files)
    head = f"""---
language: [tr]
license: apache-2.0
base_model: {base}
library_name: {"gguf" if gguf else "transformers"}
tags: [qwen2.5, unsloth, turkish, reasoning, chain-of-thought, tool-calling, rag, cybersecurity{", gguf" if gguf else ""}]
---
# 🐺 {MODEL_NAME}{" — GGUF" if gguf else ""}

{base} tabanlı, Unsloth QLoRA ile ince ayarlanmış, **6 aşamalı hibrit CoT** ve **Tavily tool calling**
yeteneğine sahip Türkçe model. Her yanıt `<thought>` içinde şu adımlardan geçer:

1. Problem Sentezleme & Ayrıştırma · 2. İçsel Akıl Yürütme · 3. Web Araştırması (`tavily_search`)
4. Kaynak Doğrulama · 5. Yanıt Planlama · 6. Öz-Denetim
"""
    if not gguf:
        return head + f"\nGGUF sürümleri: [{repo_merged}-GGUF](https://huggingface.co/{repo_merged}-GGUF)\n"
    return head + f"""
## Dosyalar
| Dosya | Kuantizasyon | Boyut |
|---|---|---|
{files}

## Tahmini RAM (2.5 GB hedefi)
{table}

## Kullanım
**PocketPal AI / LM Studio:** GGUF'u indirip yükleyin. Sohbet şablonu gömülüdür; sistem promptu boş
bırakılırsa GökTürk 6 adımlı prompt otomatik eklenir. Önerilen: temperature 0.6, top_p 0.95, ctx 2048–4096.

**llama.cpp + Tavily RAG:**
```bash
llama-server -m {gguf_files[0].name} -c 4096 -t 4 --port 8080
export TAVILY_API_KEY=tvly-...
python tavily_rag_handler.py "OpenSSH için son kritik CVE hangisi?"
```
Kaynak kod: https://github.com/a23521745-hub/GokTurk-2.5B-Thinking

<details><summary>Sistem promptu</summary>

```
{SYSTEM_PROMPT}
```
</details>
"""


def push(repo_base: str | None, merged: Path | None, gguf_files: list[Path], base: str, table: str,
         private: bool):
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("⏭️  HF_TOKEN yok → yükleme atlandı.")
        return
    api = HfApi(token=token)
    repo_base = repo_base or f"{api.whoami()['name']}/{SLUG}"
    if merged and merged.exists():
        api.create_repo(repo_base, private=private, exist_ok=True)
        print(f"⬆️  merged 16-bit → {repo_base}")
        api.upload_folder(folder_path=str(merged), repo_id=repo_base, commit_message="Merged 16-bit weights",
                          ignore_patterns=["*.gguf", "_*"])
        api.upload_file(path_or_fileobj=model_card(base, repo_base, [], "", gguf=False).encode(),
                        path_in_repo="README.md", repo_id=repo_base)
        print(f"✅ https://huggingface.co/{repo_base}")
    if gguf_files:
        repo_gguf = f"{repo_base}-GGUF"
        api.create_repo(repo_gguf, private=private, exist_ok=True)
        for f in gguf_files:
            print(f"⬆️  {f.name} → {repo_gguf}")
            for attempt in range(3):
                try:
                    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo_gguf,
                                    commit_message=f"Add {f.name}")
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  deneme {attempt + 1}/3: {e}")
                    if attempt == 2:
                        raise
        api.upload_file(path_or_fileobj=model_card(base, repo_base, gguf_files, table).encode(),
                        path_in_repo="README.md", repo_id=repo_gguf)
        print(f"✅ https://huggingface.co/{repo_gguf}")


# ---------------------------------------------------------------------------
def main():
    out_default, scratch_default = gokturk_env.default_dirs()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", type=Path, default=out_default / "gokturk-2.5b-thinking/lora_adapter")
    ap.add_argument("--out", type=Path, default=out_default / "gokturk-2.5b-thinking")
    ap.add_argument("--scratch", type=Path, default=scratch_default)
    ap.add_argument("--base", help="16-bit taban modeli elle belirt (ör. Qwen/Qwen2.5-3B-Instruct)")
    ap.add_argument("--merge", choices=["peft", "unsloth"], default="peft")
    ap.add_argument("--gguf-backend", choices=["llamacpp", "unsloth"], default="llamacpp")
    ap.add_argument("--quants", nargs="*", default=["q4_k_m", "q8_0"])
    ap.add_argument("--keep-f16", action="store_true")
    ap.add_argument("--llama-cpp-dir", type=Path, default=None)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--no-push-merged", dest="push_merged", action="store_false")
    ap.add_argument("--repo", help="kullanici/GokTurk-2.5B-Thinking (boşsa HF kullanıcı adınızla)")
    ap.add_argument("--private", action="store_true")
    a = ap.parse_known_args()[0]

    gokturk_env.load_secrets(("HF_TOKEN",))
    assert (a.lora / "adapter_config.json").exists(), f"LoRA bulunamadı: {a.lora}"
    base = a.base or resolve_base(a.lora)
    merged = a.scratch / "merged_16bit"          # büyük → scratch (Kaggle kotası dışında)
    gguf_dir = a.out / "gguf"
    a.scratch.mkdir(parents=True, exist_ok=True)

    need_merged = a.gguf_backend == "llamacpp" or (a.push and a.push_merged)
    if need_merged and not (merged / "config.json").exists():
        (merge_peft(a.lora, merged, base) if a.merge == "peft" else merge_unsloth(a.lora, merged))
    if need_merged:
        patch_tokenizer_config(merged)

    files: list[Path] = []
    if a.quants:
        files = (gguf_llamacpp(merged, gguf_dir, a.quants, a.llama_cpp_dir or a.scratch / "llama.cpp", a.keep_f16)
                 if a.gguf_backend == "llamacpp" else gguf_unsloth(a.lora, gguf_dir, a.quants))
    cfg = merged / "config.json"
    table = ram_table(files, cfg) if files else ""
    if table:
        print("\n📊 Tahmini RAM kullanımı (model + KV-cache f16 + ~0.25 GB çalışma zamanı):\n" + table)
        print("💡 2.5 GB sınırında 3B için ctx ≤ 4096 ve llama.cpp'de `-ctk q8_0 -ctv q8_0 -fa on` önerilir.")

    if a.push:
        push(a.repo, merged if a.push_merged else None, files, base, table, a.private)

    for f in files:
        print(f"📦 {f} ({f.stat().st_size / 1024**3:.2f} GB)")


if __name__ == "__main__":
    main()
