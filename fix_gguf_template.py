#!/usr/bin/env python3
"""GökTürk GGUF sohbet şablonu düzeltici — tek dosya, repo gerekmez.

Sorun: İlk GGUF'a orijinal Qwen şablonu gömüldü. Sistem mesajı yoksa bu şablon
"You are Qwen, created by Alibaba Cloud" ekler; model kendini Qwen sanır ve
6 aşamalı <thought> düşüncesini başlatmaz.

Çözüm: Model ağırlıklarına DOKUNMADAN GGUF içindeki şablonu GökTürk şablonuyla
değiştirir (eğitimdeki sistem promptu otomatik eklenir).

Kullanım (Linux/Pardus, Windows, macOS):
    pip install gguf
    python fix_gguf_template.py GokTurk2.5-3B-Thinking-Q4_K_M.gguf
    → GokTurk2.5-3B-Thinking-Q4_K_M-fixed.gguf   (orijinal dosya korunur)

İsteğe bağlı HF yükleme:
    HF_TOKEN=hf_xxx python fix_gguf_template.py model.gguf --push ALPRO2023/GokTurk2.5-3B-Thinking-GGUF
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MODEL_NAME = "GökTürk2.5-3B-Thinking"
# gokturk_cot.GGUF_CHAT_TEMPLATE ile birebir aynı (tests/test_cot6.py denetler)
CHAT_TEMPLATE = "{%- if messages[0]['role'] != 'system' -%}{{- '<|im_start|>system\\n' + 'Sen GökTürk2.5-3B-Thinking, Türkçe konuşan, dürüst ve titiz bir yapay zekâ asistanısın.\\nHer yanıttan önce <thought> içinde sırasıyla 6 adımı uygula:\\n[STEP 1] isteği, kısıtları ve beklenen formatı ayrıştır;\\n[STEP 2] kendi bilgini tara ve \"Dış kaynak: GEREKLİ\" ya da \"Dış kaynak: GEREKSİZ\" diye karar ver;\\n[STEP 3] gerekiyorsa en iyi arama sorgusunu kur ve tavily_search aracını çağır;\\n[STEP 4] araç sonuçlarını güvenilirlik ve tutarlılık açısından süz, çelişkileri ele;\\n[STEP 5] yanıtı yapılandır;\\n[STEP 6] mantık, dil ve eksiklik denetimi yap; son satıra \"Doğrulanan sonuç: ...\" yaz.\\n</thought> sonrasında nihai yanıtı açık Türkçe ile ver. Nihai yanıt, doğrulanan sonuçla BİREBİR aynı olmalı;\\ndüşüncede bulduğun sonucu değiştirme. Web kaynaklarını [1], [2] diye an.\\n\\n# Araçlar\\n<tools>\\n{\"type\": \"function\", \"function\": {\"name\": \"tavily_search\", \"description\": \"Güncel/canlı web bilgisi için Tavily araması yapar.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\", \"description\": \"Kısa, anahtar kelime odaklı sorgu\"}, \"topic\": {\"type\": \"string\", \"enum\": [\"general\", \"news\"]}, \"time_range\": {\"type\": \"string\", \"enum\": [\"day\", \"week\", \"month\", \"year\"]}}, \"required\": [\"query\"]}}}\\n</tools>\\nAraç çağırmak için yalnızca şunu yaz:\\n<tool_call>\\n{\"name\": \"tavily_search\", \"arguments\": {\"query\": \"...\"}}\\n</tool_call>' + '<|im_end|>\\n' -}}{%- endif -%}{%- for message in messages -%}{%- if message['role'] == 'tool' -%}{{- '<|im_start|>user\\n<tool_response>\\n' + message['content'] + '\\n</tool_response><|im_end|>\\n' -}}{%- else -%}{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}{%- endif -%}{%- endfor -%}{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"


def read_meta(path: Path) -> dict:
    from gguf import GGUFReader
    r = GGUFReader(str(path))

    def field(k):
        f = r.fields.get(k)
        if f is None:
            return None
        v = f.parts[f.data[0]]
        return bytes(v).decode("utf-8", "replace") if v.dtype.kind == "u" and v.itemsize == 1 else v.tolist()[0]
    return {"arch": field("general.architecture"), "name": field("general.name"),
            "template": field("tokenizer.chat_template") or "", "tensors": len(r.tensors)}


def main() -> int:
    ap = argparse.ArgumentParser(description="GökTürk sohbet şablonunu GGUF'a göm")
    ap.add_argument("gguf", nargs="+", type=Path, help="düzeltilecek .gguf dosya(lar)ı")
    ap.add_argument("--inplace", action="store_true", help="orijinalin üzerine yaz (varsayılan: -fixed kopya)")
    ap.add_argument("--push", metavar="REPO", help="düzeltilen dosyaları bu HF deposuna yükle")
    args = ap.parse_args()
    try:
        import gguf  # noqa: F401
    except ImportError:
        print("❌ 'gguf' paketi yok →  pip install gguf"); return 1

    done = []
    for src in args.gguf:
        if not src.exists():
            print(f"❌ bulunamadı: {src}"); return 1
        meta = read_meta(src)
        print(f"📄 {src.name}: arch={meta['arch']} tensör={meta['tensors']} | eski şablon: "
              + ("GökTürk ✓" if "GökTürk" in meta["template"] else "Qwen/diğer ✗"))
        if meta["arch"] != "qwen2":
            print("❌ Qwen2 mimarisi değil; yanlış dosya?"); return 1
        dst = src.with_name(src.stem + "-fixed.gguf")
        with tempfile.NamedTemporaryFile("w", suffix=".jinja", delete=False, encoding="utf-8") as f:
            f.write(CHAT_TEMPLATE)
            tpl_file = f.name
        cmd = [sys.executable, "-m", "gguf.scripts.gguf_new_metadata", str(src), str(dst),
               "--chat-template-file", tpl_file, "--general-name", MODEL_NAME, "--force"]
        print("⏳ yazılıyor (dosya boyutu kadar disk alanı gerekir, ~1 dk)...")
        r = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(tpl_file)
        if r.returncode != 0:
            print("❌ gguf_new_metadata hatası:\n" + (r.stderr or r.stdout)[-1500:]); return 1
        new = read_meta(dst)
        ok = (new["template"] == CHAT_TEMPLATE and new["tensors"] == meta["tensors"]
              and dst.stat().st_size >= src.stat().st_size * 0.99)
        if not ok:
            print(f"❌ doğrulama başarısız: {new['tensors']} tensör, şablon eşleşmesi={new['template'] == CHAT_TEMPLATE}")
            return 1
        if args.inplace:
            dst.replace(src); dst = src
        print(f"✅ {dst.name}: şablon GökTürk ✓ | ad: {new['name']} | tensör {new['tensors']} (değişmedi)")
        done.append(dst)

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("⚠️ HF_TOKEN ortam değişkeni yok; yükleme atlandı."); return 0
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(args.push, exist_ok=True)
        for d in done:
            name = d.name.replace("-fixed.gguf", ".gguf")
            print(f"☁️ {name} yükleniyor...")
            api.upload_file(path_or_fileobj=str(d), path_in_repo=name, repo_id=args.push,
                            commit_message="GökTürk sohbet şablonu (6 aşamalı CoT sistem promptu)")
        print(f"🎉 https://huggingface.co/{args.push}")
    print("\n📱 PocketPal: eski modeli silin → yeni -fixed.gguf dosyasını ekleyin → "
          "model ayarlarında Sistem İstemi (System prompt) BOŞ olsun.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
