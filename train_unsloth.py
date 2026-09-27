#!/usr/bin/env python3
"""GökTürk2.5-3B-Thinking — Unsloth QLoRA eğitimi (Kaggle T4 / Colab T4 / yerel GPU).

    python train_unsloth.py                                   # 3B, 4-bit, 2 epoch
    python train_unsloth.py --size 1.5b                       # daha hafif (≈1.0 GB Q4_K_M)
    python train_unsloth.py --max-steps 60                    # ~5 dk duman testi
    python train_unsloth.py --push-lora                       # eğitim + LoRA'yı HF'e yükle
    python train_unsloth.py --push --export q4_k_m q8_0       # eğitim + LoRA + merged 16-bit + GGUF → HF

Hugging Face: HF_TOKEN (ortam değişkeni / Kaggle Secret / Colab Secret) ve isteğe bağlı
HF_REPO=kullanici/GokTurk2.5-3B-Thinking (ya da --hf-repo). Token asla yazdırılmaz.

Veri: data/gokturk6/train.jsonl (+ val.jsonl). Yoksa sentetik veri otomatik üretilir.
Metin `gokturk_cot.render_chatml` ile işlenir → eğitim ve çıkarım biçimi birebir aynı.
Kayıp yalnızca asistan turlarında hesaplanır (sistem, kullanıcı ve <tool_response> maskelenir).
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import gokturk_env  # noqa: E402  (torch'u import etmez)

MODELS = {
    "1.5b": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",   # Q4_K_M ≈ 1.0 GB → 2.5 GB RAM'de çok rahat
    "3b": "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",       # Q4_K_M ≈ 1.9 GB → 2.5 GB RAM'de sınırda (ctx ≤ 4096)
}


def parse_args():
    out_default, _ = gokturk_env.default_dirs()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=list(MODELS), default="3b")
    ap.add_argument("--model", help="özel taban model (--size'ı geçersiz kılar)")
    ap.add_argument("--train", default=str(ROOT / "data/gokturk6/train.jsonl"))
    ap.add_argument("--val", default=str(ROOT / "data/gokturk6/val.jsonl"))
    ap.add_argument("--output", default=str(out_default / "gokturk"))
    ap.add_argument("--max-seq-len", type=int, default=2048)
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    # optimizasyon (T4 16 GB: 2 × 8 = efektif 16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--resume", action="store_true", help="son checkpoint'ten devam et")
    ap.add_argument("--skip-smoke-test", action="store_true")
    # Hugging Face
    ap.add_argument("--hf-repo", default=os.environ.get("HF_REPO"),
                    help="kullanici/GokTurk2.5-3B-Thinking (boşsa HF kullanıcı adınızla)")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--push-lora", action="store_true", help="LoRA adaptörünü <repo>-LoRA deposuna yükle")
    ap.add_argument("--push", action="store_true", help="LoRA + merged 16-bit (+ --export GGUF) yükle")
    ap.add_argument("--export", nargs="*", default=[], help="GGUF: q4_k_m q8_0")
    return ap.parse_known_args()[0]   # Jupyter'in eklediği argümanları yok say


# ---------------------------------------------------------------------------
def ensure_data(args):
    train = Path(args.train)
    if train.exists() and train.stat().st_size > 0:
        return
    print("⚙️  Eğitim verisi yok → sentetik veri üretiliyor (API gerekmez)...")
    tmp = train.parent / "_synth6.jsonl"
    subprocess.check_call([sys.executable, str(ROOT / "prepare_dataset.py"), "synthetic", "-o", str(tmp)])
    subprocess.check_call([sys.executable, str(ROOT / "prepare_dataset.py"), "merge", str(tmp),
                           "-o", str(train.parent), "--max-tokens", str(args.max_seq_len)])


def load_split(path: str):
    from gokturk_cot import SYSTEM_PROMPT, render_chatml, validate_messages
    rows, bad = [], 0
    p = Path(path)
    if not p.exists():
        return rows, 0
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        if msgs[0]["role"] == "system":
            msgs[0]["content"] = SYSTEM_PROMPT      # eski veride bile tek tip, güncel sistem promptu
        else:
            msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        ok, _ = validate_messages(msgs)
        if ok:
            rows.append({"text": render_chatml(msgs)})
        else:
            bad += 1
    return rows, bad


def push_lora(lora_dir: Path, repo: str, private: bool, meta: dict):
    from huggingface_hub import HfApi
    from gokturk_cot import MODEL_NAME
    api = HfApi(token=os.environ["HF_TOKEN"])
    gokturk_env.retry(lambda: api.create_repo(repo, private=private, exist_ok=True), what="LoRA repo oluşturma")
    card = (f"---\nbase_model: {meta['base_model']}\nlibrary_name: peft\nlanguage: [tr]\nlicense: apache-2.0\n"
            f"tags: [lora, qlora, unsloth, turkish, reasoning, chain-of-thought]\n---\n"
            f"# {MODEL_NAME} — LoRA adaptörü\n\nr={meta['lora_r']}, alpha={meta['lora_alpha']}, "
            f"epoch={meta['epochs']}, örnek={meta['train_examples']}, "
            f"train_loss={meta['metrics'].get('train_loss', 0):.4f}\n")
    (lora_dir / "README.md").write_text(card, encoding="utf-8")
    gokturk_env.retry(lambda: api.upload_folder(folder_path=str(lora_dir), repo_id=repo,
                                                commit_message="LoRA adaptörü"), what="LoRA yükleme")
    print(f"✅ LoRA → https://huggingface.co/{repo}")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    gokturk_env.require_gpu()              # CPU'da saatlerce boşa beklemeden net hata
    gokturk_env.setup_gpu_env()            # torch'tan ÖNCE
    gokturk_env.load_secrets()
    gokturk_env.ensure_unsloth()
    ensure_data(args)

    # HF token'ı eğitimden ÖNCE doğrula: geçersizse eğitim yine yapılır, yalnızca yükleme atlanır
    want_hf = args.push or args.push_lora
    repo = None
    if want_hf:
        user = gokturk_env.check_hf_token()
        if user is None:
            want_hf = False
            print("⏭️  Hugging Face yüklemesi atlanacak; model /kaggle/working/outputs altına kaydedilecek.")
        else:
            from gokturk_cot import HF_SLUG
            repo = args.hf_repo or f"{user}/{HF_SLUG}"
            print(f"🎯 Hedef HF deposu: {repo}  (+ -LoRA, -GGUF)")

    from unsloth import FastLanguageModel, is_bfloat16_supported   # transformers'tan ÖNCE
    from unsloth.chat_templates import train_on_responses_only
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from gokturk_cot import CHAT_TEMPLATE, SYSTEM_PROMPT, render_chatml, validate_text

    props = torch.cuda.get_device_properties(0)
    print(f"🖥️  {props.name} ({props.total_memory / 1024**3:.1f} GB) | ortam: {gokturk_env.detect_env()}")

    base = args.model or MODELS[args.size]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    _, scratch = gokturk_env.default_dirs()
    ckpt_dir = scratch / "checkpoints"

    # ---------------- Model + LoRA (ağ kesintisine dayanıklı) ----------------
    gokturk_env.prefetch_model(base)     # önbelleğe indir (yeniden denemeli); yüklemede DEPO ADI kullanılır ki
    model, tokenizer = gokturk_env.retry(lambda: FastLanguageModel.from_pretrained(   # adaptöre yol değil ad yazılsın
        model_name=base, max_seq_length=args.max_seq_len, dtype=None, load_in_4bit=True),
        what="model yükleme", tries=3)
    tokenizer.chat_template = CHAT_TEMPLATE          # GGUF'a gömülecek şablon (varsayılan sistem promptlu)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed, use_rslora=False)

    # ---------------- Veri ----------------
    train_rows, bad = load_split(args.train)
    val_rows, _ = load_split(args.val)
    assert len(train_rows) >= 10, f"Yetersiz geçerli veri: {len(train_rows)} ({bad} geçersiz)"
    train_ds = Dataset.from_list(train_rows).shuffle(seed=args.seed)
    val_ds = Dataset.from_list(val_rows) if val_rows else None

    def fits(x):
        return len(tokenizer(x["text"], add_special_tokens=False).input_ids) <= args.max_seq_len

    n0 = len(train_ds)
    train_ds = train_ds.filter(fits)
    if val_ds is not None:
        val_ds = val_ds.filter(fits)
        val_ds = val_ds if len(val_ds) else None
    print(f"📚 train={len(train_ds)} (uzun atılan {n0 - len(train_ds)}, geçersiz {bad}) "
          f"val={len(val_ds) if val_ds else 0}")

    # ---------------- Trainer (trl sürümlerine dayanıklı) ----------------
    bf16 = is_bfloat16_supported()
    cfg = dict(
        output_dir=str(ckpt_dir), dataset_text_field="text",
        max_seq_length=args.max_seq_len, max_length=args.max_seq_len,   # eski / yeni trl
        packing=False, per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, max_steps=args.max_steps,
        learning_rate=args.lr, warmup_ratio=args.warmup_ratio, lr_scheduler_type="cosine",
        optim="adamw_8bit", weight_decay=0.01, max_grad_norm=1.0,
        fp16=not bf16, bf16=bf16,                                         # T4 → fp16
        logging_steps=10, save_strategy="steps", save_steps=100, save_total_limit=2,
        eval_strategy="steps" if val_ds else "no", evaluation_strategy="steps" if val_ds else "no",
        eval_steps=100, seed=args.seed, report_to="none", dataset_num_proc=2,
    )
    valid = set(inspect.signature(SFTConfig.__init__).parameters)
    if "eval_strategy" in valid:
        cfg.pop("evaluation_strategy")
    sft_cfg = SFTConfig(**{k: v for k, v in cfg.items() if k in valid})
    tk = dict(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=sft_cfg)
    tk["processing_class" if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters
       else "tokenizer"] = tokenizer
    trainer = SFTTrainer(**tk)
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")

    t0 = time.time()
    resume = args.resume and any(ckpt_dir.glob("checkpoint-*"))
    stats = trainer.train(resume_from_checkpoint=True if resume else None)
    print(f"✅ Eğitim: {(time.time() - t0) / 60:.1f} dk | loss {stats.metrics.get('train_loss', 0):.4f} | "
          f"tepe VRAM {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

    lora_dir = out / "lora_adapter"
    model.save_pretrained(str(lora_dir))
    tokenizer.save_pretrained(str(lora_dir))
    # adaptöre taban modelin DEPO ADINI yaz (yerel önbellek yolu başka makinede geçersizdir)
    ac = lora_dir / "adapter_config.json"
    acj = json.loads(ac.read_text())
    acj["base_model_name_or_path"] = base
    ac.write_text(json.dumps(acj, indent=2))
    meta = {"base_model": base, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "epochs": args.epochs,
            "lr": args.lr, "train_examples": len(train_ds), "metrics": stats.metrics}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (lora_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))  # HF'e de gitsin
    print(f"💾 LoRA → {lora_dir}")

    # LoRA'yı HEMEN yükle: sonraki adımlar (merge/GGUF) başarısız olsa bile eğitim kaybolmaz
    if want_hf:
        try:
            push_lora(lora_dir, f"{repo}-LoRA", args.private, meta)
        except Exception as e:  # noqa: BLE001 — eğitim sonucu diskte güvende; yükleme sonra tekrarlanabilir
            print(f"⚠️  LoRA yüklenemedi ({str(e)[:200]}). Dosyalar {lora_dir} altında duruyor; "
                  "push_to_hf.py ile tekrar deneyebilirsiniz.")

    # ---------------- Format + tutarlılık duman testi ----------------
    if not args.skip_smoke_test:
        FastLanguageModel.for_inference(model)
        tests = ["5 işçi bir işi 12 günde bitiriyorsa, 3 işçi aynı işi kaç günde bitirir?",
                 "Bir PHP formunda dosya yükleme açığını nasıl güvenli hale getiririm?",
                 "OpenSSH için bu ay yayımlanan kritik CVE var mı?"]
        passed = 0
        for q in tests:
            prompt = render_chatml([{"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": q}], add_generation_prompt=True)
            ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=900, temperature=0.3, top_p=0.9, do_sample=True,
                                     repetition_penalty=1.05, pad_token_id=tokenizer.pad_token_id)
            text = tokenizer.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
            text = text.split("<|im_end|>")[0]
            if "<tool_call>" in text:    # araç turunda durdu → buraya kadar biçim doğru mu
                ok, why = ("[STEP 3" in text and "</tool_call>" in text), "araç çağrısı"
            else:
                ok, why = validate_text(text)
            passed += ok
            print(f"\n{'=' * 70}\n❓ {q}\n{text[:1800]}\n→ {'✓' if ok else '✗'} ({why})")
        print(f"\n📐 Format + tutarlılık: {passed}/{len(tests)}")

    # ---------------- Merge + GGUF + HF (ayrı süreç, GPU belleği serbest) ----------------
    if args.export or (args.push and want_hf):
        del trainer, model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        cmd = [sys.executable, str(ROOT / "push_to_hf.py"), "--lora", str(lora_dir), "--out", str(out),
               "--quants", *args.export]
        if args.push and want_hf:
            cmd += ["--push", "--repo", repo] + (["--private"] if args.private else [])
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
