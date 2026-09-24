# -*- coding: utf-8 -*-
"""GökTürk-2.5B-Thinking — Unsloth QLoRA eğitim betiği (Google Colab T4 16GB).

Colab'de:  !python train/train_unsloth.py            (veya notebook'u çalıştırın)
Yerelde:   python train/train_unsloth.py --model unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit

Adımlar: kurulum kontrolü → veri (yoksa sentetik üretim) → 4-bit model + LoRA →
yalnızca asistan yanıtları üzerinde SFT → format testi → LoRA/merged/GGUF kayıt.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
                    help="T4 için 3B (4-bit) veya daha hızlı: unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "data/gokturk_synth.jsonl")])
    ap.add_argument("--synth-n", type=int, default=3000, help="Veri yoksa üretilecek sentetik örnek")
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help=">0 ise epochs yerine kullanılır")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--output", default=str(ROOT / "outputs/gokturk-2.5b-thinking"))
    ap.add_argument("--gguf", nargs="*", default=[], help="Örn: --gguf q4_k_m q8_0 (Colab'de yavaş; CI önerilir)")
    ap.add_argument("--push-to-hub", default=None, help="HF repo id (HF_TOKEN gerekir)")
    ap.add_argument("--seed", type=int, default=3407)
    # Jupyter/Colab'in eklediği argümanları yok say
    return ap.parse_known_args()[0]


def ensure_unsloth():
    try:
        import unsloth  # noqa: F401
    except ImportError:
        print("📦 Unsloth kuruluyor...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth"])


def load_data(args):
    from datasets import Dataset
    files = [Path(p) for p in args.data]
    if not any(f.exists() for f in files):
        print("⚙️  Veri bulunamadı, sentetik veri üretiliyor...")
        subprocess.check_call([sys.executable, str(ROOT / "scripts/generate_dataset.py"),
                               "synthetic", "-n", str(args.synth_n), "-o", str(files[0])])
    from gokturk_format import is_valid
    rows, bad = [], 0
    for f in files:
        if not f.exists():
            print(f"[uyarı] {f} yok, atlanıyor")
            continue
        for line in open(f, encoding="utf-8"):
            rec = json.loads(line)
            if is_valid(rec["messages"][-1]["content"]):
                rows.append({"messages": rec["messages"]})
            else:
                bad += 1
    print(f"📚 {len(rows)} geçerli örnek yüklendi ({bad} bozuk örnek atıldı)")
    return Dataset.from_list(rows).shuffle(seed=args.seed)


def main():
    args = parse_args()
    ensure_unsloth()
    from unsloth import FastLanguageModel, is_bfloat16_supported  # önce import edilmeli
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    import torch
    from trl import SFTConfig, SFTTrainer

    print(f"🖥️  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'YOK'}")

    # 1) Model (4-bit QLoRA)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model, max_seq_length=args.max_seq_len,
        dtype=None, load_in_4bit=True)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    # 2) Veri → ChatML metni
    ds = load_data(args)
    ds = ds.map(lambda b: {"text": [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                                    for m in b["messages"]]},
                batched=True, remove_columns=["messages"])
    split = ds.train_test_split(test_size=0.02, seed=args.seed) if len(ds) > 200 else None
    train_ds = split["train"] if split else ds

    # 3) Eğitim
    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer, train_dataset=train_ds,
        eval_dataset=split["test"] if split else None,
        args=SFTConfig(
            dataset_text_field="text", max_seq_length=args.max_seq_len, packing=False,
            per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs, max_steps=args.max_steps,
            learning_rate=args.lr, warmup_ratio=0.05, lr_scheduler_type="cosine",
            optim="adamw_8bit", weight_decay=0.01,
            fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),  # T4 → fp16
            logging_steps=10, save_strategy="steps", save_steps=200, save_total_limit=2,
            eval_strategy="steps" if split else "no", eval_steps=200,
            output_dir=str(Path(args.output) / "checkpoints"), seed=args.seed, report_to="none"))
    # Kayıp yalnızca asistan yanıtında (5 etiketli bölüm) hesaplanır
    trainer = train_on_responses_only(trainer,
                                      instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")
    stats = trainer.train()
    print(f"✅ Eğitim bitti: {stats.metrics}")

    # 4) Hızlı format testi
    from gokturk_format import SYSTEM_PROMPT, parse_response
    FastLanguageModel.for_inference(model)
    tests = ["3x + 7 = 25 denklemini çöz.",
             "Flask uygulamamda SQL injection'ı nasıl önlerim?",
             "nginx için en son kritik güvenlik açıkları neler?"]
    ok = 0
    for q in tests:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
        out = model.generate(input_ids=ids, max_new_tokens=768, temperature=0.6, top_p=0.95, do_sample=True)
        text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        parsed = parse_response(text)
        ok += parsed is not None
        print(f"\n{'=' * 70}\n❓ {q}\n{text}\n→ format {'✓' if parsed else '✗'}"
              + (f" | arama: {parsed['search_query']}" if parsed else ""))
    print(f"\n📐 Format uyumu: {ok}/{len(tests)}")

    # 5) Kayıt
    out = Path(args.output)
    model.save_pretrained(out / "lora"); tokenizer.save_pretrained(out / "lora")
    model.save_pretrained_merged(str(out / "merged_16bit"), tokenizer, save_method="merged_16bit")
    print(f"💾 LoRA → {out / 'lora'} | merged → {out / 'merged_16bit'}")
    for q in args.gguf:
        model.save_pretrained_gguf(str(out / "gguf"), tokenizer, quantization_method=q)
        print(f"💾 GGUF ({q}) → {out / 'gguf'}")
    if args.push_to_hub:
        tok = os.environ.get("HF_TOKEN")
        model.push_to_hub_merged(args.push_to_hub, tokenizer, save_method="merged_16bit", token=tok)
        print(f"☁️  Hub'a yüklendi: {args.push_to_hub}")


if __name__ == "__main__":
    main()
