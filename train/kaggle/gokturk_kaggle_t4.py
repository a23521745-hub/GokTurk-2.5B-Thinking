# %% [markdown]
# # 🐺 GökTürk-2.5B-Thinking — Kaggle T4 QLoRA Eğitimi (Unsloth)
#
# **Kaggle ayarları (sağ panel):**
# - *Accelerator*: **GPU T4 x2** (Unsloth tek GPU kullanır, ikincisi boşta kalır)
# - *Internet*: **On**
# - *Add-ons → Secrets*: `HF_TOKEN` adlı secret ekleyin ve bu notebook'a bağlayın (write yetkili token)
#
# **Arka planda çalıştırma:** Sağ üstte *Save Version → Save & Run All (Commit)*.
# Notebook hiçbir etkileşim beklemez; tarayıcıyı kapatabilirsiniz. Çıktılar `/kaggle/working/` altında
# kalır ve GGUF otomatik olarak Hugging Face Hub'a yüklenir.
#
# Akış: kurulum → veri → 4-bit model + LoRA → SFT → test → Q4_K_M GGUF → Hugging Face Hub

# %%
# ============================================================================
# 0) AYARLAR — yalnızca bu hücreyi düzenlemeniz yeterli
# ============================================================================
import os

# NOT: Qwen2.5 ailesinde 2.5B boyutunda bir model YOKTUR (0.5B, 1.5B, 3B, 7B ...).
# "GökTürk-2.5B" proje adıdır; T4'e en uygun gerçek taban model 3B-Instruct (4-bit)'tir.
# Daha hızlı eğitim için: "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
FALLBACK_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

MODEL_DISPLAY_NAME = "GökTürk-2.5B-Thinking"
MODEL_SLUG = "GokTurk-2.5B-Thinking"       # HF repo adlarında Türkçe karakter kullanılamaz
HF_REPO_NAME = f"{MODEL_SLUG}-GGUF"         # → <kullanıcı_adınız>/GokTurk-2.5B-Thinking-GGUF
HF_PRIVATE = False                           # True: özel repo

MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
LORA_R = 16
LORA_ALPHA = 16

NUM_EPOCHS = 1          # 3000 örnek ≈ 375 adım ≈ 45–70 dk (3B, T4)
MAX_STEPS = -1          # >0 verilirse NUM_EPOCHS yerine kullanılır (hızlı deneme: 60)
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM = 4
SYNTH_SAMPLES = 3000    # /kaggle/input'ta JSONL yoksa üretilecek sentetik örnek sayısı
SEED = 3407

GGUF_QUANT = "q4_k_m"

# --- Dizinler ---------------------------------------------------------------
WORK_DIR = "/kaggle/working"                 # kalıcı çıktılar (20 GB sınırı!)
SCRATCH_DIR = "/tmp/gokturk"                 # büyük ara dosyalar (merge, F16 GGUF) → kota dışı
DATA_DIR = f"{WORK_DIR}/data"
OUTPUT_DIR = f"{WORK_DIR}/outputs"
LORA_DIR = f"{OUTPUT_DIR}/lora_adapter"
FINAL_GGUF = f"{OUTPUT_DIR}/{MODEL_SLUG}-{GGUF_QUANT.upper()}.gguf"
REPO_RAW = ("https://raw.githubusercontent.com/a23521745-hub/GokTurk-2.5B-Thinking/"
            "arena/01a0d4d3-gokturk-2-5b-thinking")

for d in (WORK_DIR, SCRATCH_DIR, DATA_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)
os.chdir(WORK_DIR)

# --- Ortam değişkenleri (torch import edilmeden ÖNCE) -----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"      # T4 x2'de tek GPU: Unsloth çoklu GPU'da hata verebilir
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HOME"] = f"{SCRATCH_DIR}/hf_cache"   # model önbelleği /kaggle/working kotasını doldurmasın
print("✓ Ayarlar yüklendi. Çalışma dizini:", os.getcwd())

# %%
# ============================================================================
# 1) KURULUM — unsloth, triton, xformers, huggingface_hub
# ============================================================================
import importlib.util
import subprocess
import sys


def pip(*args):
    print("$ pip install", " ".join(args))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])


# Unsloth; uyumlu torch / triton / xformers / trl / peft / bitsandbytes sürümlerini kendisi çeker.
pip("--upgrade", "unsloth", "unsloth_zoo")
pip("--upgrade", "huggingface_hub", "hf_transfer")

# triton ve xformers eksikse ekle (--no-deps: Kaggle'ın torch sürümünü BOZMAMAK için)
for pkg in ("triton", "xformers"):
    if importlib.util.find_spec(pkg) is None:
        try:
            pip("--no-deps", pkg)
        except subprocess.CalledProcessError:
            print(f"⚠️  {pkg} kurulamadı — Unsloth yedek çekirdeklerle devam eder.")

import importlib.metadata as md
for pkg in ("unsloth", "torch", "transformers", "trl", "peft", "bitsandbytes", "triton", "xformers", "huggingface_hub"):
    try:
        print(f"  {pkg:16s} {md.version(pkg)}")
    except md.PackageNotFoundError:
        print(f"  {pkg:16s} (yok)")

# %%
# ============================================================================
# 2) HUGGING FACE TOKEN — Kaggle Secrets
# ============================================================================
HF_TOKEN = os.environ.get("HF_TOKEN")
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN") or HF_TOKEN
except Exception as e:  # secret bağlı değilse
    print(f"ℹ️  Kaggle secret okunamadı ({type(e).__name__}).")

HF_USER = None
if HF_TOKEN:
    from huggingface_hub import HfApi, login
    login(token=HF_TOKEN, add_to_git_credential=False)
    HF_USER = HfApi(token=HF_TOKEN).whoami()["name"]
    print(f"✓ Hugging Face girişi: {HF_USER}")
else:
    print("⚠️  HF_TOKEN yok → eğitim yapılacak, ancak Hub'a yükleme ATLANACAK.")

# %%
# ============================================================================
# 3) VERİ — /kaggle/input'taki JSONL'ler veya sentetik üretim
# ============================================================================
import glob
import json
import re
import urllib.request

TAGS = ["think", "plan", "search_query", "verify", "output"]
_blocks = [rf"<{t}>(.*?)</{t}>" for t in TAGS]
_blocks[2] += r"(?:\s*<search_results>.*?</search_results>)?"
FORMAT_RE = re.compile(r"^\s*" + r"\s*".join(_blocks) + r"\s*$", re.DOTALL)

SYSTEM_PROMPT = (
    "Sen GökTürk-2.5B-Thinking adlı Türkçe bir yapay zekâ asistanısın. "
    "Her yanıtında sırasıyla <think>, <plan>, <search_query>, <verify> ve <output> "
    "etiketlerini kullan. Güncel veya doğrulanması gereken bilgi gerekiyorsa "
    "<search_query> içine kısa bir web sorgusu yaz; gerekmiyorsa YOK yaz. "
    "Nihai yanıtı yalnızca <output> içinde, açık ve doğru bir Türkçeyle ver."
)


def is_valid(text: str) -> bool:
    m = FORMAT_RE.match(text or "")
    return bool(m) and all(g.strip() for g in m.groups())


# 3a) Kaggle Dataset olarak eklenmiş JSONL dosyaları (ör. distill çıktıları)
data_files = sorted(f for f in glob.glob("/kaggle/input/**/*.jsonl", recursive=True)
                    if not f.endswith("template.jsonl"))

# 3b) Yoksa: GitHub'daki üreticiyle sentetik veri
if not data_files:
    gen_dir = f"{SCRATCH_DIR}/scripts"
    os.makedirs(gen_dir, exist_ok=True)
    for name in ("gokturk_format.py", "generate_dataset.py"):
        urllib.request.urlretrieve(f"{REPO_RAW}/scripts/{name}", f"{gen_dir}/{name}")
    out = f"{DATA_DIR}/gokturk_synth.jsonl"
    subprocess.check_call([sys.executable, f"{gen_dir}/generate_dataset.py", "synthetic",
                           "-n", str(SYNTH_SAMPLES), "-o", out, "--seed", str(SEED)])
    data_files = [out]

rows, dropped = [], 0
for f in data_files:
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                msgs = json.loads(line)["messages"]
            except (json.JSONDecodeError, KeyError):
                dropped += 1
                continue
            if msgs and msgs[-1]["role"] == "assistant" and is_valid(msgs[-1]["content"]):
                if msgs[0]["role"] != "system":
                    msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
                rows.append({"messages": msgs})
            else:
                dropped += 1

assert len(rows) >= 10, f"Yeterli geçerli örnek yok ({len(rows)}). Veri dosyalarını kontrol edin."
print(f"✓ {len(rows)} geçerli örnek ({dropped} atıldı) — kaynaklar: {data_files}")

# %%
# ============================================================================
# 4) MODEL + TOKENIZER + LoRA
# ============================================================================
from unsloth import FastLanguageModel, is_bfloat16_supported  # unsloth, transformers'tan ÖNCE import edilmeli
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import torch

assert torch.cuda.is_available(), "GPU bulunamadı! Kaggle'da Accelerator → GPU T4 seçin."
gpu = torch.cuda.get_device_properties(0)
print(f"🖥️  {gpu.name} — {gpu.total_memory / 1024**3:.1f} GB")

try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,                 # T4 → otomatik float16
        load_in_4bit=LOAD_IN_4BIT, token=HF_TOKEN)
except Exception as e:  # model adı hatalıysa (ör. var olmayan 2.5B) yedek modele geç
    print(f"⚠️  '{BASE_MODEL}' yüklenemedi ({e}). Yedek: {FALLBACK_MODEL}")
    BASE_MODEL = FALLBACK_MODEL
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH, dtype=None,
        load_in_4bit=LOAD_IN_4BIT, token=HF_TOKEN)

tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")   # ChatML

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,                                   # Unsloth'ta 0 = optimize edilmiş yol
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",             # T4 VRAM tasarrufu (~%30)
    random_state=SEED,
    use_rslora=False,
    loftq_config=None,
)
model.print_trainable_parameters()

# %%
# ============================================================================
# 5) VERİYİ CHATML METNİNE ÇEVİR
# ============================================================================
from datasets import Dataset

ds = Dataset.from_list(rows).shuffle(seed=SEED)
ds = ds.map(lambda b: {"text": [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                                for m in b["messages"]]},
            batched=True, remove_columns=["messages"])

# Çok uzun örnekleri at (kesilen <output> formatı bozar)
ds = ds.filter(lambda x: len(tokenizer(x["text"]).input_ids) <= MAX_SEQ_LENGTH)
print(f"✓ Eğitim örneği: {len(ds)}")
print(ds[0]["text"][:600])

# %%
# ============================================================================
# 6) EĞİTİM — SFTTrainer (trl sürümlerine dayanıklı kurulum)
# ============================================================================
import inspect
from trl import SFTConfig, SFTTrainer

cfg = dict(
    output_dir=f"{SCRATCH_DIR}/checkpoints",
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,     # eski trl
    max_length=MAX_SEQ_LENGTH,         # yeni trl
    packing=False,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    max_steps=MAX_STEPS,
    learning_rate=LEARNING_RATE,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    weight_decay=0.01,
    fp16=not is_bfloat16_supported(),  # T4 bf16 desteklemez → fp16
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=1,
    seed=SEED,
    report_to="none",
    dataset_num_proc=2,
)
valid = set(inspect.signature(SFTConfig.__init__).parameters)
sft_config = SFTConfig(**{k: v for k, v in cfg.items() if k in valid})

trainer_kwargs = dict(model=model, train_dataset=ds, args=sft_config)
if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer
trainer = SFTTrainer(**trainer_kwargs)

# Kayıp yalnızca asistan yanıtı (5 etiketli bölüm) üzerinde hesaplanır
trainer = train_on_responses_only(trainer,
                                  instruction_part="<|im_start|>user\n",
                                  response_part="<|im_start|>assistant\n")

torch.cuda.reset_peak_memory_stats()
stats = trainer.train()
print(f"✅ Eğitim bitti — {stats.metrics.get('train_runtime', 0) / 60:.1f} dk, "
      f"kayıp {stats.metrics.get('train_loss', float('nan')):.4f}, "
      f"tepe VRAM {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

# LoRA adaptörünü hemen kaydet (GGUF adımı başarısız olsa bile emek kaybolmasın)
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"💾 LoRA → {LORA_DIR}")

# %%
# ============================================================================
# 7) HIZLI TEST — format uyumu
# ============================================================================
FastLanguageModel.for_inference(model)
tests = [
    "3x + 7 = 25 denklemini çöz.",
    "Flask uygulamamda SQL injection'ı nasıl önlerim?",
    "OpenSSH için en son kritik CVE'ler hangileri?",
]
ok = 0
for q in tests:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=700, temperature=0.6, top_p=0.95,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    good = is_valid(text)
    ok += good
    print(f"\n{'=' * 70}\n❓ {q}\n{text}\n→ format {'✓' if good else '✗'}")
print(f"\n📐 Format uyumu: {ok}/{len(tests)}")

# %%
# ============================================================================
# 8) MERGE + Q4_K_M GGUF
# ============================================================================
import gc
import shutil

# Eğitim nesnelerini bırak → dönüştürme için RAM/VRAM aç
del trainer
gc.collect()
torch.cuda.empty_cache()
shutil.rmtree(f"{SCRATCH_DIR}/checkpoints", ignore_errors=True)

gguf_build_dir = f"{SCRATCH_DIR}/gguf_build"
# Unsloth: LoRA'yı 16-bit ağırlıklarla birleştirir → llama.cpp ile F16 GGUF → Q4_K_M
# Unsloth llama.cpp'yi çalışma dizinine klonlar → /kaggle/working çıktısını şişirmesin diye /tmp'de çalış
os.chdir(SCRATCH_DIR)
try:
    model.save_pretrained_gguf(gguf_build_dir, tokenizer, quantization_method=GGUF_QUANT)
finally:
    os.chdir(WORK_DIR)

# Unsloth sürümüne göre dosya adı/dizini değişebilir → her yerde ara
candidates = [p for pat in (f"{gguf_build_dir}*/**/*.gguf", f"{WORK_DIR}/*.gguf", f"{SCRATCH_DIR}/**/*.gguf")
              for p in glob.glob(pat, recursive=True)]
q_files = [p for p in candidates if GGUF_QUANT.replace("_", "").lower() in os.path.basename(p).replace("_", "").replace("-", "").lower()]
assert q_files, f"Q4_K_M GGUF bulunamadı! Bulunanlar: {candidates}"
src = max(q_files, key=os.path.getmtime)
shutil.move(src, FINAL_GGUF)

# Ara dosyaları sil (F16 GGUF, merged ağırlıklar) — disk temizliği
for p in candidates:
    if os.path.exists(p) and os.path.abspath(p) != os.path.abspath(FINAL_GGUF):
        os.remove(p)
shutil.rmtree(gguf_build_dir, ignore_errors=True)
print(f"💾 GGUF → {FINAL_GGUF} ({os.path.getsize(FINAL_GGUF) / 1024**3:.2f} GB)")

# %%
# ============================================================================
# 9) HUGGING FACE HUB'A YÜKLE (HfApi)
# ============================================================================
import time

MODEL_CARD = f"""---
language: [tr]
license: apache-2.0
base_model: {BASE_MODEL}
tags: [gguf, qwen2.5, unsloth, turkish, reasoning, cybersecurity, chain-of-thought]
---
# {MODEL_DISPLAY_NAME} (GGUF)

{BASE_MODEL} üzerine Unsloth QLoRA (r={LORA_R}, alpha={LORA_ALPHA}) ile ince ayar yapılmış,
5 aşamalı akıl yürütme formatı kullanan Türkçe model:
`<think>` → `<plan>` → `<search_query>` → `<verify>` → `<output>`

| Dosya | Kuantizasyon |
|---|---|
| `{os.path.basename(FINAL_GGUF)}` | {GGUF_QUANT.upper()} |

```bash
llama-server -m {os.path.basename(FINAL_GGUF)} -c 4096 --port 8080
```
Kaynak: https://github.com/a23521745-hub/GokTurk-2.5B-Thinking
"""


def with_retry(fn, tries=3, wait=30):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            print(f"  deneme {i}/{tries} başarısız: {e}")
            if i == tries:
                raise
            time.sleep(wait)


if HF_TOKEN and HF_USER:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    repo_id = f"{HF_USER}/{HF_REPO_NAME}"
    api.create_repo(repo_id, repo_type="model", private=HF_PRIVATE, exist_ok=True)
    print(f"⬆️  {repo_id} deposuna yükleniyor...")
    with_retry(lambda: api.upload_file(path_or_fileobj=FINAL_GGUF, path_in_repo=os.path.basename(FINAL_GGUF),
                                       repo_id=repo_id, commit_message=f"{GGUF_QUANT.upper()} GGUF"))
    with_retry(lambda: api.upload_folder(folder_path=LORA_DIR, path_in_repo="lora_adapter",
                                         repo_id=repo_id, commit_message="LoRA adaptörü"))
    api.upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md", repo_id=repo_id,
                    commit_message="Model kartı")
    print(f"✅ Yüklendi: https://huggingface.co/{repo_id}")
else:
    print("⏭️  HF_TOKEN olmadığı için yükleme atlandı. GGUF, Kaggle 'Output' sekmesinden indirilebilir.")

# %%
# ============================================================================
# 10) ÖZET
# ============================================================================
print("\n📦 /kaggle/working içeriği:")
for root, _, files in os.walk(OUTPUT_DIR):
    for f in files:
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, WORK_DIR):60s} {os.path.getsize(p) / 1024**2:9.1f} MB")
print(f"\n🐺 {MODEL_DISPLAY_NAME} hazır.")
