"""GokTurk_GGUF_Export.ipynb üreticisi.  python notebooks/make_gguf_notebook.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gokturk_cot import GGUF_CHAT_TEMPLATE  # noqa: E402  — eğitimle birebir aynı sohbet biçimi

MD = []
CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": s.strip("\n")})


md(r"""
# 🐺 GökTürk2.5-3B-Thinking → GGUF (Q4_K_M + Q8_0)

Bu notebook **eğitim yapmaz**. Hugging Face'teki hazır LoRA adaptörünü 16-bit Qwen2.5-3B-Instruct ile birleştirir, llama.cpp ile GGUF'a çevirir, doğrular ve yükler.

**Kaggle ayarları (sağ panel):**
1. **Accelerator: None** (GPU gerekmez, kota harcanmaz)
2. **Internet: On**
3. **Add-ons → Secrets → `HF_TOKEN`** (yetkisi **Write** olan token) → notebook'a bağlı olsun

Sonra **Save Version → Save & Run All**. Süre ≈ 25–40 dk.

Her adım kendini doğrular; bir sorun olursa hücre **Türkçe açıklamayla durur**. Tekrar çalıştırırsanız biten adımlar atlanır.
Yükleme başarısız olsa bile GGUF dosyaları **Output** sekmesinde (`/kaggle/working/gguf`) indirilebilir kalır.
""")

code(r'''
# ⚙️ 1) AYARLAR — yalnızca burayı değiştirin
LORA_REPO   = "ALPRO2023/GokTurk2.5-3B-Thinking-LoRA"   # eğitilmiş LoRA (HF deposu veya yerel klasör)
BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"                # 16-bit taban (4-bit/bnb OLMAMALI)
GGUF_REPO   = "ALPRO2023/GokTurk2.5-3B-Thinking-GGUF"   # GGUF'ların yükleneceği depo
MODEL_NAME  = "GokTurk2.5-3B-Thinking"                  # dosya adı öneki
QUANTS      = ["Q4_K_M", "Q8_0"]                         # Q4_K_M ≈ 1.9 GB (telefon), Q8_0 ≈ 3.3 GB
PUSH        = True                                      # HF'ye yükle
PRIVATE     = False                                     # depo gizli mi
SMOKE_TEST  = True                                      # GGUF ile kısa üretim testi (hata verirse sadece uyarır)

# GökTürk sohbet şablonu: sistem mesajı yoksa eğitimdeki 6 aşamalı CoT sistem promptunu ekler.
# (Qwen'in şablonu "You are Qwen..." ekler → model 6 aşamayı başlatmaz. DEĞİŞTİRMEYİN.)
CHAT_TEMPLATE = ''' + json.dumps(GGUF_CHAT_TEMPLATE, ensure_ascii=False) + r'''

# ── Yardımcılar (değiştirmeyin) ──────────────────────────────────────────────
import os, sys, json, time, shutil, subprocess, textwrap
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").exists()
WORK = Path("/tmp/gokturk_gguf")                                   # büyük ara dosyalar
OUT  = Path("/kaggle/working/gguf") if ON_KAGGLE else Path("gguf_out").resolve()   # nihai dosyalar
LLAMA = WORK / "llama.cpp"
for d in (WORK, OUT):
    d.mkdir(parents=True, exist_ok=True)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["PIP_BREAK_SYSTEM_PACKAGES"] = "1"   # bazı Debian tabanlı imajlarda pip engeli
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Dur(RuntimeError):
    """Kullanıcıya anlaşılır mesajla durdurma."""

def run(cmd, cwd=None, env=None, tail=60):
    """Komutu çalıştırır, çıktıyı canlı gösterir; hata olursa DURUR (! komutlarının aksine)."""
    print("$", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    p = subprocess.Popen(cmd, shell=isinstance(cmd, str), cwd=cwd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    last = []
    for line in p.stdout:
        last.append(line); last = last[-tail:]
        if not line.startswith("\r") and "it/s]" not in line and "B/s]" not in line:
            print(line, end="", flush=True)
    if p.wait() != 0:
        raise Dur(f"Komut başarısız (kod {p.returncode}). Son satırlar:\n" + "".join(last[-25:]))

def retry(fn, what, tries=5, wait=10):
    for i in range(1, tries + 1):
        try:
            return fn()
        except Dur:
            raise
        except Exception as e:  # ağ hataları vb.
            if i == tries:
                raise Dur(f"{what}: {tries} denemede başarısız → {type(e).__name__}: {e}")
            print(f"⚠️ {what} başarısız ({type(e).__name__}: {str(e)[:200]}) → {wait*i}s sonra tekrar ({i}/{tries})")
            time.sleep(wait * i)

def free_gb(p):
    return shutil.disk_usage(p).free / 1e9

def need_disk(p, gb, what):
    f = free_gb(p)
    print(f"💾 {p}: {f:.1f} GB boş ({what} için ~{gb} GB gerekli)")
    if f < gb:
        raise Dur(f"Disk yetersiz: {what} için {gb} GB gerekli, {f:.1f} GB var.")

MIN_MERGED_GB = 4.0      # 3B bf16 ≈ 6.2 GB; daha küçükse bir şeyler ters gitmiştir
MIN_TENSORS = 300        # Qwen2.5-3B ≈ 434 tensör

def step(n, t):
    print("\n" + "═" * 70 + f"\n{n}) {t}\n" + "═" * 70, flush=True)

try:
    import psutil
except ImportError:
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "psutil"]); import psutil
print(f"🖥️ Kaggle={ON_KAGGLE} | CPU={os.cpu_count()} | RAM={psutil.virtual_memory().total/1e9:.1f} GB | "
      f"WORK boş={free_gb(WORK):.0f} GB | OUT boş={free_gb(OUT):.0f} GB")
assert all(q in {"Q4_K_M","Q5_K_M","Q6_K","Q8_0","Q4_0","Q5_0","Q3_K_M","Q2_K"} for q in QUANTS), "QUANTS geçersiz"
if psutil.virtual_memory().total < 14e9:
    print("⚠️ RAM 14 GB'tan az; birleştirme yavaş olabilir veya bellek yetmeyebilir.")
print("✅ Ayarlar tamam")
''')

code(r'''
# 📦 2) KURULUM — kütüphaneler + llama.cpp (≈ 5–10 dk)
step(2, "Kurulum")
# Kaggle'ın eski torchao'su (0.10) peft'i çökertir ("incompatible version of torchao"); bize gerekmiyor.
subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
# Uyumlu, sabit sürümler (torch'a dokunulmaz — Kaggle'ınki kullanılır)
PKGS = ["transformers==4.57.6", "peft==0.17.1", "huggingface_hub==0.36.2", "accelerate>=1.0,<2",
        "safetensors>=0.4.3", "sentencepiece", "protobuf", "numpy<2.4", "psutil"]
retry(lambda: run([sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", *PKGS]), "pip install", tries=3)

# llama.cpp: kaynak + yalnızca gereken araçlar (CPU derlemesi)
if not (LLAMA / "convert_hf_to_gguf.py").exists():
    shutil.rmtree(LLAMA, ignore_errors=True)
    retry(lambda: run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(LLAMA)]), "llama.cpp indirme", tries=4)
retry(lambda: run([sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", "--no-deps", str(LLAMA / "gguf-py")]), "gguf-py kurulumu", tries=3)
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "--no-warn-conflicts", "pyyaml", "tqdm"])

QUANT_BIN = LLAMA / "build/bin/llama-quantize"
SIMPLE_BIN = LLAMA / "build/bin/llama-simple"
if not QUANT_BIN.exists():
    if shutil.which("cmake") is None:
        run([sys.executable, "-m", "pip", "install", "-q", "cmake"])
    jobs = str(max(2, os.cpu_count() or 2))
    run(["cmake", "-S", str(LLAMA), "-B", str(LLAMA / "build"), "-DCMAKE_BUILD_TYPE=Release",
         "-DGGML_CUDA=OFF", "-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
         "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_EXAMPLES=ON"], tail=40)
    run(["cmake", "--build", str(LLAMA / "build"), "--target", "llama-quantize", "-j", jobs], tail=40)
if not QUANT_BIN.exists():
    raise Dur("llama-quantize derlenemedi.")
if SMOKE_TEST and not SIMPLE_BIN.exists():
    try:   # isteğe bağlı: kısa üretim testi aracı (başarısız olursa sadece uyarır)
        run(["cmake", "--build", str(LLAMA / "build"), "--target", "llama-simple", "-j", str(max(2, os.cpu_count() or 2))], tail=20)
    except Dur:
        print("⚠️ llama-simple derlenemedi; üretim testi atlanacak.")

# Doğrulama AYRI süreçte (çekirdekte eski sürümler önbellekte kalmasın)
run([sys.executable, "-c",
     "import torch, transformers, peft, huggingface_hub as h, gguf, safetensors; "
     "print('✅ torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, "
     "'| hub', h.__version__, '| gguf ok')"])
print("✅ Kurulum tamam")
''')

code(r'''
# 🔑 3) HUGGING FACE TOKEN
step(3, "Token kontrolü")
TOKEN = None
if ON_KAGGLE:
    try:
        from kaggle_secrets import UserSecretsClient
        TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print(f"ℹ️ Kaggle secret okunamadı: {type(e).__name__}")
TOKEN = (TOKEN or os.environ.get("HF_TOKEN") or "").strip().strip('"').strip("'") or None

USER = None
if TOKEN:
    import subprocess as _sp
    r = _sp.run([sys.executable, "-c",
                 "import sys,json; from huggingface_hub import whoami; "
                 "w=whoami(token=sys.argv[1]); a=(w.get('auth') or {}).get('accessToken') or {}; "
                 "print(json.dumps({'name':w.get('name'),'role':a.get('role'),'fg':a.get('fineGrained')}))", TOKEN],
                capture_output=True, text=True)
    if r.returncode == 0:
        info = json.loads(r.stdout.strip().splitlines()[-1])
        USER = info["name"]
        print(f"✅ Token geçerli → {USER} (yetki: {info['role']})")
        if info["role"] == "read":
            print("⚠️ Token READ yetkili → yükleme yapılamaz. huggingface.co/settings/tokens → Write token oluşturun.")
            PUSH = False
    else:
        print("❌ Token geçersiz:", (r.stderr or r.stdout).strip().splitlines()[-1][:300])
        TOKEN = None
if not TOKEN:
    print("⚠️ Geçerli HF_TOKEN yok → dosyalar yalnızca Output'a kaydedilecek (yükleme atlanır).\n"
          "   Düzeltmek için: Add-ons → Secrets → HF_TOKEN (Write) ekleyip notebook'a bağlayın.")
    PUSH = False
else:
    os.environ["HF_TOKEN"] = TOKEN
if PUSH and USER and not GGUF_REPO.startswith(USER + "/"):
    print(f"ℹ️ GGUF_REPO '{GGUF_REPO}' başka bir hesaba ait görünüyor; o hesapta yazma yetkiniz olmalı.")
print(f"➡️ Yükleme: {'AÇIK → ' + GGUF_REPO if PUSH else 'KAPALI'}")
''')

code(r'''
# ⬇️ 4) İNDİRME — LoRA + 16-bit taban model (≈ 6.5 GB)
step(4, "İndirme")
from huggingface_hub import snapshot_download

def fetch(repo, sub, patterns):
    if Path(repo).is_dir():
        print(f"📁 yerel: {repo}")
        return Path(repo)
    dst = WORK / sub
    def _dl():
        return Path(snapshot_download(repo, local_dir=str(dst), allow_patterns=patterns, token=TOKEN))
    p = retry(_dl, f"{repo} indirme", tries=6, wait=15)
    print(f"✅ {repo} → {p}")
    return p

LORA_DIR = fetch(LORA_REPO, "lora", ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"])
if not (LORA_DIR / "adapter_config.json").exists():
    raise Dur(f"{LORA_REPO} içinde adapter_config.json yok — doğru LoRA deposu mu?")
if not any((LORA_DIR / f).exists() for f in ("adapter_model.safetensors", "adapter_model.bin")):
    raise Dur(f"{LORA_REPO} içinde adapter_model.safetensors yok — eğitim yüklemesi tamamlanmamış olabilir.")

MERGED = WORK / "merged_16bit"
if (MERGED / "config.json").exists() and list(MERGED.glob("*.safetensors")):
    print("⏭️ Birleştirilmiş model zaten var; taban indirme atlandı.")
    BASE_DIR = None
else:
    need_disk(WORK, 14, "taban + birleştirme")
    BASE_DIR = fetch(BASE_MODEL, "base", ["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja", "tokenizer*", "merges.txt", "vocab.json"])
    bcfg = json.loads((BASE_DIR / "config.json").read_text())
    if bcfg.get("quantization_config"):
        raise Dur(f"{BASE_MODEL} kuantize (4-bit/bnb) bir model! llama.cpp dönüştüremez. 16-bit taban kullanın (ör. Qwen/Qwen2.5-3B-Instruct).")
    print(f"✅ Taban: {bcfg.get('architectures')} | katman={bcfg.get('num_hidden_layers')} | gizli={bcfg.get('hidden_size')} | dtype={bcfg.get('torch_dtype')}")

# adapter_config'i temizle: yeni peft sürümlerinin alanları (kasa_config vb.) + yerel önbellek yolu
acfg_p = LORA_DIR / "adapter_config.json"
acfg = json.loads(acfg_p.read_text())
import dataclasses, subprocess as _sp
fields = json.loads(_sp.run([sys.executable, "-c",
    "import json,dataclasses; from peft import LoraConfig; print(json.dumps([f.name for f in dataclasses.fields(LoraConfig)]))"],
    capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1])
dropped = [k for k in acfg if k not in fields]
clean = {k: v for k, v in acfg.items() if k in fields}
clean["base_model_name_or_path"] = BASE_MODEL
clean["inference_mode"] = True
if acfg.get("peft_type", "LORA") != "LORA":
    raise Dur(f"Adaptör tipi LORA değil: {acfg.get('peft_type')}")
FIXED_LORA = WORK / "lora_clean"
FIXED_LORA.mkdir(exist_ok=True)
for f in LORA_DIR.iterdir():
    if f.name.startswith("adapter_model"):
        dst = FIXED_LORA / f.name
        if not dst.exists():
            shutil.copy2(f, dst)
(FIXED_LORA / "adapter_config.json").write_text(json.dumps(clean, indent=2))
print(f"✅ LoRA: r={clean.get('r')} alpha={clean.get('lora_alpha')} hedef={clean.get('target_modules')}"
      + (f" | temizlenen alanlar: {dropped}" if dropped else ""))
print(f"   eski taban kaydı: {acfg.get('base_model_name_or_path')} → {BASE_MODEL}")
''')

code(r'''
# 🔗 5) BİRLEŞTİRME — LoRA + 16-bit taban (CPU, ≈ 5–10 dk)
step(5, "Birleştirme (merge)")
MERGE_PY = WORK / "merge.py"
MERGE_PY.write_text(textwrap.dedent("""
    import sys, json, torch
    from pathlib import Path
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base, lora, out, tpl = map(Path, sys.argv[1:5])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = AutoModelForCausalLM.from_pretrained(str(base), dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    assert not getattr(model.config, "quantization_config", None), "taban kuantize!"

    # Birleştirme öncesi bir ağırlığın kopyası (LoRA'nın gerçekten uygulandığını kanıtlamak için)
    probe = next(n for n, _ in model.named_parameters() if n.endswith("self_attn.q_proj.weight"))
    before = dict(model.named_parameters())[probe].detach().clone()

    # Dosyadaki LoRA anahtar sayısı ↔ modele yüklenen LoRA modülleri
    sf = lora / "adapter_model.safetensors"
    if sf.exists():
        with safe_open(str(sf), "pt") as f:
            keys = list(f.keys())
    else:
        keys = list(torch.load(lora / "adapter_model.bin", map_location="cpu").keys())
    n_file_a = sum(".lora_A." in k for k in keys)

    pm = PeftModel.from_pretrained(model, str(lora), is_trainable=False)
    named = dict(pm.named_parameters())
    lora_a = [n for n in named if ".lora_A." in n]
    lora_b = [n for n in named if ".lora_B." in n]
    nz_b = sum(float(named[n].abs().sum()) > 0 for n in lora_b)
    print(f"LoRA: dosyada {n_file_a} A-anahtarı | modelde {len(lora_a)} A-modülü | sıfır olmayan B: {nz_b}/{len(lora_b)}")
    assert n_file_a > 0, "adaptör dosyası boş"
    assert len(lora_a) == n_file_a, f"LoRA anahtarları eşleşmedi ({n_file_a} ≠ {len(lora_a)}) — yanlış taban model?"
    assert nz_b > len(lora_b) * 0.9, "LoRA B ağırlıkları sıfır → adaptör yüklenmemiş"

    merged = pm.merge_and_unload()
    after = dict(merged.named_parameters())[probe].detach()
    delta = float((after.float() - before.float()).abs().max())
    print(f"Birleştirme etkisi ({probe}): max |Δ| = {delta:.3e}")
    assert delta > 0, "birleştirme ağırlıkları değiştirmedi"

    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True, max_shard_size="2GB")
    tok = AutoTokenizer.from_pretrained(str(base))
    tok.chat_template = tpl.read_text(encoding="utf-8")   # Qwen şablonu yerine GökTürk şablonu
    tok.save_pretrained(str(out))
    cfg = json.loads((out / "config.json").read_text())
    cfg.pop("quantization_config", None)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in ("generation_config.json",):
        if (base / f).exists() and not (out / f).exists():
            (out / f).write_text((base / f).read_text())
    print("OK")
"""))

if (MERGED / "config.json").exists() and list(MERGED.glob("*.safetensors")):
    print("⏭️ Birleştirilmiş model zaten var.")
else:
    shutil.rmtree(MERGED, ignore_errors=True)
    t = time.time()
    TPL = WORK / "gokturk_chat_template.jinja"
    TPL.write_text(CHAT_TEMPLATE, encoding="utf-8")
    run([sys.executable, str(MERGE_PY), str(BASE_DIR), str(FIXED_LORA), str(MERGED), str(TPL)])
    print(f"⏱️ {time.time()-t:.0f}s")
mcfg = json.loads((MERGED / "config.json").read_text())
if mcfg.get("quantization_config"):
    raise Dur("Birleştirilmiş model kuantize görünüyor — beklenmeyen durum.")
size = sum(f.stat().st_size for f in MERGED.glob("*.safetensors")) / 1e9
print(f"✅ 16-bit birleştirilmiş model: {size:.2f} GB → {MERGED}")
if size < MIN_MERGED_GB:
    raise Dur(f"Birleştirilmiş model beklenenden küçük ({size:.2f} GB < {MIN_MERGED_GB} GB).")
if BASE_DIR is not None and str(BASE_DIR).startswith(str(WORK)):
    shutil.rmtree(BASE_DIR, ignore_errors=True)   # disk aç
    print("🧹 taban indirmesi silindi (disk)")
''')

code(r'''
# 🔄 6) GGUF DÖNÜŞTÜRME + KUANTİZASYON (≈ 5–15 dk)
step(6, "GGUF")
F16 = WORK / f"{MODEL_NAME}-BF16.gguf"
if not F16.exists():
    need_disk(WORK, 8, "BF16 GGUF")
    tmp = F16.with_suffix(".part")
    run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(MERGED),
         "--outtype", "bf16", "--outfile", str(tmp)], tail=40)
    tmp.rename(F16)
print(f"✅ {F16.name}: {F16.stat().st_size/1e9:.2f} GB")

GGUFS = []
for q in QUANTS:
    dst = OUT / f"{MODEL_NAME}-{q}.gguf"
    if not dst.exists():
        need_disk(OUT, 4, q)
        tmp = WORK / (dst.name + ".part")
        run([str(QUANT_BIN), str(F16), str(tmp), q, str(os.cpu_count() or 2)], tail=15)
        shutil.move(str(tmp), str(dst))
    print(f"✅ {dst.name}: {dst.stat().st_size/1e9:.2f} GB")
    GGUFS.append(dst)
''')

code(r'''
# 🔍 7) DOĞRULAMA — başlık, tensörler, sohbet şablonu (+ kısa üretim testi)
step(7, "Doğrulama")
CHECK_PY = WORK / "check.py"
CHECK_PY.write_text(textwrap.dedent("""
    import sys, json
    from gguf import GGUFReader
    r = GGUFReader(sys.argv[1])
    def field(k):
        f = r.fields.get(k)
        if f is None: return None
        v = f.parts[f.data[0]]
        if v.dtype.kind == "u" and v.itemsize == 1:
            return bytes(v).decode("utf-8", "replace")
        return v.tolist()[0] if len(v) == 1 else v.tolist()
    arch = field("general.architecture")
    tmpl = field("tokenizer.chat_template") or ""
    print(json.dumps({"arch": arch, "tensors": len(r.tensors), "ftype": field("general.file_type"),
                      "chat_template": bool(tmpl), "gokturk_template": "GökTürk" in tmpl and "STEP 6" in tmpl}))
"""))
ok = True
for g in GGUFS:
    res = subprocess.run([sys.executable, str(CHECK_PY), str(g)], capture_output=True, text=True)
    if res.returncode != 0:
        raise Dur(f"{g.name} okunamadı:\n{res.stderr[-1500:]}")
    info = json.loads(res.stdout.strip().splitlines()[-1])
    print(f"   {g.name}: {info}")
    if info["arch"] != "qwen2" or info["tensors"] < MIN_TENSORS or not info["gokturk_template"]:
        raise Dur(f"{g.name} doğrulamayı geçemedi: {info}")
print("✅ GGUF başlıkları geçerli (qwen2, GökTürk 6 aşamalı sohbet şablonu gömülü)")

if SMOKE_TEST and SIMPLE_BIN.exists():
    import jinja2   # gerçek kullanımdaki gibi: GökTürk şablonu + otomatik sistem promptu
    prompt = jinja2.Environment().from_string(CHAT_TEMPLATE).render(
        messages=[{"role": "user", "content": "Bir tren saatte 80 km hızla 3 saat gidiyor. Kaç km yol alır?"}],
        add_generation_prompt=True)
    try:
        r = subprocess.run([str(SIMPLE_BIN), "-m", str(GGUFS[0]), "-n", "400", prompt],
                           capture_output=True, text=True, timeout=600)
        txt = (r.stdout or "").split("<|im_start|>assistant")[-1][:2500]
        print(f"🧪 Üretim testi ({GGUFS[0].name}):\n{txt}")
        if r.returncode != 0:
            print("⚠️ test süreci sıfırdan farklı kodla çıktı:", (r.stderr or "")[-400:])
    except Exception as e:
        print(f"⚠️ Üretim testi atlandı: {type(e).__name__}: {e}")
''')

code(r'''
# ☁️ 8) HUGGING FACE'E YÜKLEME
step(8, "Yükleme")
card = f"""---
language: [tr]
license: apache-2.0
base_model: {BASE_MODEL}
tags: [gguf, llama.cpp, turkish, reasoning, qwen2]
---
# {MODEL_NAME} — GGUF

Türkçe akıl yürütme modeli (6 aşamalı `<thought>` düşünce zinciri). Taban: `{BASE_MODEL}`, LoRA: `{LORA_REPO}`.

| Dosya | Boyut |
|---|---|
""" + "\n".join(f"| `{g.name}` | {g.stat().st_size/1e9:.2f} GB |" for g in GGUFS) + f"""

Kullanım (llama.cpp):
```bash
llama-cli -m {GGUFS[0].name} -cnv
```
Telefon: PocketPal / Layla → **{GGUFS[0].name}** (≈2.5 GB RAM).
"""
(OUT / "README.md").write_text(card, encoding="utf-8")

if not PUSH:
    print("⏭️ Yükleme kapalı/atlandı. Dosyalar: Output → gguf/")
else:
    from huggingface_hub import HfApi
    api = HfApi(token=TOKEN)
    retry(lambda: api.create_repo(GGUF_REPO, repo_type="model", private=PRIVATE, exist_ok=True), "depo oluşturma")
    for f in [*GGUFS, OUT / "README.md"]:
        retry(lambda f=f: api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=GGUF_REPO,
                                          commit_message=f"Upload {f.name}"),
              f"{f.name} yükleme", tries=6, wait=20)
        print(f"   ✅ {f.name}")
    remote = {s.rfilename: s.size for s in api.model_info(GGUF_REPO, files_metadata=True).siblings}
    for g in GGUFS:
        if remote.get(g.name) != g.stat().st_size:
            raise Dur(f"{g.name} uzak boyutu uyuşmuyor ({remote.get(g.name)} ≠ {g.stat().st_size})")
    print(f"🎉 Yüklendi ve boyutlar doğrulandı → https://huggingface.co/{GGUF_REPO}")
''')

code(r'''
# 📋 9) ÖZET
step(9, "Özet")
for f in sorted(OUT.iterdir()):
    print(f"   {f.name:45s} {f.stat().st_size/1e9:6.2f} GB" if f.suffix == ".gguf" else f"   {f.name}")
print(f"\n📂 Kaggle: sağ panel → Output → gguf/  (indirilebilir)")
if PUSH:
    print(f"🌐 https://huggingface.co/{GGUF_REPO}")
print("\n🐺 GökTürk GGUF hazır!")
''')

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"c{i:02d}"
p = Path(__file__).with_name("GokTurk_GGUF_Export.ipynb")
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("yazıldı:", p)
