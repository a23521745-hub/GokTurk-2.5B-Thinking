"""Ortam yardımcıları: Kaggle / Colab / yerel algılama, secret yükleme, GPU kontrolü,
bağımlılık kurulumu (sürüm kilitli) ve ağ kesintilerine dayanıklı yeniden deneme.

ÖNEMLİ: Bu modül torch'u import ETMEZ; `setup_gpu_env()` torch'tan önce çağrılmalıdır.
"""
from __future__ import annotations

import functools
import importlib.util
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

ROOT = Path(__file__).resolve().parent
SECRET_NAMES = ("HF_TOKEN", "TAVILY_API_KEY", "GEMINI_API_KEY")
T = TypeVar("T")


# ---------------------------------------------------------------------------
# Ortam
# ---------------------------------------------------------------------------
def detect_env() -> str:
    if Path("/kaggle/working").is_dir():
        return "kaggle"
    if importlib.util.find_spec("google") and importlib.util.find_spec("google.colab"):
        return "colab"
    return "local"


def default_dirs() -> tuple[Path, Path]:
    """(kalıcı çıktı dizini, büyük geçici dosyalar için scratch)."""
    env = detect_env()
    if env == "kaggle":   # /kaggle/working 20 GB ve "Output" olarak saklanır → ara dosyalar /tmp'ye
        return Path("/kaggle/working/outputs"), Path("/tmp/gokturk")
    if env == "colab":
        return Path("/content/outputs"), Path("/content/tmp")
    return ROOT / "outputs", ROOT / "outputs" / "tmp"


def load_secrets(names=SECRET_NAMES) -> dict[str, bool]:
    """Kaggle Secrets / Colab userdata → os.environ. Değerleri asla yazdırmaz."""
    env = detect_env()
    for n in names:
        if os.environ.get(n):
            continue
        val = None
        try:
            if env == "kaggle":
                from kaggle_secrets import UserSecretsClient
                val = UserSecretsClient().get_secret(n)
            elif env == "colab":
                from google.colab import userdata
                val = userdata.get(n)
        except Exception:  # noqa: BLE001 — secret tanımlı/bağlı değil
            val = None
        if val:
            os.environ[n] = val.strip()
    status = {n: bool(os.environ.get(n)) for n in names}
    print("🔑 Secrets: " + ", ".join(f"{n}={'✓' if ok else '✗'}" for n, ok in status.items()))
    return status


def setup_gpu_env():
    """torch import edilmeden ÖNCE çağırın."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # Kaggle T4 x2 → Unsloth tek GPU
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Hugging Face ağ zaman aşımları (yavaş/kesintili bağlantılar için)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    _, scratch = default_dirs()
    os.environ.setdefault("HF_HOME", str(scratch / "hf_cache"))   # model önbelleği kotayı doldurmasın
    if "torch" in sys.modules:
        print("⚠️  torch zaten import edilmiş; CUDA_VISIBLE_DEVICES etkisiz olabilir.", file=sys.stderr)


def gpu_available() -> tuple[bool, str]:
    """torch import etmeden GPU kontrolü (nvidia-smi)."""
    if not shutil.which("nvidia-smi"):
        return False, "nvidia-smi bulunamadı"
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                      text=True, timeout=20).strip()
        return bool(out), out.splitlines()[0] if out else "GPU yok"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def require_gpu():
    ok, info = gpu_available()
    if ok:
        print(f"🖥️  GPU: {info}")
        return
    msg = ("\n❌ GPU BULUNAMADI — eğitim CPU'da yapılamaz (3B model için günler sürer).\n"
           "   Kaggle: sağ panel → Settings → Accelerator → 'GPU T4 x2' seçin, oturumu yeniden başlatın.\n"
           "   Colab : Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU.\n"
           "   (Haftalık Kaggle GPU kotanız bittiyse 'GPU' seçeneği pasif görünür.)\n")
    raise SystemExit(msg)


# ---------------------------------------------------------------------------
# Kurulum
# ---------------------------------------------------------------------------
def pip_install(*pkgs: str, constraints: bool = True):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *pkgs]
    if constraints and (ROOT / "constraints.txt").exists():
        cmd += ["-c", str(ROOT / "constraints.txt")]
    subprocess.check_call(cmd)


def ensure_unsloth():
    if importlib.util.find_spec("unsloth") is None:
        print("📦 unsloth kuruluyor (2-4 dk)...")
        pip_install("-r", str(ROOT / "requirements-train.txt"))
    # datasets 5.x gibi uyumsuz sürümleri kontrol et
    try:
        import importlib.metadata as md
        v = md.version("datasets")
        if int(v.split(".")[0]) >= 5 or tuple(map(int, v.split(".")[:2])) >= (4, 4):
            print(f"🔧 datasets {v} Unsloth ile uyumsuz → düzeltiliyor")
            pip_install("datasets>=3.4.1,<4.4.0")
    except Exception:  # noqa: BLE001
        pass
    for pkg in ("triton", "xformers"):   # eksikse ekle; --no-deps: torch sürümünü bozmamak için
        if importlib.util.find_spec(pkg) is None:
            try:
                pip_install("--no-deps", pkg, constraints=False)
            except subprocess.CalledProcessError:
                print(f"ℹ️  {pkg} kurulamadı; Unsloth yedek çekirdeklerle çalışır.")


# ---------------------------------------------------------------------------
# Ağ dayanıklılığı
# ---------------------------------------------------------------------------
def retry(fn: Callable[[], T], *, tries: int = 5, base_delay: float = 5.0, max_delay: float = 120.0,
          what: str = "işlem") -> T:
    """Üstel geri çekilme + jitter ile yeniden dener. Kalıcı hatalarda (401/403/404) hemen bırakır."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            text = f"{type(e).__name__}: {e}"
            permanent = any(s in text for s in ("401", "403", "404", "RepositoryNotFound", "GatedRepo",
                                                "Invalid user token", "EntryNotFound"))
            if permanent or attempt == tries:
                print(f"❌ {what} başarısız ({attempt}/{tries}): {text[:400]}", file=sys.stderr)
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
            print(f"⚠️  {what} hata verdi ({attempt}/{tries}): {text[:200]} → {delay:.0f} sn sonra tekrar",
                  file=sys.stderr)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def retrying(what: str, tries: int = 5):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            return retry(lambda: fn(*a, **kw), tries=tries, what=what)
        return wrapper
    return deco


def prefetch_model(repo_id: str, allow_patterns: list[str] | None = None) -> str:
    """Modeli yeniden denemeli olarak önbelleğe indirir; yarım kalan indirmeler kaldığı yerden sürer."""
    if Path(repo_id).exists():
        return repo_id
    from huggingface_hub import snapshot_download
    return retry(lambda: snapshot_download(repo_id, token=os.environ.get("HF_TOKEN"),
                                           allow_patterns=allow_patterns or
                                           ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"]),
                 what=f"{repo_id} indirme", tries=6)


def hf_repo_default(suffix: str = "") -> str | None:
    """HF_REPO ortam değişkeni veya <kullanıcı>/<GOKTURK_SLUG>."""
    from gokturk_cot import HF_SLUG
    repo = os.environ.get("HF_REPO", "").strip()
    if repo:
        return repo + suffix
    token = os.environ.get("HF_TOKEN")
    if not token:
        return None
    from huggingface_hub import HfApi
    user = retry(lambda: HfApi(token=token).whoami()["name"], what="HF kullanıcı sorgusu", tries=3)
    return f"{user}/{HF_SLUG}{suffix}"
