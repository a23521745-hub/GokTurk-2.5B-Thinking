"""Ortam yardımcıları: Kaggle / Colab / yerel algılama, secret yükleme, Unsloth kurulumu.

ÖNEMLİ: Bu modül torch'u import ETMEZ; `setup_gpu_env()` torch'tan önce çağrılmalıdır.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

SECRET_NAMES = ("HF_TOKEN", "TAVILY_API_KEY", "GEMINI_API_KEY")


def detect_env() -> str:
    if Path("/kaggle/working").is_dir():
        return "kaggle"
    if importlib.util.find_spec("google.colab") is not None:
        return "colab"
    return "local"


def default_dirs() -> tuple[Path, Path]:
    """(kalıcı çıktı dizini, büyük geçici dosyalar için scratch)."""
    env = detect_env()
    if env == "kaggle":   # /kaggle/working 20 GB ve "Output" olarak saklanır → ara dosyalar /tmp'ye
        return Path("/kaggle/working/outputs"), Path("/tmp/gokturk")
    if env == "colab":
        return Path("/content/outputs"), Path("/content/tmp")
    root = Path(__file__).resolve().parent
    return root / "outputs", root / "outputs" / "tmp"


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
            os.environ[n] = val
    status = {n: bool(os.environ.get(n)) for n in names}
    print("🔑 Secrets: " + ", ".join(f"{n}={'✓' if ok else '✗'}" for n, ok in status.items()))
    return status


def setup_gpu_env():
    """torch import edilmeden ÖNCE çağırın."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # Kaggle T4 x2 → Unsloth tek GPU
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    out, scratch = default_dirs()
    os.environ.setdefault("HF_HOME", str(scratch / "hf_cache"))   # model önbelleği kotayı doldurmasın
    if "torch" in sys.modules:
        print("⚠️  torch zaten import edilmiş; CUDA_VISIBLE_DEVICES etkisiz olabilir.", file=sys.stderr)


def pip_install(*pkgs: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def ensure_unsloth():
    if importlib.util.find_spec("unsloth") is None:
        print("📦 unsloth kuruluyor (2-4 dk)...")
        pip_install("unsloth")
    for pkg in ("triton", "xformers"):   # eksikse ekle; --no-deps: torch sürümünü bozmamak için
        if importlib.util.find_spec(pkg) is None:
            try:
                pip_install("--no-deps", pkg)
            except subprocess.CalledProcessError:
                print(f"ℹ️  {pkg} kurulamadı; Unsloth yedek çekirdeklerle çalışır.")
