#!/usr/bin/env python3
"""Convert merged HF safetensors -> F16 GGUF -> Q4_K_M GGUF and smoke-test it.

The model directory must be a full merged checkpoint (config.json + safetensors + tokenizer),
not a PEFT adapter. llama.cpp is built from a pinned ref when it is not supplied locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gokturk_distill.config import ConfigurationError, load_yaml
from gokturk_distill.schema import json_line

LOG = logging.getLogger("gokturk_gguf")


def _resolve_path(raw: str, default: str = "") -> Path:
    path = Path(raw or default).expanduser()
    return path if path.is_absolute() else ROOT / path


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    LOG.info("run: %s", " ".join(command))
    return subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        timeout=timeout,
    )


def _git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _ensure_llama_cpp(
    repo_value: str, ref: str, build_jobs: int, build_if_missing: bool
) -> tuple[Path, str]:
    repo = _resolve_path(repo_value, "outputs/tools/llama.cpp")
    fetched_ref_sha: str | None = None
    if not repo.exists():
        if not ref:
            raise ConfigurationError(
                "llama.cpp yok; tekrarlanabilir build için LLAMA_CPP_REF=<tag/commit> ayarlayın"
            )
        repo.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/ggml-org/llama.cpp",
                str(repo),
            ]
        )
        _run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", ref])
        fetched_ref_sha = _git_output(repo, "rev-parse", "FETCH_HEAD^{commit}")
        _run(["git", "-C", str(repo), "checkout", "--detach", "FETCH_HEAD"])
    if not (repo / ".git").exists():
        raise ConfigurationError(f"llama.cpp dizini git checkout değil: {repo}")
    if ref:
        requested_sha = fetched_ref_sha
        if requested_sha is None:
            candidates = (f"{ref}^{{commit}}", f"origin/{ref}^{{commit}}")
            for candidate_ref in candidates:
                try:
                    requested_sha = _git_output(repo, "rev-parse", candidate_ref)
                    break
                except subprocess.CalledProcessError:
                    continue
        if requested_sha is None:
            raise ConfigurationError(
                f"LLAMA_CPP_REF={ref!r} checkout içinde bulunamadı; ref'i önceden fetch/checkout edin"
            )
        current_sha = _git_output(repo, "rev-parse", "HEAD")
        if requested_sha != current_sha:
            raise ConfigurationError(
                f"llama.cpp HEAD ({current_sha[:12]}) LLAMA_CPP_REF ({requested_sha[:12]}) ile eşleşmiyor"
            )
    else:
        LOG.warning(
            "LLAMA_CPP_REF sabitlenmedi; mevcut llama.cpp commit'i manifest'e kaydedilecek"
        )
    dirty = _git_output(repo, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ConfigurationError(
            "llama.cpp checkout tracked değişiklik içeriyor; temiz/pinned checkout kullanın"
        )

    build_dir = repo / "build"
    quantizer = build_dir / "bin" / "llama-quantize"
    if os.name == "nt":
        quantizer = build_dir / "bin" / "Release" / "llama-quantize.exe"
    if not quantizer.is_file():
        if not build_if_missing:
            raise FileNotFoundError(f"llama-quantize bulunamadı: {quantizer}")
        _run(
            [
                "cmake",
                "-S",
                str(repo),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DLLAMA_CURL=OFF",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_EXAMPLES=ON",
                "-DLLAMA_BUILD_SERVER=OFF",
                "-DGGML_NATIVE=OFF",
            ]
        )
        _run(
            [
                "cmake",
                "--build",
                str(build_dir),
                "--config",
                "Release",
                "-j",
                str(max(1, build_jobs)),
            ]
        )
    if not quantizer.is_file():
        raise FileNotFoundError(
            "llama.cpp build tamamlandı ancak llama-quantize bulunamadı"
        )
    converter = repo / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise FileNotFoundError(f"HF → GGUF converter bulunamadı: {converter}")
    return repo, _git_output(repo, "rev-parse", "HEAD")


def _kv_cache_estimate(
    config: dict[str, Any], context_size: int, cache_type: str
) -> float | None:
    layers = config.get("num_hidden_layers", config.get("n_layer"))
    hidden = config.get("hidden_size", config.get("n_embd"))
    heads = config.get("num_attention_heads", config.get("n_head"))
    kv_heads = config.get("num_key_value_heads", heads)
    head_dim = config.get("head_dim")
    try:
        layers, hidden, heads, kv_heads = (
            int(layers),
            int(hidden),
            int(heads),
            int(kv_heads),
        )
        head_dim = int(head_dim) if head_dim is not None else hidden // heads
        if min(layers, hidden, heads, kv_heads, head_dim, context_size) <= 0:
            return None
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    bytes_per_element = 1.0625 if cache_type.lower() == "q8_0" else 2.0
    byte_count = context_size * layers * kv_heads * head_dim * 2 * bytes_per_element
    return byte_count / 1_000_000_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _find_cli(build_dir: Path) -> Path | None:
    candidates = [
        build_dir / "bin" / "llama-cli",
        build_dir / "bin" / "llama-completion",
    ]
    if os.name == "nt":
        candidates = [
            build_dir / "bin" / "Release" / "llama-cli.exe",
            build_dir / "bin" / "Release" / "llama-completion.exe",
        ] + candidates
    return next((item for item in candidates if item.is_file()), None)


def _smoke_test(cli: Path | None, model_path: Path) -> None:
    if cli is None:
        raise FileNotFoundError(
            "llama-cli/llama-completion binary not found after build"
        )
    command = [
        str(cli),
        "-m",
        str(model_path),
        "-p",
        "2 + 2 kaç eder? Kısa yanıt ver.",
        "-n",
        "8",
        "--temp",
        "0",
    ]
    try:
        _run(command, timeout=180, capture=True)
    except subprocess.CalledProcessError as exc:
        tail = (exc.stdout or "")[-1200:]
        raise RuntimeError(
            "GGUF yükleme/üretim smoke test başarısız:\n" + tail
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir", required=True, help="Merged full HF safetensors directory"
    )
    parser.add_argument("--output-dir", default="outputs/gokturk-3.3b-gguf")
    parser.add_argument("--config", default=str(ROOT / "configs/gguf.yaml"))
    parser.add_argument(
        "--llama-cpp-dir", help="Existing llama.cpp checkout; overrides YAML/env"
    )
    parser.add_argument(
        "--llama-cpp-ref", help="Exact local ref/commit expected (pin for releases)"
    )
    parser.add_argument("--build-jobs", type=int, help="CMake parallel jobs")
    parser.add_argument(
        "--install-converter-deps",
        action="store_true",
        help="pip install llama.cpp converter requirements into the active environment",
    )
    parser.add_argument(
        "--keep-f16", action="store_true", help="keep the intermediate F16 GGUF"
    )
    parser.add_argument(
        "--strict-size",
        action="store_true",
        help="fail if Q4_K_M is outside target size range",
    )
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    scratch: Path | None = None
    try:
        config = load_yaml(args.config)
        model_cfg = dict(config.get("mobile", {}))
        model_dir = _resolve_path(args.model_dir)
        output_dir = _resolve_path(args.output_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model dizini bulunamadı: {model_dir}")
        if (model_dir / "adapter_config.json").exists():
            raise ValueError(
                "Bu bir PEFT adapter dizini; önce scripts/merge_lora.py ile full safetensors üretin"
            )
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(
                "config.json yok: merged Hugging Face model dizinini seçin"
            )
        safetensors = list(model_dir.glob("*.safetensors"))
        if not safetensors:
            raise FileNotFoundError("Model dizininde *.safetensors bulunamadı")
        if not any(
            (model_dir / name).is_file()
            for name in (
                "tokenizer.json",
                "tokenizer.model",
                "spiece.model",
                "tokenizer_config.json",
                "vocab.json",
            )
        ):
            raise FileNotFoundError(
                "Tokenizer dosyası yok (tokenizer.json/model/config, vocab.json veya spiece.model)"
            )
        if output_dir.exists():
            raise FileExistsError(f"Çıktı zaten var; üzerine yazılmadı: {output_dir}")

        repo_value = args.llama_cpp_dir or str(config.get("llama_cpp_dir", ""))
        ref = args.llama_cpp_ref or str(config.get("llama_cpp_ref", ""))
        jobs = args.build_jobs or int(config.get("build_jobs", os.cpu_count() or 4))
        if jobs < 1:
            raise ValueError("build-jobs pozitif olmalı")
        repo, llama_sha = _ensure_llama_cpp(
            repo_value,
            ref,
            jobs,
            bool(config.get("build_if_missing", True)),
        )
        converter_requirements = (
            repo / "requirements" / "requirements-convert_hf_to_gguf.txt"
        )
        if args.install_converter_deps:
            if not converter_requirements.is_file():
                raise FileNotFoundError(
                    f"Converter requirements yok: {converter_requirements}"
                )
            _run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(converter_requirements),
                ]
            )

        total_source_bytes = sum(path.stat().st_size for path in safetensors)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(output_dir.parent).free
        estimated_need = int(total_source_bytes * 1.15 + 1_000_000_000)
        if free_bytes < estimated_need:
            raise RuntimeError(
                f"Yetersiz disk alanı: {free_bytes / 1e9:.2f} GB boş, yaklaşık {estimated_need / 1e9:.2f} GB gerekli"
            )
        scratch = Path(tempfile.mkdtemp(prefix="gokturk-gguf-", dir=output_dir.parent))
        model_name = str(config.get("model_name", "GokTurk-3.3B-Thinking"))
        f16_path = scratch / f"{model_name}-F16.gguf"
        quant_path = scratch / f"{model_name}-Q4_K_M.gguf"

        converter_path = repo / "convert_hf_to_gguf.py"
        _run(
            [
                sys.executable,
                str(converter_path),
                str(model_dir),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ]
        )
        quantizer = repo / "build" / "bin" / "llama-quantize"
        if os.name == "nt":
            quantizer = repo / "build" / "bin" / "Release" / "llama-quantize.exe"
        _run(
            [
                str(quantizer),
                str(f16_path),
                str(quant_path),
                str(config.get("quantization", "Q4_K_M")),
            ]
        )
        if not quant_path.is_file() or quant_path.stat().st_size == 0:
            raise RuntimeError("Kuantizasyon çıktısı yok veya boş")
        if not args.skip_smoke_test and bool(config.get("smoke_test", True)):
            _smoke_test(_find_cli(repo / "build"), quant_path)

        size_bytes = quant_path.stat().st_size
        size_gb = size_bytes / 1_000_000_000
        min_gb = float(model_cfg.get("target_file_size_gb_min", 1.8))
        max_gb = float(model_cfg.get("target_file_size_gb_max", 1.95))
        strict_size = args.strict_size or bool(
            model_cfg.get("strict_target_size", False)
        )
        if not min_gb <= size_gb <= max_gb:
            note = f"Q4_K_M boyutu {size_gb:.3f} GB; hedef {min_gb:.2f}–{max_gb:.2f} GB. Model mimarisi/vocab boyutu sonucu değiştirebilir."
            if strict_size:
                raise RuntimeError(note)
            LOG.warning(note)

        hf_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        context_size = int(model_cfg.get("context_size", 4096))
        kv_type = str(model_cfg.get("kv_cache_type", "q8_0"))
        kv_gb = _kv_cache_estimate(hf_config, context_size, kv_type)
        reserve_gb = float(model_cfg.get("runtime_reserve_gb", 0.45))
        free_ram_gb = float(model_cfg.get("free_ram_gb", 3.6))
        estimated_total = size_gb + reserve_gb + (kv_gb or 0.0)
        if kv_gb is None:
            LOG.warning(
                "KV-cache tahmini için config.json'da katman/attention boyutları eksik"
            )
        if estimated_total > free_ram_gb:
            LOG.warning(
                "Tahmini model+KV+runtime %.2f GB, belirtilen boş RAM %.2f GB'ı aşıyor",
                estimated_total,
                free_ram_gb,
            )

        keep_f16 = args.keep_f16 or bool(config.get("keep_f16", False))
        if not keep_f16:
            f16_path.unlink(missing_ok=True)
        final_files = [quant_path] + ([f16_path] if keep_f16 else [])
        quant_hash = _sha256(quant_path)
        manifest = {
            "model_name": model_name,
            "source_model_dir": str(model_dir),
            "source_safetensors_bytes": total_source_bytes,
            "llama_cpp_commit": llama_sha,
            "quantization": str(config.get("quantization", "Q4_K_M")),
            "gguf_file": quant_path.name,
            "gguf_bytes": size_bytes,
            "gguf_gb_decimal": round(size_gb, 6),
            "sha256": quant_hash,
            "mobile_estimate": {
                "context_size": context_size,
                "kv_cache_type": kv_type,
                "estimated_kv_cache_gb": round(kv_gb, 6) if kv_gb is not None else None,
                "runtime_reserve_gb": reserve_gb,
                "estimated_total_gb": round(estimated_total, 6),
                "available_free_ram_gb_assumed": free_ram_gb,
                "estimate_is_not_a_device_benchmark": True,
            },
        }
        (scratch / "manifest.json").write_text(
            json_line(manifest) + "\n", encoding="utf-8"
        )
        (scratch / "SHA256SUMS.txt").write_text(
            "".join(f"{_sha256(path)}  {path.name}\n" for path in final_files),
            encoding="utf-8",
        )
        os.replace(scratch, output_dir)
        scratch = None
        final_quant = output_dir / quant_path.name
        LOG.info(
            "Q4_K_M ready: %s (%.3f GB, SHA256 %s)", final_quant, size_gb, quant_hash
        )
        LOG.info(
            "Mobile estimate: file %.2f + KV %s + runtime %.2f = %.2f GB (free RAM assumption %.2f GB)",
            size_gb,
            f"{kv_gb:.2f} GB" if kv_gb is not None else "unknown",
            reserve_gb,
            estimated_total,
            free_ram_gb,
        )
        return 0
    except (
        ConfigurationError,
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        OSError,
    ) as exc:
        LOG.error("%s", exc)
        return 2
    finally:
        if scratch is not None and scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
