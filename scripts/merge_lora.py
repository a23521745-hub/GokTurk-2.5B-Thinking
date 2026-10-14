#!/usr/bin/env python3
"""Merge a saved PEFT/QLoRA adapter into a BF16/F16 safetensors model directory.

Run on an H200 (recommended) or a machine with enough CPU RAM. Never pass the 4-bit
bnb checkpoint itself to llama.cpp; this script reloads the declared full-precision base.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gokturk_distill.config import ConfigurationError, load_yaml
from gokturk_distill.schema import json_line

LOG = logging.getLogger("gokturk_merge")


def _resolve(path: str) -> Path:
    item = Path(path).expanduser()
    return item if item.is_absolute() else ROOT / item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter", required=True, help="train_qlora_h200.py output directory"
    )
    parser.add_argument(
        "--base-model", help="overrides model.base_model_id from train config"
    )
    parser.add_argument(
        "--output", required=True, help="new merged safetensors directory"
    )
    parser.add_argument("--config", default=str(ROOT / "configs/train_h200.yaml"))
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--allow-base-mismatch",
        action="store_true",
        help="allow adapter_config.json base id to differ from --base-model",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    temp_path: Path | None = None
    try:
        config = load_yaml(args.config)
        configured_base = str(config.get("model", {}).get("base_model_id", "")).strip()
        base_model = args.base_model or configured_base
        if not base_model:
            raise ConfigurationError(
                "base model tanımlı değil; --base-model veya STUDENT_BASE_MODEL ayarlayın"
            )
        adapter_dir = _resolve(args.adapter)
        output_dir = _resolve(args.output)
        if not adapter_dir.is_dir():
            raise FileNotFoundError(f"Adapter dizini bulunamadı: {adapter_dir}")
        adapter_config_path = adapter_dir / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise ValueError(
                f"PEFT adapter_config.json bulunamadı: {adapter_config_path}"
            )
        if output_dir.exists():
            raise FileExistsError(f"Çıktı zaten var; üzerine yazılmadı: {output_dir}")
        adapter_config: dict[str, Any] = json.loads(
            adapter_config_path.read_text(encoding="utf-8")
        )
        adapter_base = str(adapter_config.get("base_model_name_or_path", "")).strip()
        if adapter_base and adapter_base != base_model and not args.allow_base_mismatch:
            raise ValueError(
                f"Adapter base={adapter_base!r}, istenen base={base_model!r}; yanlış ağırlıkları birleştirmemek için durduruldu"
            )
        if adapter_config.get("r") is not None and int(adapter_config["r"]) != int(
            config.get("lora", {}).get("r", 64)
        ):
            LOG.warning(
                "Adapter LoRA r=%s, config r=%s",
                adapter_config["r"],
                config.get("lora", {}).get("r", 64),
            )

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Merge bağımlılıkları eksik; pip install -r requirements-train.txt -c constraints.txt"
            ) from exc
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda seçildi ancak CUDA GPU bulunamadı")
        dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": bool(
                config.get("model", {}).get("trust_remote_code", False)
            ),
        }
        if args.device == "cuda":
            load_kwargs["device_map"] = {"": torch.cuda.current_device()}

        LOG.info("Base model yükleniyor: %s", base_model)
        base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        peft_model = PeftModel.from_pretrained(
            base, str(adapter_dir), is_trainable=False
        )
        merged = peft_model.merge_and_unload(safe_merge=True)
        merged.config.use_cache = True

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(
            tempfile.mkdtemp(prefix=output_dir.name + ".tmp-", dir=output_dir.parent)
        )
        merged.save_pretrained(
            str(temp_path),
            safe_serialization=True,
            max_shard_size="4GB",
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(adapter_dir),
                trust_remote_code=bool(
                    config.get("model", {}).get("trust_remote_code", False)
                ),
            )
        except (OSError, ValueError):
            tokenizer = AutoTokenizer.from_pretrained(
                base_model,
                trust_remote_code=bool(
                    config.get("model", {}).get("trust_remote_code", False)
                ),
            )
        tokenizer.save_pretrained(str(temp_path))
        manifest = {
            "model_name": config.get("model", {}).get(
                "artifact_name", "GokTurk-3.3B-Thinking"
            ),
            "base_model_id": base_model,
            "adapter_dir": str(adapter_dir),
            "merge_dtype": args.dtype,
            "format": "safetensors",
        }
        (temp_path / "merge_meta.json").write_text(
            json_line(manifest) + "\n", encoding="utf-8"
        )
        if not list(temp_path.glob("*.safetensors")):
            raise RuntimeError("Merge tamamlandı ancak safetensors dosyası üretilmedi")
        os.replace(temp_path, output_dir)
        temp_path = None
        LOG.info("Merged full model: %s", output_dir)
        return 0
    except (
        ConfigurationError,
        FileNotFoundError,
        FileExistsError,
        ValueError,
        RuntimeError,
        OSError,
    ) as exc:
        LOG.error("%s", exc)
        return 2
    finally:
        if temp_path is not None and temp_path.exists():
            shutil.rmtree(temp_path, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
