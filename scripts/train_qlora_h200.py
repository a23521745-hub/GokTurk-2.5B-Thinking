#!/usr/bin/env python3
"""QLoRA supervised fine-tuning on one allocated H200 GPU.

Input must be the verified ACCEPTED JSONL emitted by verify_teacher_data.py. The script trains
on compact six-step JSON answers, masks the system/user prompt from loss, uses 4-bit NF4 base
weights and BF16 compute, and saves a PEFT adapter (not a merged full model).
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from gokturk_distill.config import ConfigurationError, load_yaml
from gokturk_distill.records import validate_accepted_record
from gokturk_distill.schema import DataValidationError, json_line, strict_json_loads

LOG = logging.getLogger("gokturk_train")

TRAIN_SYSTEM_PROMPT = """Sen GökTürk adlı Türkçe bir yapay zekâ asistanısın. Her soruya yalnızca tek bir geçerli JSON nesnesiyle yanıt ver. Anahtarlar tam olarak understand, plan, solve, verify, answer, confidence olmalı. İlk beş alan kısa Türkçe metin; confidence 0 ile 1 arasında JSON sayısıdır. Gereksiz uzun düşünce metni üretme; çözümün kısa, denetlenebilir özeti yeterlidir. Sonuçtan emin değilsen confidence değerini düşür."""


class EncodedRecords:
    """Minimal torch Dataset that holds already tokenized response-only SFT examples."""

    def __init__(self, rows: list[dict[str, list[int]]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.rows[index]


class ResponseOnlyCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in features)
        if self.pad_to_multiple_of:
            width = (
                (width + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of
            ) * self.pad_to_multiple_of
        batch_size = len(features)
        input_ids = torch.full((batch_size, width), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
        labels = torch.full((batch_size, width), -100, dtype=torch.long)
        for row_index, item in enumerate(features):
            length = len(item["input_ids"])
            input_ids[row_index, :length] = torch.tensor(
                item["input_ids"], dtype=torch.long
            )
            attention_mask[row_index, :length] = 1
            labels[row_index, :length] = torch.tensor(item["labels"], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def _absolute_or_root(path_value: str, default: str = "") -> Path:
    path = Path(path_value or default).expanduser()
    return path if path.is_absolute() else ROOT / path


def _read_accepted(
    path: Path, policy: dict[str, Any], *, limit: int | None = None
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid = 0
    if not path.is_file():
        raise FileNotFoundError(f"Eğitim JSONL dosyası bulunamadı: {path}")
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = strict_json_loads(line)
            except DataValidationError as exc:
                invalid += 1
                LOG.warning(
                    "%s:%d malformed_json (%s)", path, line_number, type(exc).__name__
                )
                continue
            checked, errors = validate_accepted_record(record, policy)
            if checked is None:
                invalid += 1
                LOG.warning(
                    "%s:%d rejected_record (%s)",
                    path,
                    line_number,
                    "; ".join(errors[:4]),
                )
                continue
            example_id = str(record["example_id"])
            if example_id in seen:
                invalid += 1
                LOG.warning("%s:%d duplicate_id=%s", path, line_number, example_id)
                continue
            seen.add(example_id)
            rows.append({"id": example_id, **checked})
            if limit is not None and len(rows) >= limit:
                break
    return rows, invalid


def _split_data(
    train_rows: list[dict[str, Any]],
    validation_path: Path | None,
    policy: dict[str, Any],
    fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    invalid_val = 0
    if validation_path is not None:
        val_rows, invalid_val = _read_accepted(validation_path, policy)
        train_ids = {row["id"] for row in train_rows}
        overlap = train_ids & {row["id"] for row in val_rows}
        if overlap:
            raise ValueError(f"train/validation id çakışması: {len(overlap)} örnek")
        return train_rows, val_rows, invalid_val
    if not 0.0 <= fraction < 0.5:
        raise ValueError("validation_fraction 0 ile 0.5 arasında olmalı")
    if len(train_rows) < 50 or fraction == 0:
        return train_rows, [], invalid_val
    shuffled = list(train_rows)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * fraction))
    validation_count = min(validation_count, len(shuffled) - 1)
    return shuffled[validation_count:], shuffled[:validation_count], invalid_val


def _tokenize_rows(
    rows: list[dict[str, Any]], tokenizer: Any, max_seq_len: int
) -> tuple[EncodedRecords, int]:
    encoded: list[dict[str, list[int]]] = []
    too_long = 0
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": TRAIN_SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]["question"]},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if prompt and isinstance(prompt[0], list):
            prompt = prompt[0]
        target = json.dumps(
            row["steps"], ensure_ascii=False, allow_nan=False, separators=(",", ":")
        )
        answer_ids = tokenizer.encode(target, add_special_tokens=False)
        eos = tokenizer.eos_token_id
        if eos is not None and (not answer_ids or answer_ids[-1] != eos):
            answer_ids.append(eos)
        input_ids = [int(token) for token in prompt] + [
            int(token) for token in answer_ids
        ]
        labels = [-100] * len(prompt) + [int(token) for token in answer_ids]
        if len(input_ids) > max_seq_len:
            too_long += 1
            continue
        encoded.append({"input_ids": input_ids, "labels": labels})
    return EncodedRecords(encoded), too_long


def _build_training_arguments(
    training: dict[str, Any], output_dir: Path, has_eval: bool
) -> Any:
    from transformers import TrainingArguments

    raw: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(training.get("num_train_epochs", 2.0)),
        "max_steps": int(training.get("max_steps", -1)),
        "per_device_train_batch_size": int(
            training.get("per_device_train_batch_size", 2)
        ),
        "per_device_eval_batch_size": int(
            training.get("per_device_eval_batch_size", 2)
        ),
        "gradient_accumulation_steps": int(
            training.get("gradient_accumulation_steps", 16)
        ),
        "learning_rate": float(training.get("learning_rate", 1e-4)),
        "lr_scheduler_type": str(training.get("lr_scheduler_type", "cosine")),
        "warmup_ratio": float(training.get("warmup_ratio", 0.03)),
        "weight_decay": float(training.get("weight_decay", 0.01)),
        "max_grad_norm": float(training.get("max_grad_norm", 1.0)),
        "logging_steps": int(training.get("logging_steps", 10)),
        "save_strategy": str(training.get("save_strategy", "steps")),
        "save_steps": int(training.get("save_steps", 100)),
        "save_total_limit": int(training.get("save_total_limit", 3)),
        "dataloader_num_workers": int(training.get("dataloader_num_workers", 4)),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", True)),
        "optim": str(training.get("optim", "adamw_torch_fused")),
        "seed": int(training.get("seed", 3407)),
        "bf16": True,
        "fp16": False,
        "report_to": "none",
        "remove_unused_columns": False,
        "save_safetensors": True,
        "logging_first_step": True,
        "dataloader_pin_memory": True,
        "ddp_find_unused_parameters": False,
    }
    params = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = (
        "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    )
    raw[strategy_name] = "steps" if has_eval else "no"
    if has_eval:
        raw["eval_steps"] = int(training.get("eval_steps", 100))
        raw["load_best_model_at_end"] = False
    if "gradient_checkpointing_kwargs" in params:
        raw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    accepted = {key: value for key, value in raw.items() if key in params}
    return TrainingArguments(**accepted)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(ROOT / "configs/train_h200.yaml"))
    parser.add_argument(
        "--train-data", help="verified ACCEPTED JSONL (overrides config)"
    )
    parser.add_argument(
        "--validation-data",
        help="optional verified validation JSONL (overrides config)",
    )
    parser.add_argument(
        "--output", help="adapter/checkpoint output directory (overrides config)"
    )
    parser.add_argument("--resume-from-checkpoint", help="Trainer checkpoint directory")
    parser.add_argument(
        "--limit", type=int, help="cap valid training records for a short smoke run"
    )
    parser.add_argument(
        "--allow-non-h200",
        action="store_true",
        help="only for development/smoke testing",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    try:
        config = load_yaml(args.config)
        model_cfg = dict(config.get("model", {}))
        quant_cfg = dict(config.get("quantization", {}))
        lora_cfg = dict(config.get("lora", {}))
        data_cfg = dict(config.get("data", {}))
        training = dict(config.get("training", {}))
        if (
            not bool(training.get("bf16", True))
            or str(quant_cfg.get("compute_dtype", "bfloat16")).lower() != "bfloat16"
        ):
            raise ConfigurationError(
                "Production H200 config BF16 compute ve bf16 training gerektirir"
            )
        if (
            not bool(quant_cfg.get("load_in_4bit", True))
            or str(quant_cfg.get("quant_type", "nf4")).lower() != "nf4"
        ):
            raise ConfigurationError(
                "Production H200 profili 4-bit NF4 QLoRA gerektirir"
            )
        if int(lora_cfg.get("r", 64)) != 64 or int(lora_cfg.get("alpha", 128)) != 128:
            raise ConfigurationError(
                "Production H200 profili LoRA r=64 ve alpha=128 gerektirir"
            )
        if int(data_cfg.get("max_seq_len", 4096)) != 4096:
            raise ConfigurationError(
                "Production H200 profili max_seq_len=4096 gerektirir"
            )

        train_path = _absolute_or_root(
            args.train_data or data_cfg.get("train_jsonl", "")
        )
        validation_value = (
            args.validation_data
            if args.validation_data is not None
            else data_cfg.get("validation_jsonl", "")
        )
        validation_path = (
            _absolute_or_root(validation_value)
            if str(validation_value).strip()
            else None
        )
        output_dir = _absolute_or_root(
            args.output or training.get("output_dir", "outputs/gokturk-3.3b-h200")
        )
        base_model = str(model_cfg.get("base_model_id", "")).strip()
        if not base_model:
            raise ConfigurationError("model.base_model_id boş")
        if args.limit is not None and args.limit < 1:
            raise ConfigurationError("--limit pozitif olmalı")

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA GPU bulunamadı; bu betik H200 CUDA işinde çalıştırılmalı"
            )
        if (
            torch.cuda.device_count() != 1
            or int(os.environ.get("WORLD_SIZE", "1")) != 1
        ):
            raise RuntimeError(
                "Bu betik tek H200 GPU için yapılandırıldı; 1 GPU'luk Slurm allocation kullanın"
            )
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("Bu GPU PyTorch BF16 desteği sunmuyor")
        device_name = torch.cuda.get_device_name(0)
        if (
            bool(model_cfg.get("require_h200", True))
            and "H200" not in device_name.upper()
            and not args.allow_non_h200
        ):
            raise RuntimeError(
                f"H200 bekleniyordu, bulunan GPU: {device_name}. Test için --allow-non-h200 kullanın."
            )

        seed = int(training.get("seed", 3407))
        train_rows, invalid_train = _read_accepted(
            train_path,
            {
                "min_confidence": float(data_cfg.get("min_confidence", 0.55)),
                "require_reference_for_math": bool(
                    data_cfg.get("require_reference_for_math", True)
                ),
                "min_distinct_teachers": int(
                    config.get("generation", {}).get("min_distinct_teachers", 2)
                ),
                "min_consensus": int(
                    config.get("generation", {}).get("min_consensus", 2)
                ),
            },
            limit=args.limit,
        )
        raw_validation = None
        if validation_path is not None:
            raw_validation, invalid_val = _read_accepted(
                validation_path,
                {
                    "min_confidence": float(data_cfg.get("min_confidence", 0.55)),
                    "require_reference_for_math": bool(
                        data_cfg.get("require_reference_for_math", True)
                    ),
                    "min_distinct_teachers": int(
                        config.get("generation", {}).get("min_distinct_teachers", 2)
                    ),
                    "min_consensus": int(
                        config.get("generation", {}).get("min_consensus", 2)
                    ),
                },
            )
            if invalid_val:
                LOG.warning("validation invalid kayıt=%d", invalid_val)
        if raw_validation is not None:
            train_ids = {row["id"] for row in train_rows}
            validation_ids = {row["id"] for row in raw_validation}
            overlap = train_ids & validation_ids
            if overlap:
                raise ValueError(f"train/validation id çakışması: {len(overlap)} örnek")
            val_rows = raw_validation
        else:
            train_rows, val_rows, _ = _split_data(
                train_rows,
                None,
                {
                    "min_confidence": float(data_cfg.get("min_confidence", 0.55)),
                    "require_reference_for_math": bool(
                        data_cfg.get("require_reference_for_math", True)
                    ),
                    "min_distinct_teachers": int(
                        config.get("generation", {}).get("min_distinct_teachers", 2)
                    ),
                    "min_consensus": int(
                        config.get("generation", {}).get("min_consensus", 2)
                    ),
                },
                float(data_cfg.get("validation_fraction", 0.02)),
                seed,
            )

        min_examples = int(data_cfg.get("min_train_examples", 20))
        if len(train_rows) < min_examples:
            raise RuntimeError(
                f"Geçerli train örneği yetersiz: {len(train_rows)} < {min_examples} (invalid={invalid_train})"
            )
        if (
            output_dir.exists()
            and any(output_dir.iterdir())
            and not args.resume_from_checkpoint
        ):
            raise RuntimeError(
                f"Output dizini boş değil: {output_dir}; yanlışlıkla üzerine yazmamak için durduruldu"
            )

        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
                Trainer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Eğitim bağımlılıkları eksik; pip install -r requirements-train.txt -c constraints.txt"
            ) from exc

        max_seq_len = int(data_cfg.get("max_seq_len", 4096))
        LOG.info(
            "GPU=%s | model=%s | train=%d val=%d invalid=%d seq=%d",
            device_name,
            base_model,
            len(train_rows),
            len(val_rows),
            invalid_train,
            max_seq_len,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            use_fast=True,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(
                "Taban tokenizer chat_template içermiyor; desteklenen sohbet şablonlu base model kullanın"
            )
        if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
            raise RuntimeError(
                "Tokenizer geçerli EOS token sağlamıyor; SFT completion sonunu güvenilir biçimde ayıramaz"
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        encoded_train, long_train = _tokenize_rows(train_rows, tokenizer, max_seq_len)
        encoded_val, long_val = (
            _tokenize_rows(val_rows, tokenizer, max_seq_len)
            if val_rows
            else (EncodedRecords([]), 0)
        )
        if len(encoded_train) < min_examples:
            raise RuntimeError(
                f"Max sequence uzunluğunu aşan kayıtlar çıkarıldı; train={len(encoded_train)} < {min_examples}"
            )
        if len(encoded_val) == 0:
            encoded_val = None
        LOG.info(
            "tokenized train=%d val=%d too_long_train=%d too_long_val=%d",
            len(encoded_train),
            len(encoded_val) if encoded_val else 0,
            long_train,
            long_val,
        )

        quantization = BitsAndBytesConfig(
            load_in_4bit=bool(quant_cfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type=str(quant_cfg.get("quant_type", "nf4")),
            bnb_4bit_use_double_quant=bool(quant_cfg.get("double_quant", True)),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_load_kwargs: dict[str, Any] = {
            "quantization_config": quantization,
            "torch_dtype": torch.bfloat16,
            "device_map": {"": torch.cuda.current_device()},
            "trust_remote_code": bool(model_cfg.get("trust_remote_code", False)),
            "low_cpu_mem_usage": True,
        }
        attention = model_cfg.get("attn_implementation")
        if attention:
            model_load_kwargs["attn_implementation"] = str(attention)
        model = AutoModelForCausalLM.from_pretrained(base_model, **model_load_kwargs)
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(
                training.get("gradient_checkpointing", True)
            ),
        )

        target_modules = [str(item) for item in lora_cfg.get("target_modules", [])]
        available_leaf_modules = {
            module_name.rsplit(".", 1)[-1] for module_name, _ in model.named_modules()
        }
        missing_targets = [
            name for name in target_modules if name not in available_leaf_modules
        ]
        if not target_modules or len(missing_targets) == len(target_modules):
            raise RuntimeError(
                f"LoRA target_modules modelde bulunamadı: {target_modules}"
            )
        if missing_targets:
            LOG.warning(
                "Modelde bulunmayan bazı target_modules atlanabilir: %s",
                missing_targets,
            )
        lora = LoraConfig(
            r=int(lora_cfg.get("r", 64)),
            lora_alpha=int(lora_cfg.get("alpha", 128)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            use_rslora=bool(lora_cfg.get("use_rslora", False)),
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

        output_dir.mkdir(parents=True, exist_ok=True)
        trainer = Trainer(
            model=model,
            args=_build_training_arguments(
                training, output_dir, has_eval=encoded_val is not None
            ),
            train_dataset=encoded_train,
            eval_dataset=encoded_val,
            data_collator=ResponseOnlyCollator(int(tokenizer.pad_token_id)),
            processing_class=tokenizer,
        )
        start_time = time.time()
        resume = (
            args.resume_from_checkpoint
            or str(training.get("resume_from_checkpoint", "")).strip()
            or None
        )
        train_result = trainer.train(resume_from_checkpoint=resume)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        metrics = dict(train_result.metrics)
        metadata = {
            "model_name": model_cfg.get("artifact_name", "GokTurk-3.3B-Thinking"),
            "base_model_id": base_model,
            "device": device_name,
            "quantization": {
                "base_load_in_4bit": True,
                "type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
            },
            "lora": {
                "r": int(lora_cfg.get("r", 64)),
                "alpha": int(lora_cfg.get("alpha", 128)),
                "dropout": float(lora_cfg.get("dropout", 0.05)),
                "target_modules": target_modules,
            },
            "max_seq_len": max_seq_len,
            "train_examples": len(encoded_train),
            "validation_examples": len(encoded_val) if encoded_val else 0,
            "filtered_invalid_records": invalid_train,
            "filtered_too_long_train": long_train,
            "metrics": metrics,
            "wall_time_seconds": round(time.time() - start_time, 2),
        }
        (output_dir / "train_meta.json").write_text(
            json_line(metadata) + "\n", encoding="utf-8"
        )
        LOG.info("Adapter saved: %s", output_dir)
        return 0
    except (
        ConfigurationError,
        DataValidationError,
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as exc:
        LOG.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOG.error("Training interrupted; existing checkpoints are preserved")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
