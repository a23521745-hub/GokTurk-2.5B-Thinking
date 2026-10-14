#!/usr/bin/env python3
"""Generate concise six-step Turkish JSONL with a multi-teacher ensemble.

Example:
  python scripts/generate_teacher_data.py \
    --input data/questions.jsonl --accepted data/teacher.accepted.jsonl \
    --rejected data/teacher.rejected.jsonl --allow-external-teachers

External teachers receive the question text. Use public/sanitized questions only.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gokturk_distill.config import (
    ConfigurationError,
    load_env_file,
    load_yaml,
    resolve_provider,
)
from gokturk_distill.generator import run_generation
from gokturk_distill.schema import (
    DataValidationError,
    strict_json_loads,
    validate_question_row,
)
from gokturk_distill.teachers import create_teacher

LOG = logging.getLogger("gokturk_teacher_cli")


def _parse_names(text: str | None, defaults: list[str]) -> list[str]:
    names = (
        defaults
        if text is None
        else [part.strip() for part in text.split(",") if part.strip()]
    )
    if len(set(names)) != len(names):
        raise ValueError("--teachers içinde tekrar eden provider adı var")
    return names


def _dry_run(path: Path) -> tuple[int, int, Counter[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Soru bankası bulunamadı: {path}")
    valid_count = invalid_count = 0
    categories: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = strict_json_loads(line)
                row, errors = validate_question_row(raw)
            except DataValidationError as exc:
                row, errors = None, [str(exc)]
            if row is None:
                invalid_count += 1
                LOG.warning("line=%d invalid_question (%s)", line_no, "; ".join(errors))
            else:
                valid_count += 1
                categories[row["category"]] += 1
    return valid_count, invalid_count, categories


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(ROOT / "configs/distillation.yaml"))
    parser.add_argument(
        "--input",
        help="Soru bankası JSONL; varsayılan configs/distillation.yaml içinden alınır",
    )
    parser.add_argument(
        "--accepted", help="Yalnızca kabul edilen kayıtların JSONL çıktısı"
    )
    parser.add_argument(
        "--rejected", help="REJECTED kayıtlar ve denetim notları JSONL çıktısı"
    )
    parser.add_argument(
        "--teachers",
        help="Virgülle ayrılmış provider adları; varsayılan etkin provider'lar",
    )
    parser.add_argument(
        "--env-file", help="İsteğe bağlı, chmod 600 ile korunan .env dosyası"
    )
    parser.add_argument(
        "--allow-external-teachers",
        action="store_true",
        help="Gerekli açık onay: soru metnini dış API sağlayıcılarına gönderir",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Daha önce işlenen id'leri atla ve JSONL'e ekle",
    )
    parser.add_argument(
        "--max-questions", type=int, help="Bu çalıştırmada işlenecek en çok soru sayısı"
    )
    parser.add_argument(
        "--workers", type=int, help="Eşzamanlı API istekleri (varsayılan config değeri)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Soru bankası ve config doğrula; API çağrısı yapma",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    try:
        load_env_file(args.env_file)
        config = load_yaml(args.config)
        providers_raw = config.get("providers", {})
        generation_raw = config.get("generation", {})
        paths = config.get("paths", {})
        if not isinstance(providers_raw, dict) or any(
            not isinstance(provider, dict) for provider in providers_raw.values()
        ):
            raise ConfigurationError(
                "providers her biri mapping olan bir YAML mapping olmalı"
            )
        if not isinstance(generation_raw, dict) or not isinstance(paths, dict):
            raise ConfigurationError("generation ve paths YAML mapping olmalı")
        generation = dict(generation_raw)
        input_path = Path(
            args.input or paths.get("question_bank", "data/questions.jsonl")
        )
        if not input_path.is_absolute():
            input_path = ROOT / input_path
        accepted_path = Path(
            args.accepted or paths.get("accepted", "data/teacher.accepted.jsonl")
        )
        rejected_path = Path(
            args.rejected or paths.get("rejected", "data/teacher.rejected.jsonl")
        )
        if not accepted_path.is_absolute():
            accepted_path = ROOT / accepted_path
        if not rejected_path.is_absolute():
            rejected_path = ROOT / rejected_path

        default_names = [
            name for name, raw in providers_raw.items() if raw.get("enabled", False)
        ]
        teacher_names = _parse_names(args.teachers, default_names)
        unknown = [name for name in teacher_names if name not in providers_raw]
        if unknown:
            raise ConfigurationError("Bilinmeyen provider: " + ", ".join(unknown))
        disabled = [
            name
            for name in teacher_names
            if not providers_raw[name].get("enabled", False)
        ]
        if disabled:
            raise ConfigurationError(
                "Seçilen provider config içinde kapalı: "
                + ", ".join(disabled)
                + ". Önce configs/distillation.yaml içinde enabled: true yapın."
            )
        if args.max_questions is not None and args.max_questions < 1:
            raise ConfigurationError("--max-questions pozitif olmalı")
        if args.workers is not None:
            generation["max_workers"] = args.workers
        try:
            temperature = float(generation.get("temperature", 0.15))
            max_output_tokens = int(generation.get("max_output_tokens", 900))
            samples = int(generation.get("samples_per_provider", 1))
            max_workers = int(generation.get("max_workers", 6))
            min_confidence = float(generation.get("min_confidence", 0.55))
            min_distinct = int(generation.get("min_distinct_teachers", 2))
            min_consensus = int(generation.get("min_consensus", 2))
            default_timeout = float(generation.get("timeout_seconds", 90))
            default_retries = int(generation.get("max_retries", 3))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConfigurationError(
                "generation ayarlarında sayısal tür geçersiz"
            ) from exc
        if not math.isfinite(temperature) or not 0.0 <= temperature <= 2.0:
            raise ConfigurationError(
                "generation.temperature sonlu ve 0–2 arasında olmalı"
            )
        if not 32 <= max_output_tokens <= 8192:
            raise ConfigurationError(
                "generation.max_output_tokens 32–8192 arasında olmalı"
            )
        if not 1 <= samples <= 8:
            raise ConfigurationError(
                "generation.samples_per_provider 1–8 arasında olmalı"
            )
        if not 1 <= max_workers <= 64:
            raise ConfigurationError("generation.max_workers 1–64 arasında olmalı")
        if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
            raise ConfigurationError("generation.min_confidence 0–1 arasında olmalı")
        if min_distinct < 1 or min_consensus < 1:
            raise ConfigurationError("teacher quorum değerleri pozitif tamsayı olmalı")
        if len(set(teacher_names)) < min_distinct:
            raise ConfigurationError(
                f"En az {min_distinct} ayrı provider seçilmeli; --teachers ile provider seçimini düzeltin"
            )
        if len(set(teacher_names)) < min_consensus:
            raise ConfigurationError(
                f"min_consensus={min_consensus} seçili provider sayısını aşıyor"
            )
        if not math.isfinite(default_timeout) or default_timeout <= 0:
            raise ConfigurationError(
                "generation.timeout_seconds pozitif ve sonlu olmalı"
            )
        if not 0 <= default_retries <= 10:
            raise ConfigurationError("generation.max_retries 0–10 arasında olmalı")
        generation.update(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            samples_per_provider=samples,
            max_workers=max_workers,
            min_confidence=min_confidence,
            min_distinct_teachers=min_distinct,
            min_consensus=min_consensus,
        )

        if args.dry_run:
            valid, invalid, categories = _dry_run(input_path)
            print(
                f"dry-run: valid={valid} invalid={invalid} categories={dict(sorted(categories.items()))}",
                file=sys.stderr,
            )
            return 0 if valid else 2

        external = [
            name
            for name in teacher_names
            if bool(providers_raw[name].get("external", True))
        ]
        if external and not args.allow_external_teachers:
            raise ConfigurationError(
                "Bu çalışma soruları harici API'lere gönderecek: "
                + ", ".join(external)
                + ". Yalnızca açık/sanitize edilmiş veriler için yeniden çalıştırırken "
                "--allow-external-teachers ekleyin. Hassas/SSB sınıflı veri göndermeyin."
            )

        teachers = [
            resolve_provider(
                name,
                providers_raw[name],
                timeout_seconds=default_timeout,
                max_retries=default_retries,
            )
            for name in teacher_names
        ]
        clients = [create_teacher(provider) for provider in teachers]
        LOG.info("Başlıyor: teachers=%s input=%s", ",".join(teacher_names), input_path)
        result = run_generation(
            input_path=input_path,
            accepted_path=accepted_path,
            rejected_path=rejected_path,
            teachers=clients,
            policy=generation,
            resume=args.resume,
            limit=args.max_questions,
        )
        LOG.info(
            "Bitti: accepted=%d rejected=%d skipped=%d",
            result["accepted"],
            result["rejected"],
            result["skipped"],
        )
        LOG.info("accepted=%s rejected=%s", accepted_path, rejected_path)
        return 0
    except (
        ConfigurationError,
        DataValidationError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        LOG.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOG.error("Kullanıcı tarafından durduruldu; tamamlanan satırlar diske yazıldı")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
