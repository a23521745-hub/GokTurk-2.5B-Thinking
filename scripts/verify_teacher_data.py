#!/usr/bin/env python3
"""Re-check generated records and split them into ACCEPTED and REJECTED JSONL files.

No model/API call is made. Generated Python is parsed for syntax only and is never executed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gokturk_distill.config import ConfigurationError, load_yaml
from gokturk_distill.records import validate_accepted_record
from gokturk_distill.schema import (
    SCHEMA_VERSION,
    DataValidationError,
    json_line,
    strict_json_loads,
)

LOG = logging.getLogger("gokturk_verify_cli")


def validate_record(record: Any, policy: dict[str, Any]) -> list[str]:
    _normalized, errors = validate_accepted_record(record, policy)
    return errors


def _write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            for row in rows:
                handle.write(json_line(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        if tmp_name:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Generator JSONL output")
    parser.add_argument(
        "--accepted", required=True, help="Validated training-ready JSONL"
    )
    parser.add_argument(
        "--rejected", required=True, help="Rejected rows/audit reasons JSONL"
    )
    parser.add_argument("--config", default=str(ROOT / "configs/distillation.yaml"))
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )

    try:
        config = load_yaml(args.config)
        policy = dict(config.get("generation", {}))
        source = Path(args.input).expanduser().resolve()
        accepted_path = Path(args.accepted).expanduser().resolve()
        rejected_path = Path(args.rejected).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if accepted_path == rejected_path or source in {accepted_path, rejected_path}:
            raise ValueError(
                "input/accepted/rejected dosyaları birbirinden farklı olmalı"
            )

        accepted_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = strict_json_loads(line)
                except DataValidationError as exc:
                    rejected_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "status": "REJECTED",
                            "line_number": line_number,
                            "rejection_reasons": ["invalid_json", type(exc).__name__],
                            "raw_excerpt": line.rstrip()[:1000],
                        }
                    )
                    continue
                errors = validate_record(record, policy)
                if errors:
                    rejected_rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "status": "REJECTED",
                            "line_number": line_number,
                            "example_id": record.get("example_id")
                            if isinstance(record, dict)
                            else None,
                            "rejection_reasons": errors,
                            "original_record": record,
                        }
                    )
                else:
                    accepted_rows.append(record)

        _write_atomic(accepted_path, accepted_rows)
        _write_atomic(rejected_path, rejected_rows)
        LOG.info("accepted=%d rejected=%d", len(accepted_rows), len(rejected_rows))
        LOG.info("accepted=%s rejected=%s", accepted_path, rejected_path)
        return 0 if accepted_rows else 2
    except (
        ConfigurationError,
        DataValidationError,
        FileNotFoundError,
        ValueError,
    ) as exc:
        LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
