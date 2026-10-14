"""Strict data contracts shared by generation, verification and training."""

from __future__ import annotations

import json
import math
import unicodedata
from typing import Any

SCHEMA_VERSION = "gokturk.reasoning.v1"
STEP_KEYS = ("understand", "plan", "solve", "verify", "answer", "confidence")
TEXT_LIMITS = {
    "understand": 600,
    "plan": 800,
    "solve": 1600,
    "verify": 800,
    "answer": 1200,
}
CATEGORY_ALIASES = {
    "matematik": "matematik",
    "mathematics": "matematik",
    "math": "matematik",
    "mantik": "mantik",
    "mantık": "mantik",
    "logic": "mantik",
    "turkce_talimat": "turkce_talimat",
    "türkçe_talimat": "turkce_talimat",
    "turkish_instruction": "turkce_talimat",
    "instruction": "turkce_talimat",
    "planlama": "planlama",
    "planning": "planlama",
    "kodlama": "kodlama",
    "coding": "kodlama",
    "code": "kodlama",
}

# JSON Schema for the exact payload expected from each teacher. The implementation below
# also validates it without requiring a JSON Schema dependency.
REASONING_STEPS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GokTurkReasoningSteps",
    "type": "object",
    "additionalProperties": False,
    "required": list(STEP_KEYS),
    "properties": {
        "understand": {
            "type": "string",
            "minLength": 1,
            "maxLength": TEXT_LIMITS["understand"],
        },
        "plan": {"type": "string", "minLength": 1, "maxLength": TEXT_LIMITS["plan"]},
        "solve": {"type": "string", "minLength": 1, "maxLength": TEXT_LIMITS["solve"]},
        "verify": {
            "type": "string",
            "minLength": 1,
            "maxLength": TEXT_LIMITS["verify"],
        },
        "answer": {
            "type": "string",
            "minLength": 1,
            "maxLength": TEXT_LIMITS["answer"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}


class DataValidationError(ValueError):
    """Raised when an input row is malformed or unsafe to process."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataValidationError(f"JSON nesnesinde yinelenen anahtar: {key}")
        result[key] = value
    return result


def _reject_non_json_constant(value: str) -> None:
    raise DataValidationError(f"JSON standardı dışı sayısal sabit: {value}")


def strict_json_loads(text: str) -> Any:
    """Parse RFC 8259 JSON; reject duplicate keys, NaN/Infinity, BOM and trailing prose."""
    if not isinstance(text, str) or not text.strip():
        raise DataValidationError("Boş JSON metni")
    if text.startswith("\ufeff"):
        raise DataValidationError("JSON BOM ile başlamamalı")
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_non_json_constant,
        )
    except DataValidationError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DataValidationError(
            f"Geçersiz JSON: {exc.msg if hasattr(exc, 'msg') else type(exc).__name__}"
        ) from exc


def canonical_category(value: Any) -> str:
    if not isinstance(value, str):
        raise DataValidationError("category metin olmalı")
    key = (
        unicodedata.normalize("NFKC", value)
        .strip()
        .casefold()
        .replace("-", "_")
        .replace(" ", "_")
    )
    category = CATEGORY_ALIASES.get(key)
    if category is None:
        allowed = "matematik, mantik, turkce_talimat, planlama, kodlama"
        raise DataValidationError(
            f"Bilinmeyen category={value!r}; izin verilenler: {allowed}"
        )
    return category


def validate_steps(
    value: Any, *, min_confidence: float = 0.0
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate the teacher's exact six-key JSON object and return normalized values."""
    errors: list[str] = []
    if not isinstance(value, dict):
        return None, ["steps bir JSON nesnesi olmalı"]

    keys = set(value)
    expected = set(STEP_KEYS)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing:
        errors.append("eksik alanlar: " + ", ".join(missing))
    if extra:
        errors.append("beklenmeyen alanlar: " + ", ".join(extra))

    normalized: dict[str, Any] = {}
    for key, limit in TEXT_LIMITS.items():
        item = value.get(key)
        if not isinstance(item, str):
            errors.append(f"{key} metin olmalı")
            continue
        text = unicodedata.normalize("NFC", item).strip()
        if not text:
            errors.append(f"{key} boş olamaz")
        if len(text) > limit:
            errors.append(f"{key} {limit} karakter sınırını aşıyor")
        if any(ord(ch) < 32 and ch not in "\n\r\t" for ch in text):
            errors.append(f"{key} geçersiz kontrol karakteri içeriyor")
        normalized[key] = text

    confidence = value.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append("confidence sonlu bir JSON sayısı olmalı")
    elif not math.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        errors.append("confidence 0 ile 1 arasında sonlu olmalı")
    else:
        normalized["confidence"] = float(confidence)
        if float(confidence) < min_confidence:
            errors.append(f"confidence {min_confidence:.2f} eşiğinin altında")

    if errors:
        return None, errors
    # Insertion order is intentional: it is also the stable SFT JSON field order.
    return {key: normalized[key] for key in STEP_KEYS}, []


def validate_question_row(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate and normalize one question-bank object."""
    if not isinstance(value, dict):
        return None, ["soru satırı bir JSON nesnesi olmalı"]
    errors: list[str] = []

    raw_id = value.get("id", value.get("example_id"))
    if (
        not isinstance(raw_id, (str, int))
        or isinstance(raw_id, bool)
        or not str(raw_id).strip()
    ):
        errors.append("id boş olmayan metin veya tamsayı olmalı")
        example_id = ""
    else:
        example_id = str(raw_id).strip()
        if len(example_id) > 160:
            errors.append("id 160 karakter sınırını aşıyor")

    try:
        category = canonical_category(value.get("category"))
    except DataValidationError as exc:
        errors.append(str(exc))
        category = ""

    question = value.get("question")
    if not isinstance(question, str) or not question.strip():
        errors.append("question boş olmayan metin olmalı")
        question = ""
    else:
        question = unicodedata.normalize("NFC", question).strip()
        if len(question) > 20000:
            errors.append("question 20.000 karakter sınırını aşıyor")

    verifier = value.get("verifier", {})
    if verifier is None:
        verifier = {}
    if not isinstance(verifier, dict):
        errors.append("verifier bir JSON nesnesi olmalı")
        verifier = {}
    verifier = dict(verifier)
    allowed_verifier_keys = {"type", "tolerance", "expression", "python_syntax"}
    unknown_verifier_keys = sorted(set(verifier) - allowed_verifier_keys)
    if unknown_verifier_keys:
        errors.append(
            "bilinmeyen verifier alanları: "
            + ", ".join(str(key) for key in unknown_verifier_keys)
        )
    verifier_type = verifier.get("type")
    allowed_types = {"numeric", "exact", "contains", "consensus", "none"}
    if verifier_type is not None and (
        not isinstance(verifier_type, str) or verifier_type not in allowed_types
    ):
        errors.append(
            f"verifier.type şu değerlerden biri olmalı: {', '.join(sorted(allowed_types))}"
        )
    if "expression" in verifier and (
        not isinstance(verifier["expression"], str)
        or not verifier["expression"].strip()
    ):
        errors.append("verifier.expression boş olmayan metin olmalı")
    if "python_syntax" in verifier and not isinstance(verifier["python_syntax"], bool):
        errors.append("verifier.python_syntax boolean olmalı")
    tolerance = verifier.get("tolerance", 0)
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or tolerance < 0
    ):
        errors.append("verifier.tolerance negatif olmayan sonlu bir sayı olmalı")
    else:
        verifier["tolerance"] = float(tolerance)

    has_expected = "expected_answer" in value
    expected_answer = value.get("expected_answer")
    if has_expected and not isinstance(expected_answer, (str, int, float, bool, list)):
        errors.append("expected_answer JSON string/number/boolean/list olmalı")
    if isinstance(expected_answer, str) and not expected_answer.strip():
        errors.append("expected_answer boş metin olamaz")
    if isinstance(expected_answer, list) and any(
        isinstance(item, str) and not item.strip() for item in expected_answer
    ):
        errors.append("expected_answer list boş metin içeremez")
    if isinstance(expected_answer, list) and not expected_answer:
        errors.append("expected_answer list boş olamaz")
    if isinstance(expected_answer, list) and any(
        not isinstance(item, (str, int, float, bool))
        or isinstance(item, float)
        and not math.isfinite(item)
        for item in expected_answer
    ):
        errors.append(
            "expected_answer list öğeleri yalnızca sonlu scalar değerler olabilir"
        )
    if isinstance(expected_answer, float) and not math.isfinite(expected_answer):
        errors.append("expected_answer sonlu olmalı")
    if verifier_type is None:
        if (
            "expression" in verifier
            or isinstance(expected_answer, (int, float))
            and not isinstance(expected_answer, bool)
        ):
            verifier["type"] = "numeric"
        elif has_expected:
            verifier["type"] = "exact"
        else:
            verifier["type"] = "consensus"

    if has_expected and verifier.get("type") in {"consensus", "none"}:
        errors.append(
            "expected_answer istenmiş ancak verifier tarafından kullanılmıyor; numeric/exact/contains seçin veya alanı kaldırın"
        )
    if (
        verifier.get("type") in {"numeric", "exact", "contains"}
        and not has_expected
        and "expression" not in verifier
    ):
        errors.append(
            f"verifier.type={verifier['type']} için expected_answer veya expression gerekli"
        )
    if "expression" in verifier and verifier.get("type") not in {None, "numeric"}:
        errors.append("expression yalnızca numeric doğrulayıcı ile kullanılabilir")

    if errors:
        return None, errors

    normalized: dict[str, Any] = {
        "id": example_id,
        "category": category,
        "question": question,
        "verifier": verifier,
    }
    if has_expected:
        normalized["expected_answer"] = expected_answer
    for key in ("difficulty", "source"):
        if key in value:
            normalized[key] = value[key]
    return normalized, []


def json_line(value: Any) -> str:
    """Serialize one strict JSON value suitable for a JSONL dataset."""
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
