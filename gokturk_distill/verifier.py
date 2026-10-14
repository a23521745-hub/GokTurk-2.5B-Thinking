"""Deterministic rejection sampling and reference-answer checks.

This module never executes model-generated code. Numeric reference expressions are evaluated
with a deliberately tiny AST interpreter; optional Python answer checks use ast.parse only.
"""

from __future__ import annotations

import ast
import math
import re
import unicodedata
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from .schema import (
    DataValidationError,
    canonical_category,
    validate_question_row,
    validate_steps,
)

_NUMBER_RE = re.compile(
    r"(?<![\w])[-+]?(?:(?:\d{1,3}(?:[.,]\d{3})+)(?:[.,]\d+)?|\d+(?:[.,]\d+)?)(?:\s*/\s*\d+)?%?"
)
_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


class VerificationError(ValueError):
    """Raised when a reference cannot be safely evaluated."""


def _decimal_from_scalar(value: Any) -> Decimal:
    if isinstance(value, bool):
        raise VerificationError("boolean sayısal referans olarak kullanılamaz")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, int):
        result = Decimal(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise VerificationError("sonlu olmayan sayı")
        result = Decimal(str(value))
    elif isinstance(value, str):
        text = unicodedata.normalize("NFKC", value).strip().replace(" ", "")
        if not text:
            raise VerificationError("boş sayısal referans")
        if "," in text and "." in text:
            # Treat the rightmost separator as a decimal separator; the other is grouping.
            decimal_sep = "," if text.rfind(",") > text.rfind(".") else "."
            grouping_sep = "." if decimal_sep == "," else ","
            text = text.replace(grouping_sep, "")
            if decimal_sep == ",":
                text = text.replace(",", ".")
        else:
            text = text.replace(",", ".")
        if text.endswith("%"):
            return _decimal_from_scalar(text[:-1]) / Decimal(100)
        if "/" in text:
            left, right = text.split("/", 1)
            denominator = _decimal_from_scalar(right)
            if denominator == 0:
                raise VerificationError("sıfıra bölme")
            return _decimal_from_scalar(left) / denominator
        try:
            result = Decimal(text)
        except InvalidOperation as exc:
            raise VerificationError(f"sayısal değer okunamadı: {value!r}") from exc
    else:
        raise VerificationError(f"desteklenmeyen sayısal tür: {type(value).__name__}")
    if not result.is_finite():
        raise VerificationError("sonlu olmayan sayı")
    return result


def _safe_decimal_expression(expression: str) -> Decimal:
    """Evaluate arithmetic over numeric literals without eval or arbitrary names/calls."""
    if (
        not isinstance(expression, str)
        or not expression.strip()
        or len(expression) > 256
    ):
        raise VerificationError("numeric expression 1–256 karakter olmalı")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise VerificationError("numeric expression sözdizimi geçersiz") from exc

    def visit(node: ast.AST) -> Decimal:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            return _decimal_from_scalar(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            number = visit(node.operand)
            return number if isinstance(node.op, ast.UAdd) else -number
        if isinstance(node, ast.BinOp) and isinstance(
            node.op,
            (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow),
        ):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise VerificationError("numeric expression sıfıra bölüyor")
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                if right == 0:
                    raise VerificationError("numeric expression sıfıra bölüyor")
                return (left // right).to_integral_value()
            if isinstance(node.op, ast.Mod):
                if right == 0:
                    raise VerificationError("numeric expression sıfıra bölüyor")
                return left % right
            if isinstance(node.op, ast.Pow):
                if right != right.to_integral_value() or abs(int(right)) > 30:
                    raise VerificationError(
                        "üs tamsayı ve mutlak değeri en fazla 30 olmalı"
                    )
                return left ** int(right)
        raise VerificationError(
            f"numeric expression desteklenmeyen yapı içeriyor: {type(node).__name__}"
        )

    try:
        with localcontext() as context:
            context.prec = 50
            result = visit(tree)
    except (ArithmeticError, InvalidOperation, OverflowError) as exc:
        raise VerificationError("numeric expression hesaplanamadı") from exc
    if not result.is_finite():
        raise VerificationError("numeric expression sonlu değil")
    return result


def _number_variants(token: str) -> list[Decimal]:
    """Parse a localized token, retaining both decimal and thousands interpretations."""
    variants: list[Decimal] = []
    try:
        variants.append(_decimal_from_scalar(token))
    except VerificationError:
        return variants
    raw = unicodedata.normalize("NFKC", token).strip().replace(" ", "")
    percent = raw.endswith("%")
    if percent:
        raw = raw[:-1]
    fraction = "/" in raw
    if not fraction and ("." in raw) != ("," in raw):
        separator = "." if "." in raw else ","
        integer, fractional = raw.rsplit(separator, 1)
        if (
            len(fractional) == 3
            and integer.lstrip("+-").isdigit()
            and fractional.isdigit()
        ):
            try:
                grouped = Decimal(integer + fractional)
                if percent:
                    grouped /= Decimal(100)
                if grouped not in variants:
                    variants.append(grouped)
            except InvalidOperation:
                pass
    return variants


def _number_groups_in_text(text: str) -> list[list[Decimal]]:
    found: list[list[Decimal]] = []
    for match in _NUMBER_RE.finditer(text):
        variants = _number_variants(match.group(0).strip())
        if variants:
            found.append(variants)
    return found


def _close(actual: Decimal, expected: Decimal, tolerance: Decimal) -> bool:
    return abs(actual - expected) <= tolerance


def _turkish_casefold(text: str) -> str:
    # Unicode casefold of capital dotted İ is "i" + combining dot; canonicalize that
    # sequence without merging Turkish dotless ı into i (they are distinct letters).
    return unicodedata.normalize("NFC", text).casefold().replace("i\u0307", "i")


def _normalized_text(value: Any) -> str:
    text = _turkish_casefold(unicodedata.normalize("NFKC", str(value)))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_fingerprint(answer: str, category: str) -> str:
    """A conservative, deterministic fingerprint used only for teacher agreement."""
    normalized = unicodedata.normalize("NFKC", answer).strip().replace("\r\n", "\n")
    if category == "kodlama":
        # Preserve indentation and punctuation for source-code answers.
        return (
            "code:"
            + "\n".join(line.rstrip() for line in normalized.splitlines()).strip()
        )
    normalized = _turkish_casefold(normalized)
    normalized = re.sub(r"[\*`_#]", "", normalized)
    normalized = re.sub(r"[^\w%+./-]+", " ", normalized, flags=re.UNICODE)
    return "text:" + re.sub(r"\s+", " ", normalized).strip()


def _python_syntax_check(answer: str) -> str | None:
    blocks = _CODE_FENCE_RE.findall(answer)
    candidates = [block for block in blocks if block.strip()]
    if not candidates:
        # Also accept a bare source file, but do not attempt to strip prose into code.
        source = answer.strip()
        if not source or "\n" not in source:
            return "python_syntax doğrulaması için cevapta Python kod bloğu bulunamadı"
        candidates = [source]
    for source in candidates:
        try:
            ast.parse(source)
        except SyntaxError as exc:
            return f"Python kodunun sözdizimi geçersiz (satır {exc.lineno})"
    return None


def _reference_values(
    question: dict[str, Any],
) -> tuple[str, list[Decimal] | None, Any]:
    verifier = question.get("verifier") or {}
    kind = verifier.get("type", "consensus")
    if kind != "numeric":
        return kind, None, question.get("expected_answer")

    expression = verifier.get("expression")
    if expression:
        expected = [_safe_decimal_expression(expression)]
    else:
        raw = question.get("expected_answer")
        raw_values = raw if isinstance(raw, list) else [raw]
        if not raw_values:
            raise VerificationError("expected_answer list boş olamaz")
        expected = [_decimal_from_scalar(item) for item in raw_values]
    return kind, expected, question.get("expected_answer")


def validate_question_reference(question: dict[str, Any]) -> list[str]:
    """Validate an answer key/expression before spending a teacher API call."""
    try:
        kind, expected_numbers, raw_expected = _reference_values(question)
    except (VerificationError, TypeError, ValueError) as exc:
        return [f"reference_unusable:{type(exc).__name__}"]
    if kind == "numeric" and not expected_numbers:
        return ["numeric_reference_empty"]
    if kind in {"exact", "contains"} and raw_expected is None:
        return ["text_reference_missing"]
    return []


def verify_candidate(
    question: dict[str, Any],
    candidate: Any,
    *,
    min_confidence: float = 0.0,
    require_reference_for_math: bool = True,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate structure, local arithmetic/reference agreement and code syntax only.

    General natural-language truth cannot be proved from arbitrary text. In those cases the
    caller must enforce independent-teacher answer consensus; a majority is a filter, not a
    mathematical proof or a substitute for domain review.
    """
    normalized, errors = validate_steps(candidate, min_confidence=min_confidence)
    if normalized is None:
        return None, errors

    category = question.get("category", "")
    try:
        category = canonical_category(category)
    except DataValidationError as exc:
        errors.append(str(exc))
        category = ""

    verifier = question.get("verifier") or {}
    if verifier.get("python_syntax"):
        syntax_error = _python_syntax_check(normalized["answer"])
        if syntax_error:
            errors.append(syntax_error)

    has_reference = verifier.get("type") in {"numeric", "exact", "contains"} and (
        "expected_answer" in question or bool(verifier.get("expression"))
    )
    if category == "matematik" and require_reference_for_math and not has_reference:
        errors.append(
            "matematik sorusu için bağımsız expected_answer/expression gerekli"
        )

    try:
        kind, expected_numbers, raw_expected = _reference_values(question)
    except (VerificationError, TypeError, ValueError) as exc:
        errors.append(f"referans doğrulanamadı: {exc}")
        kind, expected_numbers, raw_expected = "consensus", None, None

    if kind == "numeric" and expected_numbers is not None:
        tolerance = _decimal_from_scalar(verifier.get("tolerance", 0))
        expected_count = len(expected_numbers)
        for field_name in ("answer", "verify", "solve"):
            value_groups = _number_groups_in_text(normalized[field_name])
            if len(value_groups) < expected_count:
                errors.append(
                    f"{field_name} alanında sayısal sonuç/referans bulunamadı"
                )
                continue
            actual_groups = value_groups[-expected_count:]
            if any(
                not any(_close(actual, expected, tolerance) for actual in variants)
                for variants, expected in zip(actual_groups, expected_numbers)
            ):
                errors.append(
                    f"{field_name} alanındaki son sayısal sonuç beklenen değerle çelişiyor"
                )

    elif kind in {"exact", "contains"} and raw_expected is not None:
        raw_values = raw_expected if isinstance(raw_expected, list) else [raw_expected]
        expected_values = [_normalized_text(item) for item in raw_values]
        answer = _normalized_text(normalized["answer"])
        verify = _normalized_text(normalized["verify"])
        if kind == "exact":
            matched = next((item for item in expected_values if answer == item), None)
        else:
            matched = next((item for item in expected_values if item in answer), None)
        if matched is None:
            errors.append(f"answer expected_answer ({kind}) ile eşleşmiyor")
        elif matched not in verify:
            errors.append("verify alanı referans cevabı doğrulamıyor")

    if errors:
        return None, errors
    return normalized, []


def validate_question(value: Any) -> tuple[dict[str, Any] | None, list[str]]:
    return validate_question_row(value)


def distinct_teacher_quorum(
    candidates: Iterable[dict[str, Any]], min_teachers: int
) -> bool:
    providers = {
        str(item.get("provider", "")).strip()
        for item in candidates
        if item.get("provider")
    }
    return len(providers) >= min_teachers
