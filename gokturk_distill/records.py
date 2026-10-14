"""Validation of accepted JSONL records before they enter the SFT dataset."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .schema import SCHEMA_VERSION, validate_question_row
from .verifier import answer_fingerprint, verify_candidate


def validate_accepted_record(
    record: Any, policy: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, list[str]]:
    """Re-run local checks and validate recorded teacher-quorum evidence."""
    if not isinstance(record, dict):
        return None, ["record_not_object"]
    errors: list[str] = []
    if record.get("status") != "ACCEPTED":
        errors.append("record_status_is_not_ACCEPTED")
    if record.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported_schema_version")
    if errors:
        return None, errors

    question_data: dict[str, Any] = {
        "id": record.get("example_id"),
        "category": record.get("category"),
        "question": record.get("question"),
    }
    if "expected_answer" in record:
        question_data["expected_answer"] = record["expected_answer"]
    if "verifier" in record:
        question_data["verifier"] = record["verifier"]
    question, question_errors = validate_question_row(question_data)
    if question is None:
        errors.extend("invalid_question_metadata:" + error for error in question_errors)
        return None, errors

    steps, step_errors = verify_candidate(
        question,
        record.get("steps"),
        min_confidence=float(policy.get("min_confidence", 0.55)),
        require_reference_for_math=bool(policy.get("require_reference_for_math", True)),
    )
    if steps is None:
        errors.extend("candidate_validation:" + error for error in step_errors)

    ensemble = record.get("ensemble")
    votes = record.get("teacher_votes")
    provider_list = record.get("teacher_providers")
    if (
        not isinstance(ensemble, dict)
        or not isinstance(votes, list)
        or not isinstance(provider_list, list)
    ):
        errors.append("missing_ensemble_audit")
        return None, errors
    if any(not isinstance(item, dict) for item in votes):
        errors.append("invalid_teacher_votes")
        return None, errors

    min_teachers = int(policy.get("min_distinct_teachers", 2))
    vote_providers: list[str] = []
    vote_pairs: set[tuple[str, str]] = set()
    vote_fingerprints: list[str] = []
    for item in votes:
        provider = item.get("provider")
        model = item.get("model")
        fingerprint = item.get("answer_fingerprint")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (provider, model, fingerprint)
        ):
            errors.append("invalid_teacher_vote_fields")
            continue
        provider = provider.strip()
        model = model.strip()
        vote_providers.append(provider)
        vote_pairs.add((provider, model))
        vote_fingerprints.append(fingerprint)

    providers = set(vote_providers)
    if len(vote_providers) != len(providers):
        errors.append("duplicate_teacher_vote_provider")
    if len(providers) < min_teachers:
        errors.append(f"teacher_quorum_not_met:{len(providers)}/{min_teachers}")
    if any(not isinstance(item, str) or not item.strip() for item in provider_list):
        errors.append("invalid_teacher_providers_audit")
    else:
        normalized_provider_list = [item.strip() for item in provider_list]
        if len(normalized_provider_list) != len(set(normalized_provider_list)):
            errors.append("duplicate_teacher_providers_audit")
        if set(normalized_provider_list) != providers:
            errors.append("teacher_providers_audit_mismatch")

    selected_provider = ensemble.get("selected_provider")
    selected_model = ensemble.get("selected_model")
    if (
        not isinstance(selected_provider, str)
        or not isinstance(selected_model, str)
        or (selected_provider.strip(), selected_model.strip()) not in vote_pairs
    ):
        errors.append("selected_teacher_missing_from_audit")

    verifier = question.get("verifier", {})
    reference_checked = verifier.get("type") in {"numeric", "exact", "contains"} and (
        "expected_answer" in question or bool(verifier.get("expression"))
    )
    recorded_reference_check = ensemble.get("reference_checked")
    if (
        not isinstance(recorded_reference_check, bool)
        or recorded_reference_check != reference_checked
    ):
        errors.append("reference_check_audit_mismatch")
    expected_method = (
        "programmatic_reference+independent_teachers"
        if reference_checked
        else "exact_normalized_teacher_consensus"
    )
    if ensemble.get("verification_method") != expected_method:
        errors.append("verification_method_audit_mismatch")

    agreement = ensemble.get("answer_agreement")
    if (
        isinstance(agreement, bool)
        or not isinstance(agreement, int)
        or agreement != len(votes)
    ):
        errors.append("answer_agreement_vote_count_mismatch")

    if not reference_checked:
        min_consensus = int(policy.get("min_consensus", 2))
        fingerprints = set(vote_fingerprints)
        if len(fingerprints) != 1 or not fingerprints:
            errors.append("teacher_answer_fingerprints_disagree_or_missing")
        if steps is not None:
            selected = answer_fingerprint(steps["answer"], question["category"])
            if fingerprints and selected not in fingerprints:
                errors.append("selected_answer_not_in_teacher_votes")
        if (
            isinstance(agreement, int)
            and not isinstance(agreement, bool)
            and agreement < min_consensus
        ):
            errors.append(f"answer_consensus_not_met:{agreement}/{min_consensus}")

    if errors or steps is None:
        return None, errors
    return {"question": question, "steps": steps}, []
