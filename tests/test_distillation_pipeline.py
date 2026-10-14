from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gokturk_distill.config import ConfigurationError, resolve_provider
from gokturk_distill.generator import run_generation
from gokturk_distill.records import validate_accepted_record
from gokturk_distill.schema import (
    DataValidationError,
    strict_json_loads,
    validate_question_row,
    validate_steps,
)
from gokturk_distill.verifier import _safe_decimal_expression, verify_candidate
from scripts.train_qlora_h200 import _tokenize_rows


class FakeTeacher:
    def __init__(self, name: str, payload: dict | str):
        self.name = name
        self.model = "fake-" + name
        self.payload = payload
        self.call_count = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        self.call_count += 1
        self.last_prompt = user_prompt
        return (
            self.payload
            if isinstance(self.payload, str)
            else json.dumps(self.payload, ensure_ascii=False)
        )


def math_steps(answer: str = "42") -> dict:
    return {
        "understand": "17 ile 25'in toplamı isteniyor.",
        "plan": "İki tamsayıyı topla.",
        "solve": "17 + 25 = 42.",
        "verify": "17 + 25 işlemi 42 eder; sonuç doğrulandı.",
        "answer": answer,
        "confidence": 0.95,
    }


def planning_steps(
    answer: str = "Önce hedefi belirle, sonra kısa adımlarla ilerle.",
) -> dict:
    return {
        "understand": "Kısa ve uygulanabilir bir plan isteniyor.",
        "plan": "Hedefi ve mevcut zamanı sırala.",
        "solve": "Önce hedefi belirle, sonra kısa adımlarla ilerle.",
        "verify": "Öneri istenen süreye sığıyor ve uygulanabilir.",
        "answer": answer,
        "confidence": 0.9,
    }


class StrictJsonTests(unittest.TestCase):
    def test_rejects_duplicate_keys_and_non_json_constants(self):
        with self.assertRaises(DataValidationError):
            strict_json_loads('{"a": 1, "a": 2}')
        with self.assertRaises(DataValidationError):
            strict_json_loads('{"a": NaN}')
        with self.assertRaises(DataValidationError):
            strict_json_loads("```json\n{}\n```")

    def test_steps_are_exactly_six_fields(self):
        result, errors = validate_steps(math_steps())
        self.assertEqual(errors, [])
        self.assertEqual(
            tuple(result),
            ("understand", "plan", "solve", "verify", "answer", "confidence"),
        )
        extra = math_steps()
        extra["chain_of_thought"] = "extra"
        _, errors = validate_steps(extra)
        self.assertTrue(any("beklenmeyen" in error for error in errors))

    def test_question_aliases_and_math_expression(self):
        question, errors = validate_question_row(
            {
                "id": "q1",
                "category": "mathematics",
                "question": "1+1?",
                "verifier": {"type": "numeric", "expression": "1 + 1"},
            }
        )
        self.assertEqual(errors, [])
        self.assertEqual(question["category"], "matematik")
        self.assertEqual(_safe_decimal_expression("(8 + 4) / 3"), 4)
        _, errors = validate_question_row(
            {
                "id": "unused-reference",
                "category": "planlama",
                "question": "Bir plan yaz.",
                "expected_answer": "Önce hedefi belirle.",
                "verifier": {"type": "consensus"},
            }
        )
        self.assertTrue(any("kullanılmıyor" in error for error in errors))
        _, errors = validate_question_row(
            {
                "id": "empty-reference",
                "category": "matematik",
                "question": "Sonuç kaç?",
                "expected_answer": [],
                "verifier": {"type": "numeric"},
            }
        )
        self.assertTrue(any("boş olamaz" in error for error in errors))
        with self.assertRaises(ValueError):
            _safe_decimal_expression("__import__('os').system('echo unsafe')")


class ProviderConfigTests(unittest.TestCase):
    def test_api_key_is_resolved_from_environment_and_redacted(self):
        config = resolve_provider(
            "test",
            {
                "api_key_env": "TEST_TEACHER_KEY",
                "base_url": "https://api.example.test/v1",
                "default_model": "test-model",
            },
            environ={"TEST_TEACHER_KEY": "super-secret-value"},
        )
        self.assertEqual(config.api_key, "super-secret-value")
        self.assertNotIn("super-secret-value", repr(config))
        self.assertEqual(config.api_key_env, "TEST_TEACHER_KEY")

    def test_base_url_rejects_embedded_credentials(self):
        with self.assertRaises(ConfigurationError):
            resolve_provider(
                "test",
                {
                    "api_key_env": "TEST_TEACHER_KEY",
                    "base_url": "https://user:pass@api.example.test/v1",
                    "default_model": "test-model",
                },
                environ={"TEST_TEACHER_KEY": "secret"},
            )


class TrainingTokenizationTests(unittest.TestCase):
    def test_completion_labels_are_masked_and_eos_is_appended(self):
        class FakeTokenizer:
            eos_token_id = 9

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                self.asserted_prompt = messages
                return [1, 2]

            def encode(self, text, *, add_special_tokens):
                return [3, 4]

        tokenizer = FakeTokenizer()
        row = {
            "question": {"question": "17+25?"},
            "steps": math_steps(),
        }
        encoded, too_long = _tokenize_rows([row], tokenizer, max_seq_len=5)
        self.assertEqual(too_long, 0)
        self.assertEqual(encoded[0]["input_ids"], [1, 2, 3, 4, 9])
        self.assertEqual(encoded[0]["labels"], [-100, -100, 3, 4, 9])
        filtered, too_long = _tokenize_rows([row], tokenizer, max_seq_len=4)
        self.assertEqual(len(filtered), 0)
        self.assertEqual(too_long, 1)


class CandidateVerificationTests(unittest.TestCase):
    def test_numeric_reference_rejects_answer_or_trace_contradiction(self):
        question = {
            "id": "q1",
            "category": "matematik",
            "question": "17+25?",
            "expected_answer": 42,
            "verifier": {"type": "numeric", "tolerance": 0},
        }
        good, errors = verify_candidate(question, math_steps(), min_confidence=0.5)
        self.assertEqual(errors, [])
        self.assertEqual(good["answer"], "42")
        bad = math_steps("43")
        _, errors = verify_candidate(question, bad)
        self.assertTrue(errors)
        bad = math_steps()
        bad["verify"] = "İşlem sonucu 41, doğru."
        _, errors = verify_candidate(question, bad)
        self.assertTrue(any("verify" in error for error in errors))

    def test_turkish_decimal_and_thousands_separators_are_checked(self):
        question = {
            "id": "q2",
            "category": "matematik",
            "question": "Sonuç kaç?",
            "expected_answer": 1234,
            "verifier": {"type": "numeric", "tolerance": 0},
        }
        candidate = {
            "understand": "Bir sayısal sonuç isteniyor.",
            "plan": "Verilen işlemi hesapla.",
            "solve": "İşlem sonucu 1.234.",
            "verify": "Türkçe binlik ayırıcıyla sonuç 1.234 olarak kontrol edildi.",
            "answer": "1.234",
            "confidence": 0.9,
        }
        _, errors = verify_candidate(question, candidate)
        self.assertEqual(errors, [])

    def test_exact_verifier_accepts_declared_alternatives(self):
        question = {
            "id": "logic-alt",
            "category": "mantik",
            "question": "En uzun kişi kim?",
            "expected_answer": ["Ayşe", "Ayse"],
            "verifier": {"type": "exact"},
        }
        candidate = {
            "understand": "Kişiler karşılaştırılıyor.",
            "plan": "Verilen sıralamayı izle.",
            "solve": "Ayşe, Bora'dan; Bora, Cem'den uzundur.",
            "verify": "Sonuç Ayşe olarak doğrulandı.",
            "answer": "Ayşe",
            "confidence": 0.9,
        }
        _, errors = verify_candidate(question, candidate)
        self.assertEqual(errors, [])

    def test_python_syntax_is_parsed_not_executed(self):
        question = {
            "id": "code1",
            "category": "kodlama",
            "question": "Fonksiyon yaz.",
            "verifier": {"type": "consensus", "python_syntax": True},
        }
        candidate = {
            "understand": "Python fonksiyonu isteniyor.",
            "plan": "Kısa bir döngü kullan.",
            "solve": "Fonksiyon girdiyi kontrol ediyor.",
            "verify": "Örnek giriş için beklenen mantık izleniyor.",
            "answer": "```python\ndef f(x):\n    return x\n```",
            "confidence": 0.8,
        }
        _, errors = verify_candidate(question, candidate)
        self.assertEqual(errors, [])
        candidate["answer"] = "```python\ndef f(:\n    pass\n```"
        _, errors = verify_candidate(question, candidate)
        self.assertTrue(any("sözdizimi" in error for error in errors))


class GeneratorTests(unittest.TestCase):
    def _run(self, bank: list[dict], teachers: list[FakeTeacher]):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        base = Path(temp_dir.name)
        input_path = base / "questions.jsonl"
        accepted_path = base / "accepted.jsonl"
        rejected_path = base / "rejected.jsonl"
        input_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in bank),
            encoding="utf-8",
        )
        summary = run_generation(
            input_path=input_path,
            accepted_path=accepted_path,
            rejected_path=rejected_path,
            teachers=teachers,
            policy={
                "min_confidence": 0.5,
                "min_distinct_teachers": 2,
                "min_consensus": 2,
                "require_reference_for_math": True,
                "max_workers": 4,
                "samples_per_provider": 1,
            },
        )
        accepted = [
            json.loads(line)
            for line in accepted_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        rejected = [
            json.loads(line)
            for line in rejected_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        return summary, accepted, rejected

    def test_reference_verified_ensemble_and_record_round_trip(self):
        bank = [
            {
                "id": "math1",
                "category": "matematik",
                "question": "17+25?",
                "expected_answer": 42,
                "verifier": {"type": "numeric"},
            }
        ]
        teachers = [
            FakeTeacher("gemini", math_steps()),
            FakeTeacher("deepseek", math_steps()),
        ]
        summary, accepted, rejected = self._run(bank, teachers)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])
        self.assertEqual(accepted[0]["status"], "ACCEPTED")
        _, errors = validate_accepted_record(
            accepted[0],
            {
                "min_confidence": 0.5,
                "min_distinct_teachers": 2,
                "min_consensus": 2,
                "require_reference_for_math": True,
            },
        )
        self.assertEqual(errors, [])
        # Gold reference is not placed in the prompt sent to the external providers.
        self.assertNotIn("expected_answer", teachers[0].last_prompt)

    def test_unreferenced_data_needs_independent_answer_consensus(self):
        bank = [
            {"id": "plan1", "category": "planning", "question": "Kısa bir plan yap."}
        ]
        teachers = [
            FakeTeacher("gemini", planning_steps()),
            FakeTeacher("qwen", planning_steps()),
        ]
        summary, accepted, rejected = self._run(bank, teachers)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])

        tampered = json.loads(json.dumps(accepted[0]))
        tampered["teacher_votes"][1]["provider"] = tampered["teacher_votes"][0][
            "provider"
        ]
        _, audit_errors = validate_accepted_record(
            tampered,
            {
                "min_confidence": 0.5,
                "min_distinct_teachers": 2,
                "min_consensus": 2,
                "require_reference_for_math": True,
            },
        )
        self.assertIn("duplicate_teacher_vote_provider", audit_errors)

        disagreeing = [
            FakeTeacher("gemini", planning_steps("Önce dinlen.")),
            FakeTeacher("qwen", planning_steps("Önce çalış.")),
        ]
        summary, accepted, rejected = self._run(bank, disagreeing)
        self.assertEqual(summary["accepted"], 0)
        self.assertEqual(rejected[0]["status"], "REJECTED")
        self.assertTrue(
            any("consensus" in reason for reason in rejected[0]["rejection_reasons"])
        )

    def test_math_without_reference_is_rejected_before_provider_calls(self):
        bank = [
            {
                "id": "math-missing-reference",
                "category": "matematik",
                "question": "17+25?",
            }
        ]
        teachers = [
            FakeTeacher("gemini", math_steps()),
            FakeTeacher("deepseek", math_steps()),
        ]
        summary, accepted, rejected = self._run(bank, teachers)
        self.assertEqual(summary["accepted"], 0)
        self.assertEqual(accepted, [])
        self.assertEqual(teachers[0].call_count, 0)
        self.assertEqual(teachers[1].call_count, 0)
        self.assertIn(
            "math_reference_required_before_teacher_call",
            rejected[0]["rejection_reasons"],
        )

    def test_invalid_numeric_reference_is_rejected_before_provider_calls(self):
        bank = [
            {
                "id": "math-invalid-reference",
                "category": "matematik",
                "question": "Sonuç kaç?",
                "expected_answer": "sayı değil",
                "verifier": {"type": "numeric"},
            }
        ]
        teachers = [
            FakeTeacher("gemini", math_steps()),
            FakeTeacher("deepseek", math_steps()),
        ]
        summary, accepted, rejected = self._run(bank, teachers)
        self.assertEqual(summary["accepted"], 0)
        self.assertEqual(accepted, [])
        self.assertEqual(teachers[0].call_count, 0)
        self.assertEqual(teachers[1].call_count, 0)
        self.assertIn(
            "reference_unusable:VerificationError", rejected[0]["rejection_reasons"]
        )

    def test_bad_model_json_is_rejected_as_jsonl_audit_record(self):
        bank = [{"id": "plan2", "category": "planning", "question": "Plan yap."}]
        teachers = [
            FakeTeacher("gemini", "```json\n{}\n```"),
            FakeTeacher("qwen", planning_steps()),
        ]
        summary, accepted, rejected = self._run(bank, teachers)
        self.assertEqual(summary["accepted"], 0)
        self.assertEqual(accepted, [])
        self.assertEqual(rejected[0]["status"], "REJECTED")


if __name__ == "__main__":
    unittest.main()
