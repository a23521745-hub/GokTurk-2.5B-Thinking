"""OpenAI-compatible teacher adapters with bounded retries and redacted failures."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .config import ProviderConfig


class TeacherCallError(RuntimeError):
    """API failure safe to log: it never contains a URL, prompt or credential."""

    def __init__(self, provider: str, error_kind: str, *, retryable: bool = False):
        super().__init__(f"{provider}: {error_kind}")
        self.provider = provider
        self.error_kind = error_kind
        self.retryable = retryable


class Teacher(Protocol):
    name: str
    model: str

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        """Return the model's assistant content, which must be a single JSON object."""


@dataclass
class OpenAICompatibleTeacher:
    config: ProviderConfig
    client: Any

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model(self) -> str:
        return self.config.model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float,
        max_output_tokens: int,
    ) -> str:
        try:
            from openai import APIConnectionError, APITimeoutError, RateLimitError
        except ImportError as exc:  # pragma: no cover - dependency message
            raise TeacherCallError(self.name, "openai_sdk_missing") from exc

        last_kind = "unknown_api_error"
        for attempt in range(self.config.max_retries + 1):
            try:
                request: dict[str, Any] = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_output_tokens,
                    "stream": False,
                }
                if self.config.json_mode:
                    request["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(**request)
                if not getattr(response, "choices", None):
                    raise TeacherCallError(self.name, "empty_choices")
                choice = response.choices[0]
                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length":
                    raise TeacherCallError(self.name, "output_truncated")
                if finish_reason in {"content_filter", "refusal"}:
                    raise TeacherCallError(self.name, "refused_or_filtered")
                message = getattr(choice, "message", None)
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    pieces = []
                    for part in content:
                        text = getattr(part, "text", None)
                        if text is None and isinstance(part, dict):
                            text = part.get("text")
                        if isinstance(text, str):
                            pieces.append(text)
                    content = "".join(pieces)
                if not isinstance(content, str) or not content.strip():
                    raise TeacherCallError(self.name, "empty_content")
                return content.strip()
            except TeacherCallError:
                raise
            except Exception as exc:  # SDK and gateway errors vary by provider.
                status_code = getattr(exc, "status_code", None)
                is_retryable = isinstance(
                    exc, (APIConnectionError, APITimeoutError, RateLimitError)
                ) or status_code in {408, 409, 425, 429, 500, 502, 503, 504}
                last_kind = f"{type(exc).__name__}"[:80]
                if not is_retryable or attempt >= self.config.max_retries:
                    raise TeacherCallError(
                        self.name, last_kind, retryable=bool(is_retryable)
                    ) from exc
                delay = min(20.0, 0.75 * (2**attempt)) * random.uniform(0.75, 1.25)
                time.sleep(delay)
        raise TeacherCallError(self.name, last_kind, retryable=True)


def create_teacher(config: ProviderConfig) -> OpenAICompatibleTeacher:
    """Construct an OpenAI SDK client lazily so validators/tests need no SDK install."""
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency message
        raise RuntimeError(
            "OpenAI SDK gereklidir: pip install -r requirements.txt"
        ) from exc
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
        timeout=config.timeout_seconds,
        max_retries=0,  # Retries are explicit above and only for transient errors.
    )
    return OpenAICompatibleTeacher(config=config, client=client)
