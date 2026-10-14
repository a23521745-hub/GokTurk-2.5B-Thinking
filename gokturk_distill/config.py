"""Configuration loading and secret handling for the distillation pipeline."""

from __future__ import annotations

import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """Raised for missing or unsafe configuration."""


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    api_key_env: str = ""
    external: bool = True
    json_mode: bool = True
    timeout_seconds: float = 90.0
    max_retries: int = 3


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: Any, environ: Mapping[str, str] | None = None) -> Any:
    """Expand ${NAME} and ${NAME:-default} recursively; leave non-strings untouched."""
    env = environ if environ is not None else os.environ
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            resolved = env.get(name)
            if resolved is None or resolved == "":
                if default is None:
                    raise ConfigurationError(
                        f"Required environment variable {name} is not set"
                    )
                return default
            return resolved

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand_env(item, env) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item, env) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a UTF-8 YAML mapping, with a clear error if PyYAML is missing."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only in minimal installs
        raise ConfigurationError(
            "PyYAML gereklidir: pip install -r requirements.txt"
        ) from exc

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Yapılandırma dosyası bulunamadı: {config_path}")
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigurationError(
            f"YAML okunamadı: {config_path} ({type(exc).__name__})"
        ) from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"YAML kökü bir nesne (mapping) olmalı: {config_path}")
    return expand_env(value)


def load_env_file(path: str | Path | None) -> None:
    """Load an explicitly selected .env file without overriding the process environment.

    The file is rejected on POSIX if group/other permissions allow reading it. This keeps
    API credentials out of source control and discourages world-readable secret files.
    """
    if path is None:
        return
    env_path = Path(path).expanduser().resolve()
    if not env_path.is_file():
        raise ConfigurationError(f".env dosyası bulunamadı: {env_path}")

    if os.name == "posix":
        mode = stat.S_IMODE(env_path.stat().st_mode)
        if mode & 0o077:
            raise ConfigurationError(
                f"{env_path} dosya izinleri {mode:o}; anahtar dosyası için chmod 600 {env_path} uygulayın"
            )

    try:
        from dotenv import load_dotenv
    except ImportError as exc:  # pragma: no cover - optional local setup
        raise ConfigurationError(
            "python-dotenv gereklidir: pip install -r requirements.txt"
        ) from exc
    load_dotenv(dotenv_path=env_path, override=False, encoding="utf-8")


def _first_env_value(
    names: str | list[str], environ: Mapping[str, str]
) -> tuple[str | None, str | None]:
    candidates = [names] if isinstance(names, str) else names
    for name in candidates:
        value = environ.get(name)
        if value and value.strip():
            return value.strip(), name
    return None, None


def resolve_provider(
    name: str,
    raw: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
    timeout_seconds: float = 90.0,
    max_retries: int = 3,
) -> ProviderConfig:
    """Resolve a provider's model, URL and API key without ever logging the key."""
    env = environ if environ is not None else os.environ
    key_names = raw.get("api_key_env", [])
    if isinstance(key_names, str):
        key_names = [key_names]
    api_key, key_name = _first_env_value(key_names, env)
    if not api_key:
        names = ", ".join(str(x) for x in key_names) or "<tanımsız>"
        raise ConfigurationError(
            f"{name} için API anahtarı yok; şu ortam değişkenlerinden birini ayarlayın: {names}"
        )

    base_url = str(raw.get("base_url", "")).strip()
    base_url_env = str(raw.get("base_url_env", "")).strip()
    if base_url_env and env.get(base_url_env, "").strip():
        base_url = env[base_url_env].strip()
    if not base_url:
        raise ConfigurationError(
            f"{name} için base URL boş; {base_url_env or 'provider config'} ayarlayın"
        )
    parsed = urlparse(base_url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ConfigurationError(
            f"{name} base URL HTTPS olmalı; query/fragment veya gömülü kimlik bilgisi içermemeli"
        )

    model = str(raw.get("default_model", "")).strip()
    model_env = str(raw.get("model_env", "")).strip()
    if model_env and env.get(model_env, "").strip():
        model = env[model_env].strip()
    if not model:
        raise ConfigurationError(
            f"{name} model adı boş; {model_env or 'provider config'} ayarlayın"
        )

    resolved_timeout = float(raw.get("timeout_seconds", timeout_seconds))
    resolved_retries = int(raw.get("max_retries", max_retries))
    if not math.isfinite(resolved_timeout) or resolved_timeout <= 0:
        raise ConfigurationError(f"{name} timeout_seconds pozitif ve sonlu olmalı")
    if not 0 <= resolved_retries <= 10:
        raise ConfigurationError(f"{name} max_retries 0–10 arasında olmalı")

    return ProviderConfig(
        name=name,
        base_url=base_url,
        model=model,
        api_key=api_key,
        api_key_env=key_name or "",
        external=bool(raw.get("external", True)),
        json_mode=bool(raw.get("json_mode", True)),
        timeout_seconds=resolved_timeout,
        max_retries=resolved_retries,
    )
