"""Configuration loader + validator.

Reads runtime configuration from environment variables, validates them,
and provides sensible defaults. Raises ``ConfigurationError`` on
unparseable values so the API server fails fast at startup rather than
at the first request.

All variables are prefixed with ``IDP_`` to avoid collisions with the
host environment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from idp._logging import get_logger
from idp.errors import ConfigurationError

_log = get_logger(__name__)


@dataclass(frozen=True)
class Settings:
    """Validated runtime configuration. Construct via ``Settings.load()``.

    Attributes are read-only to prevent accidental mutation across
    request handlers. Use ``Settings.override(...)`` for test fixtures.
    """

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_workers: int = 1

    # Auth
    api_key: str | None = None
    api_key_required: bool = True
    rate_limit_per_minute: int = 60  # 0 = unlimited

    # Storage
    storage_backend: str = "memory"  # memory | json | sql
    storage_path: str | None = None  # JSON path or SQL URL
    sql_pool_size: int = 5

    # LLM
    default_backend: str = "mock"
    default_model: str | None = None
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    # Uploads
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB
    max_pdf_pages: int = 100

    # Observability
    log_level: str = "INFO"
    log_format: str = "human"  # human | json
    metrics_enabled: bool = True

    # CORS (comma-separated; empty = same-origin only)
    cors_origins: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def load(cls, env: dict[str, str] | None = None) -> Settings:
        """Build Settings from environment, validating each value.

        ``env`` defaults to ``os.environ`` but tests can pass a stub dict.
        """
        e = env if env is not None else dict(os.environ)
        s = cls()
        kwargs: dict[str, Any] = {}

        # Server
        kwargs["api_host"] = e.get("IDP_API_HOST", s.api_host)
        kwargs["api_port"] = _validate_int("IDP_API_PORT", e.get("IDP_API_PORT"), default=s.api_port, min_v=1, max_v=65535)
        kwargs["api_workers"] = _validate_int("IDP_API_WORKERS", e.get("IDP_API_WORKERS"), default=s.api_workers, min_v=1, max_v=64)

        # Auth
        kwargs["api_key"] = e.get("IDP_API_KEY") or None
        kwargs["api_key_required"] = _validate_bool("IDP_API_KEY_REQUIRED", e.get("IDP_API_KEY_REQUIRED"), default=s.api_key_required)
        kwargs["rate_limit_per_minute"] = _validate_int("IDP_RATE_LIMIT_PER_MINUTE", e.get("IDP_RATE_LIMIT_PER_MINUTE"), default=s.rate_limit_per_minute, min_v=0, max_v=1_000_000)

        # Storage
        kwargs["storage_backend"] = _validate_choice("IDP_STORAGE_BACKEND", e.get("IDP_STORAGE_BACKEND"), choices=("memory", "json", "sql"), default=s.storage_backend)
        kwargs["storage_path"] = e.get("IDP_STORAGE_PATH") or None
        kwargs["sql_pool_size"] = _validate_int("IDP_SQL_POOL_SIZE", e.get("IDP_SQL_POOL_SIZE"), default=s.sql_pool_size, min_v=1, max_v=50)

        # LLM
        kwargs["default_backend"] = e.get("IDP_BACKEND", s.default_backend)
        kwargs["default_model"] = e.get("IDP_MODEL") or None
        kwargs["llm_timeout_seconds"] = _validate_float("IDP_LLM_TIMEOUT", e.get("IDP_LLM_TIMEOUT"), default=s.llm_timeout_seconds, min_v=0.1, max_v=600.0)
        kwargs["llm_max_retries"] = _validate_int("IDP_LLM_MAX_RETRIES", e.get("IDP_LLM_MAX_RETRIES"), default=s.llm_max_retries, min_v=0, max_v=10)

        # Uploads
        kwargs["max_upload_bytes"] = _validate_int("IDP_MAX_UPLOAD_BYTES", e.get("IDP_MAX_UPLOAD_BYTES"), default=s.max_upload_bytes, min_v=1024, max_v=1024**3)
        kwargs["max_pdf_pages"] = _validate_int("IDP_MAX_PDF_PAGES", e.get("IDP_MAX_PDF_PAGES"), default=s.max_pdf_pages, min_v=1, max_v=10_000)

        # Observability
        kwargs["log_level"] = _validate_choice("IDP_LOG_LEVEL", e.get("IDP_LOG_LEVEL"), choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), default=s.log_level)
        kwargs["log_format"] = _validate_choice("IDP_LOG_FORMAT", e.get("IDP_LOG_FORMAT"), choices=("human", "json"), default=s.log_format)
        kwargs["metrics_enabled"] = _validate_bool("IDP_METRICS_ENABLED", e.get("IDP_METRICS_ENABLED"), default=s.metrics_enabled)

        # CORS
        cors = e.get("IDP_CORS_ORIGINS", "").strip()
        kwargs["cors_origins"] = tuple(o.strip() for o in cors.split(",") if o.strip()) if cors else ()

        result = cls(**kwargs)
        _log.info("config loaded: api=%s:%d backend=%s storage=%s",
                  result.api_host, result.api_port, result.default_backend, result.storage_backend)
        return result


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
def _validate_int(name: str, raw: str | None, *, default: int, min_v: int, max_v: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        v = int(raw)
    except ValueError as e:
        raise ConfigurationError(f"{name} must be an integer, got {raw!r}") from e
    if v < min_v or v > max_v:
        raise ConfigurationError(f"{name}={v} out of range [{min_v}, {max_v}]")
    return v


def _validate_float(name: str, raw: str | None, *, default: float, min_v: float, max_v: float) -> float:
    if raw is None or raw == "":
        return default
    try:
        v = float(raw)
    except ValueError as e:
        raise ConfigurationError(f"{name} must be a number, got {raw!r}") from e
    if v < min_v or v > max_v:
        raise ConfigurationError(f"{name}={v} out of range [{min_v}, {max_v}]")
    return v


def _validate_bool(name: str, raw: str | None, *, default: bool) -> bool:
    if raw is None or raw == "":
        return default
    if raw.lower() in ("true", "1", "yes", "on"):
        return True
    if raw.lower() in ("false", "0", "no", "off"):
        return False
    raise ConfigurationError(f"{name} must be true/false, got {raw!r}")


def _validate_choice(name: str, raw: str | None, *, choices: tuple[str, ...], default: str) -> str:
    if raw is None or raw == "":
        return default
    if raw not in choices:
        raise ConfigurationError(f"{name}={raw!r} not in {choices}")
    return raw


__all__ = ["Settings"]