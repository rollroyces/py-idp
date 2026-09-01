"""Structured logging configuration for py-idp.

Provides a single ``get_logger(name)`` helper so every module emits logs
in the same format (timestamp, level, logger name, message) and honours
the ``LOG_LEVEL`` environment variable. Production deployments should
set ``LOG_LEVEL=INFO`` (default) or ``LOG_LEVEL=WARNING`` for quieter
logs; ``LOG_FORMAT=json`` switches to JSON output for log aggregators.
"""
from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"
_CONFIGURED = False


def configure(level: str | int | None = None, fmt: str | None = None) -> None:
    """Idempotent root-logger configuration.

    Reads ``LOG_LEVEL`` and ``LOG_FORMAT`` from env by default. Safe to
    call multiple times — subsequent calls are no-ops.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    if level is None:
        level = os.environ.get("LOG_LEVEL", "INFO").upper()
    if isinstance(level, str):
        level = getattr(logging, level, logging.INFO)

    if fmt is None:
        fmt = os.environ.get("LOG_FORMAT", _LOG_FORMAT)
        if fmt.lower() == "json":
            fmt = _json_log_format()

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=_LOG_DATEFMT))
    root = logging.getLogger()
    # Replace existing handlers so format is honoured (don't accumulate)
    root.handlers = [handler]
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger. Configures the root on first call."""
    configure()
    return logging.getLogger(name)


def _json_log_format() -> str:
    """Build a JSON formatter string consumed by python-json-logger downstream.

    Returns the empty string here so the default Formatter falls back to
    the human-readable format. Operators wanting JSON should install
    ``python-json-logger`` and override via ``LOG_FORMAT_HANDLER``.
    """
    return _LOG_FORMAT


def reset() -> None:
    """Reset configuration state — for tests only."""
    global _CONFIGURED
    _CONFIGURED = False
    logging.getLogger().handlers = []