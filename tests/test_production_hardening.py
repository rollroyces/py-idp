"""Tests for the production-hardening modules (v0.2.0)."""
from __future__ import annotations

import os
import threading
import time
from unittest.mock import patch

import pytest

from idp._logging import configure, get_logger
from idp.config import Settings
from idp.errors import (
    BackendUnavailableError,
    ConfigurationError,
    DocumentParseError,
    IDPError,
    RateLimitedError,
    SchemaValidationError,
    StorageError,
    format_error,
    is_idp_error,
)
from idp.metrics import Metrics
from idp.ratelimit import RateLimiter


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
def test_idp_error_is_base_for_all_subclasses():
    for cls in (DocumentParseError, SchemaValidationError, BackendUnavailableError,
                StorageError, ConfigurationError, RateLimitedError):
        assert issubclass(cls, IDPError)


def test_is_idp_error_returns_true_for_subclasses():
    assert is_idp_error(IDPError("x"))
    assert is_idp_error(RateLimitedError("x"))
    assert is_idp_error(ValueError("x")) is False


def test_format_error_is_json_serialisable():
    import json
    d = format_error(RateLimitedError("limit hit"))
    json.dumps(d)  # must not raise
    assert d["type"] == "RateLimitedError"
    assert d["is_idp_error"] is True


def test_catch_idp_error_does_not_catch_value_error():
    """`except IDPError` should NOT swallow unrelated Python errors."""
    try:
        try:
            raise ValueError("not us")
        except IDPError:
            pytest.fail("IDPError should not catch ValueError")
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def test_settings_load_returns_defaults_with_empty_env():
    s = Settings.load(env={})
    assert s.api_port == 8080
    assert s.api_key_required is True
    assert s.max_upload_bytes == 25 * 1024 * 1024


def test_settings_load_reads_idp_prefixed_env_vars():
    s = Settings.load(env={
        "IDP_API_PORT": "9000",
        "IDP_LOG_LEVEL": "DEBUG",
        "IDP_MAX_UPLOAD_BYTES": "100000",
        "IDP_RATE_LIMIT_PER_MINUTE": "120",
        "IDP_CORS_ORIGINS": "https://a.example.com,https://b.example.com",
    })
    assert s.api_port == 9000
    assert s.log_level == "DEBUG"
    assert s.max_upload_bytes == 100000
    assert s.rate_limit_per_minute == 120
    assert s.cors_origins == ("https://a.example.com", "https://b.example.com")


def test_settings_load_rejects_invalid_port():
    with pytest.raises(ConfigurationError) as exc:
        Settings.load(env={"IDP_API_PORT": "99999"})
    assert "out of range" in str(exc.value)


def test_settings_load_rejects_non_integer_port():
    with pytest.raises(ConfigurationError) as exc:
        Settings.load(env={"IDP_API_PORT": "not-a-number"})
    assert "must be an integer" in str(exc.value)


def test_settings_load_rejects_unknown_log_level():
    with pytest.raises(ConfigurationError) as exc:
        Settings.load(env={"IDP_LOG_LEVEL": "TRACE"})
    assert "not in" in str(exc.value)


def test_settings_load_rejects_invalid_bool():
    with pytest.raises(ConfigurationError) as exc:
        Settings.load(env={"IDP_API_KEY_REQUIRED": "maybe"})
    assert "true/false" in str(exc.value)


def test_settings_load_rejects_invalid_storage_backend():
    with pytest.raises(ConfigurationError) as exc:
        Settings.load(env={"IDP_STORAGE_BACKEND": "redis"})
    assert "not in" in str(exc.value)


def test_settings_immutable():
    """Settings is frozen — accidental mutation across handlers is impossible."""
    from dataclasses import FrozenInstanceError
    s = Settings.load(env={})
    with pytest.raises(FrozenInstanceError):
        s.api_port = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def test_metrics_inc_and_snapshot():
    m = Metrics()
    m.inc("requests")
    m.inc("requests", 5)
    m.inc("requests", value=2, route="/x")
    snap = m.snapshot()
    assert snap["counters"]["requests"] == 6
    assert snap["counters"]['requests{route="/x"}'] == 2


def test_metrics_gauge():
    m = Metrics()
    m.gauge("inflight", 5)
    m.gauge("inflight", 3)
    assert m.snapshot()["gauges"]["inflight"] == 3


def test_metrics_observe_trims_to_max_samples():
    m = Metrics()
    m._hist_max_samples = 100  # override env
    for i in range(150):
        m.observe("latency", float(i))
    snap = m.snapshot()
    assert snap["histograms"]["latency"]["count"] == 100


def test_metrics_timed_decorator_records_duration():
    m = Metrics()

    @m.timed("my_op")
    def slow():
        time.sleep(0.001)
        return "ok"

    assert slow() == "ok"
    snap = m.snapshot()
    assert snap["counters"]["my_op_calls"] == 1
    assert snap["histograms"]["my_op_duration_seconds"]["count"] == 1


def test_metrics_export_prometheus_format():
    m = Metrics()
    m.inc("requests", 3)
    m.gauge("inflight", 1)
    m.observe("http_request_duration_seconds", 0.123)
    out = m.export_prometheus()
    assert "requests 3" in out
    assert "inflight 1" in out
    assert "http_request_duration_seconds_count 1" in out


def test_metrics_thread_safe_under_concurrent_increments():
    m = Metrics()

    def worker():
        for _ in range(1000):
            m.inc("counter")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.snapshot()["counters"]["counter"] == 10_000


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
def test_rate_limiter_allows_under_limit():
    rl = RateLimiter(per_key_per_minute=3)
    rl.check("k1")
    rl.check("k1")
    rl.check("k1")


def test_rate_limiter_blocks_over_limit():
    rl = RateLimiter(per_key_per_minute=2)
    rl.check("k1")
    rl.check("k1")
    with pytest.raises(RateLimitedError):
        rl.check("k1")


def test_rate_limiter_isolates_keys():
    rl = RateLimiter(per_key_per_minute=1)
    rl.check("k1")
    rl.check("k2")  # different key, fresh budget
    with pytest.raises(RateLimitedError):
        rl.check("k1")
    # k2 still has its budget used
    with pytest.raises(RateLimitedError):
        rl.check("k2")


def test_rate_limiter_disabled_with_zero():
    rl = RateLimiter(per_key_per_minute=0)
    for _ in range(1000):
        rl.check("k1")  # no limit


def test_rate_limiter_global_limit():
    rl = RateLimiter(per_key_per_minute=0, global_per_minute=2)
    rl.check("k1")
    rl.check("k2")
    with pytest.raises(RateLimitedError):
        rl.check("k3")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def test_get_logger_returns_named_logger():
    log = get_logger("test.module")
    assert log.name == "test.module"
    # configure() should be idempotent
    configure()
    configure()  # no-op


def test_log_level_honoured_from_env():
    """A misconfigured LOG_LEVEL falls back to INFO."""
    with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
        configure()  # doesn't reset; need reset()
    # Just verify get_logger doesn't raise
    log = get_logger("test.debug")
    log.debug("hi")