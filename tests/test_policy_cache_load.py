"""Tests for PolicyCache.load behaviour on missing vs corrupt files."""
from __future__ import annotations

import json
import logging

from idp.rl.online import PolicyCache


def test_missing_policy_file_silent_on_first_run(tmp_path, caplog):
    """First-run: no file yet. Should NOT log a warning."""
    policy_path = tmp_path / "policy.json"
    assert not policy_path.exists()
    with caplog.at_level(logging.WARNING, logger="idp.rl.online"):
        cache = PolicyCache(str(policy_path), flush_interval_sec=10, min_reviews=1)
    # No warning expected for missing file
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 0, f"expected no warnings on missing file, got: {warnings}"
    # Defaults returned
    assert cache.policy.min_reviews == 1
    cache.stop()


def test_corrupt_policy_file_logs_error(tmp_path, caplog):
    """Existing-but-corrupt file: should log at ERROR level (loud)."""
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("not valid json {")  # corrupt
    with caplog.at_level(logging.WARNING, logger="idp.rl.online"):
        cache = PolicyCache(str(policy_path), flush_interval_sec=10, min_reviews=1)
    # An error-level log is expected
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("corrupt" in r.getMessage() for r in errors), \
        f"expected 'corrupt' error message, got: {[r.getMessage() for r in errors]}"
    # Cache still functions with defaults
    assert cache.policy.min_reviews == 1
    cache.stop()


def test_invalid_policy_shape_loads_with_defaults(tmp_path, caplog):
    """Valid JSON with weird-but-coercible values: Pydantic may or may not coerce.

    This test just verifies no crash — the actual coercion behaviour is
    Pydantic's choice. Either way, the cache must come up with sensible values.
    """
    policy_path = tmp_path / "policy.json"
    # 'not-a-numeric-value' forces Pydantic to either coerce or fail.
    # Either is acceptable; we just verify no uncaught exception.
    policy_path.write_text(json.dumps({"min_reviews": "7"}))
    with caplog.at_level(logging.WARNING, logger="idp.rl.online"):
        cache = PolicyCache(str(policy_path), flush_interval_sec=10, min_reviews=1)
    # No uncaught exception; cache.policy.min_reviews is either 7 (int) or "7" (str).
    assert cache.policy.min_reviews in (7, "7")
    cache.stop()


def test_valid_policy_file_loads_silently(tmp_path, caplog):
    """Valid file: no warnings, no errors."""
    policy_path = tmp_path / "policy.json"
    policy = {
        "high_failure_threshold": 0.3,
        "high_failure_confidence_floor": 0.95,
        "base_confidence_floor": 0.6,
        "high_failure_penalty": 0.2,
        "min_reviews": 5,
        "field_floors": {"vendor_name": 0.95},
        "field_penalties": {"vendor_name": 0.2},
    }
    policy_path.write_text(json.dumps(policy))
    with caplog.at_level(logging.WARNING, logger="idp.rl.online"):
        cache = PolicyCache(str(policy_path), flush_interval_sec=10, min_reviews=1)
    assert cache.policy.min_reviews == 5  # from file
    assert cache.policy.field_floors.get("vendor_name") == 0.95
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 0
    cache.stop()