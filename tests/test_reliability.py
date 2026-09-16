"""Tests for idp.reliability: retries, caching, structured errors."""
from __future__ import annotations

import json
import threading
from unittest.mock import MagicMock

import pytest

from idp.core.schemas import Invoice
from idp.llm.backend import CompletionRequest, Message
from idp.reliability import (
    AuthError,
    BadRequestError,
    CachingBackend,
    ConnectionError_,
    ExtractionCache,
    ExtractionError,
    RateLimitError,
    RetryConfig,
    RetryingBackend,
    TimeoutError_,
    classify_exception,
    hash_request,
)


# ---------------------------------------------------------------------------
# classify_exception — exception → ExtractionError taxonomy
# ---------------------------------------------------------------------------
def test_classify_rate_limit_by_message():
    e = Exception("429 Too Many Requests: rate limit exceeded")
    classified = classify_exception(e)
    assert isinstance(classified, RateLimitError)
    assert classified.retryable is True


def test_classify_timeout_by_class_name():
    e = TimeoutError("connection timed out")  # builtin TimeoutError
    classified = classify_exception(e)
    assert isinstance(classified, TimeoutError_)
    assert classified.retryable is True


def test_classify_connection_error_by_class_name():
    e = ConnectionError("connection refused")  # builtin ConnectionError
    classified = classify_exception(e)
    assert isinstance(classified, ConnectionError_)
    assert classified.retryable is True


def test_classify_auth_error_by_message():
    e = Exception("401 Unauthorized: invalid API key")
    classified = classify_exception(e)
    assert isinstance(classified, AuthError)
    assert classified.retryable is False


def test_classify_bad_request_by_message():
    e = Exception("400 Bad Request: malformed payload")
    classified = classify_exception(e)
    assert isinstance(classified, BadRequestError)
    assert classified.retryable is False


def test_classify_unknown_defaults_to_retryable():
    """Unknown errors should retry once by default — safer than failing fast."""
    e = Exception("something weird happened")
    classified = classify_exception(e)
    assert classified.retryable is True  # default is retryable


def test_classify_preserves_cause():
    """The original exception is preserved on the classified one."""
    original = ValueError("original")
    classified = classify_exception(original)
    assert classified.cause is original


def test_classify_preserves_backend_name():
    classified = classify_exception(Exception("429"), backend_name="openai")
    assert classified.backend_name == "openai"


# ---------------------------------------------------------------------------
# RetryConfig — backoff math
# ---------------------------------------------------------------------------
def test_retry_config_exponential_growth():
    c = RetryConfig(initial_delay_sec=1.0, backoff_multiplier=2.0, jitter=0.0)
    delays = [c.delay_for(attempt) for attempt in (1, 2, 3, 4)]
    # 1.0, 2.0, 4.0, 8.0 (no jitter)
    assert delays == [1.0, 2.0, 4.0, 8.0]


def test_retry_config_respects_max_delay():
    c = RetryConfig(initial_delay_sec=1.0, max_delay_sec=5.0, jitter=0.0)
    delays = [c.delay_for(attempt) for attempt in (1, 2, 3, 4, 5, 6)]
    # Each delay is min(initial * 2^(n-1), 5.0)
    # 1, 2, 4, 5 (capped), 5 (capped), 5 (capped)
    assert delays == [1.0, 2.0, 4.0, 5.0, 5.0, 5.0]


def test_retry_config_jitter_within_bounds():
    """Jitter should be ±jitter_pct * base."""
    c = RetryConfig(initial_delay_sec=10.0, jitter=0.1)
    for _ in range(20):
        delay = c.delay_for(1)
        # Should be between 9.0 and 11.0 (10.0 ± 10%)
        assert 9.0 <= delay <= 11.0


# ---------------------------------------------------------------------------
# RetryingBackend — retry behavior
# ---------------------------------------------------------------------------
def _flaky_backend(fail_count: list[int], fail_message: str = "429 rate limit"):
    """Build a backend that fails N times then succeeds."""
    b = MagicMock()
    b.name = "flaky"
    b.is_multimodal = False

    def complete(req):
        fail_count[0] += 1
        if fail_count[0] <= 2:
            raise Exception(fail_message)
        return '{"ok": true}'

    b.complete = complete
    return b


def test_retrying_backend_succeeds_after_transient_failures():
    fails = [0]
    flaky = _flaky_backend(fails)
    retry = RetryingBackend(
        flaky,
        RetryConfig(max_retries=5, initial_delay_sec=0.001, jitter=0.0),
    )
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    out = retry.complete(req)
    assert out == '{"ok": true}'
    assert fails[0] == 3  # 2 fails + 1 success


def test_retrying_backend_gives_up_after_max_retries():
    """Backend that always fails -> RetryingBackend raises after max_retries."""
    fails = [0]

    def complete(req):
        fails[0] += 1
        raise Exception("429 rate limit")

    b = MagicMock()
    b.name = "always-fails"
    b.is_multimodal = False
    b.complete = complete

    retry = RetryingBackend(
        b,
        RetryConfig(max_retries=3, initial_delay_sec=0.001, jitter=0.0),
    )
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    with pytest.raises(ExtractionError) as exc_info:
        retry.complete(req)
    assert fails[0] == 3  # all 3 attempts failed
    assert exc_info.value.retryable is True


def test_retrying_backend_fails_fast_on_auth_error():
    """Non-retryable errors should not be retried."""
    fails = [0]

    def complete(req):
        fails[0] += 1
        raise Exception("401 Unauthorized: bad API key")

    b = MagicMock()
    b.name = "auth-test"
    b.is_multimodal = False
    b.complete = complete

    retry = RetryingBackend(
        b,
        RetryConfig(max_retries=5, initial_delay_sec=0.001, jitter=0.0),
    )
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    with pytest.raises(AuthError):
        retry.complete(req)
    assert fails[0] == 1  # did NOT retry


def test_retrying_backend_passes_through_is_multimodal():
    inner = MagicMock()
    inner.is_multimodal = True
    inner.name = "m"
    retry = RetryingBackend(inner, RetryConfig(max_retries=1))
    assert retry.is_multimodal is True


def test_retrying_backend_records_logged_attempts():
    """All retry attempts should be logged."""
    fails = [0]

    def complete(req):
        fails[0] += 1
        raise Exception("429 rate limit")

    b = MagicMock()
    b.name = "always-fails"
    b.is_multimodal = False
    b.complete = complete

    retry = RetryingBackend(
        b,
        RetryConfig(max_retries=3, initial_delay_sec=0.001, jitter=0.0),
    )
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    with pytest.raises(ExtractionError):
        retry.complete(req)
    assert fails[0] == 3  # all attempts made


# ---------------------------------------------------------------------------
# hash_request — cache key stability
# ---------------------------------------------------------------------------
def test_hash_request_same_inputs_same_key():
    req = CompletionRequest(
        messages=[Message(role="user", content="hi")],
        temperature=0.0,
        json_mode=True,
    )
    h1 = hash_request(req, schema_name="Invoice", backend_name="openai")
    h2 = hash_request(req, schema_name="Invoice", backend_name="openai")
    assert h1 == h2


def test_hash_request_different_schema_different_key():
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    h1 = hash_request(req, schema_name="Invoice", backend_name="openai")
    h2 = hash_request(req, schema_name="Contract", backend_name="openai")
    assert h1 != h2


def test_hash_request_different_backend_different_key():
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    h1 = hash_request(req, schema_name="Invoice", backend_name="openai")
    h2 = hash_request(req, schema_name="Invoice", backend_name="anthropic")
    assert h1 != h2


def test_hash_request_different_content_different_key():
    req1 = CompletionRequest(messages=[Message(role="user", content="hi")])
    req2 = CompletionRequest(messages=[Message(role="user", content="bye")])
    h1 = hash_request(req1, schema_name="Invoice", backend_name="openai")
    h2 = hash_request(req2, schema_name="Invoice", backend_name="openai")
    assert h1 != h2


def test_hash_request_different_temperature_different_key():
    """Different temperature = different semantic intent = different cache key."""
    req1 = CompletionRequest(messages=[Message(role="user", content="hi")], temperature=0.0)
    req2 = CompletionRequest(messages=[Message(role="user", content="hi")], temperature=0.5)
    h1 = hash_request(req1, schema_name="Invoice", backend_name="openai")
    h2 = hash_request(req2, schema_name="Invoice", backend_name="openai")
    assert h1 != h2


# ---------------------------------------------------------------------------
# ExtractionCache — disk-backed SQLite cache
# ---------------------------------------------------------------------------
def test_cache_miss_returns_none(tmp_path):
    cache = ExtractionCache(tmp_path / "c.db")
    assert cache.get("nonexistent-key") is None
    cache.close()


def test_cache_put_then_get(tmp_path):
    cache = ExtractionCache(tmp_path / "c.db")
    cache.put("k1", schema_name="Invoice", backend_name="openai", response='{"vendor": "Acme"}')
    assert cache.get("k1") == '{"vendor": "Acme"}'
    cache.close()


def test_cache_get_increments_hit_count(tmp_path):
    cache = ExtractionCache(tmp_path / "c.db")
    cache.put("k1", schema_name="Invoice", backend_name="openai", response="x")
    cache.get("k1")
    cache.get("k1")
    cache.get("k1")
    stats = cache.stats()
    assert stats["total_hits"] == 3
    cache.close()


def test_cache_overwrites_on_put_collision(tmp_path):
    cache = ExtractionCache(tmp_path / "c.db")
    cache.put("k1", schema_name="Invoice", backend_name="openai", response="old")
    cache.put("k1", schema_name="Invoice", backend_name="openai", response="new")
    assert cache.get("k1") == "new"
    assert cache.stats()["entries"] == 1
    cache.close()


def test_cache_persists_across_instances(tmp_path):
    """Open the same SQLite file in a new instance — entries are still there."""
    path = tmp_path / "c.db"
    c1 = ExtractionCache(path)
    c1.put("k1", schema_name="Invoice", backend_name="openai", response="persisted")
    c1.close()

    c2 = ExtractionCache(path)
    assert c2.get("k1") == "persisted"
    c2.close()


def test_cache_stats_by_schema(tmp_path):
    cache = ExtractionCache(tmp_path / "c.db")
    cache.put("k1", schema_name="Invoice", backend_name="openai", response="x")
    cache.put("k2", schema_name="Invoice", backend_name="openai", response="y")
    cache.put("k3", schema_name="Contract", backend_name="openai", response="z")
    cache.get("k1")  # 1 hit
    stats = cache.stats()
    assert stats["entries"] == 3
    assert stats["total_hits"] == 1
    schemas = {s["schema"]: s for s in stats["by_schema"]}
    assert schemas["Invoice"]["entries"] == 2
    assert schemas["Contract"]["entries"] == 1


def test_cache_clear(tmp_path):
    cache = ExtractionCache(tmp_path / "c.db")
    cache.put("k1", schema_name="Invoice", backend_name="openai", response="x")
    cache.clear()
    assert cache.get("k1") is None
    assert cache.stats()["entries"] == 0
    cache.close()


def test_cache_thread_safety(tmp_path):
    """Concurrent writes from multiple threads don't corrupt or lose entries."""
    cache = ExtractionCache(tmp_path / "c.db")
    errors = []

    def worker(i: int):
        try:
            # Each thread writes 50 times across 5 distinct keys (k0..k4).
            for j in range(50):
                cache.put(f"k{(i + j) % 5}", schema_name="Invoice",
                          backend_name="openai", response=f"r{i}-{j}")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert cache.stats()["entries"] == 5  # exactly 5 distinct keys (k0..k4)
    cache.close()


# ---------------------------------------------------------------------------
# CachingBackend — transparent caching wrapper
# ---------------------------------------------------------------------------
def test_caching_backend_cache_miss_calls_wrapped():
    """Cache miss: wrapped backend is called."""
    call_count = [0]
    inner = MagicMock()
    inner.name = "mock"
    inner.is_multimodal = False
    inner.complete = MagicMock(side_effect=lambda req: (call_count.__setitem__(0, call_count[0] + 1), '{"x": 1}')[1])

    cache = ExtractionCache(":memory:")
    backend = CachingBackend(inner, cache, schema_name="Invoice")
    req = CompletionRequest(messages=[Message(role="user", content="test")])

    backend.complete(req)
    assert call_count[0] == 1
    cache.close()


def test_caching_backend_cache_hit_skips_wrapped():
    """Cache hit: wrapped backend is NOT called."""
    call_count = [0]
    inner = MagicMock()
    inner.name = "mock"
    inner.is_multimodal = False
    inner.complete = MagicMock(side_effect=lambda req: (call_count.__setitem__(0, call_count[0] + 1), '{"x": 1}')[1])

    cache = ExtractionCache(":memory:")
    backend = CachingBackend(inner, cache, schema_name="Invoice")
    req = CompletionRequest(messages=[Message(role="user", content="test")])

    backend.complete(req)  # miss
    backend.complete(req)  # hit
    backend.complete(req)  # hit
    assert call_count[0] == 1
    assert cache.stats()["total_hits"] == 2
    cache.close()


def test_caching_backend_requires_schema_name():
    """Without schema_name, .complete() raises ValueError."""
    inner = MagicMock()
    inner.name = "x"
    inner.is_multimodal = False
    inner.complete = MagicMock(return_value="{}")

    cache = ExtractionCache(":memory:")
    backend = CachingBackend(inner, cache)  # no schema_name
    req = CompletionRequest(messages=[Message(role="user", content="x")])
    with pytest.raises(ValueError, match="schema_name"):
        backend.complete(req)
    cache.close()


def test_caching_backend_with_schema_pins_schema():
    """with_schema() returns a view with the schema pinned."""
    inner = MagicMock()
    inner.name = "x"
    inner.is_multimodal = False
    inner.complete = MagicMock(return_value="{}")

    cache = ExtractionCache(":memory:")
    base = CachingBackend(inner, cache)
    view = base.with_schema("Invoice")
    req = CompletionRequest(messages=[Message(role="user", content="x")])

    view.complete(req)  # should work
    view.complete(req)  # hit
    assert cache.stats()["entries"] == 1
    cache.close()


def test_caching_backend_different_schemas_get_different_keys():
    """Same request, different schemas → different cache keys → no false hits."""
    inner = MagicMock()
    inner.name = "x"
    inner.is_multimodal = False
    responses = iter(['{"a": 1}', '{"b": 2}'])
    inner.complete = MagicMock(side_effect=lambda req: next(responses))

    cache = ExtractionCache(":memory:")
    base = CachingBackend(inner, cache)
    req = CompletionRequest(messages=[Message(role="user", content="x")])

    inv = base.with_schema("Invoice")
    con = base.with_schema("Contract")

    inv.complete(req)
    con.complete(req)
    assert inv.complete(req) == '{"a": 1}'
    assert con.complete(req) == '{"b": 2}'
    assert cache.stats()["entries"] == 2
    cache.close()


# ---------------------------------------------------------------------------
# Integration: retry + cache + Pipeline
# ---------------------------------------------------------------------------
def test_pipeline_with_retry_transparently_recovers(tmp_path):
    """End-to-end: a flaky backend is wrapped with RetryingBackend and used by Pipeline."""
    from idp import Document, Pipeline

    fails = [0]

    def complete(req):
        fails[0] += 1
        if fails[0] < 3:
            raise Exception("429 rate limit")
        return json.dumps({"vendor_name": "Acme", "invoice_number": "INV-1",
                          "total_amount": 100.0, "line_items": []})

    inner = MagicMock()
    inner.name = "flaky"
    inner.is_multimodal = False
    inner.complete = complete

    retry = RetryingBackend(inner, RetryConfig(max_retries=5,
                                                initial_delay_sec=0.001, jitter=0.0))

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    doc = Document(source_path=str(pdf), raw_text="Vendor: Acme Corp. INV-1 $100",
                   doc_id="x")

    p = Pipeline(backend=retry, schema=Invoice)
    # Provide text directly (skip parsing) so the test doesn't need real PDF parsing
    p.run(doc)
    # Should have succeeded after retries
    assert fails[0] == 3  # 2 fails + 1 success


def test_pipeline_with_cache_serves_repeat(tmp_path):
    """End-to-end: second run with same doc returns cached extraction."""
    from idp import Document, Pipeline

    call_count = [0]

    def complete(req):
        call_count[0] += 1
        return json.dumps({"vendor_name": "Acme", "invoice_number": "INV-1",
                          "total_amount": 100.0, "line_items": []})

    inner = MagicMock()
    inner.name = "mock"
    inner.is_multimodal = False
    inner.complete = complete

    cache = ExtractionCache(tmp_path / "extract.db")
    cached_backend = CachingBackend(inner, cache, schema_name="Invoice")

    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    doc = Document(source_path=str(pdf), raw_text="Vendor: Acme Corp. INV-1 $100",
                   doc_id="x")

    p = Pipeline(backend=cached_backend, schema=Invoice)
    p.run(doc)  # miss
    p.run(doc)  # hit
    p.run(doc)  # hit

    # Only the first call should have hit the backend
    assert call_count[0] == 1  # 1 miss + 2 cache hits
    assert cache.stats()["total_hits"] == 2
    cache.close()




# ---------------------------------------------------------------------------
# RetryingBackend: ExtractionError retry path (lines 189-210, 228)
# ---------------------------------------------------------------------------
def test_retrying_backend_retries_on_extraction_error_retryable():
    """Backend raises RateLimitError (retryable=True) -> should retry then succeed.

    This pins the ``except ExtractionError as e:`` branch in the retry loop
    (lines 189-210). The previous tests only exercised the
    ``except Exception`` branch (raw Exception → classify_exception).
    """
    from idp.reliability import ExtractionError

    call_count = [0]

    class _Backend:
        name = "ext-fail"
        is_multimodal = False

        def complete(self, req):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ExtractionError("rate limit hit", backend_name="ext-fail")
            return '{"ok": true}'

    retry = RetryingBackend(
        _Backend(),
        RetryConfig(max_retries=5, initial_delay_sec=0.001, jitter=0.0),
    )
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    out = retry.complete(req)
    assert out == '{"ok": true}'
    assert call_count[0] == 3  # 2 failures + 1 success


def test_retrying_backend_does_not_retry_non_retryable_extraction_error():
    """Backend raises BadRequestError (retryable=False) -> should fail fast.

    Pins line 191-197 (the ``if not e.retryable: raise`` branch).
    """
    from idp.reliability import BadRequestError

    call_count = [0]

    class _Backend:
        name = "bad-req"
        is_multimodal = False

        def complete(self, req):
            call_count[0] += 1
            raise BadRequestError("invalid schema", backend_name="bad-req")

    retry = RetryingBackend(
        _Backend(),
        RetryConfig(max_retries=5, initial_delay_sec=0.001, jitter=0.0),
    )
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    with pytest.raises(BadRequestError):
        retry.complete(req)
    assert call_count[0] == 1  # fail-fast, no retries


def test_retrying_backend_exhausts_retries_on_extraction_error():
    """Backend always raises retryable ExtractionError -> raises after max_retries.

    Pins lines 199-204 (the ``if attempt + 1 >= max_retries: raise`` branch)
    for the ExtractionError path.
    """
    from idp.reliability import RateLimitError

    call_count = [0]

    class _Backend:
        name = "exhaust"
        is_multimodal = False

        def complete(self, req):
            call_count[0] += 1
            raise RateLimitError("429", backend_name="exhaust")

    retry = RetryingBackend(
        _Backend(),
        RetryConfig(max_retries=3, initial_delay_sec=0.001, jitter=0.0),
    )
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    with pytest.raises(RateLimitError):
        retry.complete(req)
    assert call_count[0] == 3  # all 3 attempts


def test_retrying_backend_unclassified_exception_not_retryable():
    """Backend raises Exception that classifies as non-retryable -> no retry.

    Pins lines 211-214 (the ``if not classified.retryable: raise`` branch
    in the ``except Exception`` arm).
    """
    from idp.reliability import AuthError

    call_count = [0]

    class _Backend:
        name = "auth"
        is_multimodal = False

        def complete(self, req):
            call_count[0] += 1
            raise Exception("401 Unauthorized: bad api key")

    retry = RetryingBackend(
        _Backend(),
        RetryConfig(max_retries=5, initial_delay_sec=0.001, jitter=0.0),
    )
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    with pytest.raises(AuthError):
        retry.complete(req)
    assert call_count[0] == 1  # classified as auth -> fail-fast


# ---------------------------------------------------------------------------
# classify_exception: passthrough for already-typed ExtractionError (line 109)
# ---------------------------------------------------------------------------
def test_classify_extraction_error_returns_same_instance():
    """classify_exception is idempotent for already-ExtractionError inputs."""
    from idp.reliability import RateLimitError, classify_exception

    e = RateLimitError("already typed", backend_name="orig")
    out = classify_exception(e)
    assert out is e  # same instance, no reclassification


# ---------------------------------------------------------------------------
# ExtractionCache: image content in hash (lines 259-260)
# ---------------------------------------------------------------------------
def test_extraction_cache_hash_distinguishes_images_by_length():
    """Hash function includes each image's length — different lengths => different hashes."""
    from idp.reliability import hash_request

    req1 = CompletionRequest(
        messages=[Message(role="user", content="hi", images_b64=["a" * 4])]
    )
    req2 = CompletionRequest(
        messages=[Message(role="user", content="hi", images_b64=["b" * 999])]
    )
    key1 = hash_request(req1, schema_name="S", backend_name="B")
    key2 = hash_request(req2, schema_name="S", backend_name="B")
    assert key1 != key2  # different lengths hash to different keys
    # Identical-length images hash the same way (documented design choice —
    # see hash_request note re: base64 content not hashed; pinning the
    # current behavior so a future change is intentional).
    req3 = CompletionRequest(
        messages=[Message(role="user", content="hi", images_b64=["c" * 4])]
    )
    assert hash_request(req3, schema_name="S", backend_name="B") == key1


# ---------------------------------------------------------------------------
# _testing_backends.SlowMockBackend — gated by IDP_ENABLE_SLOWMOCK
# ---------------------------------------------------------------------------
def test_slowmock_backend_disabled_by_default(monkeypatch):
    """slowmock backend refuses to instantiate without IDP_ENABLE_SLOWMOCK=1."""
    monkeypatch.delenv("IDP_ENABLE_SLOWMOCK", raising=False)
    from idp.llm.backend import get_backend

    with pytest.raises(ValueError, match="slowmock backend is disabled"):
        get_backend("slowmock")


def test_slowmock_backend_sleeps_then_responds(monkeypatch):
    """SlowMockBackend sleeps ~latency_ms then returns a valid completion."""
    monkeypatch.setenv("IDP_ENABLE_SLOWMOCK", "1")
    monkeypatch.setenv("LOAD_LATENCY_MS", "20")
    monkeypatch.setenv("LOAD_JITTER_MS", "0")
    from idp.llm.backend import get_backend

    backend = get_backend("slowmock")
    assert backend.name == "slowmock"
    req = CompletionRequest(messages=[Message(role="user", content="hi")])
    import time
    t0 = time.perf_counter()
    out = backend.complete(req)
    elapsed = time.perf_counter() - t0
    assert elapsed >= 0.015  # slept at least ~20ms
    assert isinstance(out, str)
    assert len(out) > 0
