# Reliability features

Three opt-in features for production workloads.

## RetryingBackend — automatic retries with backoff

Wraps any `Backend` with exponential-backoff retries on transient errors (rate limits, timeouts, connection errors). Auth and bad-request errors fail fast — no point retrying those.

```python
from idp import Pipeline
from idp.llm.retry import RetryingBackend

backend = RetryingBackend(
    inner=get_backend("anthropic"),
    max_retries=5,
    base_delay=0.5,        # seconds; doubles each retry (0.5, 1, 2, 4, 8)
)

Pipeline(backend=backend, schema="Invoice").run(doc)
```

## ExtractionCache — disk-backed extraction cache

Avoids re-running the LLM on the same `(doc_hash, schema, backend)` tuple. Useful for:

- Re-running an eval after a partial failure
- Iterating on a HITL workflow without re-charging the LLM
- Running the same batch twice (idempotency)

```python
from idp.llm.cache import ExtractionCache

cache = ExtractionCache(path="/var/cache/idp/extract.jsonl")
backend = cache.wrap(get_backend("anthropic"))
```

## CheckpointStore — idempotent batch processing

For long batches, `process_batch()` from `idp.batch` persists per-doc results to disk so a re-run skips the docs that already succeeded.

```python
from idp.batch import process_batch, CheckpointStore

store = CheckpointStore("/var/cache/idp/checkpoints.jsonl")

results = process_batch(
    docs,
    schema="Invoice",
    backend="anthropic",
    checkpoint=store,           # re-runs skip succeeded docs
    parallelism=8,
)
```

A re-run after a partial failure picks up where it left off.