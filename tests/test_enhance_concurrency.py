"""E8: concurrency stress — JsonFileStorage under heavy parallel writes.

The earlier smoke test used 4×20. This goes to 100 threads × 100 writes
to verify the file never gets corrupted under real concurrency.

NOTE: this test is intentionally slow-ish (~5-10s) but runs in CI.
It catches the class of bug where two threads interleave partial lines.
"""
from __future__ import annotations

import threading

from idp.storage.store import JsonFileStorage, StoredResult


def test_json_storage_100_threads_100_writes(tmp_path):
    path = tmp_path / "stress.jsonl"
    s = JsonFileStorage(str(path))

    N_THREADS = 100
    N_WRITES = 100
    errors: list[str] = []
    lock = threading.Lock()

    def worker(tid):
        try:
            for w in range(N_WRITES):
                s.put(StoredResult(
                    id=f"t{tid}_w{w}",
                    doc_id=f"doc-{tid}",
                    schema_name="Invoice",
                    backend_name="mock",
                    mode="ocr_llm",
                    classification="invoice",
                    extraction={"vendor_name": f"acme-{tid}-{w}"},
                    confidence=None,
                    validation=None,
                    source_path="/x",
                    created_at=0.0,
                ))
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(f"thread {tid}: {e}")

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # No worker should have raised
    assert errors == [], f"{len(errors)} threads raised: {errors[:3]}"

    # Re-read everything: should be N_THREADS * N_WRITES = 10,000 rows,
    # none corrupted (corrupt lines are silently skipped by _read_all,
    # so a lower count means data loss happened).
    items = s.list(limit=1_000_000)
    expected = N_THREADS * N_WRITES
    assert len(items) == expected, (
        f"expected {expected} rows, got {len(items)} — data loss under concurrency"
    )

    # Spot-check: the id is unique (no overwrite collisions)
    ids = [r.id for r in items]
    assert len(set(ids)) == expected, "duplicate ids — writes overwrote each other"
