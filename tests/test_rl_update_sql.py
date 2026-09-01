"""Tests for update_policy_from_sql and the new --db-url CLI plumbing."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_update_policy_from_sql_end_to_end(tmp_path):
    """Submit reviews to SqlStorage, then update policy via --db-url."""
    from idp.rl.policy import PolicyConfig
    from idp.rl.update import update_policy_from_sql
    from idp.storage.sql import SqlStorage
    from idp.storage.store import StoredResult

    db = tmp_path / "rl.db"
    storage = SqlStorage(f"sqlite:///{db}")

    # put + review 12 invoices (need >= min_reviews=10) with vendor_name wrong
    for i in range(12):
        rid = f"r{i}"
        storage.put(StoredResult(
            id=rid, doc_id=rid, schema_name="Invoice", backend_name="mock",
            mode="ocr_llm", classification="invoice",
            extraction={"vendor_name": "Acme", "total_amount": 100.0},
            confidence=None, validation=None, source_path="/x", created_at=0.0,
        ))
        storage.submit_review(
            result_id=rid,
            edited={"vendor_name": "Acme Widgets Ltd.", "total_amount": 100.0},
            reviewer="alice",
        )

    policy_path = tmp_path / "policy.json"
    new_policy = update_policy_from_sql(f"sqlite:///{db}", str(policy_path))
    assert "vendor_name" in new_policy.field_floors


def test_cli_rl_update_db_url_flag(tmp_path):
    """End-to-end: the CLI's idp rl-update --db-url works against SqlStorage."""
    from idp.storage.sql import SqlStorage
    from idp.storage.store import StoredResult

    db = tmp_path / "cli.db"
    storage = SqlStorage(f"sqlite:///{db}")
    for i in range(12):
        rid = f"r{i}"
        storage.put(StoredResult(
            id=rid, doc_id=rid, schema_name="Invoice", backend_name="mock",
            mode="ocr_llm", classification="invoice",
            extraction={"vendor_name": "Acme"},
            confidence=None, validation=None, source_path="/x", created_at=0.0,
        ))
        storage.submit_review(rid, {"vendor_name": "Acme Widgets"}, "alice")

    policy_path = tmp_path / "policy.json"
    r = subprocess.run(
        [
            sys.executable, "-m", "idp.pipeline.cli", "rl-update",
            "--db-url", f"sqlite:///{db}",
            "--output", str(policy_path),
        ],
        capture_output=True, text=True,
        cwd="/Users/hermes/py-idp",
    )
    assert r.returncode == 0, f"CLI failed:\nSTDOUT={r.stdout}\nSTDERR={r.stderr}"
    assert policy_path.exists()
    import json
    p = json.loads(policy_path.read_text())
    assert "vendor_name" in p["field_floors"]
