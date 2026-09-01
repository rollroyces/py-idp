"""E4: CLI subprocess tests — every command.

Uses subprocess so the CLI is tested exactly as a user invokes it
(python -m idp.pipeline.cli ...), catching arg-parsing, import, and
I/O errors that unit tests miss.
"""
from __future__ import annotations

import subprocess
import sys

REPO = "/Users/hermes/py-idp"
CLI = [sys.executable, "-m", "idp.pipeline.cli"]
INVOICE = f"{REPO}/src/idp/eval/datasets/invoices/docs/inv-001.txt"
DATASET = f"{REPO}/src/idp/eval/datasets/invoices"


def _run(*args: str, expect_ok: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(CLI + list(args), capture_output=True, text=True, cwd=REPO)
    if expect_ok:
        assert r.returncode == 0, f"CLI {' '.join(args)} failed:\nSTDOUT={r.stdout}\nSTDERR={r.stderr}"
    return r


# ---------------------------------------------------------------------------
# schemas / providers (pure listing commands)
# ---------------------------------------------------------------------------
def test_cli_schemas():
    r = _run("schemas")
    assert "Invoice" in r.stdout
    assert "Contract" in r.stdout
    assert "BankStatement" in r.stdout


def test_cli_providers():
    r = _run("providers")
    assert "deepseek" in r.stdout
    assert "qwen" in r.stdout


# ---------------------------------------------------------------------------
# run (the main pipeline command)
# ---------------------------------------------------------------------------
def test_cli_run_invoice(tmp_path):
    out = tmp_path / "out.json"
    r = _run("run", INVOICE, "--schema", "Invoice", "--backend", "mock", "--output", str(out))
    assert "classification" in r.stdout
    assert out.exists()


def test_cli_run_invoice_no_output_file():
    r = _run("run", INVOICE, "--schema", "Invoice", "--backend", "mock")
    assert "extraction" in r.stdout or "invoice_number" in r.stdout


def test_cli_run_bad_schema_fails():
    r = _run("run", INVOICE, "--schema", "NonexistentSchema", "--backend", "mock", expect_ok=False)
    assert r.returncode != 0


def test_cli_run_missing_file_fails():
    r = _run("run", "/tmp/does-not-exist.pdf", "--backend", "mock", expect_ok=False)
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def test_cli_eval(tmp_path):
    out = tmp_path / "eval.json"
    r = _run("eval", "--dataset", DATASET, "--strategy", "mock,mock-omits", "--output", str(out))
    assert "schema_valid_rate" in r.stdout or "field_f1" in r.stdout
    assert out.exists()


# ---------------------------------------------------------------------------
# rl-update
# ---------------------------------------------------------------------------
def test_cli_rl_update_reviews(tmp_path):
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        "\n".join(
            f'{{"doc_id":"x{i}","schema":"Invoice",'
            f'"model":{{"vendor_name":"Acme","total":100.0}},'
            f'"human":{{"vendor_name":"Acme Widgets Ltd.","total":100.0}}}}'
            for i in range(12)
        ) + "\n"
    )
    policy = tmp_path / "policy.json"
    r = _run("rl-update", "--reviews", str(reviews), "--output", str(policy))
    assert "vendor_name" in r.stdout or policy.exists()
    assert policy.exists()


def test_cli_rl_update_no_input_fails():
    r = _run("rl-update", "--output", "/tmp/x.json", expect_ok=False)
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# rl-eval
# ---------------------------------------------------------------------------
def test_cli_rl_eval_synthetic(tmp_path):
    policy = tmp_path / "policy.json"
    # build a policy first
    reviews = tmp_path / "reviews.jsonl"
    reviews.write_text(
        "\n".join(
            f'{{"doc_id":"x{i}","schema":"Invoice",'
            f'"model":{{"vendor_name":"Acme"}},'
            f'"human":{{"vendor_name":"Acme Widgets"}}}}'
            for i in range(12)
        ) + "\n"
    )
    _run("rl-update", "--reviews", str(reviews), "--output", str(policy))
    r = _run("rl-eval", "--policy", str(policy), "--fixtures", DATASET, "--injection-rate", "0.3")
    assert "hit_rate_when_fires" in r.stdout or "review_efficiency" in r.stdout


def test_cli_rl_eval_no_input_fails():
    r = _run("rl-eval", "--policy", "/tmp/nonexistent.json", "--fixtures", DATASET, expect_ok=False)
    assert r.returncode != 0


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------
def test_cli_help_lists_all_commands():
    r = _run("--help")
    for cmd in ("run", "schemas", "providers", "serve", "eval", "rl-update", "rl-eval"):
        assert cmd in r.stdout
