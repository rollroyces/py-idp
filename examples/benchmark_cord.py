"""Reproducible benchmark on the CORD-subset receipts.

Runs the in-tree MockBackend against all 5 receipts and prints
per-field precision / recall / F1, plus latency. No API key needed.

Reproducible by anyone with `pip install py-idp[eval]`. For real
CORD benchmarks (GPT-4o, Claude, qwen2.5-vl-72b), see the docstring
in `src/idp/eval/datasets/cord_subset/README.md`.

Usage:
    python examples/benchmark_cord.py
"""
from __future__ import annotations

import json
from pathlib import Path

from idp.eval.runner import run_dataset

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "src" / "idp" / "eval" / "datasets" / "cord_subset"


def _fmt(x: float) -> str:
    return f"{x:.3f}" if isinstance(x, float) else str(x)


def _print_table(results: dict) -> None:
    print()
    print("=" * 78)
    print("CORD-subset benchmark — MockBackend (no API key needed)")
    print("=" * 78)
    print()
    print(f"Dataset:    {DATASET.relative_to(REPO)}/")
    print(f"Documents:  {results['rows'][0]['n_docs'] if results['rows'] else 0}")
    print()
    header = f"{'strategy':<14} {'schema-valid':>13} {'field-F1':>10} {'prec':>8} {'recall':>8} {'sec/doc':>10}"
    print(header)
    print("-" * len(header))
    for row in results["rows"]:
        print(
            f"{row['strategy']:<14} "
            f"{_fmt(row['schema_valid_rate']):>13} "
            f"{_fmt(row['field_f1']):>10} "
            f"{_fmt(row['field_precision']):>8} "
            f"{_fmt(row['field_recall']):>8} "
            f"{_fmt(row['avg_sec_per_doc']):>10}"
        )
    print()
    print(
        "Interpretation: MockBackend is the deterministic CI baseline —\n"
        "  field-F1 is intentionally low (no real LLM). Replace 'mock'\n"
        "  with 'ollama', 'openai', 'anthropic', or 'china:qwen' to run\n"
        "  against a real model. The schema_valid_rate should stay 1.0\n"
        "  on any backend that respects the JSON contract."
    )
    print()


def main() -> None:
    print("Loading CORD-subset and running MockBackend...")
    results = run_dataset(str(DATASET), strategies=["mock", "mock-omits"])
    _print_table(results)

    out = REPO / "examples" / "output" / "cord_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"Full results written to {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
