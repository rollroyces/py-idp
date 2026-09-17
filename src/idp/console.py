"""Console helpers for py-idp (self-print utilities).

Used by the example scripts in the repo and (optionally) by user code
that wants a quick human-readable rendering of a ``PipelineResult`` without
importing ``rich`` or wiring up a logging handler.

Why this is named ``console`` and not ``_util``:
  The leading-underscore name was a hint that the module was
  "implementation detail, not public". In practice, the example
  scripts import it directly. We've renamed to make the public
  surface explicit; the old ``idp._util`` import path remains
  available as a thin re-export shim for backward compatibility.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from idp.pipeline.pipeline import PipelineResult


def pretty_print_result(result: PipelineResult) -> None:
    """Print a human-friendly summary of a PipelineResult to stdout.

    Layout:

        === /path/to/invoice.pdf ===
        schema:    Invoice
        backend:   ollama (ocr_llm)
        classify:  invoice (conf=0.95)
        validate:  PASS
        timings:   parse=0.123s, classify=0.456s, ...

        extraction:
        {
          "invoice_number": "INV-001",
          "vendor_name": "Acme Corp",
          ...
        }

        confidence (ascending):
          total_amount               0.65 [REVIEW]
          vendor_name                0.80
          ...

    Fields with confidence < 0.6 are flagged with ``[REVIEW]`` so they
    are easy to spot in a long batch run.
    """
    print(f"\n=== {result.document.source_path} ===")
    print(f"schema:    {result.schema_name}")
    print(f"backend:   {result.backend_name} ({result.mode})")
    print(
        f"classify:  {result.classification} "
        f"(conf={result.document.classification_confidence})"
    )
    print(f"validate:  {'PASS' if result.validation_passed else 'FAIL'}")
    print(
        "timings:   "
        + ", ".join(f"{t.name}={t.seconds:.3f}s" for t in result.timings)
    )
    print("\nextraction:")
    print(json.dumps(result.document.extraction, indent=2, default=str))
    if result.confidence:
        ordered = sorted(result.confidence.items(), key=lambda kv: kv[1])
        print("\nconfidence (ascending):")
        for k, v in ordered:
            mark = " [REVIEW]" if v < 0.6 else ""
            print(f"  {k:<24} {v:.2f}{mark}")
    if result.document.errors:
        print(f"\nerrors ({len(result.document.errors)}):")
        for e in result.document.errors:
            print(f"  - {e}")


__all__ = ["pretty_print_result"]