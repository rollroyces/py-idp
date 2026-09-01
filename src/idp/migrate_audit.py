"""Migration utility: scan stored results in a SQLite DB and flag rows
that would have validated differently between v0.1 and v0.2.

Use this if you upgraded from v0.1 → v0.2 and want to know which historical
extractions would now fail ``schema_valid=True``. The output is a report;
no data is modified.

Usage:
    python -m idp.migrate_audit --db-url sqlite:////data/idp.db --output report.json

The report contains:
  - total_rows, rows_v01_only, rows_v02_failing
  - for each affected row: doc_id, schema_name, missing_fields, current_validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def audit_db(db_url: str, output_path: str | None = None) -> dict[str, Any]:
    """Walk ``stored_results`` + apply v0.1 vs v0.2 schemas; return a report."""
    # Import here so the CLI is light when unused
    # v0.1 shapes — re-register on demand (don't pollute the global registry
    # persistently; reset at the end)
    from idp.compat_v01 import (
        BankStatement_v01,
        Contract_v01,
        Invoice_v01,
    )
    from idp.core.schemas import SCHEMA_REGISTRY
    from idp.storage.sql import SqlStorage
    v01: dict[str, type[BaseModel]] = {
        "Invoice": Invoice_v01, "Contract": Contract_v01, "BankStatement": BankStatement_v01
    }
    v02: dict[str, type[BaseModel]] = {k: v for k, v in SCHEMA_REGISTRY.items() if k in v01}

    storage = SqlStorage(db_url)

    total_rows = 0
    rows_v01_pass_v02_fail: list[dict[str, Any]] = []

    for result in storage.list():
        total_rows += 1
        schema_name = result.schema_name
        if schema_name not in v01:
            continue
        # Extraction is typed as dict[str, Any] (non-None by the dataclass).
        # An empty dict IS a v0.1-valid extraction; do not skip it.

        # Try v0.2 schema (current); fail -> flag
        v02_ok = False
        v02_error = ""
        try:
            v02[schema_name].model_validate(result.extraction)
            v02_ok = True
        except Exception as e:
            v02_error = str(e)

        # Try v0.1 schema; pass -> was OK in old version
        v01_ok = False
        try:
            v01[schema_name].model_validate(result.extraction)
            v01_ok = True
        except Exception:
            pass

        if v01_ok and not v02_ok:
            rows_v01_pass_v02_fail.append({
                "doc_id": result.doc_id,
                "schema_name": schema_name,
                "result_id": result.id,
                "extraction": result.extraction,
                "v02_error": v02_error,
            })

    report = {
        "db_url": db_url,
        "total_rows": total_rows,
        "rows_v01_pass_v02_fail": len(rows_v01_pass_v02_fail),
        "affected": rows_v01_pass_v02_fail,
        "recommendation": (
            "Rows that passed v0.1 schema but fail v0.2 are 'extractions that the "
            "old model would have accepted as valid but the new model rejects "
            "as missing required fields. To re-validate them with v0.1 semantics, "
            "either (a) switch the Pipeline's schema_name to 'Invoice_v01' / "
            "'Contract_v01' / 'BankStatement_v01' (import idp.compat_v01.register_compat_schemas()) "
            "or (b) re-extract these documents with the new model."
        ),
    }
    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-url", required=True, help="e.g. sqlite:///./idp.db")
    p.add_argument("--output", default=None, help="Write report JSON to this path")
    args = p.parse_args()
    report = audit_db(args.db_url, args.output)
    print(json.dumps({
        "db_url": report["db_url"],
        "total_rows": report["total_rows"],
        "rows_v01_pass_v02_fail": report["rows_v01_pass_v02_fail"],
        "output": args.output,
    }, indent=2))
    if args.output:
        print(f"\nfull report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())