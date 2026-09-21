# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <roycelam@umich.edu> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regenerate ``docs/eval/BASELINE.md`` from a fresh eval run.

Used by ``tests/test_eval_cord.py`` (MockBackend row) and by users
manually after running the slow real-backend test. Writes a Markdown
table summarising per-strategy precision / recall / F1 / schema-valid
/ n_docs / avg_sec_per_doc. Idempotent: re-running replaces the
table body, leaving everything outside the BASELINE markers intact.

Markers used:

* ``<!-- BASELINE-START -->\n...\n<!-- BASELINE-END -->`` for the
  table block in ``docs/eval/BASELINE.md``.

Usage::

    python tests/eval_cord/update_baseline_doc.py                # mock row only
    python tests/eval_cord/update_baseline_doc.py --backend ollama  # add real row

The script imports the eval runner and reads ``IDP_EVAL_BACKEND`` if
set, mirroring the slow test. If the requested backend cannot be
instantiated (no API key, no SDK), the row is written with ``--`` in
the metric columns and a note in the source column.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Make `tests/` importable as a package root and `idp` importable from src
_TESTS = REPO / "tests"
_SRC = REPO / "src"
for p in (str(_TESTS), str(_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Use absolute-path import to side-step Python's script-dir-on-sys.path[0]
# precedence that breaks `from tests.eval_cord import ...` when the script
# lives inside tests/eval_cord/.
import importlib.util as _ilu  # noqa: E402

from idp.eval.runner import run_dataset  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_tests_eval_cord", str(Path(__file__).resolve().parent / "__init__.py")
)
if _spec is None or _spec.loader is None:
    raise ImportError("could not load tests/eval_cord via spec_from_file_location")
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
cord_dataset_dir = _mod.cord_dataset_dir
load_manifest = _mod.load_manifest
select_env_backend = _mod.select_env_backend
write_cases_jsonl = _mod.write_cases_jsonl

BASELINE_DOC = REPO / "docs" / "eval" / "BASELINE.md"
START_MARK = "<!-- BASELINE-START -->"
END_MARK = "<!-- BASELINE-END -->"


def _run(backend_name: str) -> dict[str, object] | None:
    """Run the eval against ``backend_name``; return a row dict or ``None``.

    Returns ``None`` when the backend cannot be instantiated (no API
    key, missing SDK, etc.) so the caller can render an empty row.
    """
    ds = cord_dataset_dir()
    manifest = load_manifest(ds)
    target = write_cases_jsonl(manifest, target_dir=None)
    try:
        try:
            res = run_dataset(target, strategies=[backend_name])
        except Exception as e:  # noqa: BLE001
            print(f"[warn] backend={backend_name!r} unavailable: {e}", file=sys.stderr)
            return None
        if not res["rows"]:
            return None
        row = res["rows"][0]
        return {
            "backend": backend_name,
            "schema_valid_rate": row["schema_valid_rate"],
            "field_f1": row["field_f1"],
            "field_precision": row["field_precision"],
            "field_recall": row["field_recall"],
            "n_docs": row["n_docs"],
            "avg_sec_per_doc": row["avg_sec_per_doc"],
        }
    finally:
        # Clean up the temp staging dir
        import shutil

        if target.exists():
            shutil.rmtree(target, ignore_errors=True)


def _fmt(x: object) -> str:
    if x is None or x == "":
        return "--"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def render_table(rows: list[dict[str, object]]) -> str:
    """Render the BASELINE block as a Markdown table."""
    out = []
    out.append("| backend | schema-valid | field-F1 | precision | recall | n_docs | sec/doc |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            "| {backend} | {schema_valid} | {f1} | {prec} | {rec} | {n} | {sec} |".format(
                backend=r.get("backend", "?"),
                schema_valid=_fmt(r.get("schema_valid_rate")),
                f1=_fmt(r.get("field_f1")),
                prec=_fmt(r.get("field_precision")),
                rec=_fmt(r.get("field_recall")),
                n=_fmt(r.get("n_docs")),
                sec=_fmt(r.get("avg_sec_per_doc")),
            )
        )
    return "\n".join(out)


def update_doc(rows: list[dict[str, object]], note: str = "") -> None:
    """Replace the BASELINE block in ``docs/eval/BASELINE.md``."""
    BASELINE_DOC.parent.mkdir(parents=True, exist_ok=True)
    if not BASELINE_DOC.exists():
        BASELINE_DOC.write_text(_initial_doc(rows, note))
        return
    text = BASELINE_DOC.read_text()
    block = render_table(rows)
    if note:
        block += "\n\n" + note
    replacement = f"{START_MARK}\n{block}\n{END_MARK}"
    pattern = re.compile(
        re.escape(START_MARK) + r".*?" + re.escape(END_MARK), re.DOTALL
    )
    if pattern.search(text):
        new_text = pattern.sub(replacement, text)
    else:
        # No markers — append at end
        new_text = text.rstrip() + "\n\n" + replacement + "\n"
    BASELINE_DOC.write_text(new_text)


def _initial_doc(rows: list[dict[str, object]], note: str) -> str:
    """Initial content for a freshly-created BASELINE.md."""
    body = render_table(rows)
    if note:
        body += "\n\n" + note
    return (
        "# Baseline numbers\n\n"
        "Per-backend field precision / recall / F1 on the CORD eval "
        "(`src/idp/eval/datasets/cord/`). The MockBackend row is "
        "regenerated by `tests/test_eval_cord.py`. Real-backend rows "
        "are added by running the slow test with "
        "`IDP_EVAL_BACKEND=<name>`.\n\n"
        f"{START_MARK}\n{body}\n{END_MARK}\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--backend",
        default=select_env_backend() or "mock",
        help="Backend name to run (default: $IDP_EVAL_BACKEND or 'mock').",
    )
    ap.add_argument(
        "--add-only",
        action="store_true",
        help=(
            "Append a row for --backend to the existing table "
            "instead of regenerating the mock row."
        ),
    )
    args = ap.parse_args(argv)

    # Always include the mock row when regenerating fresh
    if args.add_only:
        existing = _read_existing_rows()
        new_row = _run(args.backend)
        if new_row is None:
            print(f"skipping: backend={args.backend!r} unavailable", file=sys.stderr)
            return 1
        rows = existing + [new_row]
        update_doc(rows)
        print(f"appended row for {args.backend}: F1={new_row['field_f1']:.3f}")
    else:
        t0 = time.perf_counter()
        mock_row = _run("mock")
        elapsed = time.perf_counter() - t0
        if mock_row is None:
            print("ERROR: mock backend unavailable (should never happen)", file=sys.stderr)
            return 2
        rows = [mock_row]
        update_doc(
            rows,
            note=(
                f"_MockBackend row regenerated {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"in {elapsed:.1f}s. To add a real-backend row, run: "
                f"`python tests/eval_cord/update_baseline_doc.py --backend <name> --add-only`._"
            ),
        )
        print(f"updated BASELINE.md (mock row F1={mock_row['field_f1']:.3f})")
    return 0


def _read_existing_rows() -> list[dict[str, object]]:
    """Parse the current BASELINE.md back into row dicts (best-effort)."""
    if not BASELINE_DOC.exists():
        return []
    text = BASELINE_DOC.read_text()
    m = re.search(
        re.escape(START_MARK) + r"(.*?)" + re.escape(END_MARK), text, re.DOTALL
    )
    if not m:
        return []
    block = m.group(1)
    rows: list[dict[str, object]] = []
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|") or "---" in line or "backend" in line.lower():
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        def _num(s: str) -> object:
            if s in ("", "--"):
                return None
            try:
                return float(s)
            except ValueError:
                return s
        rows.append(
            {
                "backend": cells[0],
                "schema_valid_rate": _num(cells[1]),
                "field_f1": _num(cells[2]),
                "field_precision": _num(cells[3]),
                "field_recall": _num(cells[4]),
                "n_docs": _num(cells[5]),
                "avg_sec_per_doc": _num(cells[6]),
            }
        )
    return rows


if __name__ == "__main__":
    raise SystemExit(main())