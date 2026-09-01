# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <royce-license-placeholder@protonmail.com> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Example: batch-process multiple documents and save JSON output."""
from __future__ import annotations

from pathlib import Path

from idp._util import pretty_print_result
from idp.core.document import Document
from idp.core.schemas import SCHEMA_REGISTRY
from idp.llm.backend import get_backend
from idp.pipeline.pipeline import Pipeline, save_result

DOCS = [
    ("src/idp/eval/datasets/invoices/docs/inv-001.txt", "Invoice"),
    ("src/idp/eval/datasets/invoices/docs/inv-002.txt", "Invoice"),
    ("src/idp/eval/datasets/contracts/docs/svc-001.txt", "Contract"),
]


def main() -> None:
    backend = get_backend("mock")
    out_dir = Path("examples/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    for rel, schema_name in DOCS:
        path = Path(rel)
        pipeline = Pipeline(backend=backend, schema=SCHEMA_REGISTRY[schema_name])
        result = pipeline.run(Document.from_path(path))
        pretty_print_result(result)
        out = out_dir / f"{path.stem}.json"
        save_result(result, out)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
