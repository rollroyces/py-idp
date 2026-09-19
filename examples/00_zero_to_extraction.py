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

"""Example: from ``pip install py-idp`` to a dict of extracted fields.

The shortest possible happy path — no extras, no API key required.
Run: ``python examples/00_zero_to_extraction.py`` after dropping any
``.pdf`` or ``.txt`` into the current directory.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from idp.easy import extract_one

logging.getLogger("idp").setLevel(logging.CRITICAL)


def _find_first_doc(start: Path) -> Path:
    for ext in (".pdf", ".txt"):
        hits = sorted(start.glob(f"*{ext}"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No .pdf or .txt found in {start}. Drop an invoice here and re-run."
    )


def main() -> None:
    doc_path = _find_first_doc(Path.cwd())
    backend = os.environ.get("IDP_BACKEND", "mock")
    fields = extract_one(str(doc_path), backend=backend, schema="Invoice")

    print(f"file:    {doc_path}")
    print(f"backend: {backend}")
    print("extracted:")
    for key, value in fields.items():
        print(f"  {key}: {value}")

    if backend == "mock" and not os.environ.get("OPENAI_API_KEY"):
        print(
            "\nHint: set IDP_BACKEND=ollama and run `ollama pull qwen2.5:0.5b` "
            "for real extraction."
        )


if __name__ == "__main__":
    main()