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

"""Example: extract an Invoice using the mock backend (no API key needed).

Run:
    python -m examples.invoice
"""
from pathlib import Path

from idp._util import pretty_print_result
from idp.core.document import Document
from idp.core.schemas import Invoice
from idp.llm.backend import get_backend
from idp.pipeline.pipeline import Pipeline

# Locally-shipped sample so this works offline / in CI
SAMPLE = Path(__file__).parent.parent / "src/idp/eval/datasets/invoices/docs/inv-001.txt"


def main() -> None:
    """Run the full pipeline on the sample invoice using the mock backend.

    Run:
        python -m examples.invoice
    """
    backend = get_backend("mock")
    pipeline = Pipeline(backend=backend, schema=Invoice)
    doc = Document.from_path(SAMPLE)
    res = pipeline.run(doc)
    pretty_print_result(res)


if __name__ == "__main__":
    main()
