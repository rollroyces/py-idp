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

"""The v0.4 cost win: TieredBackend wrapping OllamaBackend.

Most fields on most documents are easy; a small local model
(Ollama) gets them right. The expensive model should only see
the *hard* cases — those the local model couldn't parse or
where per-field confidence dips below ``ambiguity_threshold``.

Typical impact: ~10× cheaper per document at ~2-5% accuracy drop
on aggregate; the expensive tier catches the regression on the
hard fields so worst-case accuracy is preserved.

Run:
    python examples/06_tiered_pipeline.py

Requires: Ollama running locally (``ollama serve``) and the default
model pulled (``ollama pull llama3.2:3b``). Set ``ANTHROPIC_API_KEY``
or ``OPENAI_API_KEY`` for the expensive tier; otherwise we fall
back to MockBackend so the example still runs in any venv.
"""
from __future__ import annotations

import os
from pathlib import Path

from idp.console import pretty_print_result
from idp.core.document import Document
from idp.core.schemas import Invoice
from idp.llm.backend import MockBackend, get_backend
from idp.llm.ollama import OllamaBackend
from idp.llm.tiered import TieredBackend
from idp.pipeline.pipeline import Pipeline

SAMPLE = Path(__file__).parent.parent / "src/idp/eval/datasets/invoices/docs/inv-001.txt"


def main() -> None:
    expensive = (
        get_backend("anthropic") if os.environ.get("ANTHROPIC_API_KEY")
        else get_backend("openai") if os.environ.get("OPENAI_API_KEY")
        else MockBackend()
    )
    pipeline = Pipeline(
        backend=TieredBackend(
            cheap=OllamaBackend(model="llama3.2:3b"),
            expensive=expensive,
        ),
        schema=Invoice,
    )
    result = pipeline.run(Document.from_path(SAMPLE))
    pretty_print_result(result)
    tb = pipeline.backend  # type: ignore[assignment]
    if isinstance(tb, TieredBackend):
        n = tb.cheap_call_count + tb.expensive_call_count
        if n:
            pct = tb.cheap_call_count / n * 100
            print(
                f"\nTieredBackend: {tb.cheap_call_count}/{n} cheap, "
                f"{tb.expensive_call_count}/{n} escalated "
                f"({pct:.0f}% served by Ollama)"
            )


if __name__ == "__main__":
    main()
