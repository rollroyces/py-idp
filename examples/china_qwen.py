"""Example: run the pipeline against a real China LLM.

To run this for real:
    export DASHSCOPE_API_KEY=...
    python -m examples.china_qwen

Without the API key, this script falls back to the MockBackend so you
can still see the full pipeline structure.
"""
from __future__ import annotations

import os
from pathlib import Path

from idp.core.document import Document
from idp.core.schemas import Invoice
from idp._util import pretty_print_result
from idp.llm import get_backend
from idp.pipeline import Pipeline

SAMPLE = Path(__file__).parent.parent / "src/idp/eval/datasets/invoices/docs/inv-001.txt"


def main() -> None:
    try:
        backend = get_backend("china:qwen", multimodal=True)  # qwen2.5-vl-72b
        print("Using DashScope Qwen-VL")
    except Exception as e:
        print(f"  ! Falling back to mock backend: {e}")
        from idp.llm import get_backend as _gb

        backend = _gb("mock")

    pipeline = Pipeline(backend=backend, schema=Invoice)
    res = pipeline.run(Document.from_path(SAMPLE))
    pretty_print_result(res)


if __name__ == "__main__":
    main()
