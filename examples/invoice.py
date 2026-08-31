"""Example: extract an Invoice using the mock backend (no API key needed).

Run:
    python -m examples.invoice
"""
from pathlib import Path

from idp.core.document import Document
from idp.core.schemas import Invoice
from idp._util import pretty_print_result
from idp.llm.backend import get_backend
from idp.pipeline.pipeline import Pipeline
from idp.validate.validator import required_fields_rule

# Locally-shipped sample so this works offline / in CI
SAMPLE = Path(__file__).parent.parent / "src/idp/eval/datasets/invoices/docs/inv-001.txt"


def main() -> None:
    backend = get_backend("mock")           # swap for "ollama" or "openai"
    pipeline = Pipeline(
        backend=backend,
        schema=Invoice,
        business_rules=[
            required_fields_rule("invoice_number", "vendor_name", "total_amount"),
        ],
    )
    doc = Document.from_path(SAMPLE)
    result = pipeline.run(doc)
    pretty_print_result(result)


if __name__ == "__main__":
    main()
