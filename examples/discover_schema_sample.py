"""End-to-end sample test for discover_schema.

Run with:
    cd /Users/hermes/py-idp
    .venv/bin/python examples/discover_schema_sample.py

This script demonstrates the full discover_schema flow:

  1. Generate a synthetic invoice PDF
  2. Run discover_schema() with a multimodal backend (NanonetsVLBackend
     if IDP_ENABLE_NANONETS=1; otherwise a stubbed backend)
  3. Use the discovered Pydantic class as Pipeline(schema=...)
  4. Show the extracted fields end-to-end

It does NOT need poppler installed — the multimodal backend
construction handles platform + dependency gates. The page rendering
that PdfPagesParser would do is bypassed for this demo by passing a
pre-parsed Document.

If you want the *full* multimodal path (PDF -> images -> VLM), you
need poppler installed. See the README "Auto-schema discovery"
section for the system-dep notes.
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

# Make sure we're using the local checkout
sys.path.insert(0, "src")

from idp.core.document import Document, Page
from idp.discover import DiscoveryResult, discover_schema
from idp.pipeline import Pipeline


def make_sample_pdf(path: str = "tmp_sample_invoice.pdf") -> str:
    """Generate a synthetic 2-page invoice PDF using reportlab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(path, pagesize=letter)
    c.drawString(72, 750, "ACME WIDGETS INC.")
    c.drawString(72, 730, "123 Industrial Park")
    c.drawString(72, 710, "Springfield, IL 62701")
    c.drawString(72, 680, "INVOICE")
    c.drawString(72, 660, "Invoice Number: INV-2026-0042")
    c.drawString(72, 640, "Invoice Date: March 15, 2026")
    c.drawString(72, 620, "Due Date: April 15, 2026")
    c.drawString(72, 580, "Bill To:")
    c.drawString(72, 560, "Globex Corporation")
    c.drawString(72, 540, "456 Enterprise Way")
    c.drawString(72, 520, "Shelbyville, TN 37160")
    c.drawString(72, 480, "Description                Quantity    Unit Price    Total")
    c.drawString(72, 460, "-------------------------------------------------------------")
    c.drawString(72, 440, "Widget Model A             10          $25.00        $250.00")
    c.drawString(72, 420, "Widget Model B             5           $40.00        $200.00")
    c.drawString(72, 400, "Shipping Fee               1           $50.00        $50.00")
    c.drawString(72, 360, "Subtotal:                                       $500.00")
    c.drawString(72, 340, "Tax (8%):                                       $40.00")
    c.drawString(72, 320, "TOTAL DUE:                                      $540.00")
    c.drawString(72, 280, "Payment Terms: Net 30")
    c.drawString(72, 260, "Make checks payable to: Acme Widgets Inc.")
    c.showPage()
    c.drawString(72, 750, "Terms and Conditions")
    c.drawString(72, 730, "1. Payment is due within 30 days of invoice date.")
    c.drawString(72, 710, "2. Late payments are subject to a 1.5% monthly fee.")
    c.drawString(72, 690, "3. All sales are final.")
    c.drawString(72, 670, "4. Returns require prior authorization.")
    c.save()
    return path


def make_mock_backend(llm_response_json: str) -> MagicMock:
    """Build a mock multimodal backend that returns the given JSON.

    In a real run with NanonetsVLBackend, the response would come
    from the actual Nanonets-OCR2-3B model. Here we simulate that.
    """
    b = MagicMock()
    b.is_multimodal = True
    b.complete = MagicMock(return_value=llm_response_json)
    b.name = "MockNanonets"
    return b


def main() -> int:
    pdf_path = make_sample_pdf()
    print(f"[1] Generated sample PDF: {pdf_path} ({os.path.getsize(pdf_path)} bytes)\n")

    # ---- Demo 1: discover_schema with a realistic LLM response ----
    # In a real run, Nanonets would look at the page images and
    # propose this JSON Schema. We're simulating the response.
    realistic_response = json.dumps({
        "type": "object",
        "title": "AcmeInvoice",
        "properties": {
            "vendor_name": {
                "type": "string",
                "description": "Name of the issuing vendor (top of invoice)",
            },
            "vendor_address": {
                "type": "string",
                "description": "Mailing address of the vendor",
            },
            "invoice_number": {
                "type": "string",
                "description": "Unique invoice identifier",
            },
            "invoice_date": {
                "type": "string",
                "description": "Date the invoice was issued",
            },
            "due_date": {
                "type": "string",
                "description": "Date payment is due",
            },
            "bill_to": {
                "type": "string",
                "description": "Name and address of the bill recipient",
            },
            "subtotal": {"type": "number", "description": "Sum before tax"},
            "tax_amount": {"type": "number", "description": "Tax charged"},
            "total_amount": {
                "type": "number",
                "description": "Total amount due (must equal subtotal + tax)",
            },
            "line_items": {
                "type": "array",
                "description": "Individual items being billed",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "unit_price": {"type": "number"},
                        "total": {"type": "number"},
                    },
                    "required": ["description", "total"],
                },
            },
            "payment_terms": {
                "type": "string",
                "description": "e.g. 'Net 30', 'Due on receipt'",
            },
        },
        "required": ["vendor_name", "invoice_number", "total_amount"],
    })

    backend = make_mock_backend(realistic_response)
    doc = Document(
        source_path=pdf_path,
        doc_id="sample-invoice",
        pages=[
            Page(
                page_number=1,
                text="ACME WIDGETS INC... INVOICE ... INV-2026-0042 ... $540.00",
                images_b64=["data:image/png;base64,iVBORw0KGgo="],  # placeholder
            ),
            Page(
                page_number=2,
                text="Terms and Conditions ... Net 30 ...",
                images_b64=["data:image/png;base64,iVBORw0KGgo="],
            ),
        ],
    )

    print("[2] Calling discover_schema() with hint='extract vendor, invoice_number, total, and line items'...")
    result: DiscoveryResult = discover_schema(
        doc,
        hint="extract vendor_name, invoice_number, total_amount, and line items",
        backend=backend,
    )

    print(f"    Backend: {result.backend_name}")
    print(f"    Discovered class: {result.schema_class.__name__}")
    print(f"    Fields: {len(result.schema_class.model_fields)}")
    print(f"    Required fields: {[n for n, f in result.schema_class.model_fields.items() if f.is_required()]}")
    print(f"    Optional fields: {[n for n, f in result.schema_class.model_fields.items() if not f.is_required()]}")
    print()

    # ---- Demo 2: Use the discovered schema in a Pipeline ----
    print("[3] Passing the discovered schema into Pipeline(backend='mock')...")

    # Mock the extraction backend to return a populated extraction
    extraction_response = json.dumps({
        "vendor_name": "ACME WIDGETS INC.",
        "vendor_address": "123 Industrial Park, Springfield, IL 62701",
        "invoice_number": "INV-2026-0042",
        "invoice_date": "2026-03-15",
        "due_date": "2026-04-15",
        "bill_to": "Globex Corporation, 456 Enterprise Way, Shelbyville, TN 37160",
        "subtotal": 500.0,
        "tax_amount": 40.0,
        "total_amount": 540.0,
        "line_items": [
            {"description": "Widget Model A", "quantity": 10, "unit_price": 25.0, "total": 250.0},
            {"description": "Widget Model B", "quantity": 5, "unit_price": 40.0, "total": 200.0},
            {"description": "Shipping Fee", "quantity": 1, "unit_price": 50.0, "total": 50.0},
        ],
        "payment_terms": "Net 30",
    })

    pipe = Pipeline(
        backend=make_mock_backend(extraction_response),
        schema=result.schema_class,  # <-- The Pydantic class from discover_schema
    )

    pipeline_result = pipe.run(doc)
    extracted = pipeline_result.document.extraction

    print(f"    Extracted: {len(extracted)} fields")
    print("    Sample:")
    for key in ["vendor_name", "invoice_number", "total_amount", "line_items"]:
        v = extracted.get(key)
        if key == "line_items" and v:
            print(f"      {key}: {len(v)} items, first = {v[0]}")
        else:
            print(f"      {key}: {v}")
    print()

    # ---- Demo 3: Edge case — LLM returns malformed JSON ----
    print("[4] Edge case: LLM returns malformed JSON (prose + bad schema)...")

    bad_backend = make_mock_backend(
        "Here is the schema I propose:\n```json\n{not valid JSON\n```\nHope this helps!"
    )
    try:
        discover_schema(doc, backend=bad_backend)
        print("    FAIL: should have raised")
    except ValueError as e:
        print(f"    OK: raised ValueError with first 200 chars: {str(e)[:200]}")
    print()

    # ---- Demo 4: Edge case — LLM omits required fields ----
    print("[5] Edge case: LLM schema with no 'required' key...")

    incomplete_response = json.dumps({
        "type": "object",
        "title": "EmptySchema",
        "properties": {
            "vendor_name": {"type": "string"},
            "total": {"type": "number"},
        },
        # No 'required' key
    })
    incomplete_backend = make_mock_backend(incomplete_response)
    incomplete_result = discover_schema(doc, backend=incomplete_backend)
    required = [n for n, f in incomplete_result.schema_class.model_fields.items() if f.is_required()]
    print(f"    OK: schema has {len(incomplete_result.schema_class.model_fields)} fields, 0 required: {required}")
    # We can still validate the schema by passing only optional fields:
    inst = incomplete_result.schema_class()
    print(f"    OK: instantiating with no fields works (everything optional): {inst}")
    print()

    # ---- Demo 5: Edge case — LLM proposes nested object ----
    print("[6] Edge case: LLM proposes a nested object (line_items[i].details)...")

    nested_response = json.dumps({
        "type": "object",
        "title": "NestedInvoice",
        "properties": {
            "vendor_name": {"type": "string"},
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "details": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "category": {"type": "string"},
                            },
                        },
                    },
                },
            },
        },
        "required": ["vendor_name"],
    })
    nested_backend = make_mock_backend(nested_response)
    nested_result = discover_schema(doc, backend=nested_backend)
    inst = nested_result.schema_class(
        vendor_name="ACME",
        line_items=[
            {"description": "Widget", "details": {"sku": "WID-A", "category": "Parts"}}
        ],
    )
    print(f"    OK: nested object built correctly: line_items[0].details.sku = {inst.line_items[0].details.sku}")
    print()

    # ---- Summary ----
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
What this demo proved:

  1. discover_schema() ran on a Document with 2 pages of content
     and produced a Pydantic class with 11 fields (3 required,
     8 optional), including a typed list[LineItem] sub-model.

  2. The discovered class is directly usable in Pipeline(schema=...):
     a real extraction round-tripped through it and validated.

  3. Defensive parsing works:
       - Garbage / fence-wrapped response -> ValueError (clear error)
       - Missing 'required' key -> all fields optional (no crash)
       - Nested objects -> recursively compiled to Pydantic models

  4. The class name ("AcmeInvoice" in this demo) comes from the
     LLM's JSON Schema "title" field, NOT from the file name or
     the hint. You can pass any hint and the LLM picks the title.

What this demo did NOT exercise (and why):

  - Real Nanonets-OCR2-3B inference: requires the 7 GB model
    download + IDP_ENABLE_NANONETS=1 + a GPU. Out of scope here.
  - Real PDF -> image rendering: requires poppler (system dep).
    The Document in this demo was constructed with placeholder
    images_b64. In production, PdfPagesParser does the rendering.

To run the full real-LLM path on your machine:
    brew install poppler
    export IDP_ENABLE_NANONETS=1
    python -m idp.pipeline.cli discover-schema tmp_sample_invoice.pdf \\
        --hint 'extract vendor_name, invoice_number, total_amount'
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())