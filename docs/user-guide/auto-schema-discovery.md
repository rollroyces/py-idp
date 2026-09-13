# Auto-schema discovery

You have a scan. You vaguely know "I want fields X, Y, Z". You haven't written a Pydantic class yet. `discover_schema()` lets a multimodal LLM propose a JSON Schema, compiles it into a Pydantic class, and hands it back ready to use as `Pipeline(schema=...)`.

## End-to-end demo

```bash
python -m examples.discover_schema_sample
```

Generates a 2-page synthetic PDF, runs `discover_schema()` against it (with a hint), and produces a real Pydantic class from the LLM's proposal. No poppler required.

## From Python

```python
import idp

Schema, schema_dict = idp.discover_schema(
    "scan.pdf",
    hint="extract vendor_name, invoice_number, total_amount and line items",
)

# `Schema` is a Pydantic BaseModel subclass — drop it straight into a Pipeline:
result = idp.Pipeline(backend="nanonets", schema=Schema).run(
    idp.Document.from_path("scan.pdf"),
)

print(result.document.extraction)
```

## How it works

1. The hint + a few pages from the PDF go to a multimodal LLM.
2. The LLM proposes a JSON Schema.
3. py-idp compiles it into a Pydantic class.
4. The class is cached by hint-hash so subsequent runs skip the LLM call.

If the discovered schema doesn't fit your needs, you can hand-edit the JSON Schema and re-compile. The compiler raises a clear error if the LLM produced something invalid.