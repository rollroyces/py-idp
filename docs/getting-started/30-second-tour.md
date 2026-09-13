# 30-second tour

The smallest end-to-end run, with output annotations.

```python
import idp
from idp.pipeline import Pipeline

result = Pipeline(
    backend="mock",            # try "anthropic" / "openai" / "china:qwen" / "ollama" with a key
    schema="Invoice",
).run(idp.Document.from_path("invoice.pdf"))
```

## What `Pipeline.run()` returns

A `PipelineResult` with:

| attribute | type | what it contains |
|---|---|---|
| `result.document.extraction` | `dict` | the extracted fields, validated against the schema |
| `result.document.classification` | `str` | e.g. `"invoice"` |
| `result.document.classification_confidence` | `float` | 0..1 |
| `result.document.errors` | `list[str]` | any validation / parsing errors |
| `result.confidence` | `dict[str, float]` | per-field confidence scores |
| `result.validation_passed` | `bool` | did the whole document validate? |
| `result.backend_name` | `str` | which backend produced the extraction |
| `result.mode` | `str` | `"ocr_llm"` or `"multimodal"` |
| `result.timings` | `list[Timing]` | per-stage latencies in seconds |
| `result.source_path` | `str` | the file we ran on |

## Example output (real run, MockBackend)

```
=== /Users/hermes/py-idp/src/idp/eval/datasets/invoices/docs/inv-001.txt ===
schema:    Invoice
backend:   mock (ocr_llm)
classify:  invoice (conf=0.99)
validate:  FAIL
timings:   parse=0.002s, classify=0.000s, route=0.000s, extract=0.460s, assess=0.000s, validate=0.000s

extraction:
{
  "invoice_number": "",
  "vendor_name": "",
  "total_amount": 0.0,
  ...
}

confidence (ascending):
  invoice_number           0.10 [REVIEW]
  vendor_name              0.10 [REVIEW]
  ...
  total_amount             0.70

errors (1):
  - extract_schema_unvalidated: 9 validation errors for Invoice
  ...
```

`MockBackend` returns empty defaults by design — it's the no-API-key path used to validate the framework end-to-end. A real backend (`anthropic`, `openai`, `china:qwen`) produces a valid schema with high confidence on most fields.

## Where to go next

- **Use a real backend** — see [Backends](../user-guide/backends.md).
- **Process a 50-page scan** — see [Auto-chunking](../user-guide/auto-chunking.md).
- **Don't know your schema yet** — see [Auto-schema discovery](../user-guide/auto-schema-discovery.md).
- **Run on real documents in production** — see [Production deployment](../PRODUCTION.md).