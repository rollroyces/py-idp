# Pipeline overview

A py-idp extraction is six sequential stages. Each stage is a pure function on a `Document` and is independently testable, independently swappable, and independently observable (each reports a `Timing`).

```text
INGEST → PARSE → CLASSIFY → ROUTE → EXTRACT → ASSESS → VALIDATE → HITL
                                                           (Streamlit UI)
```

| stage | module | default | purpose |
|---|---|---|---|
| **parse** | `idp.parse` | Docling (PDF) · pdfplumber (fallback) · text passthrough | extract text + tables + page images |
| **classify** | `idp.classify` | rules first, LLM fallback | identify document type (invoice, contract, bank statement, ...) |
| **route** | `idp.parse.router` | heuristic | pick multimodal VLM vs. OCR+LLM based on the document |
| **extract** | `idp.extract` | Pydantic-schema-driven | structured extraction from text or page images |
| **assess** | `idp.assess` | heuristic + optional LLM self-score | per-field confidence 0..1 |
| **validate** | `idp.validate` | Pydantic + user predicates | schema validation + business rules |
| **HITL** | `idp.hitl` | Streamlit UI | review low-confidence fields, save corrections |
| **pipeline** | `idp.pipeline.pipeline` | orchestrator | compose the stages, return `PipelineResult` |

## How to customize a stage

Each stage exposes a protocol (`Parser`, `Classifier`, `Extractor`, `Assessor`, `Validator`). Implement the protocol and pass your implementation to `Pipeline`:

```python
from idp.pipeline import Pipeline

pipeline = Pipeline(
    backend="anthropic",
    schema="Invoice",
    parser=MyCustomParser(),       # default is Docling/poppler
    classifier=MyCustomClassifier(),
    # ...
)
```

The defaults are tuned for English-language invoices and contracts. If you have a specialized use case (Japanese receipts, shipping waybills, lab reports), the protocols make it straightforward to slot in your own logic without forking the framework.

## What flows through the pipeline

Every stage reads from `Document` and writes back to it. The `Document` carries:

- `source_path` / `source_bytes` — what was ingested
- `pages` — parsed text per page (set by parse)
- `classification`, `classification_confidence` — set by classify
- `extraction` — set by extract
- `errors` — accumulated across all stages

The pipeline orchestrator does not mutate any stage's output directly — it threads `Document` through and gathers per-stage `Timing`s.