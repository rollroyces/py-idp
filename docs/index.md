---
hide:
  - navigation
---

# py-idp

**General-purpose, AI-enabled Intelligent Document Processing for Python.**

Six-stage pipeline (parse → classify → extract → assess → validate → HITL), 12+ LLM backends, Pydantic-schema-driven extraction, built-in eval harness, self-hosted OCR via Nanonets-OCR2-3B, auto-chunking for oversized documents, and AI-driven schema discovery.

[:material-rocket-launch: Getting started](getting-started/installation.md){ .md-button .md-button--primary }
[:fontawesome-brands-github: View on GitHub](https://github.com/rollroyces/py-idp){ .md-button }

---

## Why py-idp?

| pain point | how py-idp helps |
|---|---|
| 12+ LLM providers, 6 SDKs, no consistency | one `Pipeline(backend=...)` interface; switch with one arg |
| PDFs with weird tables | Docling + pdfplumber fallback; tables survive extraction |
| OCR for sensitive documents | self-hosted Nanonets-OCR2-3B on your hardware (Apple Silicon / CUDA), no API key, offline |
| 50-page documents that overflow the context window | automatic chunking (page-based + token-based) |
| "We want fields X, Y, Z but haven't written the schema yet" | `discover_schema()` turns a hint + a sample into a Pydantic class |
| Small models get arithmetic wrong | confidence assessment flags low-confidence fields for human review; HITL feedback folds into runtime overrides |
| Production has to handle retries, caching, rate limits | `RetryingBackend`, `ExtractionCache`, `JsonFileStorage` (or `SqlStorage`) |
| We need to actually measure which backend is best | `idp eval` with side-by-side strategies, schema-valid rate, field-F1, latency |

---

## Quick install

```bash
pip install py-idp                          # core
pip install py-idp[docling]                 # IBM Docling PDF parser
pip install py-idp[anthropic]               # Anthropic Claude
pip install py-idp[openai]                  # OpenAI GPT-4o (+ China-LLM compatible gateways)
pip install py-idp[hf-vlm]                  # self-hosted Nanonets-OCR2-3B
pip install py-idp[api]                     # FastAPI server (idp.api:app)
pip install py-idp[dev]                     # pytest + ruff + mypy
pip install py-idp[docs]                    # this site (mkdocs-material)
```

No API key needed to run the in-tree eval, the examples, or the full test suite — the `MockBackend` is built in.

---

## 30-second tour

```python
import idp
from idp.pipeline import Pipeline

result = Pipeline(
    backend="mock",            # or "anthropic", "openai", "china:qwen", "ollama", "nanonets"...
    schema="Invoice",
).run(idp.Document.from_path("invoice.pdf"))

print(result.extraction)        # dict, validated against the Pydantic schema
print(result.confidence)        # per-field 0..1; fields <0.6 routed to HITL
print(result.validation)        # schema + custom predicate outcomes
```

See the [30-second tour](getting-started/30-second-tour.md) for the full output.

---

## Project status

- **v0.3.1** shipped ([release notes](https://github.com/rollroyces/py-idp/releases/tag/v0.3.1))
- **508 tests**, mypy clean (59 files), ruff clean
- Dual-licensed: [AGPL-3.0-or-later](https://github.com/rollroyces/py-idp/blob/main/LICENSE-AGPL) (open-source) and a [commercial license](https://github.com/rollroyces/py-idp/blob/main/LICENSE-COMMERCIAL) for closed-source embedding.