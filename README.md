# py-idp

**General-purpose, AI-enabled Intelligent Document Processing framework for Python.**

Convert PDFs, scans, images, and text into validated, structured JSON with confidence
scores, a built-in eval harness, and a human-in-the-loop review UI.

```python
import idp
from idp.llm.backend import get_backend
from idp.pipeline.pipeline import Pipeline

result = Pipeline(
    backend=get_backend("ollama"),   # or "mock", "openai", "anthropic", "china:qwen" …
    schema="Invoice",
).run(idp.Document.from_path("invoice.pdf"))

print(result.extraction)
print(result.confidence)
```

---

## Why this exists

There are excellent point solutions for OCR (Tesseract, Docling, Google Document AI), for
LLM extraction (LlamaExtract, ExtractThinker), and for IDP pipelines (the AWS GenAI IDP
Accelerator, Unstructured). Each of them makes the other parts harder than they should
be. `py-idp` is one small library that ties them together with **a single Python API**,
**a single schema-driven contract**, and **pluggable parsers + backends + confidence +
HITL** so you can run it locally, ship it to production, or build a SaaS on top of it.

Design is grounded in the real 2025 evidence: multimodal VLMs win on clean digital PDFs,
OCR+LLM chains still win on noisy scans and complex tables — so the framework
auto-routes between them per document.

Inspired by / extends:
- `aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws` (pipeline shape, HITL)
- `docling-project/docling` (PDF parsing, table extraction)
- `run-llama/llama_cloud_services` (Pydantic-schema-driven extraction)
- `Unstructured-IO/unstructured` (chunking, multi-format ingest)

---

## Install

```bash
pip install py-idp                  # base install (pydantic + typer + rich + httpx)
pip install py-idp[docling]        # add IBM Docling (best PDF table extraction)
pip install py-idp[openai]         # OpenAI SDK client
pip install py-idp[anthropic]      # Anthropic SDK client
pip install py-idp[china]          # China-LLM SDK (same `openai` package)
pip install py-idp[api]            # FastAPI + uvicorn (for `examples/api.py`)
pip install py-idp[dev]            # pytest + ruff + mypy
```

No LLM API key is required to run the test suite — `MockBackend` ships in-tree.

---

## Pipeline

```
INGEST → PARSE → CLASSIFY → EXTRACT → ASSESS → VALIDATE → HITL
                                                       (Streamlit UI)
```

Each stage is a pure function over a `Document` object that flows through the pipeline.
They are independently runnable and testable, and you can substitute any stage.

| Stage       | Module                              | Default                                   |
|-------------|-------------------------------------|-------------------------------------------|
| parse       | `idp.parse`                         | Docling (PDF), pdfplumber (fallback), plain |
| classify    | `idp.classify.classifier`           | rule-first, LLM fallback                   |
| route       | `idp.parse.router`                  | multimodal vs OCR+LLM auto-pick            |
| extract     | `idp.extract.extractor`             | Pydantic-schema-driven                     |
| assess      | `idp.assess.confidence`             | heuristic + optional LLM self-rate         |
| validate    | `idp.validate.validator`            | Pydantic + user-supplied predicates        |
| HITL        | `idp.hitl.app`                      | Streamlit                                  |
| pipeline    | `idp.pipeline.pipeline`             | Orchestrator                                |

---

## Supported backends

### International
- **OpenAI** — GPT-4o (multimodal), GPT-4.1, GPT-4.1-mini, o1
- **Anthropic** — Claude 3.5/4 Sonnet (vision)
- **Ollama** — local models via OpenAI-compat (`llama3.2-vision`, `qwen2.5-vl`)
- **vLLM / LM Studio / TGI** — any OpenAI-compatible endpoint
- **Mock** — for offline tests + CI + reproducible evals (`mock`, `mock-random`, `mock-omits`)

### China (all speak the OpenAI Chat-Completions protocol)

```bash
idp providers   # full table
```

| provider | env var              | default model         | vision model              |
|----------|----------------------|-----------------------|---------------------------|
| deepseek | `DEEPSEEK_API_KEY`   | deepseek-chat         | — (text-only)             |
| qwen     | `DASHSCOPE_API_KEY`  | qwen-plus             | qwen2.5-vl-72b-instruct   |
| zhipu    | `ZHIPUAI_API_KEY`    | glm-4-plus            | glm-4v-plus               |
| moonshot | `MOONSHOT_API_KEY`   | moonshot-v1-128k      | moonshot-v1-128k-vision-preview |
| yi       | `YI_API_KEY`         | yi-large              | yi-vision                 |
| doubao   | `ARK_API_KEY`        | doubao-pro-32k        | doubao-1-5-vision-pro-32k |
| hunyuan  | `HUNYUAN_API_KEY`    | hunyuan-pro           | hunyuan-vision            |
| baichuan | `BAICHUAN_API_KEY`   | baichuan4             | — (text-only)             |

```python
from idp.llm import get_china_backend
backend = get_china_backend("qwen", multimodal=True)   # uses qwen2.5-vl-72b-instruct
```

---

## CLI

```bash
idp run path/to/invoice.pdf --schema Invoice --backend ollama --output out.json
idp providers                                          # list all China + international providers
idp schemas                                            # list built-in Pydantic schemas
idp eval  --dataset src/idp/eval/datasets/invoices \
          --strategy mock,mock-omits --output results.json
idp serve                                              # launch Streamlit HITL UI on :8501
```

---

## Built-in schemas

| name          | fields                                                                |
|---------------|-----------------------------------------------------------------------|
| `Invoice`     | number, dates, vendor, customer, line_items, subtotal, tax, total      |
| `Contract`    | title, dates, parties, governing_law, total_value, key_obligations    |
| `BankStatement` | account holder, period, opening / closing, transactions             |

Pass your own Pydantic model — there's no requirement to use these.

```python
from pydantic import BaseModel
from idp import Document
from idp.pipeline import Pipeline

class Receipt(BaseModel):
    merchant: str
    total: float
    date: str

result = Pipeline(backend="ollama", schema=Receipt).run(Document.from_path("receipt.jpg"))
```

---

## Adding a business rule

```python
from idp.validate import required_fields_rule, numeric_range_rule

pipe = Pipeline(
    backend="ollama",
    schema="Invoice",
    business_rules=[
        required_fields_rule("invoice_number", "vendor_name", "total_amount"),
        numeric_range_rule("total_amount", min_v=0.0, max_v=10_000_000.0),
    ],
)
```

---

## Eval harness (and why it's not optional)

Honest extraction benchmarks need labeled data and multiple backends compared on the
same inputs. `py-idp` ships with two labeled fixtures and a runner that prints a
comparison table.

```bash
idp eval --dataset src/idp/eval/datasets/invoices \
         --strategy mock,mock-omits --output results.json
```

| strategy      | schema_valid | field F1 | sec/doc |
|---------------|--------------|----------|---------|
| mock          | 100 %        | 0.00     | 1.06 ms |
| mock-omits    | 100 %        | 0.00     | 0.74 ms |

(Both are empty baselines measured on the in-tree `eval/datasets/invoices/` fixture.
F1 = 0 because nothing was extracted. To get real numbers, point `--strategy` at a
real LLM, e.g. `--strategy ollama,china:qwen`.)

When you swap in a real backend, the harness reports schema-valid-rate, field-level
F1, and $/doc — same metrics you should publish claims against.

---

## Enterprise scaffolding (built-in, optional)

| need                        | shipping now                  | path                                   |
|-----------------------------|--------------------------------|----------------------------------------|
| async job queue             | `idp.queue.InProcessQueue`    | swap ARQ/Celery/SQS for production     |
| persistent storage          | `idp.storage.JsonFileStorage` | swap Postgres + S3 for production      |
| API key auth                | `idp.auth.keys`               | wire into FastAPI dependency           |
| HTTP API server             | `examples/api.py`             | FastAPI, optional `[api]` extra        |
| HITL review UI              | `idp.hitl.app` (Streamlit)    | swap React/FastAPI for production      |
| Docker / docker-compose     | `Dockerfile`, `compose.yml`   | ship to your own cloud                 |
| Permissioned eval / metric  | `idp.eval.runner`             | drop into your MLOps stack             |

**What is NOT shipping in 0.1** (and will be added on demand):
- Multi-tenant isolation, SSO, SAML, RBAC, audit-grade storage — needed for SaaS, not for
  single-tenant self-hosted use.
- Cost controls / rate limiting on top of cloud-provider SDKs — they handle it natively.
- PII auto-redaction — built-in is `regex`; production-grade needs Presidio or similar.

---

## Roadmap to "passive income"

The honest way this makes money (the OSS repo itself rarely does):

1. **Self-hosted license**, billed per install or per seat. ~3 months of work to harden
   the existing scaffolding above.
2. **Hosted SaaS** with **AWS / Aliyun / Tencent deployments**, billed per page.
   ~6-12 months, real AWS deployment cost, real enterprise sales cycle.
3. **Add-on extractor packs** (specialised schemas for medical, legal, bank statements)
   sold through a marketplace.

(0.1 is intentionally NOT one of these — it is the public framework that all three
paths can be built on. Demand signal is the only honest justification to proceed with
(2) or (3).)

---

## Development

```bash
git clone https://github.com/rollroyces/py-idp
cd py-idp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -v                  # all tests, 19 included
ruff check src tests examples  # lint

python -m examples.invoice # run a working end-to-end example
```

API layout:

```python
import idp
idp.__version__             # '0.1.0'
```

---

## License

Dual-licensed, deliberately:

- **AGPL-3.0-or-later** (`LICENSE-AGPL`) — for open-source use. Anyone can use,
  modify, and run py-idp. Modifications must be published under AGPL when run
  as a network-accessible service. This is the copyleft that prevents
  competitors from wrapping your work in a SaaS without contributing back.
- **Commercial License** (`LICENSE-COMMERCIAL`) — for organisations that need
  to embed py-idp in proprietary products / hosted SaaS without the AGPL
  copyleft. Indicative pricing: $300/yr Solo, $1,500/yr Team, contact for
  Enterprise / SaaS-OEM. Email rollroyces for a signed agreement.

This mirrors the MariaDB / Sentry / MinIO model: pay for the convenience of
running in a closed product; get the full source for free if you keep changes
open.

## Citation

If you use py-idp in research, please cite this repo (and Docling, arXiv 2408.09869,
which py-idp wraps for parsing).
