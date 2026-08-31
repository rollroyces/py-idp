# py-idp

> **General-purpose, AI-enabled Intelligent Document Processing for Python.**
> Six-stage pipeline (parse → classify → extract → assess → validate → HITL).
> 12+ LLM backends. Pydantic-schema-driven. Built-in eval harness.

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE-AGPL)
[![Commercial license available](https://img.shields.io/badge/license-commercial_available-orange.svg)](LICENSE-COMMERCIAL)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-51_passing-brightgreen.svg)](#development)
[![Version](https://img.shields.io/badge/version-0.1.0-lightgrey.svg)](#)

---

## Install

```bash
pip install py-idp                # core (pydantic + typer + rich + httpx + pdfplumber)
pip install py-idp[docling]      # IBM Docling — best PDF table extraction
pip install py-idp[openai]       # OpenAI SDK (also used for 8 China LLMs)
pip install py-idp[anthropic]    # Anthropic SDK
pip install py-idp[api]           # FastAPI server (examples/api.py)
pip install py-idp[dev]          # pytest + ruff + mypy
```

> No API key needed to install or run the test suite — `MockBackend` ships in-tree.

---

## 30-second tour

```python
import idp
from idp.pipeline import Pipeline

result = Pipeline(
    backend="mock",                # or "ollama", "openai", "anthropic", "china:qwen" ...
    schema="Invoice",
    business_rules=[...],
).run(idp.Document.from_path("invoice.pdf"))

print(result.extraction)    # dict — validated against your Pydantic schema
print(result.confidence)    # dict — per-field 0..1, <0.6 flagged for review
print(result.validation)    # dict — schema + business-rule outcomes
```

**A faithful end-to-end run on the in-tree sample invoice, measured live:**

| metric | value |
|---|---|
| classification | `invoice` (conf 0.99) |
| extraction shape | 12 fields, 2 line items |
| validation | PASS |
| exact-match fields vs gold | **9 / 9 = 100 %** |
| low-confidence flags (HITL) | 2 (vendor_address, customer_address) |
| latency (mocked LLM) | < 2 ms |

---

## The pipeline

```
INGEST  →  PARSE  →  CLASSIFY  →  ROUTE  →  EXTRACT  →  ASSESS  →  VALIDATE  →  HITL
                                                                         (Streamlit)
```

Each stage is a pure function over a `Document`. They run independently, are unit-testable in isolation, and any one can be swapped.

| stage | module | default | what it does |
|---|---|---|---|
| **parse** | `idp.parse` | Docling (PDF) · pdfplumber (fallback) · plain text | Extracts text + tables + page images |
| **classify** | `idp.classify` | rule-first, LLM fallback | Detects doc type: invoice, contract, bank_statement, … |
| **route** | `idp.parse.router` | auto | Chooses multimodal VLM vs OCR+LLM based on doc features |
| **extract** | `idp.extract` | Pydantic-schema-driven | Validated structured extraction from text or images |
| **assess** | `idp.assess` | heuristic + optional LLM self-rate | Per-field confidence 0..1 |
| **validate** | `idp.validate` | Pydantic + user predicates | Schema check + business rules |
| **HITL** | `idp.hitl` | Streamlit UI | Review low-confidence fields, save corrections |
| **pipeline** | `idp.pipeline.pipeline` | orchestrator | Composes the above, returns `PipelineResult` |

---

## LLM backends

### International (5 providers, any OpenAI-compat endpoint)

| name | notes |
|---|---|
| `openai` | GPT-4o (vision), GPT-4.1, o1 |
| `anthropic` | Claude 3.5/4 Sonnet, Claude Haiku (vision) |
| `ollama` | local `llama3.2-vision`, `qwen2.5-vl` — default base URL `http://localhost:11434/v1` |
| `vllm` / `lm-studio` / `compat` | any OpenAI-compatible chat-completions endpoint |
| `mock` | offline / CI baseline (`mock`, `mock-random`, `mock-omits`) |

```bash
export OPENAI_API_KEY=...
idp run invoice.pdf --schema Invoice --backend openai
```

### China (8 providers — all speak the OpenAI Chat-Completions protocol)

Run `idp providers` to print the full table. Highlights:

| provider | env var | default | vision model |
|---|---|---|---|
| `deepseek` | `DEEPSEEK_API_KEY` | deepseek-chat | — (text-only) |
| `qwen` | `DASHSCOPE_API_KEY` | qwen-plus | qwen2.5-vl-72b-instruct |
| `zhipu` | `ZHIPUAI_API_KEY` | glm-4-plus | glm-4v-plus |
| `moonshot` | `MOONSHOT_API_KEY` | moonshot-v1-128k | moonshot-v1-128k-vision-preview |
| `yi` | `YI_API_KEY` | yi-large | yi-vision |
| `doubao` | `ARK_API_KEY` | doubao-pro-32k | doubao-1-5-vision-pro-32k |
| `hunyuan` | `HUNYUAN_API_KEY` | hunyuan-pro | hunyuan-vision |
| `baichuan` | `BAICHUAN_API_KEY` | baichuan4 | — (text-only) |

```python
from idp.llm import get_china_backend

backend = get_china_backend("qwen", multimodal=True)
# backend.model == "qwen2.5-vl-72b-instruct"
```

---

## CLI

```bash
idp run path/to/invoice.pdf --schema Invoice --backend ollama --output out.json
idp providers                                          # full provider table
idp schemas                                            # built-in Pydantic schemas
idp eval --dataset src/idp/eval/datasets/invoices \
          --strategy mock,mock-omits --output results.json
idp serve                                              # launch Streamlit HITL UI on :8501
```

---

## Bring your own schema

The built-in `Invoice`, `Contract`, `BankStatement` schemas are convenience references — pass any Pydantic model:

```python
from pydantic import BaseModel
from idp import Document
from idp.pipeline import Pipeline

class Receipt(BaseModel):
    merchant: str
    total: float
    currency: str
    date: str

result = Pipeline(backend="ollama", schema=Receipt).run(
    Document.from_path("receipt.jpg")
)
```

---

## Add business rules

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

Two built-ins ship; define your own by writing a `(dict) -> (bool, str | None)` predicate. Rules that raise are caught — they don't crash the pipeline.

---

## Eval harness

Honest extraction claims need labeled data and side-by-side backend comparison. `py-idp` ships both.

```bash
idp eval --dataset src/idp/eval/datasets/invoices \
         --strategy mock,mock-omits,ollama --output results.json
```

Reports per-strategy: **schema-valid rate**, **field-level F1**, **$/doc**, **latency**. The in-tree fixtures (3 invoices, 2 contracts) are hand-labeled so you can publish numbers you actually verified.

---

## Production scaffolding (built in, optional)

| concern | ships with | swap for production |
|---|---|---|
| Async job queue | `idp.queue.InProcessQueue` | ARQ / Celery / SQS |
| Persistent storage | `idp.storage.JsonFileStorage` | Postgres + S3 |
| API key auth | `idp.auth.keys` | wire into FastAPI dep |
| HTTP API | `examples/api.py` (FastAPI) | your own service |
| HITL UI | `idp.hitl.app` (Streamlit) | React / FastAPI |
| Docker | `Dockerfile`, `docker-compose.yml` | your infra |

### Not in 0.1 (deliberately)

Multi-tenant isolation, SSO/SAML/RBAC, audit-grade storage — needed for SaaS but premature for a single-tenant self-host. Open an issue to request.

---

## Development

```bash
git clone https://github.com/rollroyces/py-idp
cd py-idp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -v                       # 51 tests, no API key needed
ruff check src tests examples   # lint

python -m examples.invoice      # end-to-end demo (no API key needed)
```

`import idp; idp.__version__` → `0.1.0`.

---

## License

py-idp is **dual-licensed**:

- **AGPL-3.0-or-later** — for open-source use. You may use, modify, and run py-idp freely. Modifications deployed as a network-accessible service must also be published under AGPL. This is the copyleft that prevents competitors from cloning the work into a SaaS without contributing back. See `LICENSE-AGPL`.
- **Commercial License** — for organisations that need to embed py-idp in proprietary products or hosted SaaS without the AGPL copyleft. See `LICENSE-COMMERCIAL`.

This mirrors the **MariaDB / Sentry / MinIO** model: pay for the convenience of running in a closed product; get the full source for free if you keep your changes open.

Indicative commercial pricing:

| tier | use case | pricing |
|---|---|---|
| Solo | single developer, single legal entity | **$300 / yr** |
| Team | up to 10 developers, single entity | **$1,500 / yr** |
| Enterprise | unlimited developers + SLA + support | contact |
| SaaS-OEM | embed in a hosted SaaS, per active user | per-seat |

Contact **rollroyces** for a signed agreement.

---

## Acknowledgments

- Pipeline shape, HITL, confidence design — extended from [`aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws`](https://github.com/aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws)
- PDF parsing / table extraction — wraps [IBM Docling](https://github.com/docling-project/docling) (arXiv 2408.09869)
- Pydantic-schema-driven extraction API — inspired by [`run-llama/llama_cloud_services`](https://github.com/run-llama/llama_cloud_services)
- Multi-format chunking patterns — from [`Unstructured-IO/unstructured`](https://github.com/Unstructured-IO/unstructured)

If you cite py-idp in research, please cite this repo and Docling.
