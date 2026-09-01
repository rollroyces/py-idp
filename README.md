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
| exact-match fields vs gold | **7 / 9 = 78 %** (single doc) |
| low-confidence flags (HITL) | 2 (subtotal, tax_amount — small-model arithmetic) |
| latency (mocked LLM) | < 2 ms |

**Full eval harness, 3 invoices, real local Ollama (`qwen2.5:0.5b`, 397 MB):**

| metric | value |
|---|---|
| schema-valid rate | **100 %** (3 / 3) |
| field F1 | **0.96** (precision 1.00, recall 0.93) |
| latency | **2.05 s / doc** on Apple Silicon |
| per-doc exact match | inv-001 7/9 · inv-002 9/9 · inv-003 9/9 |

The framework is honest about what small models get wrong: arithmetic on tiny models (`subtotal`/`tax_amount`) is flagged with conf 0.10 and routed to HITL review, not silently passed.

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
| **RL from HITL corrections** | `idp.rl` + `idp rl-update` | online per-review update (`PolicyCache`) |

### Not in 0.1 (deliberately)

Multi-tenant isolation, SSO/SAML/RBAC, audit-grade storage — needed for SaaS but premature for a single-tenant self-host. Open an issue to request.

---

## Learning from HITL corrections (RL)

Every human review in `idp.storage` becomes a training signal. The framework ships an **offline batch policy update** that turns "fields humans keep correcting" into higher-confidence-floor + lower-confidence-penalty for those fields — so they reliably surface to HITL review in the next run.

```bash
# Offline batch: derive rewards from accumulated reviews, write policy.json
idp rl-update --storage idp_data/results.jsonl \
               --output policy.json

# Apply policy in the pipeline:
result = Pipeline(
    backend="ollama",
    schema="Invoice",
    policy_path="policy.json",
).run(Document.from_path("invoice.pdf"))

# Or hand-craft reviews if you don't have storage yet:
idp rl-update --reviews reviews.jsonl --output policy.json
```

**What this is:** a deterministic, inspectable, version-controllable rule update. It is **not** a learned reward model, **not** a fine-tuned LLM. We're learning the post-hoc confidence adjustment that decides what to flag for HITL — not the model itself.

**Why this approach:** real-world ROI is highest at this layer. Training an LLM with RLHF/DPO gives ~2-3% F1 gain for weeks of work; a 7B model would beat that for less. Learning *which fields to send to HITL more reliably* compounds every review.

**Measured (real Ollama, `qwen2.5:0.5b`, in-tree fixture):**

| field | without policy | with policy (after 5 human corrections) | delta |
|---|---|---|---|
| `vendor_name` | 0.75 (would pass HITL) | **0.55** (now flagged) | −0.20 |
| `subtotal` | 0.10 (already flagged) | **0.0** (urgent) | −0.10 |
| `invoice_number` | 0.75 | 0.75 (no override) | 0.0 |

Online (per-review) update ships as a v0.2 hook; the offline batch is fully wired today.

### Calibration eval — does the policy actually do what it claims?

```bash
# Generate synthetic reviews from gold truth, derive a policy, evaluate it
idp rl-update --reviews reviews.jsonl --output policy.json
idp rl-eval   --policy policy.json --fixtures src/idp/eval/datasets/invoices \
              --injection-rate 0.30 --output calibration.json
```

Reports **hit rate when policy fires** (did humans correct what we flagged?) and **true-accept rate when policy silent** (did humans accept what we didn't flag?), with explicit `n=` and a `synthetic=true` flag — synthetic reviews are biased optimistic (gold truth IS the human's correction), so real HITL data will be noisier.

**Honest measured results (synthetic reviews from 3 in-tree invoices, `qwen2.5:0.5b` real Ollama run, fields × docs = 27 pairs):**

| metric | value | what it means |
|---|---|---|
| policy caught (flag → human corrected) | **21** | without the policy, these errors would have escaped HITL |
| policy silenced (was flagged, no longer flagged) | **0** | no regressions |
| already flagged by both | 2 | no change |
| model was right, not flagged | 4 | correct accepts — model was actually right |

**Honest call-out:** with `qwen2.5:0.5b` specifically, the base confidence heuristic is so pessimistic that almost every error was already escaping HITL — so the policy's gain looks dramatic. A larger model with cleaner confidence calibration would benefit less. The honest sample size here is 27 (field, doc) pairs; do not extrapolate beyond this.

### Online policy update (per-review, in-process)

The `PolicyCache` watches `storage.mark_reviewed()` and incrementally folds each new review into the in-memory policy, with debounced atomic disk flushes. The very next `Pipeline.run()` sees the updated override — no restart, no separate CLI invocation.

```python
from idp.storage import make_storage
from idp.rl import PolicyCache

storage = make_storage("sql", db_url="sqlite:///./idp.db")
cache = PolicyCache(policy_path="policy.json", flush_interval_sec=1.0)
cache.attach_to_storage(storage)   # patches mark_reviewed to fire on_review

# From now on, every human review edits the policy in the background.
```

**Defaults:** `flush_interval_sec=1.0` (debounce window), `min_reviews=10` (the small-sample guard — fields with fewer than 10 total observations get no override regardless of fail rate, because fail_rate estimates are too noisy at n<10).

**Multi-process:** only one process should hold the cache (e.g. the FastAPI server). Other processes (CLI tools, the Streamlit reviewer UI) read `policy.json` from disk. The cache uses `os.replace` for atomic writes, so a crash mid-flush leaves the previous policy intact.

### Real HITL data collection

The `SqlStorage` backend persists everything `JsonFileStorage` does plus per-field edit history in a real relational database. SQLite works out-of-the-box (zero extra deps); Postgres is opt-in via `pip install py-idp[sql]`.

```bash
# SQLite, single-file
export IDP_DB_URL="sqlite:///./idp.db"
idp serve                                  # Streamlit UI now reads/writes this DB
idp rl-update --db-url "sqlite:///./idp.db" --output policy.json
idp rl-eval  --db-url "sqlite:///./idp.db" --policy policy.json \
             --output calibration.json
```

**Schema (4 tables):** `reviewers`, `stored_results` (denormalised cache of latest review state), `reviews` (one row per review session), `review_edits` (one row per field-level diff). The split lets you compute per-reviewer agreement, per-field edit rate over time, and "did the policy flag this and the human agreed it was wrong" without scanning full result blobs.

**Why the split matters:** `review_edits` is the granular signal the RL layer consumes (one row per corrected field). Without it, you can't tell *which field* in a multi-field review the human changed.

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
