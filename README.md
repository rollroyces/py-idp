# py-idp

> **General-purpose, AI-enabled Intelligent Document Processing for Python.**
> Six-stage pipeline (parse → classify → extract → assess → validate → HITL).
> 12+ LLM backends. Pydantic-schema-driven. Built-in eval harness.
> **Auto-chunking for oversized documents. Self-hosted OCR via Nanonets-OCR2-3B. AI-driven schema discovery.**

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE-AGPL)
[![Commercial license available](https://img.shields.io/badge/license-commercial_available-orange.svg)](LICENSE-COMMERCIAL)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-399_passing-brightgreen.svg)](#development)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](#)

---

## Install

```bash
pip install py-idp                # core (pydantic + typer + rich + httpx + pdfplumber)
pip install py-idp[docling]      # IBM Docling — best PDF table extraction
pip install py-idp[openai]       # OpenAI SDK (also used for 8 China LLMs)
pip install py-idp[anthropic]    # Anthropic SDK
pip install py-idp[api]           # FastAPI server (idp.api:app — production-ready)
pip install py-idp[hf-vlm]        # Self-hosted Nanonets-OCR2-3B (Apple Silicon / CUDA)
pip install py-idp[dev]          # pytest + ruff + mypy
```

> No API key needed to install or run the test suite — `MockBackend` ships in-tree.
> `tiktoken` is installed automatically by the core package (used for token-budget chunking).

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

### Self-hosted (Nanonets-OCR2-3B on Apple Silicon / CUDA)

For documents you can't send to a third party. **No API key, no cloud
egress, fully offline after the first download** (~7 GB cached to
`~/.cache/huggingface/hub/`).

```bash
pip install py-idp[hf-vlm]    # adds torch + transformers + accelerate + safetensors
export IDP_ENABLE_NANONETS=1  # explicit opt-in (avoids surprise downloads)
export IDP_BACKEND=nanonets
idp run scan.pdf --backend nanonets
```

**Memory budget on Apple M4 16 GB** (float16, 448×448 image):
- weights + vision encoder + KV cache: ~8.3 GB
- OS + apps: ~3.5 GB
- headroom: ~4 GB (comfortable)

**Speed**: ~5-15 sec per page on M4. First call: 5-10 min to download
the model. Subsequent calls: ~10 s to load from cache.

**Why Nanonets-OCR2-3B**: open weights, no auth, Apache-2.0 (Qwen2.5-VL
base) — verify the Nanonets fine-tune license before commercial use.
Outperforms Tesseract on noisy scans and handles multilingual docs.

**Why gated**: model download is large and slow. We refuse to
auto-trigger it; you must explicitly set `IDP_ENABLE_NANONETS=1`.

**Platform support** (verified at construction time):

| Platform | Status |
|---|---|
| macOS arm64 (M1/M2/M3/M4, 16+ GB) | ✅ tested target, MPS |
| macOS arm64 (8 GB) | ❌ OOM (use Docling instead) |
| macOS x86_64 (Intel) | ❌ no MPS, eGPU CUDA flaky — fails loud |
| Linux x86_64 + CUDA | ✅ best (1-5s per page) |
| Linux x86_64 CPU-only | ⚠️ works, 30-60s per page |
| Linux arm64 | ⚠️ works, CPU only |
| Windows x86_64 + CUDA | ✅ same as Linux CUDA |
| Windows arm64 | ❌ PyTorch has no Windows-arm64 wheels — fails loud |

```python
from idp.llm.nanonets import NanonetsVLBackend
backend = NanonetsVLBackend(
    device="mps",          # or "cuda", "cpu", "auto"
    max_image_side=448,    # 4x less vision memory than 1024, ~95% acc
    load_in_4bit=False,    # True if you OOM at float16
)

# End-to-end with PdfPagesParser (renders pages to images)
from idp import Document, Pipeline
from idp.parse.parser import parse_document
from idp.core.schemas import Invoice

doc = Document.from_path("scan.pdf")
parse_document(doc, parser="pdf-pages")   # renders pages to base64 PNG
result = Pipeline(backend=backend, schema=Invoice).run(doc)
print(result.document.extraction)
```

### Auto-chunking for oversized documents

Nanonets-OCR2-3B has a 16k token context. A 50-page invoice PDF won't
fit in one call. `extract()` detects this and automatically splits
the input, runs the model once per chunk, and merges the per-chunk
extractions. **No glue code required** — it's invisible to the caller.

Two chunkers ship:

| chunker | when | default config |
|---|---|---|
| `PageChunker` | multimodal (NanonetsVLBackend + page images) | 4 pages per chunk, 1-page overlap |
| `TokenChunker` | text extractors (OCR + LLM) | 4000 tokens per chunk, 200-token overlap (tiktoken) |

```python
from idp.chunker import PageChunker, TokenChunker

# Tighter memory budget on a small M-series Mac
chunker = PageChunker(max_pages=2, overlap_pages=1)

# Or pass directly to the pipeline
from idp.pipeline import Pipeline
pipe = Pipeline(backend=backend, schema=Invoice, chunker=chunker)

# End-to-end: chunks, calls, merges, validates — one call
result = pipe.run(Document.from_path("huge-50-page-scan.pdf"))
```

The merged extraction includes a `_chunk_count` marker so you can
attribute cost and observability per chunk run.

**Per-chunk failure resilience:** if one chunk's LLM call fails, the
error is logged (`extract_chunk_failed[i]`) but other chunks' data is
still merged in. You get partial results + a clear error trail, not
a hard crash.

See [`src/idp/chunker.py`](src/idp/chunker.py) for the implementation
and [`tests/test_chunker.py`](tests/test_chunker.py) for the 34 tests.

---

## CLI

```bash
idp run path/to/invoice.pdf --schema Invoice --backend ollama --output out.json
idp providers                                          # full provider table
idp schemas                                            # built-in Pydantic schemas
idp discover-schema scan.pdf --hint "extract vendor_name, total_amount" --output schema.json
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

## Auto-schema discovery

You have a scanned PDF and a vague sense of "I want fields X, Y, Z" —
but no Pydantic class yet. `discover_schema()` asks the multimodal LLM
(NanonetsVLBackend by default) to propose a JSON Schema, then compiles
it to a Pydantic class you can pass straight into `Pipeline(schema=...)`.

```python
import idp

Schema, schema_dict = idp.discover_schema(
    "scan.pdf",
    hint="extract vendor_name, invoice_number, total_amount, and line items",
)
# Schema is a Pydantic BaseModel subclass — pass it directly:
result = idp.Pipeline(backend="nanonets", schema=Schema).run(
    idp.Document.from_path("scan.pdf")
)
print(result.document.extraction)
```

The returned `DiscoveryResult` exposes both the compiled Pydantic class
and the raw JSON Schema dict:

```python
result = idp.discover_schema("scan.pdf", hint="...")
result.schema_class    # the Pydantic class
result.json_schema     # the raw JSON Schema dict
result.raw_response    # raw LLM output (debug aid)
result.backend_name    # "NanonetsVLBackend"
result.doc             # the parsed Document (reuse for extraction)
```

**CLI equivalent:**

```bash
export IDP_ENABLE_NANONETS=1
idp discover-schema scan.pdf \
    --hint "extract vendor_name, total_amount, and line items" \
    --output schema.json
```

**Defaults:** pages capped at 4 (fits most 16k-context VLMs), Nanonets
backend (must set `IDP_ENABLE_NANONETS=1`), fallback to Mock for tests.

**Honest limits:**

- LLM-proposed field names are sometimes wrong — the user hint steers
  this but doesn't guarantee it. Always review the resulting schema
  against a few real extractions before using in production.
- Field types are inferred from the JSON Schema (string / number /
  integer / boolean / array / nested object). Required-vs-optional
  is preserved.
- LLMs sometimes emit ```json fences or wrap output in prose; the
  parser strips both. Pure garbage raises `ValueError` with the first
  200 chars for debugging.
- This is **schema discovery** — it tells you *what fields exist* and
  *what they're called*. It is not schema **validation** — pass the
  discovered schema into `Pipeline(schema=...)` and use HITL review
  for the validation step.

See [`src/idp/discover.py`](src/idp/discover.py) for the implementation
and [`tests/test_discover.py`](tests/test_discover.py) for the 31 tests.

---

## Chunking for oversized documents

Most LLMs cap context at 6k-200k tokens. A long invoice, contract, or
multi-page scan may exceed that. `py-idp` **auto-detects oversized
input, chunks it, runs the LLM once per chunk, and merges the
per-chunk extractions** — all without glue code.

| chunker | when used | default config |
|---|---|---|
| `PageChunker` | multimodal backends (NanonetsVLBackend, GPT-4o, etc.) | 4 pages per chunk, 1-page overlap |
| `TokenChunker` | text extractors (OCR + LLM) | 4000 tokens per chunk, 200-token overlap (tiktoken) |

**Defaults are tuned for the most common models:**
- 4 pages @ 200dpi ≈ 3000 image tokens → fits Nanonets-OCR2-3B (16k context)
- 4000 text tokens → fits qwen2.5:0.5b (6k context) and llama3.2 (8k)

**Per-chunk failure resilience:** if one chunk's LLM call fails, the
error is logged (`extract_chunk_failed[i]`) but other chunks' data is
still merged. Partial results > no results.

```python
from idp.chunker import PageChunker

# Tight memory budget (M-series Mac with 16 GB unified)
chunker = PageChunker(max_pages=2, overlap_pages=1)
result = Pipeline(backend="nanonets", schema="Invoice", chunker=chunker).run(
    Document.from_path("huge-50-page-scan.pdf")
)
print(result.document.extraction.get("_chunk_count"))  # ~25
```

The merged extraction is **schema-validated as a whole** after merging,
so you still get a Pydantic-typed result even though it was built from
many small extractions.

See [`src/idp/chunker.py`](src/idp/chunker.py) for the implementation
and [`tests/test_chunker.py`](tests/test_chunker.py) for the 34 tests.

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
| HTTP API | `idp.api:app` (production, FastAPI, auth+rate-limit+metrics) | your own service |
| HITL UI | `idp.hitl.app` (Streamlit) | React / FastAPI |
| Docker | `Dockerfile`, `docker-compose.yml` | your infra |
| **RL from HITL corrections** | `idp.rl` + `idp rl-update` | online per-review update (`PolicyCache`) |
| **Document chunking** | `idp.chunker` (auto for oversized input) | custom `PageChunker` / `TokenChunker` |
| **Schema discovery** | `idp.discover_schema` + `idp discover-schema` | custom multimodal backend |

### Not in 0.3 (deliberately)

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

Online (per-review) update ships in v0.2 via `PolicyCache`; the offline batch is fully wired today.

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

pytest -v                       # 399 tests, no API key needed
ruff check src tests examples   # lint
mypy src/idp                    # type-check (clean across 56 files)

python -m examples.invoice      # end-to-end demo (no API key needed)
python -m examples.nanonets_ocr2  # NanonetsVLBackend end-to-end (needs IDP_ENABLE_NANONETS=1)
python -m examples.batch        # process_batch() helper for Databricks-style batches
```

`import idp; idp.__version__` → `0.3.0`.

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
