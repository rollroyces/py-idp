---
title: "Building py-idp: a 12-backend document-extraction framework that's honest about what small models get wrong"
date: 2026-09-10
author: Royce
tags: [python, llm, ocr, document-processing, open-source]
canonical: https://github.com/rollroyces/py-idp
---

# Building py-idp: a 12-backend document-extraction framework that's honest about what small models get wrong

Extracting structured data from documents — invoices, contracts, bank statements — sounds simple until you try it. PDF tables have merged cells. Scans have noise. Real-world documents have arithmetic (`subtotal + tax_amount = total_amount`) that small language models confidently get wrong.

After watching three internal projects re-invent the same pipeline — *parse → classify → extract → assess → validate → HITL* — I shipped the work as an open-source framework. **py-idp** ([github.com/rollroyces/py-idp](https://github.com/rollroyces/py-idp)) is a general-purpose, AI-enabled Intelligent Document Processing library for Python. This post walks through the design, what's in v0.3.1, and the honest limits.

## Why another framework?

The landscape already has strong options — LlamaCloud, AWS's accelerated-IDP sample, Unstructured. Each solves part of the problem. None of them ship all of these in one place:

- **Six pipeline stages** (parse, classify, route, extract, assess, validate, HITL) where any one can be swapped independently
- **12+ LLM backends** with the same interface: OpenAI, Anthropic, Ollama (local), vLLM, LM Studio, any OpenAI-compatible endpoint, and **8 China LLM providers** (DeepSeek, Qwen, Zhipu, Moonshot, Yi, Doubao, Hunyuan, Baichuan)
- **Self-hosted OCR** via Nanonets-OCR2-3B (Qwen2.5-VL fine-tune) for documents you can't send to a third party — no API key, no cloud egress, runs on Apple Silicon MPS or CUDA
- **AI-driven schema discovery** — give it a PDF and a hint ("extract vendor_name, total_amount, line items"), it proposes a Pydantic class you can pass straight to `Pipeline(schema=...)`
- **HITL review** with a Streamlit UI that records human corrections, and a **policy update** that learns from them
- **Production hardening**: typed exception hierarchy, Prometheus `/metrics`, per-key rate limiting, validated env config, FastAPI server with healthz/readyz/version/metrics/extract endpoints, structured logging
- **AGPL-3.0 + commercial** dual license — same model as MariaDB, Sentry, MinIO

## The 30-second tour

```python
import idp
from idp.pipeline import Pipeline

result = Pipeline(
    backend="mock",                # or "ollama", "openai", "anthropic", "china:qwen" ...
    schema="Invoice",
).run(idp.Document.from_path("invoice.pdf"))

print(result.extraction)    # dict, validated against your Pydantic schema
print(result.confidence)    # per-field 0..1, low values flagged for review
print(result.validation)    # schema + business-rule outcomes
```

`MockBackend` ships in-tree so the test suite runs without an API key. A typical eval cycle looks like:

```bash
idp run path/to/invoice.pdf --schema Invoice --backend ollama --output out.json
idp eval --dataset src/idp/eval/datasets/invoices \
          --strategy mock,mock-omits --output results.json
idp serve    # Streamlit HITL UI on :8501
```

## What the eval harness actually measures

On the in-tree 3-invoice sample, with a **local Ollama qwen2.5:0.5b (397 MB)** running on Apple Silicon:

| metric | value |
|---|---|
| schema-valid rate | **100%** (3/3) |
| field F1 | **0.96** (precision 1.00, recall 0.93) |
| latency | **2.05 s/doc** |

Per-doc exact match: inv-001 7/9 fields, inv-002 9/9, inv-003 9/9.

The interesting part is which fields the small model gets wrong: `subtotal` and `tax_amount` — small-model arithmetic. Py-idp doesn't hide this. The confidence scorer flags those fields at 0.10 and routes them to human review, where `mark_reviewed()` records the correction. After enough reviews, the policy-update step folds them into a runtime override (`field_floors`) so future runs skip the LLM call for those fields.

This is the part I'm proudest of: **the framework is honest about what small models get wrong, and the HITL feedback loop actually changes runtime behavior.** Most "AI document extraction" demos quietly use GPT-4o on a hand-picked example. Py-idp tells you which fields will fail before you deploy.

## What's hard

A few things I'd flag for anyone considering it:

**1. The China-LLM story needs real testing.** I claim 8 providers. The OpenAI-compatible plumbing is identical, but each provider's prompt-response format and rate-limit behavior is its own adventure. PRs with per-provider regression tests are very welcome.

**2. The NanonetsVLBackend is heavy.** First call downloads ~7 GB to `~/.cache/huggingface/hub/`. Cold start is 5–10 minutes. Once cached, ~10s load + ~10s per page. It's gated behind `IDP_ENABLE_NANONETS=1` precisely so you don't accidentally download it on `pip install py-idp`.

**3. Docling is the default PDF parser but it's a 500 MB install.** If you only need text extraction from clean PDFs, the plain `pdfplumber` fallback is enough and avoids the bloat. Pick your extra carefully: `pip install py-idp[docling]` only when you need Docling.

**4. The "auto-chunking for oversized documents" feature is genuinely useful but easy to misuse.** A 50-page invoice PDF gets split into 4-page chunks with 1-page overlap, run through the model per chunk, then merged. The merged extraction carries a `_chunk_count` marker so you can attribute cost. If you need different chunking for your domain (e.g. 1-page chunks for legal contracts), pass a `PageChunker(max_pages=2, overlap_pages=1)` explicitly.

## What's in v0.3.1 (released today)

- **Discoverability:** `CITATION.cff` (Cite this repository button), `docs/SECURITY.md` (private vulnerability reporting), `CONTRIBUTING.md`, structured issue templates, live README badges
- **Bug fixes:**
  - `tiktoken` was promised by the README but never declared as a dependency. Now it is.
  - `pdf2image` + `Pillow` for the multimodal path moved to a `[pdf-render]` extra.
  - CI install line now includes `[api]` and `[pdf-render]`, so the security-adversarial suite collects cleanly.
  - Six test files no longer hard-code `/Users/hermes/py-idp/...` as the repo path (yes, this is embarrassing; tests now run on any machine).
  - `PolicyCache.flush_now()` had a real race with the background flusher — fixed.

458 tests pass across Python 3.10 / 3.11 / 3.12 on every commit. Ruff + mypy clean.

## How to try it

```bash
pip install py-idp
# or with the multimodal OCR stack
pip install py-idp[hf-vlm,pdf-render]
```

Then:

```python
import idp
from idp.pipeline import Pipeline

result = Pipeline(backend="mock", schema="Invoice").run(
    idp.Document.from_path("some-invoice.pdf")
)
```

`MockBackend` produces a valid extraction without an API key, so the above works in a clean venv. Swap `"mock"` for `"ollama"`, `"openai"`, or `"china:qwen"` once you've got credentials.

## What's next

I'm working on a real benchmark against the CORD and Kleister-NDA datasets with GPT-4o, Claude 3.5 Sonnet, and qwen2.5-vl-72b. That'll replace the current in-tree eval with public numbers people can verify. If you want to help — adding a backend, writing a per-domain example, or running the eval against your own dataset — the contributing guide is at [github.com/rollroyces/py-idp/blob/main/CONTRIBUTING.md](https://github.com/rollroyces/py-idp/blob/main/CONTRIBUTING.md).

If you find a real bug or a security issue, please email me or use the GitHub Security tab (private reporting). For everything else, open an issue — the templates will guide you through what's needed.

— Royce

[github.com/rollroyces/py-idp](https://github.com/rollroyces/py-idp) · AGPL-3.0-or-later + commercial license · v0.3.1
