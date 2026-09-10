# Launch cross-post — Reddit + HN

Two posts ready to copy-paste. Post them yourself (different account age, different trust signals). Best times: **Tuesday/Wednesday 9–11am US Eastern** for HN; **Tuesday/Thursday morning** for Reddit. Wait at least 24h between the two so they don't look coordinated.

---

## r/LocalLLaMA — "Show & Tell" post

**Title:** *py-idp v0.3.1 — open-source document extraction with Ollama + self-hosted NanonetsVLBackend (12 backends, no API key needed for the test suite)*

**Body:**

I shipped v0.3.1 of [py-idp](https://github.com/rollroyces/py-idp) yesterday — a Python framework for extracting structured data from PDFs / scans / images using LLMs. The thing I think is most relevant here:

- **12+ LLM backends with the same interface** — OpenAI, Anthropic, Ollama, vLLM, LM Studio, any OpenAI-compatible endpoint, plus 8 China LLM providers. Ollama + qwen2.5-vl works out of the box.
- **Self-hosted OCR via NanonetsVLBackend** — Nanonets-OCR2-3B (Qwen2.5-VL fine-tune) for documents you can't send to a third party. No API key, no cloud egress. Runs on Apple Silicon (MPS) or CUDA. Gated behind `IDP_ENABLE_NANONETS=1` so you don't accidentally download 7 GB on `pip install`.
- **Honest eval harness.** Local Ollama `qwen2.5:0.5b` (397 MB, the smallest sensible model) gets field F1 = 0.96 on the in-tree 3-invoice sample. Per-field confidence flagging routes low-confidence fields to HITL review; after enough human corrections the policy update folds them into runtime overrides so future runs skip the LLM call for those fields.
- **AI-driven schema discovery** — give it a PDF and a hint, it proposes a Pydantic class you pass straight to `Pipeline(schema=...)`. No more "first write the schema" yak-shaving.

The 30-second tour:

```python
import idp
from idp.pipeline import Pipeline

result = Pipeline(backend="ollama", schema="Invoice").run(
    idp.Document.from_path("invoice.pdf")
)
print(result.extraction)   # validated dict
print(result.confidence)   # per-field 0..1
```

Install:

```bash
pip install py-idp                # core
pip install py-idp[hf-vlm]        # NanonetsVLBackend (Apple Silicon / CUDA)
```

What I think is interesting / unusual:

1. **Six-stage pipeline (parse → classify → extract → assess → validate → HITL) where any one can be swapped.** I wanted something where if you only need classify + extract with a custom rule-based classifier, you don't have to drag in the rest.
2. **Auto-chunking for oversized documents.** Nanonets has a 16k token context; a 50-page PDF won't fit in one call. Py-idp detects this and chunks automatically. Per-chunk failure resilience — if one chunk's LLM call fails, the error is logged but other chunks' data is still merged.
3. **It's honest about what small models get wrong.** The eval table on the README shows `subtotal` and `tax_amount` getting flagged at confidence 0.10 because a 0.5B model can't do arithmetic reliably. The framework tells you which fields will fail before you deploy.

What it's not:

- **Not a hosted SaaS.** You run it yourself.
- **Not optimized for the absolute best accuracy.** If you want SOTA on a public dataset, you'll wire in your own model + eval loop.
- **The eval sample is in-tree.** I'm working on a CORD / Kleister-NDA benchmark with GPT-4o + Claude 3.5 + qwen2.5-vl-72b for the next release. Open to PRs adding public-dataset evals.

License is dual AGPL-3.0-or-later + commercial (same model as MariaDB / Sentry). For personal projects, AGPL is fine. For embedding in a proprietary product, the commercial license kicks in.

Happy to answer questions about the design or any specific stage. If you try it on your own docs and find a bug, please file an issue with the reproduction — the templates will guide you.

[github.com/rollroyces/py-idp](https://github.com/rollroyces/py-idp)

---

## Hacker News — Show HN

**Title:** *Show HN: py-idp – 12-backend document-extraction framework for Python*

**URL:** https://github.com/rollroyces/py-idp

**Body (text-only, no link shorteners):**

py-idp is a Python framework for extracting structured data from PDFs, scans, and images. It targets the "I have a folder of invoices / contracts / bank statements and I want them in a database" problem.

What's in the box:

- Six pipeline stages (parse → classify → route → extract → assess → validate → HITL), each swappable.
- 12+ LLM backends: OpenAI, Anthropic, Ollama, vLLM, LM Studio, any OpenAI-compatible endpoint, plus 8 China LLM providers.
- Self-hosted OCR via Nanonets-OCR2-3B (Qwen2.5-VL fine-tune) for documents you can't send to a third party. Runs on Apple Silicon MPS or CUDA.
- AI-driven schema discovery — give it a PDF and a hint, it proposes a Pydantic class.
- HITL review with a Streamlit UI; the human corrections feed back into a policy that learns to skip the LLM call for fields where the model consistently fails.
- Production FastAPI server with healthz/readyz/metrics endpoints, structured logging, rate limiting, typed exception hierarchy.

What I'm proud of: the framework is honest about what small models get wrong. The in-tree eval on 3 invoices with local Ollama qwen2.5:0.5b shows field F1 = 0.96 — but the two fields the model gets wrong (subtotal + tax_amount, both arithmetic) are flagged at confidence 0.10 and routed to HITL review, not silently passed. After 10 human corrections on the same field, the policy folds them into a runtime override.

What's hard:

- The China-LLM story needs more per-provider testing. The OpenAI-compatible plumbing is identical across all 8, but rate limits and prompt conventions differ.
- NanonetsVLBackend is heavy: 7 GB download on first call. Gated behind an env var so a fresh `pip install` doesn't trigger it.
- Docling (the default PDF parser) is a 500 MB install. Plain `pdfplumber` is fine for clean PDFs and much lighter.

458 tests, ruff + mypy clean. AGPL-3.0-or-later + commercial dual license.

I'm working on a public-dataset benchmark (CORD, Kleister-NDA) with GPT-4o + Claude + qwen2.5-vl-72b for v0.4. Open to PRs.

---

## Posting notes

- **r/LocalLLaMA:** paste the post, no link in the title. Engage with every reply for the first 24h. The mods are strict about self-promotion — frame it as "I built this and want feedback", not "look at my project".
- **HN:** submit as Show HN. The body should be honest about limitations — HN hates overselling. Title without marketing words ("12-backend" is fine; "amazing" is not). Be ready to respond to criticism; HN's first 5 comments set the tone.
- **Wait 24h between the two posts.** Cross-posting simultaneously looks coordinated and gets both flagged.
- **Track results:** star count over the next 7 days. If a post does well, consider a follow-up "lessons learned" post in 2 weeks — that compounds.
