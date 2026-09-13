# Auto-chunking

Vision-language models have hard context-window limits (Nanonets-OCR2-3B is 16k tokens). 50-page invoices don't fit. `extract()` automatically detects oversized input and chunks it, runs the LLM per chunk, and merges the results — the caller never sees the chunks.

## Built-in chunkers

| chunker | best for | default config |
|---|---|---|
| `PageChunker` | multimodal (NanonetsVLBackend + page images) | 4 pages per chunk, 1 page overlap |
| `TokenChunker` | text-only extractors (OCR + LLM) | 4000 tokens per chunk, 200-token overlap (tiktoken) |

## Choosing a chunker

```python
from idp.chunker import PageChunker, TokenChunker

# On memory-constrained hardware (small-M-series Macs), shrink chunks.
chunker = PageChunker(max_pages=2, overlap_pages=1)

# Or pass it to the pipeline:
from idp.pipeline import Pipeline
pipe = Pipeline(backend=backend, schema=Invoice, chunker=chunker)

# End-to-end: chunk → call → merge → validate — one call.
result = pipe.run(Document.from_path("huge-50-page-scan.pdf"))
```

## What you get back

The merged extraction includes a `_chunk_count` marker so you can attribute cost and observability metrics per chunk.

## Partial-failure handling

If one chunk's LLM call fails, the error is recorded in `extract_chunk_failed[i]` and the other chunks' extractions are still merged in. You get a "partial result + clear error chain" rather than a whole-batch crash. This is intentional — for a 50-page scan, losing all 49 successful chunks because chunk #31 timed out is worse than a clearly-labeled partial result.