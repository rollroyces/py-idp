"""Chunking strategies for documents that exceed a model's context window.

When a PDF has more pages (or text has more tokens) than the model can
fit in one call, the extractor splits the document into chunks, calls
the model once per chunk, and merges the per-chunk extractions.

Two strategies ship:

  PageChunker (multimodal)
    Splits a Document's pages into groups of K with R overlap pages.
    The first chunk has pages [1..K], the second has pages
    [K-R+1..2K-R+1], etc. So an invoice's "header on page 1" appears
    in chunks 1 and 2, giving the model a chance to re-extract.

    Default K=4, R=1 — fits 4 pages @ 200 dpi into a 16k context
    (Nanonets-OCR2-3B's limit) with margin to spare for the schema
    prompt.

  TokenChunker (text)
    Splits raw text into N-token chunks with M-token overlap, using
    tiktoken (cl100k_base). Works for the OCR+LLM path where the
    parser emits text and the model is a regular LLM.

    Default N=4000 tokens, M=200 tokens.

Merge semantics
  After the per-chunk calls, each chunk returns a dict matching the
  target schema. We merge them with:

    Scalar fields (str / int / float / bool / date): non-null wins;
      ties broken by confidence (if available) or "first non-null".
    List fields: union; deduplicated by `id` / `name` when items are
      dicts, or by value for primitives.
    Optional / nullable: skipped if missing.

  `merge_extractions()` accepts any dict-like structure and returns a
  merged dict. It does NOT re-validate against a Pydantic schema —
  that's the next stage's job.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from idp.core.document import Page

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Page chunker
# ---------------------------------------------------------------------------
@dataclass
class PageChunker:
    """Split a Document's pages into groups with overlap.

    Args:
        max_pages:     max pages per chunk (e.g. 4 for Nanonets-OCR2-3B).
        overlap_pages: pages to repeat between chunks (e.g. 1).
        include_images: when True, the chunk includes per-page images
                        (multimodal mode). When False, only text.
    """
    max_pages: int = 4
    overlap_pages: int = 1
    include_images: bool = True

    def __post_init__(self) -> None:
        if self.max_pages < 1:
            raise ValueError("max_pages must be >= 1")
        if self.overlap_pages < 0:
            raise ValueError("overlap_pages must be >= 0")
        if self.overlap_pages >= self.max_pages:
            raise ValueError(
                f"overlap_pages ({self.overlap_pages}) must be < "
                f"max_pages ({self.max_pages})"
            )

    def split(self, pages: Sequence[Page]) -> list[list[Page]]:
        """Split pages into chunks. Always returns at least one chunk."""
        if not pages:
            return [[]]
        stride = self.max_pages - self.overlap_pages
        out: list[list[Page]] = []
        n = len(pages)
        for start in range(0, n, stride):
            chunk = list(pages[start : start + self.max_pages])
            out.append(chunk)
            if start + self.max_pages >= n:
                break
        return out

    def collect_texts(self, pages: Sequence[Page]) -> list[str]:
        """Concatenate page texts per chunk."""
        return ["\n\n".join(p.text or "" for p in chunk) for chunk in self.split(pages)]

    def collect_images(self, pages: Sequence[Page]) -> list[list[str]]:
        """Collect per-page images_b64 per chunk (for multimodal)."""
        chunks = self.split(pages)
        return [
            [img for p in chunk for img in (p.images_b64 or [])]
            for chunk in chunks
        ]


# ---------------------------------------------------------------------------
# Token chunker
# ---------------------------------------------------------------------------
@dataclass
class TokenChunker:
    """Split text into token-bounded chunks with overlap.

    Args:
        max_tokens:     target max tokens per chunk (default 4000).
        overlap_tokens: tokens to repeat between adjacent chunks (default 200).
        encoding_name:  tiktoken encoding (default 'cl100k_base', the GPT-4
                        family tokenizer — works as an approximation for
                        most chat LLMs).
    """
    max_tokens: int = 4000
    overlap_tokens: int = 200
    encoding_name: str = "cl100k_base"

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens must be >= 0")
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError(
                f"overlap_tokens ({self.overlap_tokens}) must be < "
                f"max_tokens ({self.max_tokens})"
            )

    def _encoder(self):
        try:
            import tiktoken
        except ImportError as e:
            raise ImportError(
                "TokenChunker requires tiktoken. "
                "Install with:  pip install tiktoken"
            ) from e
        return tiktoken.get_encoding(self.encoding_name)

    def split(self, text: str) -> list[str]:
        """Split text into chunks. Empty input -> empty list."""
        if not text or not text.strip():
            return []
        enc = self._encoder()
        tokens = enc.encode(text)
        stride = self.max_tokens - self.overlap_tokens
        chunks: list[str] = []
        for start in range(0, len(tokens), stride):
            end = min(start + self.max_tokens, len(tokens))
            chunks.append(enc.decode(tokens[start:end]))
            if end >= len(tokens):
                break
        return chunks


# ---------------------------------------------------------------------------
# Merge per-chunk extractions
# ---------------------------------------------------------------------------
def merge_extractions(
    per_chunk: list[dict[str, Any]],
    *,
    prefer: str = "first",
) -> dict[str, Any]:
    """Merge multiple per-chunk extraction dicts into one.

    Args:
        per_chunk: list of dicts, one per chunk. Each dict conforms to
                   the same schema (keys may vary if a chunk returned None
                   for a field).
        prefer:    for scalar fields with conflicting non-null values:
                   - "first"  -> first chunk wins (deterministic, good default)
                   - "last"   -> last chunk wins
                   - "longest"-> the value with the most info wins (heuristic)

    Returns:
        A single merged dict. List fields are unioned. Scalar fields
        follow the `prefer` policy. None / missing fields are skipped.
    """
    if not per_chunk:
        return {}

    merged: dict[str, Any] = {}
    for chunk_dict in per_chunk:
        if not chunk_dict:
            continue
        for key, value in chunk_dict.items():
            if value is None:
                continue
            if key not in merged:
                merged[key] = value
                continue
            existing = merged[key]
            if isinstance(existing, list) or isinstance(value, list):
                merged[key] = _merge_list_field(existing, value)
            else:
                merged[key] = _merge_scalar_field(existing, value, prefer=prefer)

    # Metadata: chunk_count is useful for downstream observability
    merged["_chunk_count"] = len([c for c in per_chunk if c])
    return merged


def _merge_scalar_field(a: Any, b: Any, *, prefer: str) -> Any:
    if a == b:
        return a
    if a is None:
        return b
    if b is None:
        return a
    if prefer == "last":
        return b
    if prefer == "first":
        return a
    if prefer == "longest":
        # Heuristic: prefer the value with more information.
        # Strings: longer wins (more chars = more detail).
        # Numbers: ignore (precision is the same).
        if isinstance(a, str) and isinstance(b, str):
            return a if len(a) >= len(b) else b
        return a
    raise ValueError(f"Unknown prefer={prefer!r}")


def _merge_list_field(a: Any, b: Any) -> list[Any]:
    """Union two list fields with deduplication.

    - If items are dicts: dedupe by 'id' or 'name' key if present.
    - If items are primitives: dedupe by value.
    """
    if a is None:
        return list(b) if isinstance(b, list) else [b]
    if b is None:
        return list(a) if isinstance(a, list) else [a]
    if not isinstance(a, list):
        a = [a]
    if not isinstance(b, list):
        b = [b]

    seen: set[Any] = set()
    out: list[Any] = []

    def _key(item: Any) -> Any:
        if isinstance(item, dict):
            return item.get("id") or item.get("name") or repr(sorted(item.items()))
        return item

    for item in a + b:
        k = _key(item)
        # Hashable? If not (e.g. dict without id/name), keep it anyway.
        try:
            if k in seen:
                continue
            seen.add(k)
        except TypeError:
            # Unhashable key — keep all such items
            pass
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Helper: did this doc need chunking?
# ---------------------------------------------------------------------------
def should_chunk_pages(pages: Sequence[Page], *, max_pages: int) -> bool:
    """True iff the document has more pages than fit in one call."""
    return len(pages) > max_pages


def should_chunk_text(text: str, *, max_tokens: int, encoding_name: str = "cl100k_base") -> bool:
    """True iff the document has more tokens than fit in one call."""
    if not text:
        return False
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text)) > max_tokens
    except ImportError:
        # Conservative fallback: ~4 chars per token.
        return len(text) / 4 > max_tokens


__all__ = [
    "PageChunker",
    "TokenChunker",
    "merge_extractions",
    "should_chunk_pages",
    "should_chunk_text",
]