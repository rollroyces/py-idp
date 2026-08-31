# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <royce-license-placeholder@protonmail.com> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Document / Page / Block data model.

A Document is the single artifact that flows through every stage of the pipeline.
Each stage enriches it (adds parsed_text, classification, extraction, confidence).
"""
from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Block:
    """Atomic chunk of document content with provenance."""

    text: str
    block_type: str = "paragraph"  # paragraph | table | heading | list | image_text
    page: int = 0
    bbox: tuple[float, float, float, float] | None = None  # x0,y0,x1,y1 normalized
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class Page:
    page_number: int
    text: str = ""
    blocks: list[Block] = field(default_factory=list)
    width: float | None = None
    height: float | None = None
    image_path: str | None = None  # rendered page image (used by multimodal)

    @property
    def char_count(self) -> int:
        return len(self.text)


@dataclass
class Document:
    """A document flowing through the IDP pipeline.

    Stages mutate this in-place by setting the corresponding attribute
    (parsed_pages, classification, extraction, confidence, validation).
    """

    source_path: str
    doc_id: str
    pages: list[Page] = field(default_factory=list)
    raw_text: str = ""
    # populated by stages
    parsed_pages: list[dict[str, Any]] | None = None
    classification: str | None = None
    classification_confidence: float | None = None
    extraction: dict[str, Any] | None = None
    extraction_schema: str | None = None
    confidence: dict[str, float] | None = None
    validation: dict[str, Any] | None = None
    mode: str | None = None  # 'multimodal' | 'ocr_llm' (chosen by router)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_path(cls, path: str | Path) -> "Document":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No such file: {path}")
        mt, _ = mimetypes.guess_type(str(p))
        h = hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:16]
        return cls(
            source_path=str(p),
            doc_id=f"{p.stem}-{h}",
            metadata={"size": p.stat().st_size, "mime": mt or "application/octet-stream"},
        )

    @property
    def page_count(self) -> int:
        return len(self.pages) or len(self.parsed_pages or [])

    @property
    def extension(self) -> str:
        return Path(self.source_path).suffix.lower().lstrip(".")

    def parser_used(self) -> str:
        """Convenience: which parser wrote this Document (or '?' if not yet parsed)."""
        return (self.metadata or {}).get("parser", "?")
