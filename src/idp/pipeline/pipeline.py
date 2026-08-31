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

"""Pipeline orchestrator.

Composes the six stages: parse -> classify -> extract -> assess -> validate.

Each stage is a pure function that mutates a `Document`. The pipeline
adds structured logging and timing, and returns a `PipelineResult` with
both the document and a per-stage timing/cost breakdown.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel

from idp.assess import assess_confidence
from idp.classify import classify_document
from idp.core.document import Document
from idp.core.types import ExtractionMode
from idp.extract import extract
from idp.llm.backend import Backend, get_backend
from idp.parse import choose_mode, parse_document
from idp.parse.parser import get_parser
from idp.validate import validate
from idp.validate.validator import BusinessRule

log = logging.getLogger(__name__)


@dataclass
class StageTiming:
    name: str
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    document: Document
    schema_name: str
    timings: list[StageTiming]
    backend_name: str
    mode: str | None
    classification: str | None
    confidence: dict[str, float] | None
    validation_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.document.doc_id,
            "source_path": self.document.source_path,
            "schema": self.schema_name,
            "backend": self.backend_name,
            "mode": self.mode,
            "classification": self.classification,
            "extraction": self.document.extraction,
            "confidence": self.confidence,
            "validation": self.document.validation,
            "errors": self.document.errors,
            "timings": [{"name": t.name, "seconds": t.seconds, **t.extra} for t in self.timings],
        }


class Pipeline:
    """Compose the six stages. Each stage can be skipped via flags."""

    def __init__(
        self,
        backend: Backend | str = "auto",
        schema: str | Type[BaseModel] = "Invoice",
        parser: str | Parser | None = "auto",
        use_llm_confidence: bool = False,
        business_rules: list[BusinessRule] | None = None,
    ):
        # resolve objects if the caller passed names
        if isinstance(backend, str):
            backend = get_backend(backend)
        self.backend = backend
        self.backend_name = getattr(backend, "name", "unknown")
        if isinstance(schema, str):
            from idp.core.schemas import get_schema

            self.schema = get_schema(schema)
            self.schema_name = schema
        else:
            self.schema = schema
            self.schema_name = schema.__name__
        self.parser = parser  # resolved lazily inside run() once we have a Document
        self.parser_name = parser if isinstance(parser, str) else None
        self.use_llm_confidence = use_llm_confidence
        self.business_rules = business_rules or []

    def _resolve_parser(self, doc: Document) -> Parser:
        # explicit parser object? take it as-is
        if self.parser is not None and not isinstance(self.parser, str):
            return self.parser
        # explicit name ('pdfplumber', 'docling', 'plain')? use legacy factory
        if isinstance(self.parser, str) and self.parser != "auto":
            return get_parser(self.parser)
        # None or "auto": pick by extension (the recommended path)
        from idp.parse.parser import _auto_pick

        return _auto_pick(doc)

    def run(self, doc: Document) -> PipelineResult:
        timings: list[StageTiming] = []

        # PARSE ------------------------------------------------------------
        t = time.perf_counter()
        parse_document(doc, parser=self._resolve_parser(doc))
        timings.append(StageTiming("parse", time.perf_counter() - t))

        # CLASSIFY ---------------------------------------------------------
        t = time.perf_counter()
        classify_document(doc, self.backend)
        timings.append(StageTiming("classify", time.perf_counter() - t))

        # ROUTE → EXTRACT --------------------------------------------------
        t = time.perf_counter()
        mode = choose_mode(doc, backend_is_multimodal=self.backend.is_multimodal)
        timings.append(StageTiming("route", time.perf_counter() - t, {"chosen": mode.value}))

        t = time.perf_counter()
        extract(doc, self.schema, self.backend, mode=mode)
        timings.append(StageTiming("extract", time.perf_counter() - t, {"mode": mode.value}))

        # ASSESS -----------------------------------------------------------
        t = time.perf_counter()
        assess_confidence(doc, backend=self.backend, use_llm=self.use_llm_confidence)
        timings.append(StageTiming("assess", time.perf_counter() - t))

        # VALIDATE ---------------------------------------------------------
        t = time.perf_counter()
        validate(doc, rules=self.business_rules)
        timings.append(StageTiming("validate", time.perf_counter() - t))

        return PipelineResult(
            document=doc,
            schema_name=self.schema_name,
            timings=timings,
            backend_name=self.backend_name,
            mode=mode.value,
            classification=doc.classification,
            confidence=doc.confidence,
            validation_passed=bool((doc.validation or {}).get("passed", False)),
        )


def run_file(
    path: str | Path,
    schema: str | Type[BaseModel] = "Invoice",
    backend: str = "auto",
    parser: str = "auto",
) -> PipelineResult:
    """Convenience: build the document, run the pipeline, return the result."""
    p = Pipeline(backend=backend, schema=schema, parser=parser)
    doc = Document.from_path(path)
    return p.run(doc)


def save_result(result: PipelineResult, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(result.to_dict(), indent=2, default=str))
