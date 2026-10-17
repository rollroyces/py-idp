# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <roycelam@umich.edu> for terms.
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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from idp.assess import assess_confidence
from idp.classify import classify_document
from idp.core.document import Document
from idp.extract import extract
from idp.llm.backend import Backend, get_backend
from idp.parse import choose_mode, parse_document
from idp.parse.parser import get_parser
from idp.reliability import (
    CachingBackend,
    ExtractionCache,
    RetryConfig,
    RetryingBackend,
)
from idp.validate import validate
from idp.validate.validator import BusinessRule

if TYPE_CHECKING:
    from idp.parse.parser import Parser
    from idp.rl.policy import PolicyConfig

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
    template_name: str | None = None
    template_version: int | None = None

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
            "template_name": self.template_name,
            "template_version": self.template_version,
            "errors": self.document.errors,
            "timings": [{"name": t.name, "seconds": t.seconds, **t.extra} for t in self.timings],
        }


class Pipeline:
    """Compose the six stages. Each stage can be skipped via flags.

    Template support: pass a :class:`idp.templates.Template` (or a
    string name resolved against a registry) and the template's
    Markdown body is prepended to every LLM extraction call as
    document-type-specific guidance. See ``idp/templates.py``.
    """

    def __init__(
        self,
        backend: Backend | str = "auto",
        schema: str | type[BaseModel] = "Invoice",
        parser: str | Parser | None = "auto",
        use_llm_confidence: bool = False,
        business_rules: list[BusinessRule] | None = None,
        policy: PolicyConfig | None = None,
        policy_path: str | None = None,
        retry: RetryConfig | bool = False,
        cache: ExtractionCache | bool = False,
        template: Any = None,  # idp.templates.Template | str | None
    ):
        # Resolve backend to a real object first, then optionally wrap it
        # with retry + cache. We wrap lazily (not at construction) so the
        # caller can still see the original backend for debugging, but the
        # wrapped one is what gets called inside the pipeline.
        if isinstance(backend, str):
            backend = get_backend(backend)
        # Save the "logical" name (before wrapping) for cache keying
        _logical_backend_name = getattr(backend, "name", "unknown")
        # Retry wrapper
        if retry is True:
            retry = RetryConfig()
        if isinstance(retry, RetryConfig):
            backend = RetryingBackend(backend, retry)  # type: ignore[assignment]
        # Cache wrapper (applied AFTER retry so cached hits skip retries)
        if cache is True:
            cache = ExtractionCache(
                str(Path.home() / ".cache" / "idp" / "extract.db")
            )
        if isinstance(cache, ExtractionCache):
            # Need schema_name; resolve it first by getting the schema
            pass  # done after self.schema is set
        self.backend: Backend = backend  # type: ignore[assignment]
        self.backend_name = getattr(backend, "name", _logical_backend_name)
        if isinstance(schema, str):
            from idp.core.schemas import get_schema

            self.schema = get_schema(schema)
            self.schema_name = schema
        else:
            self.schema = schema
            self.schema_name = schema.__name__
        # Now that schema_name is known, wrap with cache if requested.
        # We re-wrap because CachingBackend requires schema_name.
        if isinstance(cache, ExtractionCache):
            self.backend = CachingBackend(
                backend, cache, schema_name=self.schema_name
            )  # type: ignore[assignment]
            self.backend_name = self.backend.name
        self.cache = cache if isinstance(cache, ExtractionCache) else None
        self.parser = parser  # resolved lazily inside run() once we have a Document
        self.parser_name = parser if isinstance(parser, str) else None
        self.use_llm_confidence = use_llm_confidence
        self.business_rules = business_rules or []
        # Template: accept either a Template object, a string name
        # (loaded from the in-process registry if set via
        # ``Pipeline.set_template_registry``), or None for no template.
        # The body is read at run() time, not at construction, so the
        # template can be edited and hot-reloaded between calls.
        self._template_obj: Any = None
        self._template_name: str | None = None
        self._template_body: str = ""
        self._template_version: int | None = None
        self._template_registry: Any = None
        if template is not None:
            if isinstance(template, str):
                self._template_name = template
            else:
                # Assume it's a Template-shaped object
                self._template_obj = template
                self._template_name = template.name
                self._template_body = template.body
                self._template_version = template.version
        # Policy: load from path if given, else use the passed object
        self.policy = None
        if policy is not None:
            self.policy = policy
        elif policy_path is not None:
            try:
                from idp.rl.policy import PolicyConfig
                self.policy = PolicyConfig.load(policy_path)
            except Exception as e:  # noqa: BLE001
                log.warning("failed to load policy from %s: %s", policy_path, e)

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

        # Resolve template body if a name was provided (not a Template obj).
        # The body is read at run() time, not at construction, so the
        # template registry can hot-reload changes between calls.
        template_name = self._template_name
        template_version = self._template_version
        template_body = self._template_body
        if template_name and not template_body and self._template_registry is not None:
            try:
                t = self._template_registry.get(template_name)
                template_body = t.body
                template_version = t.version
            except Exception:  # noqa: BLE001
                log.warning("template %r not found in registry; running without it", template_name)
                template_name = None

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
        extract(doc, self.schema, self.backend, mode=mode,
                template_body=template_body,
                template_name=template_name,
                template_version=template_version)
        timings.append(StageTiming("extract", time.perf_counter() - t, {"mode": mode.value}))

        # ASSESS -----------------------------------------------------------
        t = time.perf_counter()
        assess_confidence(
            doc, backend=self.backend, use_llm=self.use_llm_confidence, policy=self.policy,
        )
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
            template_name=template_name,
            template_version=template_version,
        )

    async def arun(self, doc: Document) -> PipelineResult:
        """Async variant of ``run()`` — runs the blocking pipeline in a
        worker thread so the FastAPI event loop stays responsive.

        Without this, ``Pipeline.run()`` (which calls sync LLM clients and
        on Nanonets runs torch inference) blocks the event loop for
        5–15 s per request, freezing the entire server. ``arun`` uses
        ``asyncio.to_thread`` to dispatch the blocking work to the default
        thread pool, freeing the event loop to handle health checks,
        other concurrent requests, etc.

        Performance note: this trades parallelism-on-the-loop for one
        blocking operation per request. For sustained throughput you still
        want a separate worker process; ``arun`` is the minimum-viable
        fix to keep the HTTP tier responsive when Nanonets (or any
        synchronous backend) is the chosen LLM.
        """
        import asyncio
        return await asyncio.to_thread(self.run, doc)

    def set_template_registry(self, registry: Any) -> None:
        """Attach a TemplateRegistry for resolving string template names.

        Usage::

            from idp.templates import TemplateRegistry
            registry = TemplateRegistry.load("./templates")
            pipeline = Pipeline(backend="mock", template="invoice")
            pipeline.set_template_registry(registry)
            result = pipeline.run(doc)

        The registry is consulted at run() time, so calls benefit from
        hot-reload if ``watch=True`` was passed when loading.
        """
        self._template_registry = registry


def run_file(
    path: str | Path,
    schema: str | type[BaseModel] = "Invoice",
    backend: str = "auto",
    parser: str = "auto",
) -> PipelineResult:
    """Convenience: build the document, run the pipeline, return the result."""
    p = Pipeline(backend=backend, schema=schema, parser=parser)
    doc = Document.from_path(path)
    return p.run(doc)


def save_result(result: PipelineResult, output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(result.to_dict(), indent=2, default=str))
