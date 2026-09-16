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

"""py-idp: General-purpose, AI-enabled Intelligent Document Processing framework.

A six-stage pipeline: parse -> classify -> extract -> assess -> validate -> HITL.
Each stage is a pure function over a Document, pluggable, and independently testable.

Design draws from:
  - aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws
    (pipeline shape, HITL, confidence assessment)
  - docling-project/docling                   (parser: PDF, tables, reading order)
  - run-llama/llama_cloud_services            (Pydantic-schema-driven extraction API)
  - Unstructured-IO/unstructured              (chunking + multi-format ingest)
"""

from idp.batch import BatchItemResult, process_batch
from idp.core.document import Block, Document, Page
from idp.discover import DiscoveryResult, discover_schema
from idp.errors import IDPError, error_envelope
from idp.pipeline.pipeline import Pipeline, PipelineResult
from idp.templates import Template, TemplateRegistry, load_template

# Kept in sync with `version` in pyproject.toml.
# pyproject.toml is the source of truth (used by `python -m build` and
# trusted-publisher publish); this __version__ is mirrored here so users
# can introspect it at runtime via `import idp; idp.__version__`.
__version__ = "0.3.6"
__all__ = [
    "BatchItemResult",
    "Block",
    "DiscoveryResult",
    "Document",
    "IDPError",
    "Page",
    "Pipeline",
    "PipelineResult",
    "Template",
    "TemplateRegistry",
    "discover_schema",
    "error_envelope",
    "load_template",
    "process_batch",
]
