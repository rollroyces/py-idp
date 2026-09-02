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

from idp.core.document import Block, Document, Page
from idp.pipeline.pipeline import Pipeline, PipelineResult

__version__ = "0.3.0"
__all__ = ["Document", "Page", "Block", "Pipeline", "PipelineResult"]
