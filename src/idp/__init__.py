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

from idp.core.document import Document, Page, Block
from idp.pipeline.pipeline import Pipeline, PipelineResult

__version__ = "0.1.0"
__all__ = ["Document", "Page", "Block", "Pipeline", "PipelineResult"]
