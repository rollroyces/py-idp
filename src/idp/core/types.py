"""Shared types: enums, protocol hints."""

from enum import Enum


class ExtractionMode(str, Enum):
    MULTIMODAL = "multimodal"  # VLM sees page images directly
    OCR_LLM = "ocr_llm"        # parser -> text -> LLM


class ConfidenceLevel(str, Enum):
    HIGH = "high"     # >=0.85
    MEDIUM = "medium"  # 0.6-0.85
    LOW = "low"       # <0.6 -> route to HITL
