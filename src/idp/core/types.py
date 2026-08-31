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

"""Shared types: enums, protocol hints."""

from enum import Enum


class ExtractionMode(str, Enum):
    MULTIMODAL = "multimodal"  # VLM sees page images directly
    OCR_LLM = "ocr_llm"        # parser -> text -> LLM


class ConfidenceLevel(str, Enum):
    HIGH = "high"     # >=0.85
    MEDIUM = "medium"  # 0.6-0.85
    LOW = "low"       # <0.6 -> route to HITL
