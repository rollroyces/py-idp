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

"""Pretty-print a PipelineResult in a terminal-friendly way."""
from __future__ import annotations

import json


def pretty_print_result(result) -> None:
    """Console-friendly rendering of a pipeline result."""
    print(f"\n=== {result.document.source_path} ===")
    print(f"schema:    {result.schema_name}")
    print(f"backend:   {result.backend_name} ({result.mode})")
    print(f"classify:  {result.classification} (conf={result.document.classification_confidence})")
    print(f"validate:  {'PASS' if result.validation_passed else 'FAIL'}")
    print(f"timings:   " + ", ".join(f"{t.name}={t.seconds:.3f}s" for t in result.timings))
    print("\nextraction:")
    print(json.dumps(result.document.extraction, indent=2, default=str))
    if result.confidence:
        ordered = sorted(result.confidence.items(), key=lambda kv: kv[1])
        print("\nconfidence (ascending):")
        for k, v in ordered:
            mark = " [REVIEW]" if v < 0.6 else ""
            print(f"  {k:<24} {v:.2f}{mark}")
    if result.document.errors:
        print(f"\nerrors ({len(result.document.errors)}):")
        for e in result.document.errors:
            print(f"  - {e}")
