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

"""Public eval API for py-idp.

Exposes the dataset runner, regression gate, and pretty-printer used
by CI to fail builds when extraction quality regresses.
"""
from idp.eval.runner import (
    EvalRegressionError,
    check_regression,
    format_report,
    run_dataset,
    verify_against_thresholds,
)

__all__ = [
    "EvalRegressionError",
    "check_regression",
    "format_report",
    "run_dataset",
    "verify_against_thresholds",
]
