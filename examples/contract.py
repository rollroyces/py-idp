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

"""Example: extract a Contract using the mock backend."""
from pathlib import Path

from idp._util import pretty_print_result
from idp.core.document import Document
from idp.core.schemas import Contract
from idp.llm.backend import get_backend
from idp.pipeline.pipeline import Pipeline
from idp.validate.validator import required_fields_rule

SAMPLE = Path(__file__).parent.parent / "src/idp/eval/datasets/contracts/docs/svc-001.txt"


def main() -> None:
    backend = get_backend("mock")
    pipeline = Pipeline(
        backend=backend,
        schema=Contract,
        business_rules=[required_fields_rule("title", "effective_date", "expiration_date")],
    )
    doc = Document.from_path(SAMPLE)
    result = pipeline.run(doc)
    pretty_print_result(result)


if __name__ == "__main__":
    main()
