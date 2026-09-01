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

"""Built-in example schemas that ship with py-idp.

Users can supply their own Pydantic models; these are reference implementations.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: float = 1.0
    unit_price: float
    total: float


class Invoice(BaseModel):
    """Standard invoice extraction schema."""

    invoice_number: str | None = None
    invoice_date: str | None = None
    due_date: str | None = None
    vendor_name: str | None = None
    vendor_address: str | None = None
    customer_name: str | None = None
    customer_address: str | None = None
    subtotal: float | None = None
    tax_amount: float | None = None
    total_amount: float | None = None
    currency: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class ContractParty(BaseModel):
    name: str
    role: str | None = None  # e.g., "Licensor", "Licensee"


class Contract(BaseModel):
    """Basic contract extraction schema."""

    title: str | None = None
    effective_date: str | None = None
    expiration_date: str | None = None
    parties: list[ContractParty] = Field(default_factory=list)
    governing_law: str | None = None
    total_value: float | None = None
    currency: str | None = None
    key_obligations: list[str] = Field(default_factory=list)


class BankTransaction(BaseModel):
    date: str
    description: str
    amount: float
    balance: float | None = None


class BankStatement(BaseModel):
    account_holder: str | None = None
    account_number_last4: str | None = None
    statement_period_start: str | None = None
    statement_period_end: str | None = None
    opening_balance: float | None = None
    closing_balance: float | None = None
    transactions: list[BankTransaction] = Field(default_factory=list)


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "Invoice": Invoice,
    "Contract": Contract,
    "BankStatement": BankStatement,
}


def get_schema(name: str) -> type[BaseModel]:
    if name not in SCHEMA_REGISTRY:
        raise KeyError(
            f"Unknown schema '{name}'. Available: {list(SCHEMA_REGISTRY)} "
            "or pass your own Pydantic model to extract()."
        )
    return SCHEMA_REGISTRY[name]
