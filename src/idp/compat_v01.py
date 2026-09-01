"""v0.1 backwards-compatibility shims.

If you wrote code against py-idp v0.1 and relied on
``validate({}, schema='Invoice')`` returning ``passed=True``, the v0.2
required-fields upgrade turns that into ``passed=False``. To restore the
old behaviour without forking the framework, use these permissive
schemas:

    from idp.compat_v01 import Invoice_v01
    from idp.validate.validator import validate
    validate(doc, rules=[], schema_name='Invoice_v01')  # behaves like v0.1

Register them in ``SCHEMA_REGISTRY`` if you want ``Pipeline(...)
    .extract(doc)`` to use them directly:

    from idp.compat_v01 import register_compat_schemas
    register_compat_schemas()
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Invoice_v01(BaseModel):
    """The v0.1 Invoice shape: every field Optional. Use this if your
    v0.1 code relied on ``validate({}, schema='Invoice')`` passing.
    """
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
    line_items: list[dict[str, Any]] = Field(default_factory=list)


class Contract_v01(BaseModel):
    title: str | None = None
    effective_date: str | None = None
    expiration_date: str | None = None
    parties: list[dict[str, Any]] = Field(default_factory=list)
    governing_law: str | None = None
    total_value: float | None = None
    currency: str | None = None
    key_obligations: list[str] = Field(default_factory=list)


class BankStatement_v01(BaseModel):
    account_holder: str | None = None
    account_number_last4: str | None = None
    statement_period_start: str | None = None
    statement_period_end: str | None = None
    opening_balance: float | None = None
    closing_balance: float | None = None
    transactions: list[dict[str, Any]] = Field(default_factory=list)


_COMPAT_SCHEMAS = {
    "Invoice_v01": Invoice_v01,
    "Contract_v01": Contract_v01,
    "BankStatement_v01": BankStatement_v01,
}


def register_compat_schemas() -> None:
    """Add the v0.1 schemas to ``idp.core.schemas.SCHEMA_REGISTRY``.

    Idempotent — calling twice does not duplicate.
    """
    from idp.core.schemas import SCHEMA_REGISTRY
    for name, cls in _COMPAT_SCHEMAS.items():
        if name not in SCHEMA_REGISTRY:
            SCHEMA_REGISTRY[name] = cls  # type: ignore[index,assignment]  # BaseModel subclass vs ModelMetaclass


__all__ = [
    "BankStatement_v01",
    "Contract_v01",
    "Invoice_v01",
    "register_compat_schemas",
]