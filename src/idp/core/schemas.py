"""Built-in example schemas that ship with py-idp.

Users can supply their own Pydantic models; these are reference implementations.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: str
    quantity: float = 1.0
    unit_price: float
    total: float


class Invoice(BaseModel):
    """Standard invoice extraction schema."""

    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    due_date: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    customer_name: Optional[str] = None
    customer_address: Optional[str] = None
    subtotal: Optional[float] = None
    tax_amount: Optional[float] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)


class ContractParty(BaseModel):
    name: str
    role: Optional[str] = None  # e.g., "Licensor", "Licensee"


class Contract(BaseModel):
    """Basic contract extraction schema."""

    title: Optional[str] = None
    effective_date: Optional[str] = None
    expiration_date: Optional[str] = None
    parties: list[ContractParty] = Field(default_factory=list)
    governing_law: Optional[str] = None
    total_value: Optional[float] = None
    currency: Optional[str] = None
    key_obligations: list[str] = Field(default_factory=list)


class BankTransaction(BaseModel):
    date: str
    description: str
    amount: float
    balance: Optional[float] = None


class BankStatement(BaseModel):
    account_holder: Optional[str] = None
    account_number_last4: Optional[str] = None
    statement_period_start: Optional[str] = None
    statement_period_end: Optional[str] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
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
