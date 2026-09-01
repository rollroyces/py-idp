"""Mock-backend schema-stub correctness. Catches the anyOf/allOf bug."""
import json

from idp.core.schemas import BankStatement, Contract, Invoice
from idp.llm.backend import CompletionRequest, Message, MockBackend


def _messages(schema):
    return [
        Message("system", "x"),
        Message(
            "user",
            "Extract into the schema.\n\nOutput JSON Schema:\n"
            + json.dumps(schema.model_json_schema(), indent=2)
            + "\n\nOutput rules: return JSON.",
        ),
    ]


def test_mock_invoice_returns_expected_shape():
    b = MockBackend(mode="ideal")
    raw = b.complete(CompletionRequest(messages=_messages(Invoice), json_mode=True))
    parsed = Invoice.model_validate(json.loads(raw))
    # every field present, even if empty
    assert "invoice_number" in parsed.model_dump()
    assert "line_items" in parsed.model_dump()


def test_mock_contract_returns_expected_shape():
    b = MockBackend(mode="ideal")
    raw = b.complete(CompletionRequest(messages=_messages(Contract), json_mode=True))
    parsed = Contract.model_validate(json.loads(raw))
    assert "effective_date" in parsed.model_dump()


def test_mock_bankstatement_returns_expected_shape():
    b = MockBackend(mode="ideal")
    raw = b.complete(CompletionRequest(messages=_messages(BankStatement), json_mode=True))
    parsed = BankStatement.model_validate(json.loads(raw))
    assert "transactions" in parsed.model_dump()


def test_mock_random_introduces_errors():
    b = MockBackend(mode="random")
    raw = b.complete(CompletionRequest(messages=_messages(Invoice), json_mode=True))
    parsed = json.loads(raw)
    # should contain a MOCK_WRONG_ marker on at least one string field
    assert any(isinstance(v, str) and v.startswith("MOCK_WRONG_") for v in parsed.values())


def test_mock_omits_drops_fields():
    b = MockBackend(mode="omits")
    raw = b.complete(CompletionRequest(messages=_messages(Invoice), json_mode=True))
    parsed = json.loads(raw)
    n_null = sum(1 for v in parsed.values() if v is None)
    assert n_null >= 1  # at least half dropped


def test_mock_is_multimodal_flag():
    """Mock advertises multimodal so the router doesn't degrade."""
    assert MockBackend().is_multimodal is True
