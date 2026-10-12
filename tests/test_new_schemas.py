"""Verify the new W2, IDCard, MedicalClaim schemas + templates register cleanly."""
from pathlib import Path

import pytest

from idp.core.schemas import (
    SCHEMA_REGISTRY,
    W2,
    IDCard,
    MedicalClaim,
    get_schema,
)
from idp.templates import TemplateRegistry


def test_schemas_in_registry():
    assert "W2" in SCHEMA_REGISTRY
    assert "IDCard" in SCHEMA_REGISTRY
    assert "MedicalClaim" in SCHEMA_REGISTRY


def test_get_schema_returns_classes():
    assert get_schema("W2") is W2
    assert get_schema("IDCard") is IDCard
    assert get_schema("MedicalClaim") is MedicalClaim


def test_w2_minimum_required():
    w = W2(employee_name="Jane Doe", employer_name="ACME Inc", wages_tips_compensation=50000.0)
    assert w.employee_name == "Jane Doe"
    assert w.tax_year is None  # optional


def test_idcard_minimum_required():
    id_ = IDCard(document_type="Driver's License", full_name="Jane Doe")
    assert id_.full_name == "Jane Doe"
    assert id_.expiration_date is None  # optional


def test_medical_claim_minimum_required():
    claim = MedicalClaim(patient_name="Jane Doe", provider_name="Mercy Hospital")
    assert claim.patient_name == "Jane Doe"
    assert claim.line_items == []  # default empty list


def test_templates_load():
    """The 3 new templates should be discoverable from the templates/ directory."""
    repo_root = Path(__file__).resolve().parent.parent
    templates_dir = repo_root / "templates"
    registry = TemplateRegistry.load(templates_dir)
    assert "w2" in registry
    assert "id_card" in registry
    assert "medical_claim" in registry


def test_templates_metadata():
    repo_root = Path(__file__).resolve().parent.parent
    templates_dir = repo_root / "templates"
    registry = TemplateRegistry.load(templates_dir)
    w2 = registry.get("w2")
    assert w2.schema == "W2"
    assert w2.version >= 1
    assert w2.mime_types  # not empty
    assert w2.filename_patterns  # not empty
    assert w2.classification_hints  # not empty


def test_filename_match_w2():
    repo_root = Path(__file__).resolve().parent.parent
    templates_dir = repo_root / "templates"
    registry = TemplateRegistry.load(templates_dir)
    assert registry.find_for_filename("acme_w2_2024.pdf") is not None
    assert registry.find_for_filename("acme_w2_2024.pdf").name == "w2"


def test_filename_match_idcard():
    repo_root = Path(__file__).resolve().parent.parent
    templates_dir = repo_root / "templates"
    registry = TemplateRegistry.load(templates_dir)
    t = registry.find_for_filename("john-doe-drivers-license.jpg")
    assert t is not None
    assert t.name == "id_card"


def test_filename_match_medical():
    repo_root = Path(__file__).resolve().parent.parent
    templates_dir = repo_root / "templates"
    registry = TemplateRegistry.load(templates_dir)
    t = registry.find_for_filename("claim_2024_Q4_cms1500.pdf")
    assert t is not None
    assert t.name == "medical_claim"


def test_validation_required_fields():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        W2(employer_name="ACME")  # missing employee_name
    with pytest.raises(ValidationError):
        IDCard(full_name="Jane")  # missing document_type
    with pytest.raises(ValidationError):
        MedicalClaim(provider_name="Mercy")  # missing patient_name
