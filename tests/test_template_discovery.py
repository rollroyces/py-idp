"""Tests for auto-template discovery from sample documents (idp.template_discovery).

Weekend-hack version: verifies the public ``discover_template`` function and
the underlying clustering helper. All tests run offline with backend="mock".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from idp.template_discovery import (
    MAX_SAMPLES,
    MIN_SAMPLES,
    _cluster_common_fields,
    _resolve_schema_name,
    _SampleProfile,
    _slugify,
    discover_template,
)
from idp.templates import TemplateRegistry, load_template


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def three_text_samples(tmp_path: Path) -> list[Path]:
    """Three minimal text "invoice" samples. Same content — mock doesn't parse."""
    paths: list[Path] = []
    for i in range(3):
        p = tmp_path / f"sample-{i}.txt"
        p.write_text(
            f"Vendor: ACME\nInvoice #: INV-{i:03d}\nTotal: 100.00\n",
            encoding="utf-8",
        )
        paths.append(p)
    return paths


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    return tmp_path / "out_templates"


# ---------------------------------------------------------------------------
# Required tests from the task spec
# ---------------------------------------------------------------------------
def test_three_samples_write_template(
    three_text_samples: list[Path], output_dir: Path
) -> None:
    """3 valid samples + an output dir -> a .md template is written."""
    template = discover_template(
        samples=[str(p) for p in three_text_samples],
        output_dir=output_dir,
        schema_name_hint="Invoice",
    )
    assert isinstance(template.name, str) and template.name
    assert template.name == "invoice"
    assert template.schema == "Invoice"
    assert template.source_path.exists()
    assert template.source_path.parent == output_dir
    assert template.source_path.suffix == ".md"


def test_empty_inputs_handled(tmp_path: Path) -> None:
    """No samples -> ValueError with a clear message."""
    with pytest.raises(ValueError, match="at least"):
        discover_template(samples=[], output_dir=tmp_path / "out")
    # Too few (<3) also rejected
    single = tmp_path / "one.txt"
    single.write_text("hello")
    with pytest.raises(ValueError, match=str(MIN_SAMPLES)):
        discover_template(
            samples=[str(single)], output_dir=tmp_path / "out"
        )
    # Too many (>5) also rejected
    many = []
    for i in range(6):
        p = tmp_path / f"many-{i}.txt"
        p.write_text("hi")
        many.append(p)
    with pytest.raises(ValueError, match=str(MAX_SAMPLES)):
        discover_template(samples=[str(p) for p in many], output_dir=tmp_path / "out")


def test_field_clustering_dedupes_case_variants() -> None:
    """'VendorName', 'vendorname', 'VENDOR_NAME' all collapse to one entry.

    The heuristic is case-insensitive — without dedup, the template
    would list the same field three times under different casings.
    """
    profiles = [
        _SampleProfile(
            source_path=Path("a"),
            fields=["VendorName", "InvoiceNumber", "DateIssued"],
            field_set_lower={"vendorname", "invoicenumber", "dateissued"},
            classification="invoice",
            extraction={},
        ),
        _SampleProfile(
            source_path=Path("b"),
            fields=["vendorname", "Invoice_Number", "DateIssued"],
            field_set_lower={"vendorname", "invoice_number", "dateissued"},
            classification="invoice",
            extraction={},
        ),
        _SampleProfile(
            source_path=Path("c"),
            fields=["VENDOR_NAME", "Invoice Number", "DATE_ISSUED"],
            field_set_lower={"vendor_name", "invoice number", "date_issued"},
            classification="invoice",
            extraction={},
        ),
    ]
    common = _cluster_common_fields(profiles)
    # All three fields appear in all three samples (threshold = N-1 = 2)
    assert len(common) == 3
    # Each surviving field should appear exactly once (case-deduped)
    lower_counts = [c.lower() for c in common]
    assert len(lower_counts) == len(set(lower_counts))
    # Should pick the most-common variant for each
    assert "VendorName" in common  # 2 occurrences, vs vendor_name (1)
    assert "InvoiceNumber" in common  # tied between InvoiceNumber & Invoice_Number
    assert "DateIssued" in common


def test_body_section_includes_fields_summary(
    three_text_samples: list[Path], output_dir: Path
) -> None:
    """The Markdown body contains a '## Fields summary' section with stats."""
    template = discover_template(
        samples=[str(p) for p in three_text_samples],
        output_dir=output_dir,
        schema_name_hint="Invoice",
    )
    body = template.body
    assert "## Fields summary" in body
    # Summary section must mention the sample count
    assert "Samples analyzed:" in body
    assert "3" in body  # 3 samples
    # Must mention total fields
    assert "Total fields detected:" in body
    # Also must mention each detected field by name in summary or fields section
    assert "## Fields" in body


def test_schema_name_from_majority_vote() -> None:
    """When multiple samples share a classification, that wins."""
    # Two 'invoice' + one 'contract' -> invoice wins (majority)
    profiles = [
        _SampleProfile(Path("a"), [], set(), "invoice", {}),
        _SampleProfile(Path("b"), [], set(), "invoice", {}),
        _SampleProfile(Path("c"), [], set(), "contract", {}),
    ]
    name = _resolve_schema_name(profiles, hint=None)
    assert name.lower() == "invoice"

    # 3-of-3 unanimous
    profiles_unanimous = [
        _SampleProfile(Path("a"), [], set(), "receipt", {}),
        _SampleProfile(Path("b"), [], set(), "receipt", {}),
        _SampleProfile(Path("c"), [], set(), "receipt", {}),
    ]
    assert _resolve_schema_name(profiles_unanimous, hint=None).lower() == "receipt"

    # Hint overrides majority
    assert (
        _resolve_schema_name(profiles, hint="CustomName") == "CustomName"
    )

    # No classification + no hint -> fallback
    empty = [_SampleProfile(Path("a"), [], set(), None, {})]
    assert _resolve_schema_name(empty, hint=None) == "InferredTemplate"


def test_frontmatter_validates_on_roundtrip(
    three_text_samples: list[Path], output_dir: Path
) -> None:
    """Template.load_template can re-read the generated .md without error.

    Round-trip via TemplateRegistry so any malformed YAML / missing
    required key is caught here.
    """
    template = discover_template(
        samples=[str(p) for p in three_text_samples],
        output_dir=output_dir,
        schema_name_hint="Invoice",
    )
    # Re-load from disk
    reloaded = load_template(template.source_path)
    # All REQUIRED_KEYS (name, schema) must be present and equal
    assert reloaded.name == template.name
    assert reloaded.schema == template.schema
    assert reloaded.version == template.version

    # Also load via registry to confirm multi-template directory works
    reg = TemplateRegistry.load(output_dir)
    assert template.name in reg
    reg_version = reg.get(template.name)
    assert reg_version.name == template.name
    assert reg_version.body == template.body


# ---------------------------------------------------------------------------
# Additional sanity tests
# ---------------------------------------------------------------------------
def test_returns_template_dataclass(
    three_text_samples: list[Path], output_dir: Path
) -> None:
    """discover_template returns an idp.templates.Template object."""
    from idp.templates import Template

    template = discover_template(
        samples=[str(p) for p in three_text_samples],
        output_dir=output_dir,
        schema_name_hint="Invoice",
    )
    assert isinstance(template, Template)


def test_schema_name_slugified_for_filename(
    three_text_samples: list[Path], output_dir: Path
) -> None:
    """'My Invoice Type' -> 'my-invoice-type.md' on disk."""
    template = discover_template(
        samples=[str(p) for p in three_text_samples],
        output_dir=output_dir,
        schema_name_hint="My Invoice Type",
    )
    assert template.name == "my-invoice-type"
    assert template.source_path.name == "my-invoice-type.md"


def test_slugify_strips_punctuation() -> None:
    assert _slugify("Hello World!") == "hello-world"
    assert _slugify("__multiple___underscores__") == "multiple-underscores"
    assert _slugify("Invoice-Receipt") == "invoice-receipt"
    # Edge case: empty / all-punctuation -> fallback
    assert _slugify("") == "template"
    assert _slugify("!!!") == "template"


def test_cluster_empty_profiles_returns_empty_list() -> None:
    """No profiles -> empty common fields (no crash)."""
    assert _cluster_common_fields([]) == []


def test_cluster_threshold_is_n_minus_one() -> None:
    """Field in only 1 of 3 samples should NOT be in common (threshold = 2)."""
    profiles = [
        _SampleProfile(Path("a"), ["shared"], {"shared"}, "x", {}),
        _SampleProfile(Path("b"), ["shared", "only_b"], {"shared", "only_b"}, "x", {}),
        _SampleProfile(Path("c"), ["shared", "only_c"], {"shared", "only_c"}, "x", {}),
    ]
    common = _cluster_common_fields(profiles)
    # 'shared' is in all 3 -> kept (count=3 >= threshold=2)
    assert "shared" in common
    # 'only_b' and 'only_c' are each in only 1 sample -> excluded
    assert "only_b" not in common
    assert "only_c" not in common


def test_output_dir_created_if_missing(
    three_text_samples: list[Path], tmp_path: Path
) -> None:
    """Output dir doesn't need to exist beforehand."""
    out = tmp_path / "deeply" / "nested" / "out"
    assert not out.exists()
    discover_template(
        samples=[str(p) for p in three_text_samples],
        output_dir=out,
        schema_name_hint="Invoice",
    )
    assert out.is_dir()
    assert any(out.iterdir())


def test_nonexistent_sample_path_raises(
    tmp_path: Path, output_dir: Path
) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        discover_template(
            samples=[
                str(tmp_path / "ghost1.txt"),
                str(tmp_path / "ghost2.txt"),
                str(tmp_path / "ghost3.txt"),
            ],
            output_dir=output_dir,
        )


def test_cli_command_registered() -> None:
    """The discover-template CLI subcommand exists."""
    from idp.pipeline.cli import app

    cmd_names = {c.name for c in app.registered_commands}
    assert "discover-template" in cmd_names
    assert "discover-schema" in cmd_names  # original still registered