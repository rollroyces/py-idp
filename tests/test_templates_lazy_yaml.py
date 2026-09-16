"""Tests for lazy PyYAML import in idp.templates.

PyYAML is an optional dependency — templates are a feature, not a core
dependency. ``import idp`` must work on a bare pip install. These tests
pin that contract.
"""
from __future__ import annotations

import sys

import pytest


def test_load_template_without_yaml_raises_helpful_error(tmp_path) -> None:
    """load_template() raises ImportError pointing to the right dep."""
    # Write a minimal valid template file
    template = tmp_path / "tpl.md"
    template.write_text("---\nname: t1\nschema: Invoice\n---\nbody\n")

    # Force idp.templates' lazy yaml import to fail by hiding it
    # from sys.modules. We also drop our cached reference so the next
    # call goes through _require_yaml again.
    import idp.templates as tpl_mod
    saved_yaml = tpl_mod.yaml
    tpl_mod.yaml = None
    sys.modules.pop("yaml", None)
    # Make `import yaml` raise so the lazy loader hits its except branch
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name == "yaml":
                raise ImportError("blocked by test")
        def find_module(self, name, path=None):
            if name == "yaml":
                raise ImportError("blocked by test")
    sys.meta_path.insert(0, _Blocker())

    try:
        load_template = tpl_mod.load_template
        with pytest.raises(ImportError) as exc_info:
            load_template(template)
        msg = str(exc_info.value)
        assert "PyYAML" in msg or "yaml" in msg.lower()
        assert "pip install" in msg
    finally:
        sys.meta_path.pop(0)
        sys.modules.pop("yaml", None)
        tpl_mod.yaml = saved_yaml


def test_load_template_with_yaml_works(tmp_path) -> None:
    """Happy path: yaml is loaded lazily and template parses."""
    template = tmp_path / "tpl.md"
    template.write_text("---\nname: t1\nschema: Invoice\n---\nbody\n")

    from idp.templates import load_template

    tpl = load_template(template)
    assert tpl.name == "t1"
    assert tpl.schema == "Invoice"
    assert tpl.body.strip() == "body"