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

"""Regression tests for template packaging.

The sample templates live in ``src/idp/templates_pkg/`` and are declared
as package data, so ``pip install py-idp`` actually ships them. Before
this fix they lived only at the repo root (outside ``src/``), which
meant both the wheel and the sdist contained **zero** templates and every
installed user got an empty TemplateRegistry.
"""
from __future__ import annotations

import re
import tarfile
import zipfile
from pathlib import Path

import pytest

from idp.templates import TemplateRegistry, bundled_template_dir

REPO_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_TEMPLATES = {
    "bank_statement",
    "contract",
    "id_card",
    "invoice",
    "medical_claim",
    "w2",
}


# ---------------------------------------------------------------------------
# The bundled directory exists and is loadable
# ---------------------------------------------------------------------------
def test_bundled_template_dir_resolves():
    d = bundled_template_dir()
    assert d is not None, "bundled_template_dir() returned None"
    assert d.is_dir(), f"{d} is not a directory"


def test_bundled_dir_contains_expected_templates():
    d = bundled_template_dir()
    assert d is not None
    found = {p.stem for p in d.glob("*.md")}
    assert found >= EXPECTED_TEMPLATES, (
        f"missing from bundled templates: {sorted(EXPECTED_TEMPLATES - found)}"
    )


def test_bundled_templates_all_parse():
    """Every bundled .md must parse as a valid Template."""
    d = bundled_template_dir()
    assert d is not None
    reg = TemplateRegistry.load(d)
    assert set(reg.list_names()) >= EXPECTED_TEMPLATES


# ---------------------------------------------------------------------------
# load_with_fallback precedence
# ---------------------------------------------------------------------------
def test_load_with_fallback_prefers_user_directory(tmp_path):
    """A user dir with .md files wins over the bundled set."""
    (tmp_path / "custom.md").write_text(
        "---\nname: custom\nschema: Invoice\n---\n\n# Custom\n",
        encoding="utf-8",
    )
    reg = TemplateRegistry.load_with_fallback(tmp_path)
    assert reg.list_names() == ["custom"]


def test_load_with_fallback_uses_bundled_when_dir_missing(tmp_path):
    """A nonexistent directory falls back to the packaged templates."""
    reg = TemplateRegistry.load_with_fallback(tmp_path / "does-not-exist")
    assert set(reg.list_names()) >= EXPECTED_TEMPLATES


def test_load_with_fallback_uses_bundled_when_dir_empty(tmp_path):
    """An empty directory also falls back (it holds no user templates)."""
    reg = TemplateRegistry.load_with_fallback(tmp_path)
    assert set(reg.list_names()) >= EXPECTED_TEMPLATES


def test_load_with_fallback_accepts_none(tmp_path):
    """Passing None means 'no user dir' -> bundled set."""
    reg = TemplateRegistry.load_with_fallback(None)
    assert set(reg.list_names()) >= EXPECTED_TEMPLATES


# ---------------------------------------------------------------------------
# Packaging config (guards against the .md files being dropped again)
# ---------------------------------------------------------------------------
def test_pyproject_declares_package_data():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.setuptools.package-data]" in text, (
        "pyproject.toml has no [tool.setuptools.package-data] block; the "
        "bundled .md templates will be excluded from the wheel"
    )
    assert "templates_pkg" in text, (
        "package-data does not reference the templates_pkg data directory"
    )


@pytest.mark.skipif(
    not (REPO_ROOT / "dist").is_dir(),
    reason="dist/ not built; run `python -m build` first",
)
def test_built_wheel_contains_templates():
    """If dist/ has been built, the newest wheel must contain the .md files."""
    wheels = sorted((REPO_ROOT / "dist").glob("py_idp-*.whl"))
    if not wheels:
        pytest.skip("no wheel in dist/")
    newest = wheels[-1]
    with zipfile.ZipFile(newest) as z:
        names = z.namelist()
    tpl = [n for n in names if n.endswith(".md") and "templates_pkg" in n]
    assert tpl, f"{newest.name} contains no templates_pkg/*.md entries"
    stems = {Path(n).stem for n in tpl}
    assert stems >= EXPECTED_TEMPLATES, (
        f"{newest.name} is missing {sorted(EXPECTED_TEMPLATES - stems)}"
    )


@pytest.mark.skipif(
    not (REPO_ROOT / "dist").is_dir(),
    reason="dist/ not built; run `python -m build` first",
)
def test_built_sdist_contains_templates():
    """If dist/ has been built, the newest sdist must contain the .md files."""
    sdists = sorted((REPO_ROOT / "dist").glob("py_idp-*.tar.gz"))
    if not sdists:
        pytest.skip("no sdist in dist/")
    newest = sdists[-1]
    with tarfile.open(newest) as tf:
        names = tf.getnames()
    tpl = [n for n in names if n.endswith(".md") and "templates_pkg" in n]
    assert tpl, f"{newest.name} contains no templates_pkg/*.md entries"
    stems = {Path(n).stem for n in tpl}
    assert stems >= EXPECTED_TEMPLATES, (
        f"{newest.name} is missing {sorted(EXPECTED_TEMPLATES - stems)}"
    )

# ---------------------------------------------------------------------------
# PyYAML dependency (found by installing the published wheel clean)
# ---------------------------------------------------------------------------
def test_pyyaml_is_a_core_dependency():
    """Bundled templates load by default, so PyYAML must be a core dep.

    Regression: v0.6.1 shipped the templates but left PyYAML optional, so
    a clean `pip install py-idp==0.6.1` then hitting the bundled fallback
    raised ImportError: idp.templates requires PyYAML.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r"^dependencies = \[(.*?)^\]", text, re.MULTILINE | re.DOTALL)
    assert m, "could not locate the core dependencies array"
    assert "pyyaml" in m.group(1).lower(), (
        "PyYAML is missing from core dependencies; the bundled templates "
        "load by default and cannot parse without it"
    )


def test_templates_extra_exists():
    """The error message promises `pip install py-idp[templates]`; honor it.

    Regression: that extra was referenced in idp.templates but never
    declared in pyproject.toml, so the suggested command would fail.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "\ntemplates = [" in text, (
        "no `templates = [...]` extra declared in pyproject.toml, but "
        "idp.templates tells users to run `pip install py-idp[templates]`"
    )


def test_load_with_fallback_degrades_when_yaml_unavailable(monkeypatch):
    """A missing PyYAML must not crash server startup.

    Regression: the fallback path called cls.load(bundled) unguarded, so
    ImportError propagated and took down app startup. It should warn and
    fall back to an empty registry.
    """
    import idp.templates as tmod


    def _boom():
        raise ImportError("idp.templates requires PyYAML. Install it with: pip install PyYAML")

    monkeypatch.setattr(tmod, "_require_yaml", _boom)

    reg = tmod.TemplateRegistry.load_with_fallback(REPO_ROOT / "nonexistent-dir")
    assert reg.list_names() == [], "should degrade to an empty registry, not raise"


def test_load_with_fallback_still_works_with_yaml(monkeypatch):
    """The guard above must not break the normal path."""
    reg = TemplateRegistry.load_with_fallback(None)
    assert set(reg.list_names()) >= EXPECTED_TEMPLATES
