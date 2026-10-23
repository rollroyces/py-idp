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

"""Version-consistency guards.

The project declares its version in three places. Historically they
drifted apart: v0.5.0 bumped ``pyproject.toml`` but left
``src/idp/__init__.py`` at ``0.4.0``, so a user who ran
``pip install py-idp==0.5.0`` and then checked ``idp.__version__``
got the wrong answer. ``CITATION.cff`` drifted even further, sitting
at ``0.3.1.post1`` across two releases.

These tests pin the three declarations to the same string so a future
release cannot silently reintroduce the bug.
"""
from __future__ import annotations

import re
from pathlib import Path

import idp

REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_version() -> str:
    """Read ``[project] version`` out of pyproject.toml.

    Deliberately regex-based rather than ``tomllib``: ``tomllib`` only
    exists on Python 3.11+, and this project supports 3.10 (CI runs a
    3.10 job). ``tomli`` would work too but would add a test-only
    dependency just to read one string.
    """
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Anchor on the [project] table so we don't pick up a `version` key
    # from some other table (e.g. a dependency specifier elsewhere).
    m = re.search(r"^\[project\]$(.*?)^\[", text, re.MULTILINE | re.DOTALL)
    assert m, "could not locate the [project] table in pyproject.toml"
    vm = re.search(
        r'^version\s*=\s*["\']([^"\']+)["\']', m.group(1), re.MULTILINE
    )
    assert vm, "no `version = \"...\"` key found in the [project] table"
    return vm.group(1)


def test_runtime_version_matches_pyproject():
    """``idp.__version__`` must equal ``[project] version`` in pyproject.toml."""
    assert idp.__version__ == _pyproject_version(), (
        f"version drift: idp.__version__={idp.__version__!r} but "
        f"pyproject.toml={_pyproject_version()!r}. Bump both together."
    )


def test_citation_cff_version_matches_pyproject():
    """``CITATION.cff`` must declare the same version as pyproject.toml."""
    text = (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    # Two occurrences: top-level `version:` and the preferred-citation block.
    found = re.findall(r"^\s*version:\s*[\"']?([0-9][^\"'\s]*)[\"']?\s*$", text, re.M)
    assert found, "no `version:` key found in CITATION.cff"
    expected = _pyproject_version()
    stale = [v for v in found if v != expected]
    assert not stale, (
        f"CITATION.cff declares {stale} but pyproject.toml is {expected!r}. "
        f"All CITATION.cff version entries must match."
    )


def test_changelog_has_entry_for_current_version():
    """CHANGELOG.md must document the version currently being packaged."""
    version = _pyproject_version()
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{version}]" in text, (
        f"CHANGELOG.md has no `## [{version}]` section for the packaged version"
    )


def test_version_is_not_a_placeholder():
    """Guard against someone parking on 0.0.0 / x.y.z during a release."""
    version = _pyproject_version()
    assert version != "0.0.0", "version is still the 0.0.0 placeholder"
    assert not re.search(r"[xX]", version), f"version looks templated: {version!r}"
    # Major.minor.patch with no fourth component
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
        f"version {version!r} is not a plain major.minor.patch string"
    )