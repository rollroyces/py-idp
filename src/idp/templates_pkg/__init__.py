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

"""Bundled sample document templates (package data).

This directory holds the ``.md`` template files that ship inside the
wheel and sdist so ``pip install py-idp`` gets a working
``TemplateRegistry`` out of the box.

It is a package (has this ``__init__.py``) purely so that
``importlib.resources.files("idp.templates_pkg")`` resolves and so
setuptools includes the ``.md`` files via
``[tool.setuptools.package-data]`` in ``pyproject.toml``.

There is no Python API here — the templates are data, not code. Use
:func:`idp.templates.bundled_template_dir` to locate this directory, or
:meth:`idp.templates.TemplateRegistry.load_with_fallback` to load with
user-supplied templates taking precedence.
"""