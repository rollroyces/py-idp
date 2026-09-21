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

"""Test-side helpers for the CORD eval (P4 / C2).

This package exists so the eval runner (``idp.eval.runner``) can stay
single-purpose and format-agnostic. The CORD dataset ships a
``manifest.json`` (one JSON object per doc, with ``doc_path``,
``schema``, ``gold``); the runner reads ``cases.jsonl``. We
materialise the manifest into the runner's expected shape here, at
test-time, without touching the runner.

Two helpers:

* :func:`load_manifest` — returns the parsed manifest as a list of
  case dicts (already in runner shape).
* :func:`write_cases_jsonl` — materialises the manifest into a
  ``cases.jsonl`` file in a target directory and returns the path.
* :func:`cord_dataset_dir` — convenience for tests that want the
  on-disk dataset path.
* :func:`IDP_EVAL_BACKEND` — the env-var the slow real-backend test
  reads to decide which backend to instantiate.

The slow real-backend test (``test_eval_real_dataset.py``) also lives
in this directory so that ``pytest tests/`` does not pick it up
unless ``IDP_EVAL_BACKEND`` is set.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

# Repo-relative path so tests work both from repo root and from CI
REPO = Path(__file__).resolve().parents[2]
CORD_DATASET_DIR = REPO / "src" / "idp" / "eval" / "datasets" / "cord"

# Slow test marker — also registered in pyproject.toml so the default
# suite skips with `-m "not slow"`.
SLOW_MARK = "slow"

# Real-backends the slow test accepts (the ones we ship adapters for)
VALID_REAL_BACKENDS = frozenset({"ollama", "openai", "anthropic"})


def cord_dataset_dir() -> Path:
    """Return the absolute path to the CORD eval dataset directory."""
    return CORD_DATASET_DIR


def load_manifest(dataset_dir: Path | None = None) -> dict[str, Any]:
    """Read the CORD dataset's ``manifest.json`` and return it as a dict.

    Returns the manifest in its native shape:

    .. code-block:: python

       {
         "dataset": "cord",
         "schema": "Receipt",
         "n_docs": 30,
         "docs": [{"doc_path": "docs/receipt-001.txt", "schema": "Receipt",
                   "gold": {...}}, ...]
       }
    """
    ds = dataset_dir or CORD_DATASET_DIR
    path = ds / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no manifest.json at {path}")
    return json.loads(path.read_text())


def to_runner_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert the manifest's ``docs`` list into the eval-runner case shape.

    The eval runner (``idp.eval.runner``) reads ``cases.jsonl`` where
    each line is a JSON object ``{"doc_path", "schema", "gold"}``.
    This helper produces that exact shape from the manifest.
    """
    return [
        {"doc_path": d["doc_path"], "schema": d["schema"], "gold": d.get("gold", {})}
        for d in manifest["docs"]
    ]


def write_cases_jsonl(
    manifest: dict[str, Any], target_dir: Path | None = None
) -> Path:
    """Materialise the manifest into a ``cases.jsonl`` the runner can read.

    If ``target_dir`` is ``None``, a temporary directory is created
    and a copy of the docs tree is also staged so the runner can
    resolve ``doc_path`` relative to it. Returns the target dir path.
    """
    cases = to_runner_cases(manifest)
    if target_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="cord-eval-"))
        # Mirror the docs/ tree next to the cases.jsonl so the runner
        # can resolve "docs/receipt-001.txt" relative to target_dir.
        src_docs = (manifest.get("_source_dir") or CORD_DATASET_DIR) / "docs"
        dst_docs = target_dir / "docs"
        if src_docs.exists():
            shutil.copytree(src_docs, dst_docs)
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "cases.jsonl"
    with out.open("w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    return target_dir


def select_env_backend() -> str | None:
    """Read ``IDP_EVAL_BACKEND`` and return the selected backend name.

    Returns ``None`` when the env var is unset / empty / unrecognised.
    This is the gate the slow real-backend test uses: a missing or
    invalid value means the test is skipped.
    """
    name = os.environ.get("IDP_EVAL_BACKEND", "").strip().lower()
    if not name:
        return None
    if name not in VALID_REAL_BACKENDS:
        return None
    return name


def available_backends() -> list[str]:
    """Names of real backends that the user has credentials for.

    Checks env vars only (no network) — this is a "is the SDK likely
    importable + will auth work" probe, not a liveness check. The
    intent is to give a fast skip in CI without hitting the network.
    """
    candidates = []
    if os.environ.get("OPENAI_API_KEY"):
        candidates.append("openai")
    if os.environ.get("ANTHROPIC_API_KEY"):
        candidates.append("anthropic")
    if os.environ.get("OLLAMA_HOST") or os.environ.get("IDP_OLLAMA_HOST"):
        candidates.append("ollama")
    # Fall through: explicit env wins
    explicit = select_env_backend()
    if explicit and explicit not in candidates:
        candidates.insert(0, explicit)
    return candidates