# py-idp: general-purpose, AI-enabled Intelligent Document Processing.
# Copyright (c) 2026 Royce.
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)
# with the following addition: a commercial license is also available for organizations
# that wish to embed py-idp in proprietary products / hosted SaaS without the AGPL
# copyleft obligations. See LICENSE and LICENSE-COMMERCIAL at the repo root, or
# contact <royce-license-placeholder@protonmail.com> for terms.
#
# This Source Code Form is subject to the terms of the AGPL-3.0-or-later.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Streamlit HITL review UI.

Launch:
    idp run <doc> --output out.json
    streamlit run src/idp/hitl/app.py

It also accepts `--results <path>` and `--save-as <path>` via sys.argv.

Reviewers see extracted JSON with per-field confidence. Fields with
conf < 0.6 are highlighted. They can edit any field and save.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import streamlit as st


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--save-as", default="reviewed.json")
    # Streamlit passes its own flags; ignore unknown ones
    return parser.parse_known_args()[0]


_ARGS = _parse_args()


def main() -> None:
    st.set_page_config(page_title="py-idp review", layout="wide")
    st.title("py-idp human-in-the-loop review")

    results_path = Path(_ARGS.results)
    if not results_path.exists():
        st.error(
            f"no results file at {results_path}. "
            f"Generate one with: idp run <file> --output {_ARGS.results}"
        )
        st.stop()

    result = json.loads(results_path.read_text())
    st.subheader(f"Document: {result.get('source_path', '?')}")
    st.caption(
        f"schema={result.get('schema')} • backend={result.get('backend')} "
        f"({result.get('mode')}) • classification={result.get('classification')}"
    )

    extraction = result.get("extraction", {})
    confidence = result.get("confidence") or {}

    st.subheader("Review fields")
    st.caption("Fields with conf < 0.6 are highlighted in red.")

    edited = {}
    items = list(extraction.items())
    cols = st.columns(3)
    for i, (k, v) in enumerate(items):
        col = cols[i % 3]
        conf = confidence.get(k)
        label = f"{k}" + (f" • conf={conf:.2f}" if conf is not None else "")
        with col:
            if conf is not None and conf < 0.6:
                st.markdown(f":red[**{label}**]")
            else:
                st.markdown(label)
            if isinstance(v, (dict, list)):
                new_v = st.text_area(
                    f"{k}_raw",
                    value=json.dumps(v, indent=2),
                    key=f"{k}_edit",
                    height=160,
                )
                try:
                    edited[k] = json.loads(new_v)
                except json.JSONDecodeError:
                    edited[k] = new_v
                    st.warning(f"{k}: JSON invalid — saved as raw text")
            else:
                edited[k] = st.text_input(
                    f"{k}_val",
                    value="" if v is None else str(v),
                    key=f"{k}_edit",
                )

    st.subheader("Validation")
    st.json(result.get("validation") or {})

    with st.expander("Errors / warnings"):
        st.write("\n".join(result.get("errors", [])) or "(none)")

    c1, c2 = st.columns(2)
    if c1.button("Save reviewed JSON"):
        out = dict(result)
        out["extraction"] = edited
        out["review"] = {"reviewed": True, "reviewer": "streamlit"}
        Path(_ARGS.save_as).write_text(json.dumps(out, indent=2, default=str))
        st.success(f"saved: {_ARGS.save_as}")
    if c2.button("Download original JSON"):
        st.download_button(
            "download",
            results_path.read_text(),
            file_name=results_path.name,
        )


# Streamlit always runs the script body, so we always call main().
# The argparse parse is moved out of the imperative top-level to make the
# module importable for tests without a results file present.
main()
