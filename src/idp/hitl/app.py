"""Streamlit HITL review UI (storage-backed).

Run:
    idp serve --storage sql --db-url sqlite:///./idp.db
    streamlit run src/idp/hitl/app.py

The UI now reads from a Storage backend (default: IDP_DB_URL or IDP_STORAGE_BACKEND,
falling back to JsonFileStorage), and writes reviews back via
`storage.submit_review()` — which is the trigger for online RL policy update.

Pages:
  - Review queue: lists unreviewed results, lets the reviewer edit fields
    and save.
  - Review history: shows past reviews with reviewer, timestamp, edit counts.
"""
from __future__ import annotations

import argparse
import os

import streamlit as st


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--storage", default=None,
                        help="Backend name (memory/json/sql); env IDP_STORAGE_BACKEND")
    parser.add_argument("--db-url", default=None,
                        help="SQL URL for sql backend; env IDP_DB_URL")
    parser.add_argument("--json-path", default=None,
                        help="Path for json backend; default ./idp_data/results.jsonl")
    return parser.parse_known_args()[0]


_ARGS = _parse_args()


@st.cache_resource
def get_storage():
    """Resolve a Storage backend. Cached across reruns."""
    from idp.storage import make_storage

    return make_storage(
        backend=_ARGS.storage,
        json_path=_ARGS.json_path,
        db_url=_ARGS.db_url,
    )


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
def sidebar() -> None:
    storage = get_storage()
    st.sidebar.title("py-idp review")
    st.sidebar.caption(f"backend: `{storage.__class__.__name__}`")
    # counts
    try:
        total = len(storage.list(limit=100000))
        reviewed = len(storage.list(reviewed_only=True, limit=100000))
        st.sidebar.metric("total results", total)
        st.sidebar.metric("reviewed", reviewed)
        st.sidebar.metric("pending", total - reviewed)
    except Exception as e:  # noqa: BLE001
        st.sidebar.warning(f"could not count: {e}")
    st.sidebar.divider()
    page = st.sidebar.radio("page", ["Review queue", "Review history", "About"])
    if page == "Review queue":
        st.session_state.page = "queue"
    elif page == "Review history":
        st.session_state.page = "history"
    else:
        st.session_state.page = "about"


# ----------------------------------------------------------------------
# Page: Review queue
# ----------------------------------------------------------------------
def page_queue() -> None:
    storage = get_storage()
    st.header("Review queue")
    pending = storage.list(reviewed_only=False, limit=200)
    if not pending:
        st.info("No unreviewed results. Run `idp run` on a document to add some.")
        return
    # most recent first
    pending = sorted(pending, key=lambda r: -r.created_at)
    # pickable list
    labels = [f"{r.id}  —  {r.schema_name}  —  {r.source_path}" for r in pending]
    chosen_label = st.selectbox("choose a result to review", labels)
    chosen = next(r for r in pending if r.id in chosen_label)

    st.subheader(f"Result {chosen.id}")
    st.caption(f"schema: {chosen.schema_name} · backend: {chosen.backend_name} · mode: {chosen.mode}")
    st.caption(f"source: {chosen.source_path}")

    # Show model extraction in editable form
    extraction = chosen.extraction or {}
    confidence = chosen.confidence or {}
    edited: dict = {}
    cols = st.columns(3)
    for i, (field_name, val) in enumerate(extraction.items()):
        col = cols[i % 3]
        conf = confidence.get(field_name)
        label = f"{field_name}" + (f" · conf={conf:.2f}" if conf is not None else "")
        with col:
            if conf is not None and conf < 0.6:
                st.markdown(f":red[**{label}**]")
            else:
                st.markdown(label)
            if isinstance(val, (dict, list)):
                new_v = st.text_area(
                    f"{field_name}_raw",
                    value=str(val),
                    key=f"{chosen.id}_{field_name}_raw",
                    height=160,
                )
                try:
                    import json as _json
                    edited[field_name] = _json.loads(new_v)
                except Exception:  # noqa: BLE001
                    edited[field_name] = new_v
                    st.warning(f"{field_name}: JSON invalid; saved as raw")
            else:
                edited[field_name] = st.text_input(
                    f"{field_name}_val",
                    value="" if val is None else str(val),
                    key=f"{chosen.id}_{field_name}_val",
                )

    col_save, col_revert = st.columns(2)
    reviewer = st.text_input("reviewer", value=os.environ.get("USER", "streamlit"))
    with col_save:
        if st.button("Save review", type="primary"):
            try:
                # SqlStorage exposes submit_review; legacy storage
                # falls back to mark_reviewed.
                if hasattr(storage, "submit_review"):
                    storage.submit_review(
                        result_id=chosen.id,
                        edited=edited,
                        reviewer=reviewer,
                    )
                else:
                    storage.mark_reviewed(chosen.id, edited, reviewer)
                st.success(f"Saved review for {chosen.id}")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"save failed: {e}")
    with col_revert:
        if st.button("Skip / clear edits"):
            st.warning("Edits cleared. (Not saved.)")


# ----------------------------------------------------------------------
# Page: Review history
# ----------------------------------------------------------------------
def page_history() -> None:
    storage = get_storage()
    st.header("Review history")
    # SqlStorage can hand us the rich join; for JsonFileStorage we just show reviewed list
    rows = storage.list(reviewed_only=True, limit=200)
    if not rows:
        st.info("No reviews yet.")
        return
    rows = sorted(rows, key=lambda r: -r.created_at)
    for r in rows:
        with st.expander(f"{r.id}  ·  {r.schema_name}  ·  reviewer={r.reviewer}"):
            st.caption(f"source: {r.source_path}")
            if r.last_reviewed_at:
                st.caption(f"last_reviewed_at: {r.last_reviewed_at}")
            n_corrections = sum(
                1
                for k, v in (r.extraction or {}).items()
                if (r.reviewed_extraction or {}).get(k) != v
            )
            st.metric("fields corrected", n_corrections)


# ----------------------------------------------------------------------
# Page: About
# ----------------------------------------------------------------------
def page_about() -> None:
    st.header("About py-idp HITL")
    st.markdown(
        """
        This UI writes reviews to the configured `Storage` backend.
        Reviews flow through to the RL policy update at
        `idp.rl.PolicyCache` if one is attached to the same process
        (typically via `idp serve --storage sql`).

        Without a PolicyCache, reviews are persisted but the policy
        is updated only when you run `idp rl-update` against the same
        storage path.
        """
    )


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="py-idp review", layout="wide")
    sidebar()
    page = st.session_state.get("page", "queue")
    if page == "queue":
        page_queue()
    elif page == "history":
        page_history()
    else:
        page_about()


# Streamlit always runs the script body; the storage cache decorator
# means the backend is resolved only once per server lifetime.
main()
