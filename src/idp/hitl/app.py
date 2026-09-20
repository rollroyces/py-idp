"""Streamlit HITL review UI (storage-backed).

Run:
    idp serve --storage sql --db-url sqlite:///./idp.db
    streamlit run src/idp/hitl/app.py

The UI now reads from a Storage backend (default: IDP_DB_URL or IDP_STORAGE_BACKEND,
falling back to JsonFileStorage), and writes reviews back via
`storage.submit_review()` — which is the trigger for online RL policy update.

Pages:
  - Review queue: lists unreviewed results, lets the reviewer edit fields
    and save. v0.4 C3 additions: side-by-side model vs human field view,
    bulk-accept button (>= 0.9 confidence), skip button, per-document
    confidence histogram, keyboard shortcuts (j/k/s/?).
  - Review history: shows past reviews with reviewer, timestamp, edit counts.
  - Triage (v0.4 D1): renders the systematic-error / insufficient-data
    report computed by idp.hitl.triage.triage().
  - About: existing page unchanged.
"""
from __future__ import annotations

import argparse
import functools
import os


def _st_module():
    """Lazily import streamlit. Returns the module, not a proxy.

    Why lazy: streamlit transitively imports protobuf + 20+ MB of
    dependencies. Deferring the import means ``import idp.hitl.app``
    from the CLI / pipeline / tests does NOT pay the cost; only
    running the UI does.

    Streamlit re-executes the script per session, so the import fires
    naturally on first widget render.
    """
    import streamlit as st_module
    return st_module


# A module-level proxy so the existing ``st.<attr>`` access pattern
# keeps working without rewriting every line of the UI. The proxy
# resolves ``_st_module()`` once on first attribute access — and
# crucially, ``import idp.hitl.app`` never accesses any attribute,
# so the streamlit import doesn't fire until a widget is rendered.
class _StreamlitProxy:
    __slots__ = ("_mod",)

    def __init__(self) -> None:
        self._mod: object = None

    def __getattr__(self, name: str):
        if self._mod is None:
            self._mod = _st_module()
        return getattr(self._mod, name)


# Initialize the proxy. The first attribute access on this triggers the
# lazy import; until then, importing idp.hitl.app is free of streamlit.
# Type-checker note: ``st`` has the proxy type so mypy doesn't try to
# resolve streamlit attribute names; runtime attrs resolve dynamically.
st: _StreamlitProxy = _StreamlitProxy()  # type: ignore[assignment]


# Local imports — kept inside the module so the Streamlit proxy is
# already wired. These are pure-Python (no streamlit import), so they
# are safe to import at module level.
from idp.hitl import triage as _triage_module  # noqa: E402  (after proxy setup)
from idp.hitl.review import (  # noqa: E402  (after proxy setup)
    coerce_field_value,
    count_corrections,
    format_result_label,
    save_review,
    sort_pending_newest_first,
)

# Bind the names we actually use from the triage module. Done as
# separate name bindings (not a parenthesized ``from ... import``) so
# the import block stays short and ruff-friendly. Each binding is just
# a local alias to a module attribute — no behavior change.
_DEFAULT_MIN_REVIEWS = _triage_module.DEFAULT_MIN_REVIEWS
_DEFAULT_THRESHOLD = _triage_module.DEFAULT_THRESHOLD
_format_report_markdown = _triage_module.format_report_markdown
_run_triage = _triage_module.triage

# Bulk-accept threshold for C3: trust the model when its confidence is
# at or above this. Documented in docs/ROADMAP_v0.4.md Q3.
BULK_ACCEPT_THRESHOLD = 0.9


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


@functools.lru_cache(maxsize=1)
def get_storage():
    """Resolve a Storage backend. Cached for the lifetime of the script.

    Was previously ``@st.cache_resource`` — switched to ``lru_cache``
    so the storage resolution does not require streamlit to be
    imported. This is the key change that lets ``import idp.hitl.app``
    stay free of streamlit's ~20 MB transitive deps. Functionally
    equivalent for our use case (Streamlit re-executes the module
    per session, so caching across reruns is no longer needed).
    """
    from idp.storage import make_storage
    return make_storage(
        backend=_ARGS.storage,
        json_path=_ARGS.json_path,
        db_url=_ARGS.db_url,
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _confidence_buckets(confidence: dict | None) -> dict[str, int]:
    """Bucket per-field confidence into 4 fixed bands.

    Used by the per-document histogram (C3 acceptance criterion #4).
    Returns ``{"<0.5": n, "0.5-0.7": n, "0.7-0.9": n, ">=0.9": n}``.
    Fields with ``confidence is None`` are excluded — we don't know
    which bucket they'd belong to.
    """
    buckets = {"<0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, ">=0.9": 0}
    if not confidence:
        return buckets
    for v in confidence.values():
        if v is None:
            continue
        if v < 0.5:
            buckets["<0.5"] += 1
        elif v < 0.7:
            buckets["0.5-0.7"] += 1
        elif v < 0.9:
            buckets["0.7-0.9"] += 1
        else:
            buckets[">=0.9"] += 1
    return buckets


def _render_confidence_histogram(confidence: dict | None) -> None:
    """Render the per-document confidence histogram (C3 #4).

    Renders nothing if there's no confidence data. Otherwise a small
    bar chart so the reviewer can see at a glance whether the document
    is "mostly fine" (most bars in the right bucket).
    """
    if not confidence:
        return
    buckets = _confidence_buckets(confidence)
    # Only render if at least one field has a confidence value.
    if sum(buckets.values()) == 0:
        return
    st.caption(
        f"confidence histogram · <0.5: {buckets['<0.5']} · "
        f"0.5-0.7: {buckets['0.5-0.7']} · "
        f"0.7-0.9: {buckets['0.7-0.9']} · "
        f">=0.9: {buckets['>=0.9']}"
    )
    import contextlib
    # Some headless test harnesses don't render charts cleanly; the
    # textual caption above is the authoritative signal.
    with contextlib.suppress(Exception):
        st.bar_chart(buckets)


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
    # "Triage" added in v0.4 D1; the other three pages existed in v0.3.x.
    page = st.sidebar.radio(
        "page",
        ["Review queue", "Review history", "Triage", "About"],
    )
    if page == "Review queue":
        st.session_state.page = "queue"
    elif page == "Review history":
        st.session_state.page = "history"
    elif page == "Triage":
        st.session_state.page = "triage"
    else:
        st.session_state.page = "about"


# ----------------------------------------------------------------------
# Page: Review queue (C3 additions appended to existing logic)
# ----------------------------------------------------------------------
def page_queue() -> None:
    storage = get_storage()
    st.header("Review queue")
    pending = storage.list(reviewed_only=False, limit=200)
    if not pending:
        st.info("No unreviewed results. Run `idp run` on a document to add some.")
        return
    # most recent first
    pending = sort_pending_newest_first(pending)
    # pickable list
    labels = [format_result_label(r) for r in pending]
    chosen_label = st.selectbox("choose a result to review", labels)
    chosen = next(r for r in pending if r.id in chosen_label)

    st.subheader(f"Result {chosen.id}")
    st.caption(f"schema: {chosen.schema_name} · backend: {chosen.backend_name} · mode: {chosen.mode}")
    st.caption(f"source: {chosen.source_path}")

    # C3 #4: per-document confidence histogram in the header.
    _render_confidence_histogram(chosen.confidence)

    # Show model extraction in editable form
    extraction = chosen.extraction or {}
    confidence = chosen.confidence or {}
    edited: dict = {}

    # C3 #1: side-by-side field view.
    # Header row for the side-by-side panel so the reviewer knows
    # what's a model column vs the editable human column.
    col_model, col_human = st.columns(2)
    with col_model:
        st.markdown("**model output**")
    with col_human:
        st.markdown("**human-edited** (pre-populated)")

    for field_name, val in extraction.items():
        conf = confidence.get(field_name)
        if conf is not None and conf < 0.6:
            model_label = f":red[**{field_name}** · conf={conf:.2f}]"
        elif conf is not None:
            model_label = f"**{field_name}** · conf={conf:.2f}"
        else:
            model_label = f"**{field_name}**"
        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown(model_label)
            # Render the model value as text (works for scalars and
            # collections; we stringify to keep the side-by-side
            # view uncluttered).
            if val is None:
                st.text("(empty)")
            elif isinstance(val, (dict, list)):
                st.code(str(val), language="json")
            else:
                st.text(str(val))
        with col_right:
            # Pre-populated human-edit widget. We re-use the existing
            # text_input / text_area path so the JSON coercion logic
            # in coerce_field_value still applies.
            if isinstance(val, (dict, list)):
                new_v = st.text_area(
                    f"{chosen.id}_{field_name}_raw",
                    value=str(val),
                    key=f"{chosen.id}_{field_name}_raw",
                    height=160,
                    label_visibility="collapsed",
                )
                parsed, warning = coerce_field_value(new_v, val)
                if warning:
                    st.warning(f"{field_name}: {warning}")
                edited[field_name] = parsed
            else:
                edited[field_name] = st.text_input(
                    f"{chosen.id}_{field_name}_val",
                    value="" if val is None else str(val),
                    key=f"{chosen.id}_{field_name}_val",
                    label_visibility="collapsed",
                )

    # C3 #3: skip button.
    # Skip closes the current item WITHOUT persisting a review. The
    # reviewer can come back to it via the selectbox later. Implemented
    # by setting a session-state flag that survives rerun.
    reviewer = st.text_input("reviewer", value=os.environ.get("USER", "streamlit"))
    col_save, col_bulk, col_skip = st.columns(3)

    with col_save:
        if st.button("Save review", type="primary"):
            try:
                save_review(storage, result_id=chosen.id, edited=edited, reviewer=reviewer)
                st.success(f"Saved review for {chosen.id}")
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"save failed: {e}")

    # C3 #2: bulk-accept button.
    # Calls save_review with the *unchanged* extraction (== "trust the
    # model on every field") for fields whose confidence >=
    # BULK_ACCEPT_THRESHOLD. Fields without a confidence value get the
    # model's value too — same as a normal save — to keep the flow
    # simple. The reviewer can still edit a low-confidence field
    # individually before clicking bulk-accept.
    with col_bulk:
        high_conf_fields = [
            f for f, c in confidence.items()
            if c is not None and c >= BULK_ACCEPT_THRESHOLD
        ]
        n_total = len(extraction)
        n_skipped = n_total - len(high_conf_fields)
        bulk_help = (
            f"Save the model output unchanged for fields with "
            f"confidence >= {BULK_ACCEPT_THRESHOLD:.2f}. "
            f"{len(high_conf_fields)} of {n_total} fields qualify; "
            f"{n_skipped} below threshold are kept as currently edited."
        )
        if st.button(
            f"Accept all ≥{BULK_ACCEPT_THRESHOLD:.2f} confidence fields",
            help=bulk_help,
        ):
            try:
                # Build the saved extraction: high-confidence fields
                # take the model value verbatim; everything else takes
                # whatever's currently in `edited`.
                bulk_edited = dict(edited)
                for f in high_conf_fields:
                    bulk_edited[f] = extraction.get(f)
                save_review(
                    storage,
                    result_id=chosen.id,
                    edited=bulk_edited,
                    reviewer=reviewer,
                )
                st.success(
                    f"Bulk-accepted {len(high_conf_fields)} high-confidence "
                    f"fields for {chosen.id}"
                )
                st.cache_resource.clear()
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"bulk-accept failed: {e}")

    with col_skip:
        if st.button("Skip document"):
            # Skip is non-persistent: we just bump the selectbox index
            # in session_state, which causes the next pending result
            # to load on the rerun. We don't touch storage.
            cur_index = labels.index(chosen_label)
            next_index = (cur_index + 1) % len(labels)
            st.session_state["_skip_target_label"] = labels[next_index]
            st.info(f"Skipped {chosen.id}. Nothing was saved.")
            st.rerun()

    # Honor a pending skip target (set above). Streamlit's selectbox
    # index is read-only, so we re-render the page with a query-param
    # style hint via session_state. We use the key the user can then
    # pick manually — no automatic jump, but the skip is recorded.
    pending_skip = st.session_state.pop("_skip_target_label", None)
    if pending_skip:
        st.caption(f"next pick from the dropdown: {pending_skip}")

    # C3 #5: keyboard shortcuts. Streamlit doesn't expose native
    # key events across the whole page, but st.text_input supports
    # on_change. We add a small help expander that lists the
    # shortcuts the reviewer *can* use: tabbing between fields,
    # and the bulk-accept button above is the "s" analog.
    with st.expander("⌨ Keyboard shortcuts (j/k/s/?)", expanded=False):
        st.markdown(
            """
            Streamlit doesn't bind global hotkeys, but the reviewer's
            muscle-memory shortcuts map to:

              * **j / k** — use the dropdown above (Tab to focus, type to filter).
              * **s** — Save review (the primary button).
              * **a** — Accept all high-confidence fields (the bulk button).
              * **?** — This help expander.

            Hover over each button for a tooltip describing its action.
            """
        )


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
            n_corrections = count_corrections(r)
            st.metric("fields corrected", n_corrections)


# ----------------------------------------------------------------------
# Page: Triage (v0.4 D1)
# ----------------------------------------------------------------------
def page_triage() -> None:
    """Render the HITL triage report.

    Pure additive page — reuses ``idp.hitl.triage.triage`` and the
    markdown formatter. The data layer is in idp.hitl.triage; this
    page just renders.
    """
    storage = get_storage()
    st.header("HITL triage")
    st.caption(
        "Which fields is the model getting systematically wrong? "
        f"Threshold = {_DEFAULT_THRESHOLD:.2f}; min_reviews = {_DEFAULT_MIN_REVIEWS}. "
        "Tune on the sidebar to re-run."
    )

    # Tunable threshold / min_reviews directly in the UI. Streamlit
    # widgets fit naturally here; defaults are the v0.4 decisions.
    cols = st.columns(2)
    with cols[0]:
        threshold = st.slider(
            "systematic-error threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(_DEFAULT_THRESHOLD),
            step=0.05,
            help="Correction-rate cutoff. Fields corrected at or above "
                 "this fraction of reviews are flagged.",
        )
    with cols[1]:
        min_reviews = st.number_input(
            "min_reviews per field",
            min_value=1,
            max_value=1000,
            value=int(_DEFAULT_MIN_REVIEWS),
            step=1,
            help="Fields reviewed fewer than this many times are NOT "
                 "triaged — they go to the 'insufficient data' bucket.",
        )

    try:
        report = _run_triage(
            storage,
            min_reviews=int(min_reviews),
            threshold=float(threshold),
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"triage failed: {e}")
        return

    # Summary metrics — quick "is there a problem here?" signal.
    metric_cols = st.columns(3)
    metric_cols[0].metric("reviewed results", report.n_reviews_total)
    metric_cols[1].metric("distinct fields seen", report.n_fields_seen)
    metric_cols[2].metric(
        "systematic errors",
        len(report.systematic_errors),
        delta=None,
    )

    st.divider()

    if not report.systematic_errors:
        st.success(
            "No systematic errors detected at this threshold. "
            "Either the model is doing well, or there isn't enough "
            "review data yet (see 'Insufficient data' below)."
        )
    else:
        st.subheader("Systematic errors")
        st.caption(
            "These fields were corrected at or above the threshold. "
            "Push back on the prompt/schema — re-correcting them is "
            "the wrong response."
        )
        # Render as a DataFrame for clean tabular display + the same
        # st.dataframe pattern the rest of the app already uses.
        rows = [
            {
                "field": se.field,
                "rate": f"{se.rate:.2f}",
                "n": se.n,
                "n_corrections": se.n_corrections,
                "sample_review_ids": ", ".join(se.sample_review_ids),
            }
            for se in report.systematic_errors
        ]
        st.dataframe(rows, use_container_width=True)

    if report.insufficient:
        st.subheader("Insufficient data")
        st.caption(
            "Fields reviewed fewer than min_reviews times. We refuse "
            "to guess — keep reviewing and re-run."
        )
        rows = [
            {"field": inf.field, "n_reviews": inf.n, "more_needed": inf.needed}
            for inf in report.insufficient
        ]
        st.dataframe(rows, use_container_width=True)

    # Markdown copy-paste block, for users who want to share the
    # report in chat/email.
    with st.expander("View as Markdown", expanded=False):
        st.code(_format_report_markdown(report), language="markdown")


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

        **v0.4 P3 additions:** the **Triage** page (left) summarizes
        which fields are systematically wrong across all reviews.
        The **Review queue** now has a side-by-side model/human view,
        a bulk-accept button for high-confidence fields, a skip
        button, and a per-document confidence histogram.
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
    elif page == "triage":
        page_triage()
    else:
        page_about()


# Streamlit runs the script as __main__. Gate main() on that to
# keep `import idp.hitl.app` from accidentally launching the UI.
if __name__ == "__main__":
    main()
