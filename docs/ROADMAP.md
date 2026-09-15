# Roadmap

> **Status: draft, awaiting your review.** Items here are derived from the
> recent audit and a sweep of low-coverage / unhandled code paths. Items
> marked **(concrete)** are clearly actionable; items marked **(TBD)**
> need your input before I can size them.

This is a **near-term (next 2-4 weeks)** roadmap. Mid-term direction
(versions 0.4 / 0.5) is intentionally not detailed here — those
choices depend on what users do with the current 0.3.x and what
breaks in production. We can write the 0.4 / 0.5 plan after this
near-term work lands and we have real feedback.

---

## Tier A — must-do, before publishing a 0.3.5

These are **correctness, security, or data-loss risks** that are
already shipping in the 0.3.4 dist. They should land before any
`pip install py-idp` gets a meaningful number of users.

### A1. Fix `datetime.utcnow()` deprecation (concrete)

**Where**: `src/idp/storage/sql.py:497` — called 16 times in the
deprecation warnings we already saw in CI.

**Why it matters**: `datetime.utcnow()` is deprecated in Python 3.12+
and will be removed. The code still works, but every test run prints
16 deprecation warnings and any Python 3.13+ user will eventually hit
a `DeprecationWarning` → `PendingDeprecationWarning` chain.

**Fix**: `datetime.now(timezone.utc).replace(tzinfo=None)` (or keep
tz-aware throughout). ~5 lines, plus 1 test that exercises a
storage roundtrip and asserts no warnings.

**Effort**: 30 min · 1 PR · 1 test.

### A2. Close the remaining 0%-coverage CLI + queue + eval + migrate gaps (concrete)

**Where**: We went from 0% → 83% / 97% / 96% / 76% on those 4 files in
PR #14. The remaining 17% / 3% / 4% / 24% is mostly error paths.

**Why it matters**: Every uncovered line is a place where a real user
hits a bug and we have no test signal. The CLI 17% gap, in particular,
includes the `discover-schema` command's *non-`-help`* code path,
which the current tests don't exercise. A bad `discover-schema` would
silently produce wrong schemas.

**Fix**: Add ~10-15 more tests, focused on:
- CLI: discover-schema with a real mock backend, with various
  failure modes (no hint, file not found, backend error)
- migrate_audit: the SQL path (currently only SQLite tested)
- queue: the `worker tick failed` exception handler

**Effort**: 2-3 hours · 1 PR · 10-15 tests · coverage should hit >90%
on all four files.

### A3. `hitl/app.py` data logic refactor (concrete, but bigger)

**Where**: `src/idp/hitl/app.py` is 126 lines, 0% covered, and
imports Streamlit at module load (already fixed once, see git log
for `lru_cache` swap). The actual *data* logic — review queueing,
field diffing, storage writes — is intertwined with the UI.

**Why it matters**: HITL is the only path for human correction
feedback. If a future Streamlit upgrade or design change breaks it,
we have no tests. Worse, the RL feature *depends* on the review data
shape being correct, so a bug here silently corrupts the policy.

**Fix**: Extract the data-handling functions into a pure module
(`idp/hitl/review.py`), keep the Streamlit app thin (just renders
+ dispatches), add tests against the pure module. ~150 lines of
refactor + ~15 tests.

**Effort**: 3-4 hours · 1 PR · 15 tests · 0% → ~85% coverage on hitl.

### A4. SQLite + Postgres parity tests (concrete)

**Where**: `src/idp/storage/sql.py` is 82% covered, with 46 uncovered
lines. Most of those are Postgres-specific SQL (different syntax for
`INSERT ... ON CONFLICT`, `RETURNING`, `SERIAL` vs `AUTOINCREMENT`).
The current `psycopg` import is inside a `try/except ImportError`, so
Postgres is not actually tested anywhere in CI.

**Why it matters**: The README claims "Postgres via `pip install
py-idp[sql]`". That's only true if the Postgres code path works.
Today it's "untested code that may or may not work."

**Fix**: Add a test that runs against `testing.postgresql` or
`pytest-postgresql` if available. If we don't want a Postgres
dependency in CI, at minimum add a SQL-dialect-branch test that
asserts the right SQL is generated for each dialect (using
`_strip_postgres_only` as the oracle).

**Effort**: 2-3 hours · 1 PR · 5-8 tests.

---

## Tier B — should-do, before publishing a 0.4

Real production users will hit these gaps within weeks.

### B1. Auto-discovery of templates via sample docs (TBD)

**The idea**: User uploads 3-5 sample PDFs. The system proposes a
template (Markdown body + JSON Schema) by clustering fields across
the examples. This is the natural extension of the existing
`discover_schema` (single-doc → schema) to
`discover_templates` (multi-doc → reusable template).

**Why it matters**: This was the "design C" I flagged in the audit
turn as the actual endgame for non-technical users. It would also
let py-idp auto-build the `templates/*.md` files from a customer's
historical PDFs.

**Prerequisites**: I need to know the rough budget. Is this a
"weekend hack" or a "two-week feature"? It depends on how good the
field clustering needs to be.

**Effort**: TBD · needs sizing from you.

### B2. CI: add `mmdc` Mermaid validation (concrete)

**Where**: Add to `.githooks/pre-commit` and `.github/workflows/tests.yml`
a step that runs `mmdc` on every Mermaid block in the 3 READMEs.

**Why it matters**: We just shipped a Mermaid syntax bug (PR #18) that
only got caught by a human reading the rendered diagram on GitHub.
This exact class of bug will recur with more Mermaid blocks added.

**Fix**: A small bash script that:
1. Extracts every `mermaid` block from each README
2. Pipes each to `mmdc -i - -o /dev/null` (mmdc reads from stdin)
3. Exits non-zero on parse failure

**Effort**: 1 hour · 1 PR · no new tests (CI step).

### B3. PyPI publish for 0.3.5 (concrete)

**Why it matters**: 0.3.4 has been the published version for a while.
Every fix in this roadmap is pointless if no one can `pip install`
the result.

**Status**: blocked on you running `twine upload dist/*` in your
shell (security rule, not something I can do). The dist artifacts
are built and `twine check --strict` passes.

**Effort**: 30 seconds on your end.

---

## Tier C — nice-to-have, in 0.4 or 0.5

### C1. Example notebooks (concrete)

The current `examples/` directory has 7-8 scripts but no Jupyter
notebooks. For an audience discovering py-idp, a notebook walking
through "load PDF → extract → confidence → HITL → corrected
extraction" would be more discoverable than scripts.

**Effort**: 2-3 hours · 3-5 notebooks.

### C2. Real-world eval datasets (TBD)

`src/idp/eval/datasets/invoices/` has 3 hand-crafted docs.
`cord_subset/` has 5 receipts. Both are useful for testing but
small for benchmarking. A real benchmark would be 50-200 docs
across 3-5 document types.

**Needs**: agreement on license + source. CUAD, CORD (already
mentioned in code), InvoiceNet, or build our own from Kaggle
"document-ai" data.

**Effort**: TBD · depends on dataset choice.

### C3. Streamlit HITL UX polish (TBD)

The HITL app is functional but not pretty. Real users will want:
- Side-by-side field view (model output vs human edit)
- Bulk-accept for fields the model got right
- A "skip this document" button

**Needs**: a designer or a user who can articulate what they want.

**Effort**: TBD.

### C4. Rerun-with-new-template-version migration tool (TBD)

When you bump a template from v1 to v2, old extractions are
"orphan" — extracted under the old rules. The CHANGELOG and the
README reference a "re-run against new template" workflow but
no such tool exists.

**Effort**: TBD · 1-2 days once the workflow is specified.

---

## Items I will NOT work on (and why)

- **Pydantic v3 migration**: Pydantic v3 is pre-release. v2 works
  fine. Migrating now = busy-work for unstable upstream.
- **HIPAA / SOC2 / FedRAMP compliance**: These are organizational
  certifications, not code changes. Need a security team + auditor.
- **A SaaS hosted version of py-idp**: Explicitly out of scope per
  the project license model (AGPL + commercial carve-out). Build
  one if/when the commercial channel materializes.
- **A from-scratch rewrite in Rust / Go / TypeScript**: PyPI
  is the delivery target. Python is fine.
- **Multi-tenant auth (SSO / SAML / RBAC)**: Per the README's
  "Not in 0.3.x" section. Explicit non-goal.

---

## Decision log

- **2026-09-15**: This roadmap written, draft state.
- (TBD after review): adopt as the canonical roadmap, link from
  README, surface in release notes
