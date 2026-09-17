# Roadmap

> **Status (2026-09-17):** Tier A and Tier B are **done**. This roadmap
> is kept for historical context. The remaining items below (B1, C1-C4)
> were flagged TBD and still need scope input from you before I can
> size them. See "Decision log" at the bottom for what landed.

This is a **near-term (next 2-4 weeks)** roadmap. Mid-term direction
(versions 0.4 / 0.5) is intentionally not detailed here — those
choices depend on what users do with the current 0.3.x and what
breaks in production. We can write the 0.4 / 0.5 plan after this
near-term work lands and we have real feedback.

---

## Tier A — must-do, before publishing a 0.3.5 ✅ DONE in PR #20

These are **correctness, security, or data-loss risks** that are
already shipping in the 0.3.4 dist. They should land before any
`pip install py-idp` gets a meaningful number of users.

### A1. Fix `datetime.utcnow()` deprecation ✅ DONE (PR #20)
* Migrated to `datetime.now(timezone.utc)`. Zero deprecations from our code.

### A2. Close the remaining 0%-coverage CLI + queue + eval + migrate gaps ✅ DONE (PR #20)
* CLI: 0% → 97%
* queue: 0% → 96%
* eval: 0% → 96%
* migrate_audit: 0% → 76% (further to 94% in v0.3.7)

### A3. `hitl/app.py` data logic refactor ✅ DONE (PR #20)
* Pure module `hitl/review.py` extracted; 100% coverage.
* `hitl/app.py` itself: smoke-tested via Streamlit `AppTest` harness (48% in v0.3.7).

### A4. SQLite + Postgres parity tests ✅ DONE (PR #20 + v0.3.6)
* `tests/test_sql_dialect.py`: 23 tests covering URL parsing, dialect branch, schema bootstrap.

---

## Tier B — should-do, before publishing a 0.4 ✅ DONE in v0.3.7

Real production users will hit these gaps within weeks.

### B1. Auto-discovery of templates via sample docs (TBD)
* Still pending. Needs your scope.

### B2. CI: add `mmdc` Mermaid validation ✅ DONE (PRs #18, #20)
* `.github/workflows/validate-mermaid.yml` runs mmdc on every Mermaid block.
* Advisory due to GitHub Actions AppArmor sandbox limitations.

### B3. PyPI publish for 0.3.x ✅ BLOCKED ON YOU
* 0.3.6 and 0.3.7 dist artifacts built and `twine check --strict` passes.
* Upload is your move (security rule).

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
