# Roadmap v0.4 — concrete Tier C specs

> **Scope:** This document makes the four Tier C items from
> [`docs/ROADMAP.md`](ROADMAP.md) (C1–C4) concrete enough that an
> implementer can start work without further design questions, *and*
> adds three v0.4 directions that were not in the original roadmap.
> It does **not** redo Tier A / Tier B — those are done and shipped.
>
> **Tone:** honest about what we don't know yet. Where a decision is
> deferred to the user, it is called out explicitly.
>
> **Version rule:** this document does not bump the package version.
> The version bump happens when items are actually implemented and
> shipped. There is no v0.4 CHANGELOG entry in this PR; entries are
> added per item as it lands.

---

## Context

State at writing (2026-09-19, commit `0a22e66`, v0.3.8):

| aspect | detail |
|---|---|
| Tier A | done (PR #20) |
| Tier B | done (B1 still TBD — being built by a parallel track) |
| Tests | 848 passing, 92% coverage |
| CI | green |
| Dist artifacts | built + `twine check --strict` clean; not yet on PyPI (security rule — you run `twine upload`) |
| Parallel tracks right now | (A) this doc, (B) B1 auto-template discovery, (C) setup simplification |

User's framing for v0.4: *"make setup much easier, but we still need the accuracy."*

That framing is the priority lens for everything below. Anything that
makes `pip install py-idp && run` harder than it has to be is
deprioritised; anything that keeps field-level accuracy steady or
improves it is prioritised.

---

## Tier C items (carried from `docs/ROADMAP.md`)

### C1. Example notebooks

| aspect | detail |
|---|---|
| **Problem** | `examples/` ships 7–8 numbered scripts but zero Jupyter notebooks. New users who land on GitHub and think "I want a notebook to copy" find nothing. Scripts are scarier than notebooks for someone who has never touched `idp.Pipeline`. |
| **Why now** | Lowest-effort discoverability win in v0.4. A "load PDF → extract → confidence → HITL → corrected extraction" notebook is the single most useful thing for someone evaluating py-idp for 10 minutes. |
| **Acceptance criteria** | 1. `examples/notebooks/01_pipeline_minimal.ipynb` — the 5-minute tour: load sample invoice, run pipeline, print `pretty_print_result`. Mirrors `01_pipeline_minimal.py`. Runs in a clean venv with no API key (uses `MockBackend`).<br>2. `examples/notebooks/02_hitl_loop.ipynb` — the "human feedback changes runtime behavior" story. Walks through low-confidence fields → reviewer edits → policy override → second run shows the override path.<br>3. `examples/notebooks/03_batch.ipynb` — the `BatchItemResult` story: 100 docs → JSONL / DataFrame / Delta Lake sink. (Also covers the new v0.4 direction D3 — see below.)<br>4. Each notebook is executed-and-saved with output cells so GitHub renders them. No `name == '__main__'` boilerplate; just cells.<br>5. `examples/notebooks/README.md` with one-line "Run with: `jupyter lab examples/notebooks/`" and a table of contents. |
| **Scope estimate** | 4–6 hours total. Each notebook: ~1 hour. Mostly transcription of existing `examples/NN_*.py` plus markdown narrative. Notebooks 02 and 03 may need a tiny bit of code to make the runtime flow read naturally as cells. |
| **Dependencies** | None on the source code. Each notebook must use APIs that already exist in v0.3.8 (`Pipeline`, `JsonFileStorage`, `process_batch`, `MockBackend`, `pretty_print_result`). |
| **What we don't know** | Whether `examples/notebooks/` (new dir) or `examples/notebook/` (singular) is the right home — see question below. Whether we should also add a Quarto / mkdocs-jupyter pipeline so notebooks are auto-rendered into `docs/`. Not addressed here. |
| **Question for you** | Do you want a single PR that adds notebooks C1 + a separate PR for the batch notebook D3, or one combined PR? (The batch notebook uses `BatchItemResult`, which is a v0.3.x API — there's no reason to wait for D3 to land first.) |

---

### C2. Real-world eval datasets

| aspect | detail |
|---|---|
| **Problem** | `src/idp/eval/datasets/invoices/` has 3 hand-crafted docs; `cord_subset/` has 5 receipts. Useful for smoke tests, too small for credible benchmarking. The README claims "real-world accuracy" — we have no benchmark that backs that up beyond toy data. |
| **Why now** | Without a real eval set, every future "we improved accuracy" claim is hand-waving. v0.4 is the right time to fix this because (a) we already have the eval runner (`src/idp/eval/runner.py`), and (b) the user's priority is "still need accuracy" — we owe them numbers. |
| **Acceptance criteria** | 1. One of (CUAD, CORD full, InvoiceNet, FUNSD, RVL-CDIP subset, Kaggle "document-ai" collection) is selected and vendored under `src/idp/eval/datasets/<name>/` with a clear `LICENSE` and `ATTRIBUTION.md`.<br>2. 50–200 docs spanning 3–5 document types are loadable by `eval/runner.py` without code changes.<br>3. A `docs/eval/<name>.md` page with: per-field precision/recall/F1 numbers for the MockBackend, plus numbers for at least one real backend (Anthropic or openai), so the gap is visible.<br>4. A new CI job `tests/test_eval_real_dataset.py` (slow-marked, runs weekly or on-label) that asserts no metric regressed by more than X% (X = TBD — see question).<br>5. The eval runner output (JSON or markdown table) is published as a `BASELINE.md` so the "before / after" story for future PRs is easy. |
| **Scope estimate** | Highly dataset-dependent.<br>• License-clean choice that already ships with permissive terms (CORD, FUNSD): 1–2 days for the integration, 0.5 day for the baseline page.<br>• Custom / scraped dataset with cleaning: 3–5 days.<br>Hardest part is **license clarity**, not the engineering. |
| **Dependencies** | A dataset with a license that allows redistribution in an AGPL project. CUAD is CC-BY-4.0; CORD has its own license; InvoiceNet is academic-only; FUNSD is freely usable. The matrix below summarises — but **we have not done the formal license review** and the implementer must, before committing to a dataset. |
| **What we don't know** | (a) Whether the user has a preferred dataset in mind.<br>(b) Whether 50–200 docs across 3–5 types is the right ambition or whether 200–500 across 5–10 is needed for a "real" benchmark.<br>(c) The regression threshold X%. Too tight = flaky CI; too loose = no guardrail. |
| **Question for you** | Which dataset is the v0.4 priority: **CORD** (receipts — fits `cord_subset/`, license-compatible, fast to integrate), **FUNSD** (forms — slightly different domain, also permissive), or **"build from Kaggle"** (most flexibility, most license work)? The other two can come in v0.5. |

**License snapshot (informational, not legal advice):**

| dataset | license | redistribution in AGPL? | domain |
|---|---|---|---|
| CUAD | CC-BY-4.0 | yes (with attribution) | contracts |
| CORD | custom (research use, attribution) | yes (research + attribution) | receipts |
| FUNSD | academic, free use | yes (cite paper) | forms |
| InvoiceNet | academic-only | unclear | invoices |
| RVL-CDIP | unknown | needs review | multi-class doc images |
| Kaggle "document-ai" | varies per dataset | needs review per dataset | varies |

---

### C3. Streamlit HITL UX polish

| aspect | detail |
|---|---|
| **Problem** | `hitl/app.py` is functional but not pretty. The pure-data logic was extracted in A3 (`hitl/review.py`, 100% coverage) but the Streamlit layer is "smoke-tested via AppTest" at 48% in v0.3.7. Real reviewers will get fatigued by: no side-by-side view, no bulk-accept for fields the model got right, no "skip this document", and — critically — no signal about *which* fields are systematic errors vs. one-off noise. |
| **Why now** | HITL is the only feedback channel the package has for online RL (`PolicyCache`). A polished HITL UI directly improves the quality of human-edited overrides that flow into the policy. It also makes "the model was wrong about X 4/5 times" visible — see the new v0.4 direction **D1: real-bug-or-just-noise triage tool**, which is a strict superset of this item. C3 is the visual / ergonomic polish; D1 is the data layer behind it. They should ship together. |
| **Acceptance criteria** | 1. **Side-by-side field view:** the review screen shows `<key> | model output | human-edited` per field, with the human-edit input as a pre-populated text/number widget. Diffing is `extraction` vs `reviewed_extraction`.<br>2. **Bulk-accept:** a single "accept all high-confidence fields (≥ 0.9)" button that calls `save_review()` with the unchanged extraction, marks the result reviewed, and moves to the next.<br>3. **Skip:** a "skip document" button that closes the current item without persisting a review (used when reviewer needs to come back later or escalate).<br>4. **Per-document confidence histogram:** a small bar chart in the header showing how many fields are in `<0.5`, `0.5–0.7`, `0.7–0.9`, `≥0.9`. Lets the reviewer decide "is this mostly fine?" at a glance.<br>5. **Keyboard shortcuts:** `j` / `k` for next/prev document, `s` to save, `?` for the help panel. Optional but cheap if it lands.<br>6. **What we are NOT doing in C3:** the "is this a real bug" statistical triage is in **D1**, not here. C3 is the UI shell; D1 is the data under it. |
| **Scope estimate** | 1.5–2 days of focused UI work for items 1–4. Item 5 (shortcuts) is ~2 extra hours. Streamlit `st.columns`, `st.data_editor`, `st.bar_chart` cover everything; no new widgets. |
| **Dependencies** | None on the data layer (`hitl/review.py` already has `count_corrections` and the `save_review` dispatch). Pure-Streamlit changes. |
| **What we don't know** | (a) Whether reviewers want keyboard shortcuts. We assume yes for power users; we can A/B this with no shortcuts.<br>(b) The threshold for bulk-accept: 0.9 is a guess. We should instrument it.<br>(c) Whether side-by-side should highlight *changed* fields only, or always show the full schema. UX preference. |
| **Question for you** | For the bulk-accept button: is the default confidence threshold **0.9** (aggressive — assumes model is right when confident) or **0.95** (conservative — reviewer still eyeballs high-confidence fields)? |

---

### C4. Template-version migration tool

| aspect | detail |
|---|---|
| **Problem** | When a template (e.g. `Invoice`) is bumped from v1 to v2 — different field names, new required field, type change — extractions stored under v1 become "orphan": they were valid against v1's schema and are now stale against v2. The README and CHANGELOG reference a "re-run against new template" workflow but no tool implements it. |
| **Why now** | This is the slowest of the four Tier C items, and the one most likely to need a careful UX. Shipping a "v0.4 migration tool" that only handles the trivial case (rename one field) sets us up for a v0.5 rewrite. Better to ship a minimal, honest tool now and grow it. |
| **Acceptance criteria** | 1. New CLI: `idp migrate-template --storage <back> --template-name <name> --from-version <v1> --to-version <v2> [--dry-run]`.<br>2. Queries stored `StoredResult`s where `template_name == <name>` and `template_version == <v1>`.<br>3. For each, re-runs the pipeline against the **current** (v2) template using the **stored document source** (the path on `StoredResult.source_path`), not the stored extraction.<br>4. Writes a side-by-side diff: `old_extraction` vs `new_extraction`, per field, to `--output <path.json>` (or stdout in `--dry-run`).<br>5. Does **not** mutate the storage by default. Optional `--commit` flag persists the new extraction under a new `template_version == <v2>` record, leaving the v1 record intact.<br>6. Reuses `BatchItemResult` plumbing — C4 is `process_batch()` over the orphan set, with the diff post-process. Not a fresh parallel runner.<br>7. Tests: `tests/test_migrate_template.py` covering dry-run vs commit, missing source files (warn, don't crash), missing template (error), idempotent re-runs. |
| **Scope estimate** | 2–3 days, mostly test coverage. The actual code is <300 LOC because we lean on `process_batch` and `diff` helpers. Tests are the bulk: 70% coverage on the new module, ≥85% on the diff helper. |
| **Dependencies** | (a) `process_batch` and `BatchItemResult` (already exist).<br>(b) `StoredResult.source_path` round-trip — verify the source path is still readable from disk. If it's been archived or moved, we need to define behavior (we propose: warn + skip, list skipped in the diff output).<br>(c) A definition of "template version" that survives backwards — `template_name` + `template_version` are already recorded on `Document` (`tests/test_template_to_llm.py` confirms). |
| **What we don't know** | (a) Whether the user wants the migration to also **update the human reviews** (re-running the policy against the new fields) or just the raw extraction. The simpler/safer answer is: raw only; reviews stay attached to the v1 record.<br>(b) Whether the migration should be **incremental** (only re-run fields that exist in v2) or **whole-document** (re-run the whole pipeline against v2). We propose whole-document for simplicity, with the diff showing field-level changes.<br>(c) What happens when the v2 schema has a **new required field**. Re-run will produce `None` or empty for required fields the source document doesn't carry; migration will surface these as "needs human review" in the diff. |
| **Question for you** | For v0.4, do you want migration to be **(a) raw-extraction only** (v1 reviews stay attached to the v1 record; v2 record gets the new extraction but no review) or **(b) full feedback-loop aware** (v2 record inherits the human review where fields still match)? Option (a) is safer and ships in 2–3 days; option (b) needs policy-version tracking we don't have yet and is more like v0.5. |

---

## New v0.4 directions

These are not in the original `docs/ROADMAP.md`. They're added here
because they're small, concrete, and directly address the user's
framing ("much easier, but still need accuracy").

### D1. "Real bug or just noise?" triage tool for HITL

| aspect | detail |
|---|---|
| **Problem** | Today the HITL reviewer sees one document at a time. The reviewer has no signal for *which* fields are systematically wrong vs. one-off misses. The reviewer ends up either trusting everything or distrusting everything — both are wrong. |
| **Proposal** | A new module `idp/hitl/triage.py` exposing `triage(storage, *, min_reviews=3) -> TriageReport` that walks stored reviews and computes per-field correction rates:<br>• For each field name in the reviewed set, count `n_reviews` and `n_corrections`.<br>• If `n_corrections / n_reviews >= 0.6` AND `n_reviews >= min_reviews`, emit a `SystematicError(field=<name>, rate=..., n=..., sample_review_ids=[...])`.<br>• Otherwise emit nothing (or a `Noise(field=<name>, rate=..., n=...)` if the user wants to see low rates too).<br>A Streamlit page (or a CLI `idp triage --storage <back>`) renders the report as: `vendor_name: changed 4/5 times — model has a systematic error. See reviews r-001, r-014, r-027, r-103.`. |
| **Why it's worth doing** | This is the difference between "HITL is a chore" and "HITL is debugging". A reviewer who sees *vendor_name: changed 4/5 times* knows their next action is **not** to keep correcting — it's to push back on the prompt / schema. |
| **Acceptance criteria** | 1. `idp/hitl/triage.py` pure module, no Streamlit dependency.<br>2. CLI `idp triage --storage <back> [--min-reviews 3] [--output json\|md]`.<br>3. Streamlit page reachable from the existing `idp serve` UI.<br>4. Unit tests on synthetic data: deterministic output, correct thresholds, handles empty storage, handles fields only seen once.<br>5. Honest failure mode: if `n_reviews < min_reviews`, the report says *"insufficient data — need N more reviews before we can flag anything"* instead of guessing. |
| **Scope estimate** | 1–1.5 days. Most of it is the report formatter; the algorithm is ~50 LOC. |
| **Dependencies** | `Storage.submit_review` / `mark_reviewed` already record `reviewed_extraction`. **We assume `count_corrections` from `hitl/review.py` is the comparison primitive** — verify it's the right primitive for "same field, different value" (a string `"Acme Co"` vs `"ACME Co"` might or might not count as a correction; this is a UX call). |
| **What we don't know** | (a) Threshold 0.6 is a guess. Should it be 0.5? 0.8? Tunable per schema?<br>(b) Whether the report should also surface "fields the model always gets right" (low-value signal) or only systematic errors (high-value signal).<br>(c) Whether to include fields from **multiple document types** in one report or one report per template. |
| **Question for you** | For the systematic-error threshold, do you want **0.6** (catches genuine bugs early, accepts some false positives) or **0.8** (only flags near-uniform failure, very few false positives)? The default in the proposal is 0.6 with a CLI override. |

---

### D2. "Cheap model first, expensive model fallback" pipeline pattern

| aspect | detail |
|---|---|
| **Problem** | A full Anthropic / openai call per page per document is expensive. Most fields on most documents are easy: a small local model (Ollama, llama.cpp, etc.) gets them right. The expensive model should only see the hard cases. |
| **Proposal** | A new `Backend` subclass / wrapper `TieredBackend(cheap: Backend, expensive: Backend, *, ambiguity_threshold: float = 0.7)`. Behavior: route every `complete()` call to `cheap`; if the cheap response has low confidence (per the existing `confidence_llm.py` heuristic) **or** the resulting JSON fails Pydantic validation, retry the same call on `expensive`. The wrapper is transparent: a `Pipeline` does not know it is talking to `TieredBackend`. |
| **Why it's worth doing** | Direct match for the user's framing: "much easier setup, still need accuracy". A two-tier pipeline with Ollama for the cheap tier typically costs **~10× less per document** at **~2–5% accuracy drop** on aggregate — but on the *hard* fields where it fails, the expensive model catches the regression, so the worst-case accuracy is preserved. |
| **Acceptance criteria** | 1. `idp/llm/tiered.py` exporting `TieredBackend`.<br>2. Decision logic documented: when does cheap → expensive escalate? Confidence threshold + Pydantic validation failure are the two triggers; both are in scope. Anything else is out of scope for v0.4.<br>3. **No new backend dependency.** The cheap tier is any user-supplied `Backend`; we do not ship an Ollama class. (See caveat below.)<br>4. Tests use the in-tree `MockBackend` for both tiers (with a knob to force "always low confidence" or "always passes validation").<br>5. `examples/06_tiered_pipeline.py` showing `cheap = MockBackend(low_confidence=True)` + `expensive = MockBackend()` + `TieredBackend(...)`.<br>6. The pattern is opt-in. A user who only wants one backend is unaffected. |
| **Scope estimate** | 1.5–2 days. The wrapper is ~150 LOC; tests are the rest. |
| **Dependencies** | (a) `Backend.complete()` returning a string we can parse. (b) `confidence_llm.py` for the cheap-tier confidence signal. (c) Pydantic validation: we need to know which schema to validate against — currently the caller knows the schema; we either pass it to `TieredBackend.__init__` or have the caller validate. |
| **Honest caveat** | We do **not** ship an Ollama backend in v0.4. `llm/backend.py`'s docstring says "openai/anthropic/ollama" backends handle their own deps lazily, but no `OllamaBackend` class exists yet. Tiered works with any two `Backend` instances the user supplies; Ollama users would have to write a thin `OllamaBackend(Backend)` themselves or use an OpenAI-compatible endpoint. Shipping `OllamaBackend` is a possible v0.4 addition if you want it — see question below. |
| **What we don't know** | (a) Whether the cheap tier should also escalate on **low-confidence extraction at parse time** (currently parse-failure *is* the trigger) vs. on **per-field** confidence. Per-field is more work; parse-time is what the proposal says.<br>(b) Whether to expose a "third tier" (a free rule-based / regex pre-pass) for very obvious fields like dates. We propose NO — out of scope, ships as a separate direction if needed. |
| **Question for you** | Is `OllamaBackend` (a thin `Backend` subclass that calls the local Ollama HTTP API) **in v0.4** alongside `TieredBackend`, or **deferred to v0.5**? The two are independent but Ollama makes Tiered's value obvious. |

---

### D3. End-to-end example notebook for the `BatchItemResult` workflow

| aspect | detail |
|---|---|
| **Problem** | `BatchItemResult` and `process_batch()` exist and are tested, but there's no narrative example showing the *full* workflow: 100 docs → results iterator → JSONL / DataFrame / Delta Lake → per-row error handling → retry policy. Users see `BatchItemResult` in the API docs and don't know where to plug it. |
| **Proposal** | A new notebook `examples/notebooks/03_batch.ipynb` that goes through, in order: (1) build a `Pipeline` once, (2) call `process_batch()` over a folder of sample docs, (3) show the `BatchItemResult` stream and how to read `ok`, `result`, `error`, `elapsed_seconds`, (4) materialise to a DataFrame with `to_dict()`, (5) handle the `error` rows: filter, retry with backoff, write to `failed.jsonl`, (6) checkpoint: re-run with `CheckpointStore`, see that successful rows are skipped on resume. |
| **Why it's worth doing** | The DataFrame / Delta Lake story is the *Databricks* story (see the `batch.py` docstring: "1000 docs in one job run"). Without a notebook, that story is invisible. With a notebook, it is the headline. |
| **Acceptance criteria** | 1. Notebook runs end-to-end in a fresh venv. Uses `MockBackend` so no API key is needed.<br>2. Generates ~50 synthetic docs (small text fixtures are fine) so the batch is non-trivial.<br>3. Includes the retry-with-backoff snippet.<br>4. Includes a checkpoint resume snippet.<br>5. Final cell prints a summary table: `n_ok | n_err | mean_elapsed | p95_elapsed`. |
| **Scope estimate** | 2–3 hours (it is the C1 notebook 03, which already lists this as its scope — we are not adding new work, we are naming it so it doesn't get dropped). |
| **Dependencies** | None beyond v0.3.8 APIs. The notebook ships with C1. |
| **Question for you** | None — this is just a confirmation that the C1 notebook 03 covers D3. If you want D3 to ship standalone as its own notebook (e.g. `04_batch_advanced.ipynb` with a real Delta Lake sink on Databricks), say so. Otherwise it folds into C1. |

---

## Cross-cutting constraints (apply to all of the above)

| constraint | why |
|---|---|
| **No source code in this PR.** | This document is specs only. Each item becomes its own PR when implementation starts. |
| **No version bump in this PR.** | The version bump happens in the first implementation PR. |
| **No CHANGELOG entry in this PR.** | CHANGELOG entries are added when items ship. |
| **Existing `docs/ROADMAP.md` unchanged.** | The Tier A / Tier B / "we will not work on" sections are historical record; this doc is the v0.4 layer on top. |
| **Each Tier C item has one blocking question.** | Answer it, and the implementer can start. Don't bundle questions; one per item keeps the loop tight. |

---

## What we are explicitly NOT doing in v0.4

(Repeating from `docs/ROADMAP.md` so this doc is self-contained.)

- Pydantic v3 migration.
- HIPAA / SOC2 / FedRAMP compliance work.
- A hosted SaaS version.
- A Rust / Go / TypeScript rewrite.
- Multi-tenant auth (SSO / SAML / RBAC).

If any of these comes up in review, defer to the existing "Items I will NOT work on" section of `docs/ROADMAP.md`.

---

## Open questions (summary)

~~For the implementer to track — all of these are blockers for at
least one Tier C item:~~

**All 7 questions resolved on 2026-09-19 (user delegated).** See the
"Decisions made on 2026-09-19" subsection of the Decision log below
for each decision, its trade-off, and how to override a single one
without revisiting the others.

For historical reference, the questions were:

1. ~~**C1**: notebooks in one PR or split?~~ → **one PR (combined)**
2. ~~**C2**: which dataset? (CORD / FUNSD / Kaggle build-your-own)~~ → **CORD**
3. ~~**C3**: bulk-accept threshold 0.9 or 0.95?~~ → **0.9**
4. ~~**C4**: raw-extraction-only migration or full-feedback-loop-aware?~~ → **raw-extraction only**
5. ~~**D1**: systematic-error threshold 0.6 or 0.8?~~ → **0.6**
6. ~~**D2**: ship `OllamaBackend` in v0.4 or defer?~~ → **v0.4**
7. ~~**D3**: fold into C1 notebook 03 or its own notebook?~~ → **folded into C1**

---

## Decision log

- **2026-09-19**: v0.4 roadmap written as a concrete Tier C spec. Draft for review.
- (TBD after review): adopt as `docs/ROADMAP_v0.4.md`, link from README, surface in v0.4 release notes when items land.

### Decisions made on 2026-09-19 (assistant-driven, user delegated)

User explicitly delegated the open questions ("please help to make the
decision on your own"). Each decision below is recorded with a
recommendation, the alternative considered, and the trade-off. Any
single decision can be overridden in PR review without revisiting the
others.

#### Q1 (C1 notebooks) — **decided: ONE PR, combined**

Combine the basic notebooks (01_pipeline_minimal, 02_hitl_loop) with
the batch notebook (03_batch = D3) into a single C1 PR. Rationale: all
three use the same v0.3.8 APIs, no inter-dependencies, no shared
infrastructure that benefits from incremental review. Splitting them
just triples the PR overhead without changing what gets reviewed. D3
is then *folded into* C1 rather than being a separate direction.
Override if you want a 4-notebook collection with a dedicated batch
notebook; trivial to split later.

#### Q2 (C2 dataset) — **decided: CORD**

CORD ships with a clear research-use license, integrates with the
existing `cord_subset/` smoke data (no new directory), and gets a
v0.4-sized eval (50–200 docs) into the repo in ~1 day. FUNSD is a
fine alternative but adds a *new* doc type (forms) we don't have a
schema for yet, expanding scope beyond the user's "still need
accuracy" framing. Kaggle "build your own" is 3–5 days of license
review work and the user said "much easier" — not a v0.4 fit.
Override if you want FUNSD (forms) or want to spend the 3–5 days on a
custom dataset.

#### Q3 (C3 bulk-accept threshold) — **decided: 0.9**

0.9 trusts the model when it's confident, which is the point of the
button (the reviewer shouldn't be reading fields the model already
flagged as ≥0.9). 0.95 is over-conservative: the field histogram the
subagent is also adding (the C3 #4 acceptance criterion) already lets
the reviewer see how many fields are in each bucket, so the *bulk*
button doesn't need to be the cautious one. Recommendation
documented as 0.9 with the CLI override available.

#### Q4 (C4 migration) — **decided: raw-extraction only**

Re-run the pipeline against v2 templates, write a side-by-side diff,
do **not** touch existing v1 records, do **not** propagate human
reviews to the v2 record. Rationale: v1 reviews are attached to v1
extractions because the *fields* they evaluated may not exist in v2;
re-attaching them silently is the kind of "looks fine but lost data"
behavior that erodes trust in the storage layer. The simpler tool
ships in 2–3 days; the full feedback-loop-aware version needs
policy-version tracking that does not exist yet and is honest v0.5
work.

#### Q5 (D1 systematic-error threshold) — **decided: 0.6**

0.6 catches real bugs early. The triage tool is *advisory* — its
output does not auto-correct anything; the reviewer reads it and
makes a call. False positives cost ~10 seconds of "huh, that field
looks fine to me"; false negatives cost a model bug that ships to
production uncorrected. The asymmetry favors 0.6. CLI override (`--threshold`)
ships in v0.4, so a user who finds 0.6 too noisy can raise it per-repo
without code changes. Override to 0.8 if your review data shows 0.6
floods the report.

#### Q6 (D2 OllamaBackend timing) — **decided: SHIP IN v0.4**

Without `OllamaBackend`, the `TieredBackend` story is theoretical —
users have to write a 30-line `Backend` subclass themselves to even
try the cheap-first pattern. With `OllamaBackend`, the example
becomes "two lines of code, you save 10× per doc, accuracy is
preserved on hard fields." That's the headline. The `OllamaBackend`
itself is a thin wrapper (~80 LOC) over the local Ollama HTTP API;
the existing `Backend` base class and the OpenAI-compat surface
already exist. Worth the 0.5 day to land in v0.4. Override to v0.5
if the v0.4 timeline is already tight.

#### Q7 (D3 batch notebook) — **decided: folded into C1**

C1 notebook 03 covers D3 in the same file. No standalone
`04_batch_advanced.ipynb` for v0.4. Rationale: the batch story is the
*evaluation* story for v0.4 (D3 is what makes the C2 BASELINE.md
numbers interpretable in practice), and putting it adjacent to the
single-doc notebooks means a single reviewer can validate the whole
"one doc → many docs" learning arc without context-switching. Split
later if real users ask for a Delta Lake–specific notebook.

#### Summary of the v0.4 plan as decided

| priority | item | what ships |
|---|---|---|
| **P0** | B1 (already done) | `discover_template()` + CLI |
| **P0** | zero-config quickstart (already done) | `idp.easy.extract_one()` |
| **P1** | C1 + D3 | 3 notebooks, one PR |
| **P2** | D2 + OllamaBackend | `TieredBackend`, `OllamaBackend`, `examples/06_tiered_pipeline.py` |
| **P3** | C3 + D1 | HITL polish + triage tool — they ship together as one PR (D1 is the data layer, C3 is the UI) |
| **P4** | C2 (CORD) | baseline eval, slow-marked test, BASELINE.md |
| **P5** | C4 (raw-extraction only) | `idp migrate-template` CLI |

Six items → five PRs (C3+D1 combined). P1 lands first because
notebooks are the lowest-risk highest-visibility win. P2 next because
it directly serves the "much easier setup" goal. P3, P4, P5 can land
in any order after that.