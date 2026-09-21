# CORD eval — per-field baseline numbers

This page is the per-field precision / recall / F1 baseline for
py-idp's `Receipt` schema on the
[CORD: Consolidated Receipt Dataset](https://github.com/clovaai/cord)
shape (Park et al., 2019).

The numbers below were produced by the in-tree `MockBackend` — no API
key, no network. The MockBackend's purpose is to give a reproducible
**offline baseline**. It returns an empty JSON for every extraction
prompt, so its F1 is near zero by design. The value of this page is
not the absolute number; it is the **shape** of the row (which fields
are tracked, how to read the table) so a user running the same eval
against Ollama / OpenAI / Anthropic knows what to compare against.

To reproduce these numbers:

```bash
pytest tests/test_eval_cord.py -v
```

The script that regenerates `BASELINE.md` is:

```bash
python tests/eval_cord/update_baseline_doc.py
```

## How to run against your own backend

The slow real-backend test (`tests/eval_cord/test_eval_real_dataset.py`)
runs the eval against the backend you select via `IDP_EVAL_BACKEND`:

```bash
IDP_EVAL_BACKEND=ollama    pytest tests/eval_cord/test_eval_real_dataset.py -v
IDP_EVAL_BACKEND=openai    pytest tests/eval_cord/test_eval_real_dataset.py -v
IDP_EVAL_BACKEND=anthropic pytest tests/eval_cord/test_eval_real_dataset.py -v
```

The slow test is **skipped by default** (it is marked
`@pytest.mark.slow` and the default CI command runs
`pytest -m "not slow"`). To run it you must set
`IDP_EVAL_BACKEND` explicitly. The credentials expected:

| backend   | credential / endpoint |
|-----------|----------------|
| `ollama`  | `OLLAMA_HOST` env var, or local Ollama running on `localhost:11434` |
| `openai`  | `OPENAI_API_KEY` env var |
| `anthropic` | `ANTHROPIC_API_KEY` env var |

After you run the slow test, regenerate the baseline doc with your
new row:

```bash
python tests/eval_cord/update_baseline_doc.py --backend ollama --add-only
```

This appends a new row to `docs/eval/BASELINE.md` between the
`<!-- BASELINE-START -->` / `<!-- BASELINE-END -->` markers, leaving
the MockBackend row intact.

## Per-field precision / recall / F1 — MockBackend

The MockBackend returns an empty extraction for every doc, so the
field-by-field scores are all 0.0:

| field                 | precision | recall | F1   |
|-----------------------|-----------|--------|------|
| `merchant_name`       | 0.000     | 0.000  | 0.000 |
| `date`                | 0.000     | 0.000  | 0.000 |
| `time`                | 0.000     | 0.000  | 0.000 |
| `line_items`          | 0.000     | 0.000  | 0.000 |
| `subtotal`            | 0.000     | 0.000  | 0.000 |
| `tax_amount`          | 0.000     | 0.000  | 0.000 |
| `tip_amount`          | 0.000     | 0.000  | 0.000 |
| `total`               | 0.000     | 0.000  | 0.000 |
| `payment_method`      | 0.000     | 0.000  | 0.000 |
| `credit_card_last4`   | 0.000     | 0.000  | 0.000 |
| **micro-avg**         | 0.000     | 0.000  | 0.000 |
| `schema_valid_rate`   | --        | --     | 0.000 |
| `n_docs`              | --        | --     | 30    |

Schema-valid-rate is also 0 because the empty extraction fails the
`Receipt` schema (the `merchant_name` field is required).

## What a real backend looks like

We have not run a real backend in CI at v0.4 cut. The pattern a user
should expect on a competent real backend (Claude Sonnet, GPT-4o,
Qwen2.5-VL, etc.) on the same 30 fixtures is approximately:

- **F1 ≈ 0.55–0.85** micro-aggregated across all 10 fields
- `merchant_name` is usually correct (a short distinctive string at
  the top of the receipt)
- `date` / `time` / `total` / `subtotal` / `tax_amount` are usually
  correct (they are the most-labeled fields in CORD)
- `tip_amount` is the hardest field — many receipts omit it, and the
  model has to distinguish "tip was 0" from "tip field absent"
- `credit_card_last4` is a 4-digit string; it is the noisiest
  because OCR often mangles digits (`0` ↔ `O`, `1` ↔ `l`)

Those numbers are illustrative; the slow test is the source of truth.

## Why the MockBackend baseline is zero

The MockBackend lives at `src/idp/llm/backend.py` and its
`mode="ideal"` returns `_empty_schema(schema)` — i.e. an empty JSON
object `{}` matching the requested Pydantic schema's shape. The
Pydantic `Receipt` schema rejects that empty object because
`merchant_name` is required. The eval runner then records zero
matches on every field and zero schema-valid extractions.

This is the documented behaviour and is on purpose:

* It is **deterministic** — running the eval twice gives bit-identical
  numbers, so it's a CI-stable baseline.
* It costs **zero** — no API key, no network, no GPU.
* It is a **lower bound** — any real backend that doesn't beat 0.0
  F1 is misconfigured (wrong model name, bad API key, etc.) and the
  slow test will surface that with a warning, not a silent pass.

If a future PR changes the MockBackend's behaviour (e.g. parses the
prompt for known fields), the F1 in `BASELINE.md` will go up — and
that is the right time to *raise* the offline-baseline threshold in
the slow test. Today: 0.0.

## Edge cases exercised by the 30 fixtures

The fixtures deliberately cover every shape the `Receipt` schema can
take:

| shape                              | fixtures |
|------------------------------------|---|
| cafe / quick-service restaurant    | 001, 002, 011, 013, 018, 021, 029 |
| restaurant with tip                | 003, 007, 014, 030 |
| pharmacy / grocery / convenience   | 004, 008, 009, 023 |
| specialty retail (books / wine / electronics / etc.) | 005, 012, 016, 019, 022 |
| auto / gas / parking / car wash    | 006, 010, 017, 028 |
| hotel / movie / ice cream          | 020, 026, 027 |
| dry cleaner / smoothie / hardware  | 015, 024, 025 |
| **zero `line_items`** (parking, gas) | 006, 010 |
| **OCR-noisy** (`O→0`, `l→1`)        | 008 |
| **`Cash` payment**, no card last-4  | 008, 018, 021, 027, 029 |
| **`quantity > 1`** line items      | 002, 007, 009, 011, 013, 014, 015, 024, 025, 026, 029, 030 |

This breadth is why the CORD subset is the v0.4 eval (rather than the
`cord_subset/` smoke data which only has 5 docs).

## v0.4 scope boundaries

This PR deliberately does **not** ship:

- **Per-template baselines.** The number above is the aggregate
  over all 30 fixtures. A breakdown by merchant category
  (e.g. "F1 on restaurants vs gas stations") is v0.5 work.
- **Automated regression gating** (`fail CI if F1 drops > 5%`). The
  threshold number depends on real CORD results we don't have at
  v0.4 cut, and is v0.5.
- **The original ~1000 CORD receipts.** We ship 30 synthetic
  fixtures, scoped to the 1-day v0.4 budget. The `manifest.json`
  format is forward-compatible with a mechanical converter for the
  real CORD release.

See `docs/ROADMAP_v0.4.md` (item C2) for the full v0.4 spec and the
v0.5 follow-ups.