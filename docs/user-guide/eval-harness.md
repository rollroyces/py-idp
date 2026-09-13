# Eval harness

Honest extraction claims need labeled data and side-by-side backend comparison. `py-idp` ships both.

## Running an eval

```bash
idp eval --dataset src/idp/eval/datasets/invoices \
         --strategy mock,mock-omits,ollama --output results.json
```

Reports per-strategy: **schema-valid rate**, **field-level F1**, **$/doc**, **latency**. The in-tree fixtures (3 invoices, 2 contracts, 5 CORD-style receipts) are hand-labeled so you can publish numbers you actually verified.

## CORD-style receipt benchmark

A 5-receipt hand-curated subset modeled on the [CORD: Consolidated Receipt Dataset](https://github.com/clovaai/cord) lives at `src/idp/eval/datasets/cord_subset/`. Run it with:

```bash
python examples/benchmark_cord.py
```

This runs the in-tree MockBackend against all 5 receipts and prints per-field precision / recall / F1 plus latency. **No API key needed** — the numbers are reproducible by anyone with `pip install py-idp[eval]`. To benchmark a real backend, swap `"mock"` for `"ollama"` / `"openai"` / `"anthropic"` / `"china:qwen"` in `examples/benchmark_cord.py`.

## Adding your own dataset

A dataset is a directory with:

- `cases.jsonl` — one JSON object per line with `{"doc_path": "relative/path.txt", "schema": "SchemaName", "gold": {...}}`.
- `docs/` — the actual documents, paths relative to `cases.jsonl`.

Drop the directory under `src/idp/eval/datasets/` and `idp eval --dataset your_dir` will pick it up.