# Contributing to py-idp

Thanks for taking an interest. The bar is intentionally low: PRs that
land cleanly get merged quickly.

## TL;DR

```bash
git clone https://github.com/rollroyces/py-idp
cd py-idp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,eval]"
pytest -v && ruff check src tests examples && mypy src/idp
```

If all three are green, push and open a PR. CI runs the same matrix
on Linux / Python 3.10 / 3.11 / 3.12.

## Project layout

```
src/idp/
  core/         # Document, PipelineResult, base protocols
  parse/        # PDF / image / text parsers + router
  classify/     # rule-first doc-type classification
  extract/      # schema-driven LLM extraction
  assess/       # heuristic + LLM confidence
  validate/     # Pydantic + user-rule validation
  hitl/         # Streamlit review UI
  pipeline/     # orchestrator
  llm/          # backends (openai, anthropic, ollama, china:*, nanonets)
  storage/      # JSON / SQLite / in-memory HITL stores
  rl/           # policy update from review feedback
  errors.py metrics.py config.py ratelimit.py api.py _logging.py
```

## Adding a new LLM backend

1. Subclass `idp.llm.backend.Backend` and implement `complete()`.
2. If the provider is OpenAI-compatible, subclass `OpenAICompatibleBackend`
   in `idp.llm.compat` — saves you 50 lines of HTTP plumbing.
3. Register the name in `get_backend()` (`idp/llm/backend.py`).
4. Add the env-var prefix to the `Settings` model in `idp.config`.
5. Tests: extend `tests/test_backends.py` with a mocked test that
   doesn't require live credentials.

## Adding a new schema

Drop a Pydantic `BaseModel` subclass in `src/idp/core/schemas.py`,
list it in `BUILTIN_SCHEMAS`, and the CLI auto-picks it up via
`idp schemas`.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(extract): add field-level retry budget per backend
fix(storage): skip corrupt lines instead of crashing the read
docs(readme): cross-link discover_schema_sample
perf(hitl): lazy-import streamlit to avoid 20 MB import cost
```

The first line stays under 72 chars; wrap the body at 100. CI doesn't
lint commit messages but the release tool reads them.

## Code style

- **ruff** for lint + import order (`ruff check src tests examples`).
- **mypy --strict-ish** (`mypy src/idp/`) — keep public functions typed.
- **pytest** for tests. Prefer small unit tests; one integration test
  per backend is fine if it's mocked at the HTTP boundary.
* **Hypothesis** for property tests on parsing / extraction paths.

## Stats

458 tests, 58 mypy-checked modules, ruff-clean across `src/ tests/
examples/`. Coverage is highest on the validator, parser, and store
modules (see `tests/` for per-module coverage badges in the docstrings).

## What NOT to PR

- Whitespace-only changes, mass reformatting, dependency bumps without
  a motivating issue. Open an issue first.
- New dependencies in `core` — extras only (`[docling]`, `[openai]`,
  `[anthropic]`, `[hf-vlm]`, `[api]`, `[eval]`).
- Renames of public API symbols — we are pre-1.0, but renaming
  `Document` / `Pipeline` / `Backend` would break every downstream
  user. Open an issue first.

## Releasing

Only maintainers cut releases. The flow:

```bash
git tag v0.X.Y -m "v0.X.Y — <one-line summary>"
git push --tags
gh release create v0.X.Y --notes-file CHANGELOG.md
# → .github/workflows/publish.yml picks up the tag, builds, uploads to PyPI
```

The PyPI trusted-publishing workflow is already wired up
(`.github/workflows/publish.yml`).
