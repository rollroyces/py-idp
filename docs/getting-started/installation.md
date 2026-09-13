# Installation

`py-idp` ships a small core package and a list of optional extras. Pick what you need.

## Core

```bash
pip install py-idp
```

This installs the pipeline, parsers, validators, HITL store, eval harness, and CLI. It depends on `pydantic`, `typer`, `rich`, `httpx`, `pdfplumber`, and `tiktoken`.

No API key is required to run the in-tree examples, the test suite, or the eval harness — the `MockBackend` is built in.

## Optional extras

| extra | installs | use when |
|---|---|---|
| `docling` | `docling>=2.0` | you want IBM Docling for the strongest PDF table extraction (heavy: ~500 MB transitive deps) |
| `ocr` | `pytesseract`, `Pillow` | noisy scans where you want a tesseract fallback |
| `openai` | `openai>=1.30` | you want to call OpenAI GPT-4o directly (also works for OpenAI-compatible gateways, including most China LLM providers) |
| `anthropic` | `anthropic>=0.27` | you want to call Anthropic Claude |
| `ollama` | `ollama>=0.2` | you want to call a locally-running Ollama instance |
| `hf-vlm` | `torch`, `transformers`, `accelerate`, `safetensors`, `Pillow`, `huggingface-hub` | you want to run the self-hosted Nanonets-OCR2-3B backend (heavy: ~3 GB pip install, ~7 GB model weights) |
| `api` | `fastapi`, `uvicorn` | you want to run `idp.api:app` as a FastAPI service |
| `china` | `openai>=1.30` | you want to call DeepSeek / Qwen / Zhipu / Moonshot / Yi / Doubao / Hunyuan / Baichuan through the OpenAI-compatible protocol |
| `pdf-render` | `pdf2image`, `Pillow` | the multimodal backend needs to render PDF pages to images (also requires `poppler-utils` at runtime) |
| `eval` | `datasets`, `pandas` | you want to run `idp eval` over Hugging Face datasets |
| `dev` | `pytest`, `pytest-benchmark`, `hypothesis`, `httpx`, `ruff`, `mypy`, `types-PyYAML`, `reportlab` | you want to run the test suite locally |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings`, `pymdown-extensions` | you want to build this site locally (`mkdocs serve`) |

Combine extras: `pip install py-idp[docling,anthropic,eval,dev]`.

## Verifying the install

```bash
python -c "import idp; print(idp.__version__)"      # should print 0.3.1
idp providers                                        # prints the full backend table
idp schemas                                          # prints the built-in Pydantic schemas
pytest -q                                            # runs the full suite (508 tests)
```

If `idp` isn't on your `PATH` after install, run `python -m idp.pipeline.cli` instead.

## Self-hosted OCR (Nanonets-OCR2-3B)

If you installed `[hf-vlm]`, you also need to enable the backend explicitly:

```bash
pip install py-idp[hf-vlm]
export IDP_ENABLE_NANONETS=1
export IDP_BACKEND=nanonets
idp run scan.pdf --backend nanonets
```

The first call downloads the model (~7 GB) into `~/.cache/huggingface/hub/`. Subsequent calls run fully offline.

Apple Silicon memory budget (M4 16 GB, float16, 448×448 images): ~8.3 GB for weights + visual encoder + KV cache, leaving ~4 GB headroom for the system.