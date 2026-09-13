# Backends

Every backend speaks the same protocol: given a Pydantic schema and a document, return a JSON-shaped extraction. Switching backends is a single argument.

## Listing backends

```bash
idp providers    # prints the full table
```

## Calling from Python

```python
from idp.llm import get_backend

backend = get_backend("anthropic")
# or: "openai", "ollama", "nanonets", "china:qwen", "china:deepseek", ...
```

`get_backend("auto")` resolves in this order: `anthropic` > `openai` > `ollama` (if `OLLAMA_HOST` is set) > `mock` (last resort, with an INFO log line).

## International providers

| provider | env var | default model | multimodal |
|---|---|---|---|
| `anthropic` | `ANTHROPIC_API_KEY` | claude-3-5-sonnet-latest | yes |
| `openai` | `OPENAI_API_KEY` | gpt-4o-mini | yes |
| `ollama` | (none — uses `OLLAMA_HOST`) | whatever you have running | depends on the model |
| `nanonets` | (none — local model) | Nanonets-OCR2-3B | yes (it's a VLM) |

## China LLM providers (8, OpenAI-compatible)

All eight speak the OpenAI Chat-Completions protocol, so the API-key-and-HTTP path is identical — only the env var name and the default model differ.

```python
from idp.llm import get_china_backend

backend = get_china_backend("qwen", multimodal=True)
# backend.model == "qwen2.5-vl-72b-instruct"
```

| provider | env var | default model | multimodal model |
|---|---|---|---|
| `deepseek` | `DEEPSEEK_API_KEY` | deepseek-chat | — (text only) |
| `qwen` | `DASHSCOPE_API_KEY` | qwen-plus | qwen2.5-vl-72b-instruct |
| `zhipu` | `ZHIPUAI_API_KEY` | glm-4-plus | glm-4v-plus |
| `moonshot` | `MOONSHOT_API_KEY` | moonshot-v1-128k | moonshot-v1-128k-vision-preview |
| `yi` | `YI_API_KEY` | yi-large | yi-vision |
| `doubao` | `ARK_API_KEY` | doubao-pro-32k | doubao-1-5-vision-pro-32k |
| `hunyuan` | `HUNYUAN_API_KEY` | hunyuan-pro | hunyuan-vision |
| `baichuan` | `BAICHUAN_API_KEY` | baichuan4 | — (text only) |

## MockBackend

`get_backend("mock")` returns a deterministic backend that produces empty defaults. Used for:

- CI runs without API keys (the full test suite runs against MockBackend).
- Showing the pipeline structure to a new contributor.
- Reproducing a known-bad extraction to debug a schema.

The mock backend is **not** a real LLM and shouldn't be confused with one — fields come back as defaults, the confidence is uniform 0.10, and the schema doesn't validate.

## Self-hosted OCR (Nanonets-OCR2-3B)

A standalone, fully offline alternative for sensitive documents. No API key, no network. Heavy install: ~3 GB pip packages + ~7 GB model weights.

```bash
pip install py-idp[hf-vlm]
export IDP_ENABLE_NANONETS=1
export IDP_BACKEND=nanonets
```

Speed: ~5–15 seconds per page on Apple M4 (CPU+GPU mixed, float16). First call downloads the model; subsequent calls use the `~/.cache/huggingface/hub/` cache.