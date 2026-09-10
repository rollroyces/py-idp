# py-idp

> **面向 Python 的通用 AI 智能文档处理框架。**
> 六阶段流水线（parse → classify → extract → assess → validate → HITL）。
> 支持 12+ 种 LLM 后端。以 Pydantic Schema 为驱动。内置评测工具集。
> **自动分块处理超大文档。通过 Nanonets-OCR2-3B 实现自托管 OCR。AI 驱动的 Schema 自动发现。**

**语言:** [English](README.md) · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md)

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE-AGPL)
[![Commercial license available](https://img.shields.io/badge/license-commercial_available-orange.svg)](LICENSE-COMMERCIAL)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CI](https://github.com/rollroyces/py-idp/actions/workflows/tests.yml/badge.svg)](https://github.com/rollroyces/py-idp/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/py-idp)](https://pypi.org/project/py-idp/)
[![Downloads](https://img.shields.io/pypi/dm/py-idp)](https://pypistats.org/packages/py-idp)
[![GitHub stars](https://img.shields.io/github/stars/rollroyces/py-idp)](https://github.com/rollroyces/py-idp/stargazers)
[![Cite this repository](https://img.shields.io/badge/Cite-CITATION.cff-blue)](CITATION.cff)

---

## 安装

```bash
pip install py-idp                # 核心依赖（pydantic + typer + rich + httpx + pdfplumber）
pip install py-idp[docling]      # IBM Docling — 最佳 PDF 表格抽取
pip install py-idp[openai]       # OpenAI SDK（同时也用于 8 个中国 LLM）
pip install py-idp[anthropic]    # Anthropic SDK
pip install py-idp[api]           # FastAPI 服务器（idp.api:app，生产可用）
pip install py-idp[hf-vlm]        # 自托管 Nanonets-OCR2-3B（Apple Silicon / CUDA）
pip install py-idp[dev]          # pytest + ruff + mypy
```

> 安装与运行测试套件均无需 API Key——`MockBackend` 已内置。
> `tiktoken` 由核心包自动安装（用于基于 token 预算的分块）。

---

## 30 秒快速上手

```python
import idp
from idp.pipeline import Pipeline

result = Pipeline(
    backend="mock",                # 或 "ollama"、"openai"、"anthropic"、"china:qwen" 等
    schema="Invoice",
    business_rules=[...],
).run(idp.Document.from_path("invoice.pdf"))

print(result.extraction)    # dict —— 已根据你的 Pydantic Schema 验证
print(result.confidence)    # dict —— 每个字段 0..1，<0.6 将标记为待复核
print(result.validation)    # dict —— Schema 与业务规则校验结果
```

**在内置示例发票上的端到端实测：**

| 指标 | 值 |
|---|---|
| 文档分类 | `invoice`（置信度 0.99） |
| 抽取字段规模 | 12 个字段，2 个行项目 |
| 校验 | 通过 |
| 与 gold 的字段精确匹配率 | **7 / 9 = 78 %**（单文档） |
| 低置信度标记（HITL） | 2 个（subtotal、tax_amount——小模型算术错误） |
| 延迟（模拟 LLM） | < 2 ms |

**完整评测套件，3 张发票，本地真实 Ollama（`qwen2.5:0.5b`，397 MB）：**

| 指标 | 值 |
|---|---|
| Schema 合法率 | **100 %**（3 / 3） |
| 字段 F1 | **0.96**（精确率 1.00，召回率 0.93） |
| 延迟 | **2.05 s / 文档**（Apple Silicon） |
| 单文档精确匹配 | inv-001 7/9 · inv-002 9/9 · inv-003 9/9 |

框架对小型模型的弱点是诚实的：小模型在算术运算上（`subtotal` / `tax_amount`）的置信度为 0.10，会被路由到 HITL 复核，而不是静默放行。

---

## 流水线

```text
INGEST  →  PARSE  →  CLASSIFY  →  ROUTE  →  EXTRACT  →  ASSESS  →  VALIDATE  →  HITL
                                                                         (Streamlit)
```

每个阶段都是 `Document` 上的纯函数。它们独立运行、可单独单元测试、任何一个都可以替换。

| 阶段 | 模块 | 默认实现 | 功能 |
|---|---|---|---|
| **parse** | `idp.parse` | Docling（PDF）· pdfplumber（兜底）· 纯文本 | 抽取文本 + 表格 + 页面图像 |
| **classify** | `idp.classify` | 规则优先，LLM 兜底 | 识别文档类型：invoice、contract、bank_statement…… |
| **route** | `idp.parse.router` | 自动 | 根据文档特征选择多模态 VLM 或 OCR+LLM |
| **extract** | `idp.extract` | Pydantic Schema 驱动 | 从文本或图像中抽取并校验结构化数据 |
| **assess** | `idp.assess` | 启发式 + 可选 LLM 自评 | 每个字段的置信度 0..1 |
| **validate** | `idp.validate` | Pydantic + 用户谓词 | Schema 校验 + 业务规则 |
| **HITL** | `idp.hitl` | Streamlit UI | 复核低置信度字段，保存修正 |
| **pipeline** | `idp.pipeline.pipeline` | 编排器 | 组合上述阶段，返回 `PipelineResult` |

---

## LLM 后端

### 国际（5 个 provider，支持任意 OpenAI 兼容接口）

| 名称 | 说明 |
|---|---|
| `openai` | GPT-4o（视觉）、GPT-4.1、o1 |
| `anthropic` | Claude 3.5/4 Sonnet、Claude Haiku（视觉） |
| `ollama` | 本地 `llama3.2-vision`、`qwen2.5-vl` —— 默认 base URL `http://localhost:11434/v1` |
| `vllm` / `lm-studio` / `compat` | 任意 OpenAI 兼容的 chat-completions 接口 |
| `mock` | 离线 / CI 基线（`mock`、`mock-random`、`mock-omits`） |

```bash
export OPENAI_API_KEY=...
idp run invoice.pdf --schema Invoice --backend openai
```

### 中国（8 个 provider —— 均兼容 OpenAI Chat-Completions 协议）

运行 `idp providers` 可打印完整列表。要点：

| provider | 环境变量 | 默认模型 | 视觉模型 |
|---|---|---|---|
| `deepseek` | `DEEPSEEK_API_KEY` | deepseek-chat | —（纯文本） |
| `qwen` | `DASHSCOPE_API_KEY` | qwen-plus | qwen2.5-vl-72b-instruct |
| `zhipu` | `ZHIPUAI_API_KEY` | glm-4-plus | glm-4v-plus |
| `moonshot` | `MOONSHOT_API_KEY` | moonshot-v1-128k | moonshot-v1-128k-vision-preview |
| `yi` | `YI_API_KEY` | yi-large | yi-vision |
| `doubao` | `ARK_API_KEY` | doubao-pro-32k | doubao-1-5-vision-pro-32k |
| `hunyuan` | `HUNYUAN_API_KEY` | hunyuan-pro | hunyuan-vision |
| `baichuan` | `BAICHUAN_API_KEY` | baichuan4 | —（纯文本） |

```python
from idp.llm import get_china_backend

backend = get_china_backend("qwen", multimodal=True)
# backend.model == "qwen2.5-vl-72b-instruct"
```

### 自托管（Nanonets-OCR2-3B，Apple Silicon / CUDA）

适用于不能将文档发给第三方的场景。**首次下载后完全离线，无需 API Key，不出云**（约 7 GB 缓存在 `~/.cache/huggingface/hub/`）。

```bash
pip install py-idp[hf-vlm]    # 额外安装 torch + transformers + accelerate + safetensors
export IDP_ENABLE_NANONETS=1  # 显式启用（避免意外下载）
export IDP_BACKEND=nanonets
idp run scan.pdf --backend nanonets
```

**Apple M4 16 GB 显存预算**（float16，448×448 图像）：
- 权重 + 视觉编码器 + KV cache：约 8.3 GB
- 系统 + 应用：约 3.5 GB
- 余量：约 4 GB（充裕）

**速度**：M4 上每页约 5-15 秒。首次调用需 5-10 分钟下载模型；后续调用约 10 秒从缓存加载。

**为何选择 Nanonets-OCR2-3B**：开源权重、无需鉴权、Apache-2.0 许可（基于 Qwen2.5-VL）——商用前请核实 Nanonets 微调版本的许可。在噪声扫描和复杂排版文档上优于 Tesseract，并支持多语言。

**为何默认关闭**：模型下载体积大且耗时。py-idp 拒绝自动触发；你必须显式设置 `IDP_ENABLE_NANONETS=1`。

**平台支持**（构造时验证）：

| 平台 | 状态 |
|---|---|
| macOS arm64（M1/M2/M3/M4，16 GB 及以上） | ✅ 测试目标，启用 MPS |
| macOS arm64（8 GB） | ❌ 显存不足（改用 Docling） |
| macOS x86_64（Intel） | ❌ 无 MPS，eGPU CUDA 不稳定——会显式失败 |
| Linux x86_64 + CUDA | ✅ 最佳（每页 1-5 秒） |
| Linux x86_64 仅 CPU | ⚠️ 可用，每页 30-60 秒 |
| Linux arm64 | ⚠️ 可用，仅 CPU |
| Windows x86_64 + CUDA | ✅ 与 Linux CUDA 相同 |
| Windows arm64 | ❌ PyTorch 没有 Windows arm64 的 wheel——会显式失败 |

```python
from idp.llm.nanonets import NanonetsVLBackend
backend = NanonetsVLBackend(
    device="mps",          # 或 "cuda"、"cpu"、"auto"
    max_image_side=448,    # 显存占用仅为 1024 的 1/4，准确率约 95%
    load_in_4bit=False,    # 若 float16 显存溢出则设为 True
)

# 与 PdfPagesParser 端到端配合（将页面渲染为图像）
from idp import Document, Pipeline
from idp.parse.parser import parse_document
from idp.core.schemas import Invoice

doc = Document.from_path("scan.pdf")
parse_document(doc, parser="pdf-pages")   # 将页面渲染为 base64 PNG
result = Pipeline(backend=backend, schema=Invoice).run(doc)
print(result.document.extraction)
```

### 自动分块处理超大文档

Nanonets-OCR2-3B 的上下文窗口为 16k tokens。50 页的发票 PDF 无法一次塞进去。`extract()` 会自动检测这种情况，对输入进行切分，逐块调用模型，并合并各块抽取结果。**调用方无需任何额外代码**——对调用者完全透明。

内置两种分块器：

| 分块器 | 使用场景 | 默认配置 |
|---|---|---|
| `PageChunker` | 多模态后端（NanonetsVLBackend、GPT-4o 等） | 每块 4 页，1 页重叠 |
| `TokenChunker` | 文本抽取器（OCR + LLM） | 每块 4000 tokens，200 tokens 重叠（基于 tiktoken） |

```python
from idp.chunker import PageChunker, TokenChunker

# 在小内存 M 系列 Mac 上调紧预算
chunker = PageChunker(max_pages=2, overlap_pages=1)

# 或者直接传给 Pipeline
from idp.pipeline import Pipeline
pipe = Pipeline(backend=backend, schema=Invoice, chunker=chunker)

# 端到端：分块、调用、合并、校验，一次完成
result = pipe.run(Document.from_path("huge-50-page-scan.pdf"))
```

合并后的抽取结果中包含 `_chunk_count` 标记，便于按块统计成本与可观测性。

**逐块容错**：如果某一块的 LLM 调用失败，错误会被记录（`extract_chunk_failed[i]`），但其他块的数据仍会被合并。你将获得部分结果 + 清晰的错误日志，而不是整个崩溃。

详见 [`src/idp/chunker.py`](src/idp/chunker.py) 实现与 [`tests/test_chunker.py`](tests/test_chunker.py) 34 个测试。

---

## 命令行工具

```bash
idp run path/to/invoice.pdf --schema Invoice --backend ollama --output out.json
idp providers                                          # 完整 provider 表
idp schemas                                            # 内置 Pydantic Schema
idp discover-schema scan.pdf --hint "extract vendor_name, total_amount" --output schema.json
idp eval --dataset src/idp/eval/datasets/invoices \
          --strategy mock,mock-omits --output results.json
idp serve                                              # 启动 Streamlit HITL UI，端口 :8501
```

---

## 自定义 Schema

内置的 `Invoice`、`Contract`、`BankStatement` 只是参考示例——你可以传入任意 Pydantic 模型：

```python
from pydantic import BaseModel
from idp import Document
from idp.pipeline import Pipeline

class Receipt(BaseModel):
    merchant: str
    total: float
    currency: str
    date: str

result = Pipeline(backend="ollama", schema=Receipt).run(
    Document.from_path("receipt.jpg")
)
```

---

## Schema 自动发现

> **试一试：** `python -m examples.discover_schema_sample`
> 该示例运行 6 个端到端场景（生成一份真实的 2 页发票、发现 Schema、运行抽取、覆盖边界情况）。无需 API Key 或 poppler。

你手上有一份扫描 PDF，脑海中大概知道想要哪些字段（X、Y、Z）——但还没有 Pydantic 类。`discover_schema()` 会调用多模态 LLM（默认 NanonetsVLBackend）提议一个 JSON Schema，然后编译成一个 Pydantic 类，你可以直接传给 `Pipeline(schema=...)`。

```python
import idp

Schema, schema_dict = idp.discover_schema(
    "scan.pdf",
    hint="extract vendor_name, invoice_number, total_amount, and line items",
)
# Schema 是 Pydantic BaseModel 的子类——直接传入：
result = idp.Pipeline(backend="nanonets", schema=Schema).run(
    idp.Document.from_path("scan.pdf")
)
print(result.document.extraction)
```

返回的 `DiscoveryResult` 同时暴露编译后的 Pydantic 类与原始 JSON Schema 字典：

```python
result = idp.discover_schema("scan.pdf", hint="...")
result.schema_class    # Pydantic 类
result.json_schema     # 原始 JSON Schema 字典
result.raw_response    # 原始 LLM 输出（调试用）
result.backend_name    # "NanonetsVLBackend"
result.doc             # 解析后的 Document（可复用）
```

**对应的 CLI：**

```bash
export IDP_ENABLE_NANONETS=1
idp discover-schema scan.pdf \
    --hint "extract vendor_name, total_amount, and line items" \
    --output schema.json
```

**默认设置**：页面数上限 4（适配大多数 16k 上下文 VLM）、Nanonets 后端（需设置 `IDP_ENABLE_NANONETS=1`）、测试时回退到 Mock。

**提示词匹配度（Hint grounding）**：当你提供提示词时，`discover_schema()` 会从中提取候选字段名 token，并检查它们在发现的 Schema 中出现的比例（精确匹配 + SequenceMatcher 比例 >0.8 的模糊匹配）。结果存放在 `DiscoveryResult.hint_grounding` 字典中，包含 `hint_tokens`、`schema_fields`、`grounded`、`ungrounded` 和 `grounding_score`（0.0 表示提示词 token 全部未出现，1.0 表示完全匹配）。当得分低于 0.5 时，会打印警告，列出 LLM 忽略的 token。它并不能修正错误的字段名——但能让这种错误变得可观察，从而提醒你去做复核。

```python
result = idp.discover_schema("scan.pdf", hint="...")
if result.hint_grounding and result.hint_grounding["grounding_score"] < 0.5:
    print("LLM largely ignored your hint!")
    print("missing:", result.hint_grounding["ungrounded"])
```

**诚实的局限：**

- LLM 提议的字段名有时是错的——提示词能起引导作用，但不能保证正确。`hint_grounding` 字段让这一点可观察。生产前请用真实抽取结果对发现的 Schema 做验证。
- 字段类型从 JSON Schema 推断（string / number / integer / boolean / array / 嵌套 object）。必选/可选状态会被保留。
- LLM 有时会输出 ```json 围栏或在结果外加一段说明文字；解析器会自动剥离。纯垃圾响应会抛出 `ValueError`，错误信息包含前 200 字符以便调试。
- 这是 **Schema 发现**（告诉你有哪些字段、字段叫什么名字），不是 Schema **验证**（告诉你抽取得对不对）——请将发现的 Schema 传入 `Pipeline(schema=...)`，并在 HITL 复核中完成验证步骤。

详见 [`src/idp/discover.py`](src/idp/discover.py) 实现，[`tests/test_discover.py`](tests/test_discover.py) 45 个测试，以及 [`examples/discover_schema_sample.py`](examples/discover_schema_sample.py) 可运行的端到端示例。

---

## 超大文档分块

大多数 LLM 的上下文窗口限制为 6k-200k tokens。一份很长的发票、合同或多页扫描件可能超过这一限制。`py-idp` **自动检测超限输入，对其进行切分，逐块调用 LLM，并合并各块抽取结果**——所有这些无需任何额外胶水代码。

| 分块器 | 使用场景 | 默认配置 |
|---|---|---|
| `PageChunker` | 多模态后端（NanonetsVLBackend、GPT-4o 等） | 每块 4 页，1 页重叠 |
| `TokenChunker` | 文本抽取器（OCR + LLM） | 每块 4000 tokens，200 tokens 重叠（基于 tiktoken） |

**默认值针对最常见模型进行了调优：**
- 4 页 @ 200dpi ≈ 3000 图像 tokens → 适配 Nanonets-OCR2-3B（16k 上下文）
- 4000 文本 tokens → 适配 qwen2.5:0.5b（6k 上下文）和 llama3.2（8k）

**逐块容错**：如果某一块的 LLM 调用失败，错误会被记录（`extract_chunk_failed[i]`），但其他块的数据仍会被合并。部分结果优于完全没有结果。

```python
from idp.chunker import PageChunker

# 在小内存（统一内存 16 GB）的 M 系列 Mac 上调紧预算
chunker = PageChunker(max_pages=2, overlap_pages=1)
result = Pipeline(backend="nanonets", schema="Invoice", chunker=chunker).run(
    Document.from_path("huge-50-page-scan.pdf")
)
print(result.document.extraction.get("_chunk_count"))  # 约 25
```

合并后的抽取在合并完成后会进行**整 Schema 校验**，因此即便抽取过程分布在多个小块，最终你拿到的依然是 Pydantic 类型的强校验结果。

详见 [`src/idp/chunker.py`](src/idp/chunker.py) 实现与 [`tests/test_chunker.py`](tests/test_chunker.py) 34 个测试。

---

## 添加业务规则

```python
from idp.validate import required_fields_rule, numeric_range_rule

pipe = Pipeline(
    backend="ollama",
    schema="Invoice",
    business_rules=[
        required_fields_rule("invoice_number", "vendor_name", "total_amount"),
        numeric_range_rule("total_amount", min_v=0.0, max_v=10_000_000.0),
    ],
)
```

内置两条规则；你也可以自定义一个 `(dict) -> (bool, str | None)` 形式的谓词。规则抛出异常会被捕获——不会让整个流水线崩溃。

---

## 示例演示

在内置示例发票上的端到端实时运行（MockBackend —— 无需 API Key）：

![py-idp 在示例发票上的流水线](docs/assets/demo-pipeline.svg)

同一个抽取过程在 Streamlit HITL 复核 UI 中的呈现：

![Streamlit HITL 复核界面](docs/assets/demo-hitl.svg)

（上面的 SVG 为示意效果图。如需真实录屏，请运行 `idp run path/to/your-invoice.pdf --backend ollama` 和 `idp serve`。）

---

## 评测套件

诚实的抽取结果需要标注数据和多后端横向对比。`py-idp` 内置两者。

```bash
idp eval --dataset src/idp/eval/datasets/invoices \
         --strategy mock,mock-omits,ollama --output results.json
```

按后端输出：**Schema 合法率**、**字段级 F1**、**$/文档**、**延迟**。内置样本（3 张发票、2 份合同、5 张 CORD 风格小票）已经手工标注，你可以放心地发布你亲自验证过的数字。

### CORD 风格小票基准

`src/idp/eval/datasets/cord_subset/` 下有一个 5 张小票的手工精选子集，模仿 [CORD: Consolidated Receipt Dataset](https://github.com/clovaai/cord)。运行方式：

```bash
python examples/benchmark_cord.py
```

该脚本使用内置 MockBackend 对全部 5 张小票跑一遍，输出每字段的精确率 / 召回率 / F1，以及延迟。**无需 API Key**——任何人都能用 `pip install py-idp[eval]` 复现这些数字。如需对比真实后端，将 `examples/benchmark_cord.py` 中的 `"mock"` 替换为 `"ollama"` / `"openai"` / `"anthropic"` / `"china:qwen"`。

---

## 可靠性：重试、缓存、检查点

三个面向生产场景的可选功能。

### RetryingBackend —— 自动指数退避重试

为任意 `Backend` 包装指数退避重试，捕获瞬时错误（限流、超时、网络中断）。鉴权和参数错误直接抛出——重试无意义。

```python
from idp import Pipeline
from idp.reliability import RetryConfig

pipe = Pipeline(
    backend="openai",
    schema="Invoice",
    retry=RetryConfig(max_retries=5, initial_delay_sec=2.0, max_delay_sec=60.0),
)
```

默认值：4 次尝试，1s → 2s → 4s → 8s，带 ±20% 抖动，单次最大 30s。对于不可重试的错误（`AuthError`、`BadRequestError`），底层后端立即抛出。错误类型通过消息模式匹配识别，详见 `idp.reliability.classify_exception` 的分类树。

### ExtractionCache —— 基于磁盘的抽取缓存

相同输入直接命中，不再调用 LLM。缓存键为 `(schema_name, backend_name, request payload)` 的 sha256。每次命中都会按 schema 维度统计，便于可观测。

```python
from idp import Pipeline
from idp.reliability import ExtractionCache

pipe = Pipeline(
    backend="nanonets",
    schema="Invoice",
    cache=ExtractionCache("/dbfs/mnt/idp/extract.db"),  # 默认：~/.cache/idp/extract.db
)
```

默认位置 `~/.cache/idp/extract.db`，跨进程重启仍然保留。`cache.stats()` 返回条目数、总命中数、按 schema 维度的细分。

### CheckpointStore —— 批量断点续跑

对成百上千份文档运行 `process_batch()` 时，如果中途出现中断（服务器重启、网络抖动），重跑不会丢失已完成的工作。默认行为就是幂等的——重跑时传入相同的 checkpoint 路径即可：

```python
from idp.llm.nanonets_batch import process_batch

# 第一次运行：处理第 1-1000 份文档，在第 500 份处中断
results = process_batch(paths, pipeline, checkpoint="/dbfs/.../cp.jsonl")
# 第二次运行：第 1-499 份跳过（已记录），从第 500 份开始
results = process_batch(paths, pipeline, checkpoint="/dbfs/.../cp.jsonl")
```

设置 `archive_at_start=True` 可以在两次运行之间轮换 checkpoint 文件（每次运行一份独立文件，保留历史）。使用 `CheckpointStore.clear()` 强制重新处理。

`retry=True` 和 `cache=True` 可以叠加使用：`cache` 在 `retry` 之后生效，因此缓存命中完全跳过重试逻辑。

详见 [`src/idp/reliability.py`](src/idp/reliability.py)、
[`src/idp/checkpoint.py`](src/idp/checkpoint.py)，
以及 [`tests/test_reliability.py`](tests/test_reliability.py) /
[`tests/test_checkpoint.py`](tests/test_checkpoint.py) 的完整 API。

---

## 生产级脚手架（内置，按需启用）

| 关注点 | 自带实现 | 生产可替换为 |
|---|---|---|
| 异步任务队列 | `idp.queue.InProcessQueue` | ARQ / Celery / SQS |
| 持久化存储 | `idp.storage.JsonFileStorage` | Postgres + S3 |
| API Key 鉴权 | `idp.auth.keys` | 接入 FastAPI 依赖 |
| HTTP API | `idp.api:app`（生产级 FastAPI，含鉴权/限流/指标） | 自建服务 |
| HITL UI | `idp.hitl.app`（Streamlit） | React / FastAPI |
| Docker | `Dockerfile`、`docker-compose.yml` | 自有基础设施 |
| **基于 HITL 修正的 RL** | `idp.rl` + `idp rl-update` | 在线逐条更新（`PolicyCache`） |
| **文档分块** | `idp.chunker`（自动对超限输入启用） | 自定义 `PageChunker` / `TokenChunker` |
| **Schema 发现** | `idp.discover_schema` + `idp discover-schema` | 自定义多模态后端 |

### 0.3.x 不包含的功能（有意为之）

多租户隔离、SSO/SAML/RBAC、审计级存储——SaaS 需要，但单机自托管还为时尚早。如有需要请提 issue。

---

## 从 HITL 修正中学习（RL）

`idp.storage` 中每一次人工复核都会成为训练信号。框架自带**离线批量策略更新**，将"人工反复修正的字段"转化为这些字段的更高置信度下限 + 更低置信度惩罚——确保它们在下一次运行中稳定地进入 HITL 复核。

```bash
# 离线批量：从累积的复核中派生奖励，写入 policy.json
idp rl-update --storage idp_data/results.jsonl \
               --output policy.json

# 在流水线中应用策略：
result = Pipeline(
    backend="ollama",
    schema="Invoice",
    policy_path="policy.json",
).run(Document.from_path("invoice.pdf"))

# 如果暂时没有 storage，可以手工构造复核：
idp rl-update --reviews reviews.jsonl --output policy.json
```

**它是什么**：确定性、可审计、可版本化的规则更新。**不是**学习得到的奖励模型，**不是**微调后的 LLM。我们在学的是后置的置信度调整规则——决定哪些字段应当被送进 HITL——而不是模型本身。

**为什么这样做**：真实业务中 ROI 最高的就在这一层。用 RLHF / DPO 微调 LLM 能带来 ~2-3% 的 F1 提升，但需要数周时间；一个 7B 模型往往就能轻松超过这个提升。而**学会哪些字段更值得送进 HITL**会让每一次复核都产生复利。

**实测结果**（本地真实 Ollama，`qwen2.5:0.5b`，内置样本）：

| 字段 | 无策略 | 启用策略（经过 5 次人工修正后） | 变化 |
|---|---|---|---|
| `vendor_name` | 0.75（会通过 HITL） | **0.55**（现在会标记） | −0.20 |
| `subtotal` | 0.10（已经标记） | **0.0**（紧急） | −0.10 |
| `invoice_number` | 0.75 | 0.75（未覆盖） | 0.0 |

v0.2 起通过 `PolicyCache` 提供在线（逐条）更新；离线批量在当前版本完全可用。

### 校准评估 —— 策略是否真的兑现了承诺？

```bash
# 用 gold truth 合成复核，派生策略，再评估
idp rl-update --reviews reviews.jsonl --output policy.json
idp rl-eval   --policy policy.json --fixtures src/idp/eval/datasets/invoices \
              --injection-rate 0.30 --output calibration.json
```

报告**策略触发时的命中率**（标记后人工确实修正了吗？）和**策略沉默时的真接受率**（没标记的、人工确实接受了吗？），并明确给出 `n=` 和 `synthetic=true` 标记——合成复核天然偏乐观（gold truth 本身就是人工的修正），真实 HITL 数据会更嘈杂。

**实测结果**（基于 3 张内置发票的合成复核，`qwen2.5:0.5b` 本地 Ollama 真实运行，字段×文档 = 27 对）：

| 指标 | 值 | 含义 |
|---|---|---|
| 策略捕获（标记 → 人工修正） | **21** | 没有策略的话，这些错误都会逃过 HITL |
| 策略抑制（原本被标记，现在不再标记） | **0** | 无回归 |
| 两者都已标记 | 2 | 无变化 |
| 模型实际正确、未被标记 | 4 | 正确接受——模型判断正确 |

**诚实说明**：使用 `qwen2.5:0.5b` 时，基线置信度启发式本身就过于悲观，几乎所有错误都已经逃过了 HITL——所以策略看起来收益巨大。一个置信度校准更好的大模型收益会小得多。本次实测的样本量只有 27（字段×文档），请勿外推。

### 在线策略更新（逐条，进程内）

`PolicyCache` 监听 `storage.mark_reviewed()`，把每一条新的人工复核增量折入内存中的策略，刷新是带防抖的原子磁盘写入。下一条 `Pipeline.run()` 立刻看到更新后的覆盖——无需重启、无需单独的 CLI 调用。

```python
from idp.storage import make_storage
from idp.rl import PolicyCache

storage = make_storage("sql", db_url="sqlite:///./idp.db")
cache = PolicyCache(policy_path="policy.json", flush_interval_sec=1.0)
cache.attach_to_storage(storage)   # 改写 mark_reviewed 以触发 on_review

# 此后每一次人工复核都会在后台更新策略
```

**默认值**：`flush_interval_sec=1.0`（防抖窗口）、`min_reviews=10`（小样本保护——总观测数不足 10 的字段不会获得覆盖，因为 n<10 时失败率估计过于嘈杂）。

**多进程**：只应有一个进程持有 cache（比如 FastAPI 服务器）。其他进程（CLI 工具、Streamlit 复核 UI）从磁盘读取 `policy.json`。cache 使用 `os.replace` 进行原子写入，写入中途崩溃也会留下上一次完整的策略。

### 真实 HITL 数据采集

`SqlStorage` 后端持久化了 `JsonFileStorage` 的所有内容，并把每次字段级的修改历史存入真正的关系型数据库。SQLite 开箱即用（无需额外依赖）；Postgres 通过 `pip install py-idp[sql]` 启用。

```bash
# SQLite，单文件
export IDP_DB_URL="sqlite:///./idp.db"
idp serve                                  # Streamlit UI 现在读写这个 DB
idp rl-update --db-url "sqlite:///./idp.db" --output policy.json
idp rl-eval  --db-url "sqlite:///./idp.db" --policy policy.json \
             --output calibration.json
```

**Schema（4 张表）**：`reviewers`、`stored_results`（最新复核状态的非规范化缓存）、`reviews`（每次复核会话一行）、`review_edits`（每次字段级 diff 一行）。这种拆分让你可以计算复核员间一致性、字段随时间修改率、以及"策略标记了这个、人工也认为它确实错了"——而无需扫描整个结果 blob。

**为什么这种拆分很重要**：`review_edits` 是 RL 层真正消费的细粒度信号（每个被修改的字段一行）。没有它，你就无法知道在一次多字段复核里，人工到底改了哪个字段。

---

## 开发

```bash
git clone https://github.com/rollroyces/py-idp
cd py-idp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -v                       # 508 个测试，无需 API Key
ruff check src tests examples   # 代码风格检查
mypy src/idp                    # 类型检查（59 个源文件全部通过）

python -m examples.invoice      # 端到端演示（无需 API Key）
python -m examples.nanonets_ocr2  # NanonetsVLBackend 端到端（需要 IDP_ENABLE_NANONETS=1）
python -m examples.batch        # process_batch() 批量助手，用于 Databricks 类批量任务
python -m examples.discover_schema_sample  # AI 驱动的 Schema 发现（6 个场景，生成真实 PDF）
```

`import idp; idp.__version__` → `0.3.2`.

---

## 安全

发现安全漏洞？请查阅 [`docs/SECURITY.md`](docs/SECURITY.md) —— **请勿**以公开 issue 形式提交。

---

## 引用

如果 py-idp 帮到了你的研究或产品，学术引用见 [`CITATION.cff`](CITATION.cff)。GitHub 侧栏的 "Cite this repository" 按钮可一键导出 BibTeX。

---

## 贡献

欢迎提交 Issue、PR 和 Discussion。完整指南——包括如何新增 LLM 后端或 Schema、提交消息规范、发布流程——见 [`CONTRIBUTING.md`](CONTRIBUTING.md)。Bug 报告最好附上最小复现脚本和你的 `py-idp` 版本。CI 在每次 PR 上运行 ruff + mypy + 508 个测试，覆盖 Python 3.10 / 3.11 / 3.12。

---

## 许可协议

py-idp **双重许可**：

- **AGPL-3.0-or-later** —— 开源使用。你可以自由使用、修改、运行 py-idp。若以网络可访问服务的形式部署修改版本，同样必须按 AGPL 公开。这是防止竞争者将本项目克隆为闭源 SaaS 而不回馈社区的 copyleft 条款。详见 `LICENSE-AGPL`。
- **商业许可** —— 为那些需要在闭源产品或托管 SaaS 中嵌入 py-idp、且不希望受 AGPL copyleft 约束的组织而设。详见 `LICENSE-COMMERCIAL`。

这与 **MariaDB / Sentry / MinIO** 的模式一致：若希望将其运行在闭源产品中，请付费；若保持改动开源，则免费获得完整源码。

商业许可参考价格：

| 等级 | 用途 | 价格 |
|---|---|---|
| Solo | 单个开发者，单个法人实体 | **$300 / 年** |
| Team | 最多 10 名开发者，单一法人实体 | **$1,500 / 年** |
| Enterprise | 不限开发者数量 + SLA + 技术支持 | 联系商务 |
| SaaS-OEM | 嵌入托管 SaaS，按活跃用户数计费 | 按席位计费 |

请联系 **rollroyces** 获取签约。

---

## 致谢

- 流水线结构、HITL、置信度设计 —— 扩展自 [`aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws`](https://github.com/aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws)
- PDF 解析 / 表格抽取 —— 包装自 [IBM Docling](https://github.com/docling-project/docling)（arXiv 2408.09869）
- Pydantic Schema 驱动的抽取 API —— 灵感来自 [`run-llama/llama_cloud_services`](https://github.com/run-llama/llama_cloud_services)
- 多格式分块模式 —— 来自 [`Unstructured-IO/unstructured`](https://github.com/Unstructured-IO/unstructured)

如果研究中引用了 py-idp，请同时引用本仓库与 Docling。
