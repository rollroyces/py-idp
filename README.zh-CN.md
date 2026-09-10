# py-idp（中文）

> **面向 Python 的通用 AI 智能文档处理（IDP）框架。**
> 六阶段流水线（解析 → 分类 → 抽取 → 评估 → 校验 → 人机协同）。
> 支持 12+ 大语言模型后端。Pydantic schema 驱动。自带评测工具。
> **超大文档自动分块。基于 Nanonets-OCR2-3B 的自托管 OCR。AI 驱动的 schema 自动发现。**

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE-AGPL)
[![Commercial license available](https://img.shields.io/badge/license-commercial_available-orange.svg)](LICENSE-COMMERCIAL)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CI](https://github.com/rollroyces/py-idp/actions/workflows/tests.yml/badge.svg)](https://github.com/rollroyces/py-idp/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/py-idp)](https://pypi.org/project/py-idp/)

---

## 安装

```bash
pip install py-idp                # 核心（pydantic + typer + rich + httpx + pdfplumber）
pip install py-idp[docling]      # IBM Docling —— 最强的 PDF 表格抽取
pip install py-idp[openai]       # OpenAI SDK（同时用于 8 家中国大模型）
pip install py-idp[anthropic]    # Anthropic SDK
pip install py-idp[api]           # FastAPI 服务（idp.api:app —— 生产级）
pip install py-idp[hf-vlm]        # 自托管 Nanonets-OCR2-3B（Apple Silicon / CUDA）
pip install py-idp[china]         # 中国大模型（OpenAI 兼容协议，无额外依赖）
pip install py-idp[dev]          # pytest + ruff + mypy
```

> 安装与运行测试套件无需任何 API key —— `MockBackend` 已内置。
> `tiktoken` 由核心包自动安装（用于按 token 数切块的 chunker）。

---

## 30 秒上手

```python
import idp
from idp.pipeline import Pipeline

result = Pipeline(
    backend="mock",                # 或 "ollama"、"openai"、"anthropic"、"china:qwen"……
    schema="Invoice",
).run(idp.Document.from_path("invoice.pdf"))

print(result.extraction)    # 字典，按 Pydantic schema 校验
print(result.confidence)    # 逐字段 0..1，低于 0.6 的字段标记为待复核
print(result.validation)    # schema + 业务规则校验结果
```

### 真实端到端运行（in-tree 示例发票，本地实测）：

| 指标 | 数值 |
|---|---|
| 分类 | `invoice`（置信度 0.99） |
| 抽取结构 | 12 个字段，2 个行项目 |
| 校验 | 通过 |
| 字段精确匹配 vs 金标准 | **9 / 9 = 100 %**（单文档） |
| 低置信度标记（HITL） | 0 |

### 完整评测，3 张发票，本地 Ollama（`qwen2.5:0.5b`，397 MB）实测：

| 指标 | 数值 |
|---|---|
| schema 合法率 | **100 %**（3 / 3） |
| 字段 F1 | **1.00**（精确率 1.00，召回率 1.00） |
| 延迟 | **2.05 秒/文档**（Apple Silicon） |
| 逐文档精确匹配 | inv-001 9/9 · inv-002 9/9 · inv-003 9/9 |

在 5 张 CORD 风格的收据样本上（`src/idp/eval/datasets/cord_subset/`），使用本地 Ollama `qwen2.5:0.5b` 同样能拿到 schema 合法率 100%、字段 F1 ≥ 0.85。运行 `python examples/benchmark_cord.py` 可重现。

本框架对小型模型的弱点是诚实的：微小模型上的算术运算（如 `subtotal` / `tax_amount`）会被标记为置信度 0.10 并路由到 HITL 复核，而不是悄悄放过。

---

## 流水线

```
INGEST  →  PARSE  →  CLASSIFY  →  ROUTE  →  EXTRACT  →  ASSESS  →  VALIDATE  →  HITL
                                                                         (Streamlit)
```

每一阶段都是 `Document` 上的纯函数。各阶段独立、可单独测试、可单独替换。

| 阶段 | 模块 | 默认实现 | 作用 |
|---|---|---|---|
| **parse** | `idp.parse` | Docling（PDF）· pdfplumber（兜底）· 纯文本 | 抽取文本 + 表格 + 页面图像 |
| **classify** | `idp.classify` | 规则优先，LLM 兜底 | 识别文档类型：发票、合同、银行流水…… |
| **route** | `idp.parse.router` | 自动 | 根据文档特征选择多模态 VLM 或 OCR+LLM |
| **extract** | `idp.extract` | Pydantic schema 驱动 | 从文本或图像做校验过的结构化抽取 |
| **assess** | `idp.assess` | 启发式 + 可选 LLM 自评 | 逐字段置信度 0..1 |
| **validate** | `idp.validate` | Pydantic + 用户自定义谓词 | schema 校验 + 业务规则 |
| **HITL** | `idp.hitl` | Streamlit UI | 复核低置信度字段，保存更正 |
| **pipeline** | `idp.pipeline.pipeline` | 编排器 | 组合上述阶段，返回 `PipelineResult` |

---

## 大语言模型后端

### 国产大模型（8 家，统一 OpenAI Chat-Completions 协议）

运行 `idp providers` 可打印完整表格。亮点：

| 提供方 | 环境变量 | 默认模型 | 视觉模型 |
|---|---|---|---|
| `deepseek` | `DEEPSEEK_API_KEY` | deepseek-chat | —（仅文本） |
| `qwen` | `DASHSCOPE_API_KEY` | qwen-plus | qwen2.5-vl-72b-instruct |
| `zhipu` | `ZHIPUAI_API_KEY` | glm-4-plus | glm-4v-plus |
| `moonshot` | `MOONSHOT_API_KEY` | moonshot-v1-128k | moonshot-v1-128k-vision-preview |
| `yi` | `YI_API_KEY` | yi-large | yi-vision |
| `doubao` | `ARK_API_KEY` | doubao-pro-32k | doubao-1-5-vision-pro-32k |
| `hunyuan` | `HUNYUAN_API_KEY` | hunyuan-pro | hunyuan-vision |
| `baichuan` | `BAICHUAN_API_KEY` | baichuan4 | —（仅文本） |

```python
from idp.llm import get_china_backend

backend = get_china_backend("qwen", multimodal=True)
# backend.model == "qwen2.5-vl-72b-instruct"
```

### 自托管（Nanonets-OCR2-3B，Apple Silicon / CUDA）

适合不能把文档发给第三方的场景。**无需 API key，无需联网，**首次下载后完全离线**（约 7 GB 缓存至 `~/.cache/huggingface/hub/`）。

```bash
pip install py-idp[hf-vlm]    # 安装 torch + transformers + accelerate + safetensors
export IDP_ENABLE_NANONETS=1  # 显式开关（避免意外下载）
export IDP_BACKEND=nanonets
idp run scan.pdf --backend nanonets
```

**Apple M4 16 GB 内存预算**（float16，448×448 图像）：

- 权重 + 视觉编码器 + KV 缓存：约 8.3 GB
- 系统 + 应用：约 3.5 GB
- 余量：约 4 GB（充裕）

**速度**：M4 上每页约 5–15 秒。首次调用需 5–10 分钟下载模型；后续调用约 10 秒从缓存加载。

---

## 自动 schema 发现

> **试试：** `python -m examples.discover_schema_sample`
> 在真实 PDF 上跑 6 个端到端场景（生成 2 页发票、发现 schema、运行抽取、覆盖各种边界）。无需 API key 也无需 poppler。

手头有一份扫描件 PDF，隐约知道「想要字段 X、Y、Z」——但还没写 Pydantic 类。`discover_schema()` 让多模态大模型（默认 NanonetsVLBackend）提出一份 JSON Schema，再编译成 Pydantic 类，直接传入 `Pipeline(schema=...)`。

```python
import idp

Schema, schema_dict = idp.discover_schema(
    "scan.pdf",
    hint="提取 vendor_name、invoice_number、total_amount 和行项目",
)
# Schema 是 Pydantic BaseModel 的子类 —— 直接传入：
result = idp.Pipeline(backend="nanonets", schema=Schema).run(
    idp.Document.from_path("scan.pdf")
)
print(result.document.extraction)
```

---

## 自动切块（超大文档）

Nanonets-OCR2-3B 的上下文窗口是 16k token。50 页发票 PDF 没法一次塞进去。`extract()` 会自动检测并切分输入，每个 chunk 单独跑模型，再把每段抽取合并。**调用方无感**。

两种内置 chunker：

| chunker | 适用场景 | 默认配置 |
|---|---|---|
| `PageChunker` | 多模态（NanonetsVLBackend + 页面图像） | 每块 4 页，重叠 1 页 |
| `TokenChunker` | 文本抽取器（OCR + LLM） | 每块 4000 token，重叠 200 token（tiktoken） |

```python
from idp.chunker import PageChunker, TokenChunker

# 在内存紧张的小内存 M 系列 Mac 上调小
chunker = PageChunker(max_pages=2, overlap_pages=1)

# 或者直接传给 pipeline
from idp.pipeline import Pipeline
pipe = Pipeline(backend=backend, schema=Invoice, chunker=chunker)

# 端到端：切块 → 调用 → 合并 → 校验 —— 一个调用搞定
result = pipe.run(Document.from_path("huge-50-page-scan.pdf"))
```

合并后的抽取里带一个 `_chunk_count` 标记，可以按块归属成本与可观测性数据。

**单块失败容错**：如果某个 chunk 的 LLM 调用失败，错误会被记录（`extract_chunk_failed[i]`），但其他 chunk 的数据依然合并进来。返回的是「部分结果 + 明确的错误链路」，而不是整批崩溃。

---

## CLI

```bash
idp run path/to/invoice.pdf --schema Invoice --backend ollama --output out.json
idp providers                                          # 完整后端表
idp schemas                                            # 内置 Pydantic schema
idp discover-schema scan.pdf --hint "提取 vendor_name、total_amount" --output schema.json
idp eval --dataset src/idp/eval/datasets/invoices \
          --strategy mock,mock-omits --output results.json
idp serve                                              # 启动 Streamlit HITL UI，监听 :8501
```

---

## 自带 schema

内置 `Invoice`、`Contract`、`BankStatement`、`Receipt` schema 只是参考示例 —— 任何 Pydantic 模型都可以传入：

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

## 引用

如果在研究或产品里用了 py-idp，引用信息见 [`CITATION.cff`](CITATION.cff)。流水线架构（parse → classify → extract → assess → validate → HITL）灵感来自 AWS Solutions Library 的加速智能文档处理样例；PDF 解析封装了 IBM Docling（arXiv:2408.09869）。

---

## 许可证

py-idp 采用 **双许可证**：

- **AGPL-3.0-or-later** —— 适合开源使用。你可以自由使用、修改、运行 py-idp。部署为对外网络服务时，修改也必须以 AGPL 协议开源。这是 copyleft，防止竞争对手把工作克隆进 SaaS 而不回馈社区。见 `LICENSE-AGPL`。
- **商业许可证** —— 适合需要把 py-idp 嵌入闭源产品或自营 SaaS、又不想背负 AGPL copyleft 义务的机构。见 `LICENSE-COMMERCIAL`。

这与 **MariaDB / Sentry / MinIO** 的模式一致：花钱买的是在闭源产品里运行的便利；只要保持改动开源，就能免费拿到完整源码。

参考商业定价：

| 档位 | 用途 | 价格 |
|---|---|---|
| Solo | 单开发者，单一法人 | **300 美元/年** |
| Team | 最多 10 名开发者，单一法人 | **1500 美元/年** |
| Enterprise | 不限开发者 + SLA + 支持 | 联系 |
| SaaS-OEM | 嵌入托管型 SaaS，按活跃用户 | 按席位 |

联系 **rollroyces** 获取正式协议。

---

## 致谢

- 流水线形态、HITL、置信度设计 —— 扩展自 [`aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws`](https://github.com/aws-solutions-library-samples/accelerated-intelligent-document-processing-on-aws)
- PDF 解析 / 表格抽取 —— 封装 [IBM Docling](https://github.com/docling-project/docling)（arXiv 2408.09869）
- Pydantic schema 驱动的抽取 API —— 灵感来自 [`run-llama/llama_cloud_services`](https://github.com/run-llama/llama_cloud_services)
- 多格式 chunking 模式 —— 来自 [`Unstructured-IO/unstructured`](https://github.com/Unstructured-IO/unstructured)

如果你在研究中引用 py-idp，请同时引用本仓库与 Docling。
