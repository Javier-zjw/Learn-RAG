# Learn-RAG

一个用于**学习与实验检索增强生成（RAG）**的轻量级 Python 框架。

项目的目标不是再造一个“大而全”的 RAG 框架，而是把 RAG 的每一个环节拆成**可替换、可评测**的组件：
先用零依赖的离线实现把“建库 → 检索 → 生成 → 评测”全链路跑通，再逐个替换成真实模型，用指标量化每一步改动带来的收益。

- **跑起来**：`ask` 一条命令完成建索引 + 问答，并展示证据与各阶段耗时
- **量出来**：内置检索、生成、忠实度、延迟等十余项指标，支持 HotpotQA / CMRC2018 / BEIR 等公开数据集
- **换得动**：所有组件通过注册表 + YAML 配置装配，改实验只改配置，不改代码

---

## 目录

- [当前已完成的工作](#当前已完成的工作)
- [整体架构](#整体架构)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [评测与消融实验](#评测与消融实验)
- [已知问题 / 待办](#已知问题--待办)

---

## 当前已完成的工作

### 1. 核心抽象层（`learn_rag/core`）

- **统一数据契约**（`types.py`）：`Document`、`Chunk`、`ScoredChunk`、`Answer`、`RagResult`、`EvalSample`、`EvalCase`。
  所有检索结果统一为 `ScoredChunk`，一次问答的全过程（答案、证据、耗时）统一为 `RagResult`，评测层只依赖它。
  `Chunk` 的 id 由 `doc_id + 序号 + 内容摘要` 派生，保证重建索引后 id 稳定、评测可复现。
- **窄接口抽象**（`interfaces.py`）：`DocumentSource`、`Chunker`、`TextEncoder`、`VectorIndex`、`Retriever`、`Reranker`、`QueryTransformer`、`LLM`、`Generator`、`EvalDataset`、`Metric`，每个接口原则上只有一个核心方法。
- **通用注册表**（`registry.py`）：`@registry.register(namespace, name)` 注册实现，`registry.build(namespace, spec, **injected)` 按配置构建组件，并按参数名自动注入 `llm` / `kb` 等依赖。
- **配置系统**（`config.py`）：多份 YAML/JSON 按顺序深度合并；组件 `type` 变化时整体替换而非逐字段合并；支持 `${VAR}` / `${VAR:-默认值}` 引用环境变量，并自动加载 `.env`（shell 中已设置的变量优先）。
- **中英混合文本处理**（`text.py`）：统一的归一化（全角转半角、去标点）、中文按字 + bigram、英文按词切分，BM25、Hashing 向量和 EM/F1 指标共用同一套口径。

### 2. 离线建库链路

| 环节 | 已实现组件 | 说明 |
| --- | --- | --- |
| 数据源 `source` | `memory`、`jsonl`、`directory`、`files` | 统一为 `Document` 流；`files` 解析 PDF / Office / 网页 / 图片（见下文“文档解析”） |
| 文档解析 `parser` | `markdown`、`html`、`docx`、`pptx`、`xlsx`、`csv`、`tsv`、`pdf`、`vlm_ocr`、`libreoffice`、`mineru`、`docling` | 各种文件统一解析为结构元素 `Element` |
| 切分 `chunker` | `fixed`、`recursive`、`markdown`、`structure` | 定长滑窗 / 按语义边界递归切分 / 按标题层级切分 / 基于解析结构的父子分块（见下文“结构感知的父子分块”） |
| 向量化 `encoder` | `hashing`、`openai_compat`、`sentence_transformers`、`cached` | 特征哈希（零依赖离线可跑）/ 任意 OpenAI 兼容 embedding 服务（超批量上限时自动减半重试）/ 本地模型 / 带缓存的装饰器 |
| 向量索引 `index` | `flat`、`chroma` | 暴力内积精确检索 / Chroma 持久化 HNSW（可调 `space`、`ef_construction`、`max_neighbors`、`ef_search`） |
| 倒排索引 | `BM25Index` | 与向量索引同步写入、同步落盘 |

`KnowledgeBase`（`store/knowledge_base.py`）作为离线门面，对外只暴露 `add / expand / save / load / stats`，内部完成“切分 → 区分父子块 → 攒批向量化 → 写向量索引 → 写 BM25 索引”；子块进索引，父块只存储，`expand` 把命中的子块换成父块。
持久化目录按 collection 隔离，避免不同数据集的 BM25 索引互相覆盖；Chroma 未安装时会给出带排查建议的明确报错，而不是含糊的“未注册”。

#### 文档解析（`learn_rag/parsing`）

企业文档先解析为统一的结构元素 `Element`（标题 / 段落 / 表格 / 图片 / 公式 / 代码，附带层级、页码、坐标），再由 `Document.from_elements` 渲染成 Markdown 文本，旧的切分器与检索无需改动。

| 格式 | 默认解析器 | 处理方式 |
| --- | --- | --- |
| `.md` / `.txt` | `markdown` | 识别标题、表格、代码块、公式、图片 |
| `.html` | `html` | 读取标签结构，跳过导航、页眉页脚、脚本 |
| `.docx` | `docx` | 按正文顺序读取段落与表格，标题层级取自样式 |
| `.pptx` | `pptx` | 每页一个标题，按阅读顺序读取文本框和表格，收录演讲者备注 |
| `.xlsx` | `xlsx` | 每个工作表一张表格，去掉空行空列，读取公式计算结果 |
| `.csv` / `.tsv` | `csv` / `tsv` | 零依赖读取表格，自动兼容 UTF-8 / GBK 编码，可设行数上限防止超大文件 |
| `.doc` / `.xls` / `.ppt` / `.rtf` | `libreoffice` | 调用 LibreOffice 无头转换成 OOXML 后复用上面的解析器；每次转换使用独立配置目录，支持并发 |
| `.pdf` | `pdf` | 按字号识别标题层级、检测表格、去除重复页眉页脚和页码、合并跨页段落；文字层缺失的扫描页交给 OCR |
| 图片 | `vlm_ocr` | 调用任意 OpenAI 兼容的视觉模型（PaddleOCR-VL、MinerU2.5、Qwen-VL 等）转写为 Markdown |

- `configs/parsing.yaml` 预置企业级路由：PDF 首选 MinerU、Word 首选 Docling，未安装或失败时自动退回内置解析器；两路解析出的图片统一存入内容寻址资产库（按 SHA-256 去重），图注进入索引，资产路径记录在块元数据 `assets` 字段中。
- MinerU 4.x 模型权重通过 ModelScope 自动下载到项目内的 `mineru_model_weight/`（已 gitignore），解析器通过 `MINERU_HOME` 加载该目录；CPU 部署可用 `scripts/start_mineru.sh` 读取 `.mineru.env` 并启动本地 VLM 服务，解析器读取同一个 `.mineru.env`。
- `FileSource` 按“文件内容哈希 + 解析器 + 参数”缓存解析结果，调整切分策略、重建索引时不会重复解析；缓存原子写入、损坏自动重建；目录扫描跳过隐藏文件、Office 锁文件和未下载完成的临时文件，可按 `max_file_size` 限制单文件大小；单个文件失败只记日志并跳过。
- 解析结果交给 `structure` 切分器做父子分块，见下一节。

#### 结构感知的父子分块（`learn_rag/ingest/structure.py`）

检索和生成对块大小的要求是矛盾的：块小，向量表达集中，召回才准；块大，交给大模型的上下文才完整。父子分块把两件事拆开：

```
                ┌─ 父块（≤ parent_size，默认 1200 token）：只存储，不进索引 ──────────┐
文档 ─▶ 章 / 小节 ─┤                                                               │ 命中后展开
                └─ 子块（≤ chunk_size，默认 300 token）：进向量索引和 BM25 ─▶ 召回 ─▶ 精排 ─┘─▶ 大模型
```

切分沿着解析出的结构走，规则如下：

| 规则 | 说明 |
| --- | --- |
| 章 | 文档标题之下的第一级标题算一章（PPT 的每一页就是一章）。父块不跨章，子块不跨小节 |
| 父块 | 同一章内按顺序把小节装进父块，装满换下一个；小节本身超长时按元素拆开。父块正文保留小节标题 |
| 子块 | 每个小节内按顺序装元素，块首拼上“文档标题 > 章节路径”，片段脱离原文也知道自己在讲什么 |
| 不拆开的组合 | 公式、图片紧跟前一个元素（“按下式计算：”和公式，“如图 2 所示”和图片）；以冒号结尾的引导句带上后一个元素（列表、表格） |
| 超长段落 | 按句切开；句子本身超长时在逗号、顿号、冒号处断开，连分句都超长（OCR 结果、长网址）才按字数硬切。数字里的英文逗号（4,860）不算断点 |
| 句子重叠 | 默认关闭。`overlap_sentences: 1` 时，一个长段落被切成几个子块，下一块开头重复上一块的最后一句，弥补“它”“该方案”这类指代在切口处丢失；只作用于同一段落内的子块，不跨元素，表格和父块不重叠 |
| 超长表格 | 按行切开，每块都带表题和表头，表注跟在最后一块；合并单元格先展开：跨行的值复制到每一行，多层表头合成一行（“上半年 Q1”），切出的每一行都是自包含的 |
| 代码、公式、图片 | 代码按行切，不把一行断开；公式和图片描述不切 |
| 去冗余 | 只有一个子块的父块不生成（和子块内容相同） |
| 纯文本 | 没有解析结构的文本（jsonl 语料）按空行分段后走同一套规则 |

- **大小按 token 估算**（`core/text.py` 的 `count_tokens`）：中文每字约 1 个，英文每词约 1.3 个，中英文混排时块大小一致；只计正文，不计块首路径。
- **检索流程**：召回和精排都在子块上做（短而集中，交叉编码器打分更准、不会被截断），之后 `KnowledgeBase.expand` 把子块换成父块，同一父块下的多个子块合并为一条，取够 `top_k` 个不同的上下文；命中了哪些子块记在 `debug["children"]` 中。切分器不生成父块时这一步不改变任何结果。
- **图片和图表**：图片二进制存在内容寻址资产库，图注 / 模型描述随所在小节进入子块，向量和 BM25 索引的是这些文字，图片本身不向量化；原生图表（PPT / Excel）由 MinerU 转成数据表，按表格切分。块内每张图在元数据 `assets` 中记一条（见下方示例），超长表格切成几片时每片都带着同一张表格截图。
- **持久化**：父块保存在索引目录的 `parents.jsonl` 中，`build` 后 `--reuse-index` 复用时同样可以展开。
- **溯源**：每个块记录 `section`（章节路径）、`page_start` / `page_end`、`kinds`（包含的元素类型）、`assets`，子块记录 `parent_id`。

块的元数据示例（年度报告第 2 章的父块）：

```json
{
  "title": "01_年度报告_双栏复杂版面", "path": "samples/parsing/01_年度报告_双栏复杂版面.pdf",
  "file_type": "pdf", "file_hash": "a4b2a402…", "parser": "mineru", "doc_id": "01_年度报告_双栏复杂版面.pdf",
  "section": "星河科技 2025 年度经营报告 > 2 区域经营数据",
  "kinds": ["formula", "image", "table", "text"], "page_start": 2, "page_end": 3,
  "assets": [
    {"asset": "01/01b178….jpg", "kind": "table", "caption": "表 1 2025 年各区域季度营收", "page": 2,
     "bbox": [0.118, 0.165, 0.878, 0.325], "mime": "image/jpeg"},
    {"asset": "9f/9f04a9….jpg", "kind": "image", "caption": "2025年各区域季度营收 图 1 2025 年各区域季度营收",
     "page": 2, "bbox": [0.246, 0.499, 0.754, 0.713], "mime": "image/jpeg"}
  ]
}
```

`assets` 的每一条：`asset` 是资产库内的相对路径；`kind` 为 `image`（图片、图表）或 `table`（表格截图）；`caption` 是图注和模型描述（表格只取表题，`[图片]` 这类占位文字不算）；`page`、`bbox` 沿用解析器给出的页码和坐标（MinerU 为 0～1 的相对坐标）。没有的字段不写。
- **元数据完整落盘**：flat 索引原样保存；Chroma 的元数据只接受字符串和数字，标量字段原样存入供 `where` 过滤，完整元数据另存为一个 JSON 字段，取回时原样还原（旧版本建的集合仍可读取）。
- **问答时展示出处**：`ask` 的每条证据下方列出页码、章节，以及块内每张图的图注和本地文件路径（在配置的 `assets_dir` 中查找）。

```yaml
chunker:
  type: structure
  chunk_size: 300        # 子块
  parent_size: 1200      # 父块；设为 0 关闭父块，只按小节切分
  overlap_sentences: 0   # 句子重叠，默认关闭
```

为什么默认不重叠：切口都落在元素或句子边界上，不会把一句话切成两半；被召回后交给大模型的是父块，切口两侧的内容都在里面；重叠还会让同一句话出现在两个子块里，一起被召回时白白占掉候选名额。重叠只解决“召回之前匹配不上”的问题，等评测发现子块边界导致漏召回时再打开。

`scripts/export_parsed.py` 导出的 `<文件名>.chunks.jsonl` 中，`role` 标明父块 / 子块，`tokens` 是估算的 token 数，可以直接检查切分效果（样例见 `samples/parsing_results/*/`）。

### 3. 在线检索与生成链路

| 环节 | 已实现组件 | 说明 |
| --- | --- | --- |
| 召回 `retriever` | `vector`、`bm25`、`hybrid`、`transformed` | 混合检索使用带权重的 **RRF** 融合；检索器可任意嵌套组合 |
| 查询改写 `query_transformer` | `identity`、`multi_query`、`hyde` | 多路改写 / HyDE 假设答案检索（需 LLM） |
| 精排 `reranker` | `identity`、`lexical`、`cross_encoder`、`api`、`llm` | 不精排 / 词法重叠 / 本地 CrossEncoder / 远程 Rerank API（兼容标准 `/rerank` 协议与通义 DashScope 原生协议，失败时退回召回顺序）/ LLM 打分 |
| 大模型 `llm` | `echo`、`openai_compat` | 占位 / 任意 OpenAI 兼容 Chat 接口（DeepSeek、OpenAI 等） |
| 生成 `generator` | `extractive`、`stuff` | 抽取式（无需 LLM 即可端到端评测）/ 按字符预算装配上下文、要求引用并允许拒答 |

`RagPipeline`（`pipeline/rag.py`）是系统级门面：`index(documents)` 建库、`answer(question)` 问答、`retrieve_only(question)` 只评检索，`from_config(cfg)` 一处集中完成“配置 → 对象图”的构建。

### 4. 评测体系（`learn_rag/eval`）

- **数据集适配**：`jsonl`（自定义业务集）、`hotpotqa`（英文多跳）、`squad_style`（CMRC2018 / DuReader-robust / SQuAD）、`beir_style`（SciFact、FiQA 等纯检索基准）。
- **指标**：
  - 检索：Recall@k、Precision@k、HitRate@k、MRR@k、MAP@k、NDCG@k
  - 上下文：Context Recall
  - 生成：Exact Match、Answer Contains、Token F1、ROUGE-L
  - 可信度：词法忠实度（Faithfulness）、可选 LLM-as-a-Judge
  - 性能：各阶段延迟
- **评测执行器** `Evaluator`：自动用数据集语料建索引；缺少证据标注或标准答案时**跳过**对应指标而不是记 0 分；单样本失败不影响整体；支持多线程并发；输出 Markdown 汇总 + JSON 明细。

### 5. 命令行工具（`learn_rag/cli.py`）

| 子命令 | 作用 |
| --- | --- |
| `ask` | 单次问答，展示答案、证据片段（分数/来源/标题）与各阶段耗时；`--docs` 可直接解析文档目录 |
| `build` | 离线建库并持久化向量索引与 BM25 索引（只需跑一次） |
| `eval` | 在数据集上评测，支持 `--retrieval-only`、`--judge`、`--reuse-index`、`--workers`、`--out` |
| `ls` | 列出每一层所有已注册的可用实现 |

### 6. 辅助脚本（`scripts/`）

- `make_sample_data.py`：生成内置示例语料与问答集（内容就是 RAG 的核心概念，边评测边学习）。
- `prepare_dataset.py`：把 HotpotQA / SQuAD 结构 / BEIR 数据集转换成统一的 `corpus.jsonl + qa.jsonl`。
- `inspect_dataset.py`：评测前检查 `data/` 下数据集结构是否能被正确读取。
- `run_ablation.py`：消融实验，一次只改一个变量，结果以相对基线的增量呈现并连同完整配置落盘。
  内置套件：`chunk_size`、`channel`、`rerank`、`index`、`space`、`hnsw_build`、`hnsw_search`、`top_k`。
- `make_parsing_samples.py`：生成 `samples/parsing/` 下的文档解析验证样例（22 个文件，覆盖 PDF、扫描件、Office 新旧格式、OpenDocument、RTF、CSV/TSV、网页、网页存档、EPUB、图片），说明见 `samples/README.md`。
- `export_parsed.py`：把文档解析结果导出为 Markdown、结构元素 JSON 和切分块，用于检查 MinerU 或内置解析器的效果；`--mineru` 让 Word / PPT / Excel 也优先走 MinerU。
- `start_mineru.sh` / `stop_mineru.sh`：读取 `.mineru.env`，启动 / 停止本机 MinerU 的 llama.cpp VLM 服务（停止时顺带清理残留的文档库服务），详见“使用 MinerU 解析复杂 PDF”。
- `make_agentic_data.py`：生成 Agentic RAG 演示数据（带元数据的文件、SQLite 订单库、标注了 `expected_tools` 的问答集），为后续“自主决策”模式做准备。

### 7. 预置实验配置（`configs/`）

| 文件 | 用途 |
| --- | --- |
| `default.yaml` | 完全离线基线：递归切分 + Hashing 向量 + 混合检索 + 词法精排 + 抽取式生成 |
| `models_env.yaml` | 从环境变量读取真实的 Embedding / LLM / Rerank 模型 |
| `chroma.yaml` | Chroma 持久化向量库 |
| `ds_scifact.yaml` / `ds_fiqa.yaml` | BEIR 基准，对齐 nDCG@10 口径 |
| `ds_cmrc.yaml` | 中文 CMRC2018，可同时评检索与生成 |
| `parsing.yaml` | 文档解析 + 结构感知的父子分块 |

### 8. 测试（`tests/`）

基于 `unittest`：`test_core.py` 覆盖文本切词、切分器、Hashing 向量、检索指标公式（Recall / MRR / NDCG / F1 / ROUGE-L）、端到端管线、API 精排、Chroma 索引、数据集加载与 `.env` 加载等容易写错的“接口契约”；`test_parsing.py` 覆盖各格式解析、MinerU / Docling / OCR 适配器（mock）、解析缓存与降级和端到端问答；`test_chunking.py` 覆盖父子分块的各条规则、表格展开与按行切分、知识库的父块存储、展开与持久化。样例文件在测试中现场生成。

---

## 整体架构

```
            ┌──────────────────── 离线（build） ────────────────────┐
Document ──▶ Chunker ──▶ TextEncoder ──▶ VectorIndex (flat / chroma)
                                    └──▶ BM25Index
            └──────────────── KnowledgeBase（门面） ─────────────────┘

            ┌──────────────────── 在线（ask） ──────────────────────┐
Question ──▶ QueryTransformer ──▶ Retriever (vector / bm25 / hybrid-RRF)
         ──▶ Reranker ──▶ 父块展开 ──▶ Generator (+LLM) ──▶ RagResult（答案 + 证据 + 耗时）
            └──────────────── RagPipeline（门面） ──────────────────┘

EvalDataset ──▶ Evaluator(RagPipeline, Metrics) ──▶ EvalReport（Markdown + JSON）
```

分层依赖自下而上：`core` 不依赖任何上层模块；上层只依赖接口与数据契约，所有具体实现通过注册表按配置装配。

---

## 项目结构

```
Learn-RAG/
├── learn_rag/
│   ├── core/          # 数据契约、抽象接口、注册表、配置加载、文本处理
│   ├── ingest/        # 数据源 + 切分器（含结构感知的父子分块）
│   ├── parsing/       # 文档解析：PDF / Office / 网页 / OCR / MinerU / Docling
│   ├── embedding/     # 文本向量化
│   ├── store/         # 向量索引（flat / chroma）、BM25、KnowledgeBase
│   ├── retrieval/     # 召回、查询改写、精排
│   ├── generation/    # LLM 客户端 + 生成策略
│   ├── pipeline/      # RagPipeline 端到端编排
│   ├── eval/          # 数据集适配、指标、评测执行器
│   └── cli.py         # 命令行入口
├── configs/           # YAML 实验配置
├── scripts/           # 数据准备、数据检查、消融实验、MinerU 服务启动脚本
├── tests/             # 单元测试
├── samples/           # 文档解析验证样例与解析结果（见 samples/README.md）
├── pyproject.toml     # 打包与依赖声明
└── env.example.txt    # 模型/密钥配置模板
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -e .              # 核心依赖：numpy、pyyaml，并注册 learn-rag 命令
# 可选扩展
pip install -e ".[dotenv]"    # 自动加载 .env
pip install -e ".[chroma]"    # Chroma 持久化向量库
pip install -e ".[local]"     # 本地 embedding / CrossEncoder 精排
pip install -e ".[parsing]"   # 文档解析：PDF / Word / PPT / Excel
pip install -e ".[all]"       # 以上全部
pip install -e ".[dev]"       # 跑测试所需依赖
```

运行测试：`python -m unittest discover tests`（或 `pytest`）。

安装后 `learn-rag <子命令>` 与 `python -m learn_rag.cli <子命令>` 等价。
全局参数 `--log-level DEBUG|INFO|WARNING|ERROR` 控制日志输出（放在子命令之前）。

### 2. 离线跑通（无需任何 API Key）

```bash
python scripts/make_sample_data.py

# 单次问答
python -m learn_rag.cli ask --corpus data/sample_corpus.jsonl -q "什么是RRF?"

# 评测
python -m learn_rag.cli eval --dataset jsonl \
    --qa data/sample_qa.jsonl --corpus data/sample_corpus.jsonl --out runs/base.json

# 查看所有可用组件
python -m learn_rag.cli ls
```

> CLI 总是先加载 `configs/default.yaml`，再依次叠加 `--config` 指定的文件（后者覆盖前者）。

### 3. 接入真实模型

```bash
cp env.example.txt .env    # 填写 LLM / Embedding / Rerank 的 base_url、api_key、model
python -m learn_rag.cli ask --config configs/models_env.yaml \
    --corpus data/sample_corpus.jsonl -q "HyDE 的原理是什么？"
```

### 4. 公开数据集 + 持久化索引

```bash
# 离线建库一次
python -m learn_rag.cli build --config configs/models_env.yaml --config configs/ds_scifact.yaml \
    --dataset beir_style --qa data/scifact

# 之后反复评测都复用索引，跳过向量化
python -m learn_rag.cli eval --config configs/models_env.yaml --config configs/ds_scifact.yaml \
    --dataset beir_style --qa data/scifact --reuse-index --retrieval-only
```

### 5. 解析企业文档（PDF / Word / PPT / Excel / 网页 / 图片）

```bash
pip install -e ".[parsing]"
python -m learn_rag.cli ask --config configs/parsing.yaml --docs path/to/docs -q "差旅住宿标准是多少？"

# 离线建库并持久化，之后问答不必再传 --docs：ask 会加载落盘的 BM25 索引和父块
python -m learn_rag.cli build --config configs/parsing.yaml --config configs/chroma.yaml --docs path/to/docs
python -m learn_rag.cli ask --config configs/parsing.yaml --config configs/chroma.yaml -q "差旅住宿标准是多少？"
```

每条证据下方会列出出处，例如：

```
[2] (0.7262 · rerank:lexical · 01_年度报告_双栏复杂版面) [01_年度报告_双栏复杂版面 > … > 2 区域经营数据] …
     出处：第 2-3 页　星河科技 2025 年度经营报告 > 2 区域经营数据
     表格截图：第 2 页 「表 1 2025 年各区域季度营收」 → /…/.cache/assets/01/01b178….jpg
     图片：第 2 页 「2025年各区域季度营收 图 1 2025 年各区域季度营收」 → /…/.cache/assets/9f/9f04a9….jpg
```

扫描件和图片需要配置 OCR 模型（`.env` 中的 `OCR_BASE_URL` / `OCR_API_KEY` / `OCR_MODEL`，并在 `configs/parsing.yaml` 中打开 `ocr`）。

### 6. 使用 MinerU 解析复杂 PDF

内置的 PyMuPDF 解析器适合版面规整的电子 PDF；多栏排版、复杂表格、公式和扫描件交给 MinerU 4.x。
`configs/parsing.yaml` 已默认把 PDF 路由为 `[mineru, pdf]`：MinerU 可用时优先使用，未安装、服务未启动或解析失败时自动退回 PyMuPDF，导入不会中断。

**工作流程**

```
FileSource ──▶ MinerUParser
                 └─ mineru-kit parse <文件> -o <临时目录>/<文件名>.zip --format zip --tier standard --ocr-mode auto
                      └─ 读取 zip 中的 middle_json.json ──▶ 翻译成 Element ──▶ 结构切分 / 建索引
                      └─ zip 中的图片、图表、表格截图 ──▶ 内容寻址资产库（.cache/assets）
```

解析器通过命令行（`mineru-kit parse`）调用 MinerU，命令行是它最稳定的对外接口。
`mineru-kit parse` 在当前进程内直接解析，只依赖 VLM 服务，**不需要 MinerU 的文档库服务（`mineru server`）**。

MinerU 的运行环境统一放在项目根目录的 `.mineru.env` 里。启动脚本和解析器读取的是**同一个文件**，
保证启动服务和解析文档用的是同一套模型目录和 VLM 地址。环境变量的优先级：

1. `.mineru.env` 中的值（最高）；
2. 当前 shell 中已经设置的同名环境变量；
3. 解析器默认值，见下表。

| 环境变量 | 解析器默认值 |
| --- | --- |
| `MINERU_HOME` | `options.mineru.models_dir`（默认项目内的 `mineru_model_weight/`） |
| `MINERU_MODEL_SOURCE` | `modelscope`（权重从 ModelScope 下载） |
| `MINERU_MODEL_SMALL_BACKEND` | `onnx` |
| `MINERU_MODEL_VLM_ENGINE` | `llama-cpp` |
| `MINERU_MODEL_VLM_SERVER_URL` | `options.mineru.vlm_server_url`（默认 `http://127.0.0.1:30000`） |

解析时日志会打印实际生效的值，例如
`MinerU 运行环境：已加载 /…/Learn-RAG/.mineru.env；MINERU_HOME=…；VLM 服务=http://127.0.0.1:30000`，
看到“已加载”就说明读到了 `.mineru.env`。

**第一步：安装**

```bash
pip install -e ".[mineru]"     # 即 mineru>=4.0,<5，提供 mineru 与 mineru-kit 命令
```

模型权重首次运行时从 ModelScope 自动下载到 `mineru_model_weight/`（已在 `.gitignore` 中）。内网机器可以把这个目录整体拷贝过去。

**第二步：创建本机配置 `.mineru.env`（含本机绝对路径，已在 `.gitignore` 中，不提交）**

```bash
MINERU_PYTHON_ENV=/你的/python环境                   # 安装了 MinerU 的 Python 环境（启动、停止脚本用）
MINERU_HOME=/绝对路径/Learn-RAG/mineru_model_weight     # 模型目录
MINERU_MODEL_VLM_SERVER_URL=http://127.0.0.1:30000     # VLM 服务地址
MINERU_VLM_HOST=127.0.0.1                             # VLM 服务监听地址（启动脚本用）
MINERU_VLM_PORT=30000                                 # VLM 服务端口（启动脚本用）
```

写法与 shell 相同：支持 `export` 前缀、引号、`#` 注释和 `$VAR` / `~`。

**第三步：启动和停止服务（CPU 部署）**

```bash
bash scripts/start_mineru.sh     # 启动 VLM 服务（已运行则跳过）
bash scripts/stop_mineru.sh      # 停止 VLM 服务，并清理残留的 MinerU 文档库服务
```

启动脚本依次完成：
1. 加载 `.mineru.env`，检查 `mineru-kit` 命令和模型目录是否存在（`MINERU_PYTHON_ENV` 没写在 `.mineru.env` 里时，默认 `/opt/anaconda3/envs/langchain_env`）；
2. 访问 `<VLM 服务地址>/v1/models` 检查 VLM 服务。未运行时用 `mineru-kit vlm-server --engine llama-cpp` 在后台启动，最多等待 120 秒。

VLM 服务的日志写在 `$MINERU_HOME/logs/vlm-server.log`，进程号写在 `$MINERU_HOME/vlm-server.pid`。
VLM 模型单独作为服务运行，避免每次解析都在进程内重新加载大模型。

停止脚本会先用 `mineru server stop` 正常停止文档库服务；如果该服务卡住不响应
（报错 `MinerU home [...] is currently owned by another doclib server process`），会找到 `python -m mineru.doclib.app` 进程直接结束。

**第四步：检查解析效果**

可以先用仓库自带的验证样例（`samples/parsing/`，说明和预期结果见 `samples/README.md`）：

```bash
python scripts/export_parsed.py samples/parsing --mineru --no-cache \
    --raw samples/parsing_results/mineru_raw --out samples/parsing_results/mineru
```

再换成自己的文档：

```bash
python scripts/export_parsed.py path/to/docs              # PDF 走 MinerU，其他格式按 parsing.yaml 路由
python scripts/export_parsed.py path/to/docs --mineru     # 所有 MinerU 支持的格式都优先交给 MinerU，便于对比
```

每个文档在 `runs/parsed/`（可用 `--out` 指定）下生成三个文件：

| 文件 | 内容 | 用来检查 |
| --- | --- | --- |
| `<文件名>.md` | 解析结果渲染成的 Markdown | 标题层级、表格、段落顺序 |
| `<文件名>.elements.json` | 每个结构元素的类型、层级、页码、坐标、附加信息 | 页眉页脚是否去掉、跨页段落是否合并、图片资产路径 |
| `<文件名>.chunks.jsonl` | 父子分块的结果：`role`、估算的 `tokens`、正文和元数据 | 最终进入知识库的内容、父子关系和溯源信息 |

输出目录里还会生成 `summary.md`（每个文件用的解析器、耗时、各类元素数量，以及没有导出的文件）和 `export.log`（完整日志）。
`--no-cache` 强制重新解析；`--raw <目录>` 另存 MinerU 的原始结果 zip，用于核对适配器有没有漏掉信息。
终端会打印每个文档实际使用的解析器：显示 `mineru` 说明 MinerU 生效；显示其他解析器说明 MinerU 失败后已降级，原因在上方的 WARNING 日志里。
解析结果按文件内容缓存在 `.cache/parsed/`，修改 MinerU 配置后想重新解析，先删除对应的 `mineru-*.json`。

**第五步：问答与建库**

```bash
python -m learn_rag.cli ask --config configs/parsing.yaml --docs path/to/docs -q "问题"
python -m learn_rag.cli build --config configs/parsing.yaml --config configs/chroma.yaml --docs path/to/docs
```

MinerU 4.x 只对 PDF 和图片区分质量档位；Word、PPT、Excel 等格式固定使用 `flash`（直接读取文件结构，不需要模型），解析器对这些格式不会传 `--tier`、`--ocr-mode` 等参数。

**配置项**（`configs/parsing.yaml` 的 `parsing.options.mineru`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `tier` | `standard` | 解析档位：`flash` / `basic` / `standard` / `advanced`；`standard` 适合复杂版面、表格、公式和扫描件。仅对 PDF 和图片生效 |
| `ocr_mode` | `auto` | 传给 `--ocr-mode` |
| `image_analysis` | `true` | 设为 `false` 时追加 `--disable-image-analysis` |
| `vlm_server_url` | `http://127.0.0.1:30000` | llama.cpp VLM 服务地址 |
| `models_dir` | `mineru_model_weight` | 模型权重目录，相对路径按项目根目录解析 |
| `assets_dir` | 不保存 | 图片资产库目录，`parsing.yaml` 中设为 `.cache/assets`，相对路径按项目根目录解析（Docling 同样如此） |
| `command` | `mineru-kit` | MinerU 命令行名称或绝对路径 |
| `timeout` | `1800` | 单个文件的解析超时（秒） |
| `raw_dir` | 不保存 | 另存 MinerU 原始结果 zip 的目录，用于排查 |
| `replay_dir` | 不读取 | 优先读取该目录里已保存的结果 zip，不调用 MinerU，用于修改适配器后快速验证 |
| `env_file` | `.mineru.env` | MinerU 运行环境文件，相对路径按项目根目录解析；设为 `null` 时不读取 |

**MinerU 输出到 `Element` 的映射**

这部分规则按 MinerU 4.x 在 22 个验证样例上的真实输出（`samples/parsing_results/mineru_raw/`）核对过。

| MinerU 块类型 | Element | 说明 |
| --- | --- | --- |
| `doc_title` | 标题（1 级） | |
| `paragraph_title` | 标题 | 层级取 MinerU 给出的 `level` |
| 正文等其他文字块 | 段落 | 行内公式保留为 `$...$`，超链接保留文字 |
| `list` / `index` | 段落 | 每项一行，嵌套的子列表每深一层缩进两个空格 |
| `page_footnote` | 段落（标记 `footnote`） | PDF 的页脚注释和 **PPT 的演讲者备注**在 MinerU 里都是它，属于正文内容 |
| `table` | 表格 | 表题 + 表格 + 表注拼在一起；表格截图存入资产库 |
| `chart` | 表格 / 图片 | **原生图表**（PPT、Excel 里的图表）的 `chart_body` 是图表数据，转成表格保留；图片形式的图表只有图题 |
| `image` | 图片 | 文本为图题 + 图片描述（`image_body` 只是文件名时忽略）；没有任何文字时为 `[图片]`，原图存入资产库 |
| 图片、图表的图注 | 段落 | 单独输出：MinerU 有时会把紧跟在图后面的正文段落识别成图注 |
| `equation` | 公式 | LaTeX |
| `code` | 代码 | 包含代码标题和脚注 |
| `header` / `footer` / `page_number` / `aside_text` / `discarded` | 丢弃 | 页眉、页脚、页码等噪声 |

- **表格统一表示**：没有合并单元格的表格转成 Markdown（结构切分器可以按行切分超长表格，每块都带表题和表头）；
  有合并单元格的保留为紧凑 HTML（去掉缩进和 `<p>` 等排版标签）。只有日期的单元格去掉 `00:00:00`。
- **折行合并**：MinerU 按版面块输出，同一段话常被拆成两块，它的 `continues_prev` 标记既会漏标也会误标
  （样例中真正跨页的句子没有标，扫描件里印章和下一页的条款反而被标成了一段），所以正文按版面几何判断：
  上一块不以句末标点结尾，并且以逗号、顿号结尾或写满了所在栏的宽度时，与下一块（同页下方或下一页开头）合并，记录结束页码 `page_end`。
  只作用于带坐标的 PDF 和图片；跨页续表仍按 `continues_prev` 合并，续页重复的表头会去掉。
- 图片按内容的 SHA-256 命名，同一张图只存一份。`Element.extra` 中记录 `asset`（资产库内相对路径）、`sha256` 和 `mime`，切分后汇总到块元数据的 `assets` 字段（每张图带图注、页码和坐标），回答时可以取回原图。
- zip 中的图片路径会做安全检查，不存在或路径不安全的图片只保留文字，不影响整份文档。
- **回放**：`export_parsed.py --replay <目录>`（即配置 `replay_dir`）直接读取之前用 `--raw` 保存的 MinerU 结果，
  修改适配器后几秒内就能在真实输出上重新验证，不必重跑 MinerU。

**常见问题**

| 现象 | 处理 |
| --- | --- |
| 日志提示找不到 `mineru-kit` | 当前 Python 环境没装 MinerU，或用 `options.mineru.command` 指定绝对路径 |
| 日志提示解析超时 | 调大 `options.mineru.timeout`，或换用更轻的 `tier` |
| PDF 实际走了 PyMuPDF | 查看日志中 `解析器 mineru 处理 ... 失败` 的原因；块元数据 `parser` 字段记录了实际使用的解析器 |
| VLM 服务没起来 | 查看 `$MINERU_HOME/logs/vlm-server.log`，或执行 `curl <VLM 服务地址>/v1/models` |

---

## 配置说明

一份配置描述整条链路，每个组件都是 `{type: 实现名, ...构造参数}`：

```yaml
chunker:   { type: recursive, chunk_size: 300, chunk_overlap: 60 }
encoder:   { type: hashing, dimension: 512 }
index:     { type: flat }
retriever:
  type: hybrid
  rrf_k: 60
  weights: [1.0, 1.0]
  channels: [ { type: vector }, { type: bm25 } ]
reranker:  { type: lexical }
llm:       { type: echo }
generator: { type: extractive }
pipeline:  { top_k: 5, candidate_k: 20 }
```

带查询改写的检索器可以嵌套：

```yaml
retriever:
  type: transformed
  transformer: { type: hyde }
  inner: { type: vector }
```

新增一个实现只需写一个类并加上 `@registry.register("<namespace>", "<name>")`，上层代码无需改动。

---

## 评测与消融实验

```bash
# 比较不同切分粒度
python scripts/run_ablation.py --suite chunk_size \
    --dataset jsonl --qa data/sample_qa.jsonl --corpus data/sample_corpus.jsonl

# 在 HotpotQA 上跑全部套件
python scripts/run_ablation.py --suite all --dataset hotpotqa \
    --qa data/hotpot_dev_distractor_v1.json --limit 300
```

每组实验只改变一个维度，其余配置完全一致；结果表格展示相对第一组的增量，完整配置随结果一起保存，保证结论可复现。

---

## 已知问题 / 待办

- CLI 已预留 `--mode agentic`（自主决策 Agent）与 `--mode wiki`（LLM Wiki 知识编译）两种系统形态，`make_agentic_data.py` 已能生成对应演示数据，但这两种模式的实现尚未加入，当前均回退为 `pipeline` 模式。
- 文档解析的已知局限：
  - 原生 PDF / Word / PPT 解析器暂不处理内嵌图片（图表不会生成文字描述）；需要时改用 `mineru` 或 `docling`。
  - 原生 PDF 解析按文字块顺序读取，多栏排版可能出现阅读顺序错乱，复杂版面建议使用 `mineru`。
  - MinerU 4.x 已按 `middle_json.zip` 输出适配，图片、图表和表格截图进入内容寻址资产库；复杂文档仍建议用自己的样本做召回质量验证。
- 分块的已知局限：
  - 没有描述的图片（MinerU 只输出 `[图片]`）只能靠所在章节的标题被召回；需要时可以接入视觉模型为图片生成描述。
  - 章节划分依赖解析出的标题。没有标题的长文档（如样例中的扫描合同，条款没有被识别为标题）只能按大小切分父块。
  - 尚未实现基于向量相似度的语义切分和用大模型为每个块生成上下文说明（Contextual Retrieval）；解析结构和标题路径已覆盖它们的大部分收益，可在评测后按需加入。

---

## 许可证

本项目基于 [MIT 协议](LICENSE) 开源。
