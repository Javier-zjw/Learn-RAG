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
| 数据源 `source` | `memory`、`jsonl`、`directory` | 统一为 `Document` 流 |
| 切分 `chunker` | `fixed`、`recursive`、`markdown` | 定长滑窗 / 按语义边界递归切分 / 按标题层级切分 |
| 向量化 `encoder` | `hashing`、`openai_compat`、`sentence_transformers`、`cached` | 特征哈希（零依赖离线可跑）/ 任意 OpenAI 兼容 embedding 服务（超批量上限时自动减半重试）/ 本地模型 / 带缓存的装饰器 |
| 向量索引 `index` | `flat`、`chroma` | 暴力内积精确检索 / Chroma 持久化 HNSW（可调 `space`、`ef_construction`、`max_neighbors`、`ef_search`） |
| 倒排索引 | `BM25Index` | 与向量索引同步写入、同步落盘 |

`KnowledgeBase`（`store/knowledge_base.py`）作为离线门面，对外只暴露 `add / save / load / stats`，内部完成“切分 → 攒批向量化 → 写向量索引 → 写 BM25 索引”。
持久化目录按 collection 隔离，避免不同数据集的 BM25 索引互相覆盖；Chroma 未安装时会给出带排查建议的明确报错，而不是含糊的“未注册”。

### 3. 在线检索与生成链路

| 环节 | 已实现组件 | 说明 |
| --- | --- | --- |
| 召回 `retriever` | `vector`、`bm25`、`hybrid`、`transformed` | 混合检索使用带权重的 **RRF** 融合；检索器可任意嵌套组合 |
| 查询改写 `query_transformer` | `identity`、`multi_query`、`hyde` | 多路改写 / HyDE 假设答案检索（需 LLM） |
| 精排 `reranker` | `identity`、`lexical`、`cross_encoder`、`api`、`llm` | 不精排 / 词法重叠 / 本地 CrossEncoder / 远程 Rerank API（如 SiliconFlow bge-reranker）/ LLM 打分 |
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
| `ask` | 单次问答，展示答案、证据片段（分数/来源/标题）与各阶段耗时 |
| `build` | 离线建库并持久化向量索引与 BM25 索引（只需跑一次） |
| `eval` | 在数据集上评测，支持 `--retrieval-only`、`--judge`、`--reuse-index`、`--workers`、`--out` |
| `ls` | 列出每一层所有已注册的可用实现 |

### 6. 辅助脚本（`scripts/`）

- `make_sample_data.py`：生成内置示例语料与问答集（内容就是 RAG 的核心概念，边评测边学习）。
- `prepare_dataset.py`：把 HotpotQA / SQuAD 结构 / BEIR 数据集转换成统一的 `corpus.jsonl + qa.jsonl`。
- `inspect_dataset.py`：评测前检查 `data/` 下数据集结构是否能被正确读取。
- `run_ablation.py`：消融实验，一次只改一个变量，结果以相对基线的增量呈现并连同完整配置落盘。
  内置套件：`chunk_size`、`channel`、`rerank`、`index`、`space`、`hnsw_build`、`hnsw_search`、`top_k`。
- `make_agentic_data.py`：生成 Agentic RAG 演示数据（带元数据的文件、SQLite 订单库、标注了 `expected_tools` 的问答集），为后续“自主决策”模式做准备。

### 7. 预置实验配置（`configs/`）

| 文件 | 用途 |
| --- | --- |
| `default.yaml` | 完全离线基线：递归切分 + Hashing 向量 + 混合检索 + 词法精排 + 抽取式生成 |
| `models_env.yaml` | 从环境变量读取真实的 Embedding / LLM / Rerank 模型 |
| `chroma.yaml` | Chroma 持久化向量库 |
| `ds_scifact.yaml` / `ds_fiqa.yaml` | BEIR 基准，对齐 nDCG@10 口径 |
| `ds_cmrc.yaml` | 中文 CMRC2018，可同时评检索与生成 |

### 8. 测试（`tests/test_core.py`）

基于 `unittest`，覆盖文本切词、切分器、Hashing 向量、检索指标公式（Recall / MRR / NDCG / F1 / ROUGE-L）、端到端管线、API 精排、Chroma 索引、数据集加载与 `.env` 加载等容易写错的“接口契约”。

---

## 整体架构

```
            ┌──────────────────── 离线（build） ────────────────────┐
Document ──▶ Chunker ──▶ TextEncoder ──▶ VectorIndex (flat / chroma)
                                    └──▶ BM25Index
            └──────────────── KnowledgeBase（门面） ─────────────────┘

            ┌──────────────────── 在线（ask） ──────────────────────┐
Question ──▶ QueryTransformer ──▶ Retriever (vector / bm25 / hybrid-RRF)
         ──▶ Reranker ──▶ Generator (+LLM) ──▶ RagResult（答案 + 证据 + 耗时）
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
│   ├── ingest/        # 数据源 + 切分器
│   ├── embedding/     # 文本向量化
│   ├── store/         # 向量索引（flat / chroma）、BM25、KnowledgeBase
│   ├── retrieval/     # 召回、查询改写、精排
│   ├── generation/    # LLM 客户端 + 生成策略
│   ├── pipeline/      # RagPipeline 端到端编排
│   ├── eval/          # 数据集适配、指标、评测执行器
│   └── cli.py         # 命令行入口
├── configs/           # YAML 实验配置
├── scripts/           # 数据准备、数据检查、消融实验脚本
├── tests/             # 单元测试
└── env.example.txt    # 模型/密钥配置模板
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install pyyaml numpy
# 可选
pip install python-dotenv           # 自动加载 .env
pip install chromadb                # Chroma 持久化向量库
pip install sentence-transformers   # 本地 embedding / CrossEncoder 精排
```

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

- 项目由 `minirag` 重命名为 `learn_rag`，部分位置仍残留旧包名：
  `scripts/run_ablation.py`、`scripts/prepare_dataset.py` 中的 `from minirag...` 导入，`tests/test_core.py` 中的 `__import__("minirag.core.types")`，以及 CLI 文档字符串中的示例命令，需要统一改为 `learn_rag`。
- `learn_rag/eval/runner.py` 中存在未使用的 `from streamlit import progress`，会导致未安装 streamlit 时评测模块无法导入，应当删除。
- 尚未提供 `requirements.txt` / `pyproject.toml`。
- CLI 已预留 `--mode agentic`（自主决策 Agent）与 `--mode wiki`（LLM Wiki 知识编译）两种系统形态，`make_agentic_data.py` 已能生成对应演示数据，但这两种模式的实现尚未加入，当前均回退为 `pipeline` 模式。
