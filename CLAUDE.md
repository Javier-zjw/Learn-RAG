# CLAUDE.md

本文件为在本仓库中工作的 AI 编码助手提供项目约定。

## 项目概览

Learn-RAG：可替换、可评测的 RAG 学习与实验框架（Python ≥ 3.10，包名 `learn_rag`）。
每一层都通过「抽象接口 + 注册表 + YAML 配置」装配，默认配置完全离线可跑，无需 API Key。

## 常用命令

```bash
pip install -e ".[dev]"                        # 安装（含测试依赖）
python -m unittest discover tests              # 运行全部测试（也可用 pytest）
python scripts/make_sample_data.py             # 生成 data/ 下的示例语料与问答集
learn-rag ls                                   # 列出所有已注册组件
learn-rag ask --corpus data/sample_corpus.jsonl -q "什么是RRF?"
learn-rag eval --dataset jsonl --qa data/sample_qa.jsonl --corpus data/sample_corpus.jsonl
python scripts/run_ablation.py --suite channel --dataset jsonl \
    --qa data/sample_qa.jsonl --corpus data/sample_corpus.jsonl
```

修改代码后至少跑一遍单元测试，并用示例数据跑通一次 `ask` 或 `eval`。

## 目录结构

- `learn_rag/core/`：数据契约（`types.py`）、抽象接口（`interfaces.py`）、注册表、配置加载、中英文分词。不依赖任何上层模块。
- `learn_rag/ingest/`：数据源与切分器
- `learn_rag/embedding/`：文本向量化
- `learn_rag/store/`：向量索引（flat / chroma）、BM25、`KnowledgeBase`（离线门面）
- `learn_rag/retrieval/`：召回、查询改写、精排
- `learn_rag/generation/`：LLM 客户端与生成策略
- `learn_rag/pipeline/rag.py`：`RagPipeline`（在线门面），`from_config` 是唯一的「配置 → 对象图」构建入口
- `learn_rag/eval/`：数据集适配、指标、`Evaluator`
- `configs/`：YAML 配置，CLI 总是先加载 `default.yaml`，再按顺序叠加 `--config`
- `scripts/`：数据准备、数据检查、消融实验

## 架构约定

- **新增组件 = 新类 + 注册**：实现对应抽象接口，用 `@registry.register("<namespace>", "<name>")` 注册，并确保所在模块会被 import（包的 `__init__.py` 或 `pipeline/rag.py` 的集中导入）。不要在 pipeline 或 CLI 里写 `if type == "xxx"` 分支。
- **接口保持窄**：每个抽象类原则上只有一个核心方法，复杂度藏在实现里。
- **跨层数据只用 `core/types.py` 中的结构**：检索结果统一为 `ScoredChunk`，一次问答统一为 `RagResult`。
- **依赖注入靠参数名**：`registry.build` 只会注入工厂签名里声明的同名参数（如 `llm`、`kb`、`inner`、`channels`）。
- **可选依赖延迟导入**：chromadb、sentence-transformers、python-dotenv 在用到时才 import，未安装时不能影响其他功能，报错需给出安装提示。
- **外部服务调用**要有超时、有限次重试和降级路径（参考 `ApiReranker` 失败时退回召回顺序），不能让一个组件失败拖垮整条链路。
- **评测不能自欺**：缺少标注的指标要跳过而不是记 0；持久化目录按 collection 隔离；BM25 索引与向量索引必须同时落盘。
- **配置**：密钥只通过环境变量 / `.env` 注入（YAML 中用 `${VAR}` 或 `${VAR:-默认值}`），绝不写进提交的配置文件。

## 代码风格

- 注释与 docstring 使用中文，重点解释「为什么这样设计」和容易踩的坑，与现有代码密度保持一致。
- 库代码使用 `logging.getLogger(__name__)` 输出日志，不用 `print`；`print` 只用于 CLI 与脚本面向用户的结果展示。
- 每个模块以 `from __future__ import annotations` 开头，使用 `list[str]`、`X | None` 等现代类型标注。
- 测试使用标准库 `unittest`，外部 HTTP 调用一律 mock，不依赖网络与真实 Key；可选依赖缺失时 `skipTest`。

## Git 提交规范

- 提交信息使用中文，**不超过 50 个字**，一行说清做了什么。
- 提交信息中**不要出现 "Claude Code" 字样**，也不要附加 Co-Authored-By、会话链接等署名行。
- 不要提交 `.env`、`data/`、`runs/`、`vector_store/`、`.idea/`、`.DS_Store` 等本地文件（已在 `.gitignore` 中）。
