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
learn-rag serve                                # 可视化建库页面：自动构建前端、随之启停本机 MinerU
cd web && npm run dev                          # 前端开发服务器，/api 转发到 learn-rag serve
```

修改代码后至少跑一遍单元测试，并用示例数据跑通一次 `ask` 或 `eval`。

## 目录结构

- `learn_rag/core/`：数据契约（`types.py`）、抽象接口（`interfaces.py`）、注册表、配置加载、中英文分词与 token 估算（`text.py`）、表格统一表示（`tables.py`）。不依赖任何上层模块。
- `learn_rag/ingest/`：数据源与切分器（`structure.py` 父子分块，`pieces.py` 切分单个超长元素）
- `learn_rag/parsing/`：文档解析（PDF / Office / 网页 / OCR / 外部工具适配器）与文件数据源
- `learn_rag/embedding/`：文本向量化
- `learn_rag/store/`：向量索引（flat / chroma）、BM25、`KnowledgeBase`（离线门面）
- `learn_rag/retrieval/`：召回、查询改写、精排
- `learn_rag/generation/`：LLM 客户端与生成策略
- `learn_rag/pipeline/rag.py`：`RagPipeline`（在线门面），`from_config` 是唯一的「配置 → 对象图」构建入口
- `learn_rag/eval/`：数据集适配、指标、`Evaluator`
- `learn_rag/server/`：可视化建库的后端（FastAPI），`library.py` 是门面，`app.py` 只暴露接口
- `web/`：可视化建库的前端（Vue 3 + Vite + TypeScript + Element Plus）
- `configs/`：YAML 配置，CLI 总是先加载 `default.yaml`，再按顺序叠加 `--config`
- `scripts/`：数据准备、数据检查、消融实验

## 设计哲学（《软件设计哲学》，所有新代码必须遵守）

- **接口简单，实现要深**：一个模块对外只暴露少量方法和参数，把复杂度（格式差异、重试、缓存、降级）藏在内部，调用方不需要了解细节。
- **按职责分模块**：一个文件只做一类事，模块之间只通过 `core` 中的数据结构和接口通信，不互相引用对方的内部实现。
- **简单直接，拒绝过度设计**：不为假想需求预留扩展点，不引入没必要的设计模式、元编程、多层继承或抽象工厂；能用一个函数解决的不写一个类，能用标准库解决的不引入新依赖。
- **代码要易读**：命名表达意图，函数短小，控制流平铺直叙；复杂逻辑用注释解释「为什么」，而不是复述代码。
- **消除特殊情况而不是堆 if**：优先通过统一数据表示（如统一的 `Element`）让上层没有分支，而不是在各处判断来源类型。
- **错误就近处理**：能在模块内部消化的错误（降级、重试、跳过）就不要抛给上层；必须抛出时要带上完整上下文和排查提示。

## 架构约定

- **新增组件 = 新类 + 注册**：实现对应抽象接口，用 `@registry.register("<namespace>", "<name>")` 注册，并确保所在模块会被 import（包的 `__init__.py` 或 `pipeline/rag.py` 的集中导入）。不要在 pipeline 或 CLI 里写 `if type == "xxx"` 分支。
- **接口保持窄**：每个抽象类原则上只有一个核心方法，复杂度藏在实现里。
- **跨层数据只用 `core/types.py` 中的结构**：检索结果统一为 `ScoredChunk`，一次问答统一为 `RagResult`。
- **依赖注入靠参数名**：`registry.build` 只会注入工厂签名里声明的同名参数（如 `llm`、`kb`、`inner`、`channels`）。
- **可选依赖延迟导入**：chromadb、sentence-transformers、python-dotenv 在用到时才 import，未安装时不能影响其他功能，报错需给出安装提示。
- **外部服务调用**要有超时、有限次重试和降级路径（参考 `ApiReranker` 失败时退回召回顺序），不能让一个组件失败拖垮整条链路。
- **评测不能自欺**：缺少标注的指标要跳过而不是记 0；持久化目录按 collection 隔离；BM25 等派生索引必须和片段库是同一批片段（从片段库重建，不单独落盘）。
- **配置**：密钥只通过环境变量 / `.env` 注入（YAML 中用 `${VAR}` 或 `${VAR:-默认值}`），绝不写进提交的配置文件。

## 文档解析模块（`learn_rag/parsing/`）

目标：把 PDF、Word、PPT、Excel、网页、扫描件统一解析为带结构的 `Document`，并按结构切分。

- **统一中间表示**：所有解析器只输出 `list[Element]`（`core/types.py`）。`Element` 记录类型（标题/段落/表格/图片/公式/代码）、Markdown 文本、标题层级、页码、坐标和原始附加信息。`Document.text` 由元素渲染成 Markdown，现有切分器与检索无需改动。
- **解析器接口只有一个方法**：`DocumentParser.parse(path) -> list[Element]`。每种工具（原生库、MinerU、Docling、VLM OCR）各写一个适配器，把自己的输出翻译成 `Element`，差异不外泄。
- **选择策略按文档类型**：Office 与网页直接读结构（不做 OCR）；有文字层的 PDF 用原生解析，页面缺少文字层时自动交给 OCR；扫描件与图片用 VLM OCR；需要高精度版面时可在配置中换成 MinerU / Docling。
- **分层落盘**：解析结果按「文件哈希 + 解析器」缓存为 JSON，改切分策略时不必重新 OCR；解析失败时按配置的备选解析器降级，单个文件失败不影响整批。
- **表格统一表示**：所有来源的表格都经 `core/tables.py` 的 `normalize_table` 处理，没有合并单元格的用 Markdown，有合并单元格的用紧凑 HTML。
- **外部工具的输出以真实样例为准**：改适配器前先看 `samples/parsing_results/*_raw/` 里的原始输出，改完用 `export_parsed.py --replay` 在真实输出上验证。
- 页眉页脚在解析层丢弃，切分层不再处理噪声。
- 重型依赖（pymupdf、python-docx、python-pptx、openpyxl、MinerU、Docling）一律延迟导入，测试中用 mock 或小样例文件，不依赖 GPU 与网络。

## 分块（`learn_rag/ingest/structure.py`）

- **父子分块**：子块（`chunk_size`）进向量索引和 BM25 负责召回，父块（`parent_size`）只存储，命中后由 `KnowledgeBase.expand` 换成父块交给大模型。子块用 `metadata.parent_id` 指向父块；`Chunker.split` 同时返回父块和子块，由知识库按"被引用即父块"区分，不给 `Chunk` 加类型字段。
- **沿结构切分**：父块不跨章（文档标题下的第一级标题），子块不跨小节；块首拼"文档标题 > 章节路径"；公式、图片紧跟前一个元素，冒号结尾的引导句带上后一个元素。
- **超长元素在 `pieces.py` 里切**：段落按句（长句再按逗号、顿号、冒号，最后才按字数硬切）、表格按行（每片带表题和表头，合并单元格先用 `flatten_table` 展开）、代码按行；公式和图片不切。新增元素类型时在这里补切法，不要在切分器里加分支。
- **句子重叠默认关闭**（`overlap_sentences`）：只作用于同一长段落切出的相邻子块，不跨元素，表格和父块不重叠；重叠不能让块超出预算。
- **大小按 token 估算**：一律用 `core/text.py` 的 `count_tokens`，不要用 `len()`；它向上取整，装箱时相加不会超出预算。
- **展开放在精排之后**：精排在短小的子块上做，再展开、去重、取 `top_k`；不生成父块的切分器不受影响，管线里不为它写分支。
- **版面位置**：块元数据 `regions` 是 `{page, bbox}` 列表（bbox 按页面宽高归一化到 [0,1]，左上角为原点），由 `_regions` 从元素的页码和坐标汇总，只收归一化坐标；元素被合并过时其余几块的位置在 `extra.boxes`，被切片时按 `extra.span` 截取。查看页的原文视图靠它定位扫描页上的分块。
- **图片资产**：块元数据 `assets` 是条目列表（`asset`、`kind`、`caption`、`page`、`bbox`、`mime`，没有的字段不写），由 `_describe` 从元素的 `extra.asset` 汇总，同一块内按路径去重。资产库目录的相对路径一律用 `parsing/assets.py` 的 `resolve_assets_dir` 按项目根解析。
- **元数据不能在存储层走样**：向量库只能存标量时，完整元数据序列化成 JSON 一起存、取回时还原（见 `chroma_index.py`），不要拼接或截断列表。
- 改切分规则后用 `export_parsed.py --replay` 重新生成 `samples/parsing_results/`，对照 `chunks.jsonl` 检查效果。

## 向量化与存储（`learn_rag/store/`）

- **片段库是唯一的数据来源**：`VectorIndex` 保存子块（向量 + 正文 + 元数据）和父块（`add` 不传向量，只存储）；BM25 倒排、父块表、文档清单由 `KnowledgeBase._sync` 从 `chunks()` 重建。不要再给派生数据单独落盘。
- **片段库接口**：`add`（按 `chunk_id` 覆盖写入）、`delete`、`chunks`、`search`、`__len__`。新的实现必须满足 `tests/test_store.py` 里 `_StoreCases` 的全部用例。
- **增量更新**：片段元数据里的 `doc_fingerprint`、`doc_chunks`、`embedding_model` 由知识库写入，用来跳过未变化的文档、发现写了一半的文档、拒绝混用模型。写入顺序固定为"子块 → 父块 → 删除旧版本多出的片段"，不能改成先删后写。
- **模型身份**：`TextEncoder.signature()` 必须包含所有会改变向量的参数（模型名、前缀、维度等）；新增 encoder 时要实现它。
- **Chroma 的坑**：每条记录都必须带向量，不传时它会调用内置模型下载并生成 384 维向量，所以父块放在 `<collection>-stored` 集合并写入一维占位向量；集合一律用 `embedding_function=None` 打开。
- 改动存储或向量化逻辑后，用 `learn-rag build` 连续运行两次（第二次应全部跳过），再运行 `learn-rag verify --sample <子块数>` 全量核对向量能否被正确检索。
- 后续改进方向和顺序记录在 README“向量化与 Chroma 存储 → 后续改进”，在本分支（`feature/embedding-store`）上按顺序推进。

## 召回（`learn_rag/retrieval/`）

- **过滤条件按查询传递**：`Retriever.retrieve(query, top_k, *, where=None)`。组合型检索器（混合、查询改写）必须把 `where` 原样传给每一路；新增召回通道必须实现过滤，语法与 `store/indexes.py` 的 `_match` 一致（等值或 in 列表，只能过滤标量字段）。
- **BM25 切词**：建索引用 `_terms`（通用切词 + 整体编号 + 编号碎片），查询用 `_query_terms`（有完整编号时以编号整体代替碎片）。改 BM25 切词不影响向量，已有索引打开时自动按新规则重建；不要改 `index_tokens`，它同时决定 Hashing 向量，改了必须重建索引。

## 可视化建库（`learn_rag/server/`、`web/`）

- **后端只编排、不重写建库逻辑**：解析走 `FileSource.load_file`，写入、增量、模型检查、核对都走 `KnowledgeBase`。`app.py` 只做参数解析和错误码翻译（`NotFound` → 404、`ValueError` → 400、`Busy` → 409），逻辑都放在 `Library`。
- **一个知识库一个目录，参数固定**：改参数走 `rebuild`，只有换 embedding 模型或向量库参数时才清空片段库。知识库 id 随机生成，不复用目录。
- **页面参数 → 配置的翻译只在一处**：`Library.validate` 和 `Library._config`。新增一个页面参数时同时改 `validate`、`environment.default_settings` 和前端 `SettingsForm.vue`。
- **推荐规则**在 `recommend.py`，每条推荐都要附一句理由；只从 `environment` 报告可用的选项里选，不可用的选项在前端置灰并说明原因。
- **一条命令启停全部**：`learn-rag serve` 负责构建前端（`server/frontend.py`）和启停 MinerU（`server/mineru_service.py`，复用 `scripts/start_mineru.sh` / `stop_mineru.sh`，不要在 Python 里重写启停逻辑）。MinerU 在后台启动，不能阻塞页面服务；退出路径（Ctrl+C、SIGTERM、关闭终端的 SIGHUP、atexit）都要能关掉它，`stop` 必须可以重复调用。
- **原文视图**（`server/preview.py`）：页面有文字层时用文字对齐定位分块，否则用块元数据 `regions`；Office 用 `parsing/legacy.py` 的 `convert` 转 PDF。页面图片、转换出的 PDF 和定位结果按文件内容哈希缓存在 `<数据目录>/preview/`。定位规则改动后用 `samples/` 里的 PDF、Word、PPT、Excel 核对覆盖率和未覆盖的文字是否都说得清原因。
- **密钥不出后端**：页面只看到“是否已配置”，地址和密钥由 encoder 从环境变量读取，不写进 `kb.json`。
- **前端约定**：请求都走 `src/api.ts`；颜色只用 `styles.css` 里的令牌（浅色、深色各一套）；分块底色要保证正文对比度不低于 12:1；动画要尊重 `prefers-reduced-motion`；`defineModel` 在同一轮里只赋值一次（连续赋值会读到旧值、互相覆盖）。
- 改动后跑 `python -m unittest tests.test_server`、`cd web && npm run build`（含类型检查），再用 `learn-rag serve` 在浏览器里走一遍“上传 → 推荐 → 建库 → 查看分块”。

## 代码风格

- 注释与 docstring 使用中文，重点解释「为什么这样设计」和容易踩的坑，与现有代码密度保持一致。
- 库代码使用 `logging.getLogger(__name__)` 输出日志，不用 `print`；`print` 只用于 CLI 与脚本面向用户的结果展示。
- 每个模块以 `from __future__ import annotations` 开头，使用 `list[str]`、`X | None` 等现代类型标注。
- 测试使用标准库 `unittest`，外部 HTTP 调用一律 mock，不依赖网络与真实 Key；可选依赖缺失时 `skipTest`。

## Git 提交规范

- 提交信息使用中文，**不超过 50 个字**，一行说清做了什么。
- 提交信息中**不要出现 "Claude Code" 字样**，也不要附加 Co-Authored-By、会话链接等署名行。
- 不要提交 `.env`、`data/`、`runs/`、`vector_store/`、`.idea/`、`.DS_Store`、`web/node_modules/`、`web/dist/` 等本地文件（已在 `.gitignore` 中）。
