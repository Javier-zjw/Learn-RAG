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
learn-rag tune --kb <id> --qa data/eval/<id>.jsonl   # 用评测集调融合参数，--apply 写回知识库
learn-rag chat --kb <id>                       # 多轮对话（追问补全需要 .env 配置大模型），每次问答记 trace
learn-rag stats --kb <id>                      # 运行监控：查询 P50 / P95、疑似空召回率、建库耗时
learn-rag worker                               # 建库 worker 进程（serve 默认会启动一个）
learn-rag eval ... --baseline tests/baselines/sample_eval.json   # 质量回归，有指标掉了超过容差时退出码 1
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

## 建库增强（`learn_rag/ingest/enrich.py`）

- **只增强检索，不改正文**：增强内容写在片段元数据 `enrich`（`ENRICH_KEY`）里。`context`、`description` 经 `Chunk.index_text` 拼在正文前进向量、BM25、稀疏索引；`questions`、`summary` 经 `Chunk.index_views` 变成视图记录。交给大模型、页面展示、原文对齐一律用 `chunk.text`，建索引一律用 `index_text`，不要混用。
- **视图记录**：由 `KnowledgeBase` 生成（`_views`），id 为 `<片段 id>@<类型><序号>`，`view_of` 指向所属片段。正文是所属片段的正文，向量来自 `view_text`；只进向量索引，不进 BM25 / 稀疏索引；`expand` 先换回所属片段再换父块；`chunks_of`、`stats()["chunks"]` 不含视图。新增"粒度"时只改 `index_views`，不要在知识库里加分支。
- **成本控制**：每个子块只调用一次大模型（各字段合在一个 JSON 里），结果按"任务 + 模型 + 输入"缓存；提示词或输出格式改了要把 `VERSION` 加一。单次失败记在 `enrich.failed`，不中断建库，`_is_current` 让这篇文档下次重写、只重试失败的调用。
- **指纹**：`ChunkEnricher.signature()` 进入文档指纹，要包含所有会改变增强结果的参数；不开增强时指纹不变。
- 增强需要真实大模型，`from_config` 遇到 `echo` 直接报错；测试用假 LLM（见 `tests/test_enrich.py`）。

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
- **稀疏通道**：只有 `supports_sparse` 的 encoder（`bge_m3`）能开。稀疏向量由 `KnowledgeBase` 写入子块元数据 `sparse`（`SPARSE_KEY`），稀疏倒排和 BM25 一样在 `_sync` 里从片段库重建，不单独落盘；新增能产出稀疏向量的 encoder 时实现 `encode_sparse`，并让 `signature()` 体现这一点。
- **RRF 只有一份**：混合召回和调参工具都用 `retrievers.py` 的 `rrf_fuse` 与 `fetch_size`，改融合公式只改这里，否则 `learn-rag tune` 模拟出的排序和线上对不上（`tests/test_tuning.py` 会失败）。
- **精排器的 `pointwise`**：分数只取决于（问题, 片段正文）时设为 True，调参时每题只打一次分；分数掺了召回分（词重叠）或逐批打分（大模型）的保持 False。
- **多轮问题补全**（`condense.py`）：`answer` / `retrieve_only` 带 `history` 时先补全再检索和生成，只在 `pipeline.condense: true` 且有真实大模型时启用（`echo` 补不了）；补全失败一律退回原问题。

## 可视化建库（`learn_rag/server/`、`web/`）

- **后端只编排、不重写建库逻辑**：解析走 `FileSource.load_file`，写入、增量、模型检查、核对都走 `KnowledgeBase`。`app.py` 只做参数解析和错误码翻译（`NotFound` → 404、`ValueError` → 400、`Busy` → 409），逻辑都放在 `Library`。
- **一个知识库一个目录，参数固定**：改参数走 `rebuild`，只有换 embedding 模型或向量库参数时才清空片段库。知识库 id 随机生成，不复用目录。
- **任务队列与 worker**（`server/jobs.py`、`server/worker.py`）：写知识库的操作（建库、追加、重建、删除文档）一律 `jobs.submit` 提交成任务，由 worker 调 `Library.run_job` 执行，页面服务进程不写片段库 —— 本地 Chroma 另一个进程写入的向量不会出现在本进程已打开的 HNSW 索引里。`Library` 默认不起 worker（命令行评测、调参打开它只为读，不能顺手执行队列里的任务），测试和单进程场景传 `worker=True`。
- **跨进程一致**：`kb.json` 的读改写都在 `_lock(kb_id)` 里（线程锁 + `fcntl` 文件锁，可重入）；片段库被写过后调 `_data_changed` 让 `revision` 加一，读的一方按"参数 + revision"缓存管线，变了就重新打开。不要绕过 `_data_changed` 直接写清单里的文件状态。
- **重试与中断**：文件级失败用 `FileProgress.retry` 标记能否重试（向量化失败、增强没完成可以，解析失败不行）；`run_job` 只处理状态为 `waiting` 的文件，所以重试和中断续做天然只做剩下的；在文件之间检查 `should_stop`，停止时抛 `Interrupted`。`ValueError` / `LookupError` 视为永久错误，不重试。
- **问答走 `Library.ask`**：线上问答（chat、以后的提问页）统一走它，记 trace（`pipeline/trace.py`）；评测、调参直接用 `pipeline()`，不记。页面服务进程里做向量检索前，要先解决上面说的 Chroma 跨进程缓存问题（第 7 阶段的提问页）。
- **建库增强**（`enrich`）属于建库参数，改了走 `rebuild`（不清空片段库，靠指纹逐篇重写）；结果缓存在项目的 `.cache/enrich`，各知识库共用；视觉模型的地址和密钥只在 `_config` 运行时从环境变量读取。
- **页面参数 → 配置的翻译只在一处**：`Library.validate` 和 `Library._config`。新增一个页面参数时同时改 `validate`、`environment.default_settings` 和前端 `SettingsForm.vue`。
- **推荐规则**在 `recommend.py`，每条推荐都要附一句理由；只从 `environment` 报告可用的选项里选，不可用的选项在前端置灰并说明原因。
- **一条命令启停全部**：`learn-rag serve` 负责构建前端（`server/frontend.py`）和启停 MinerU（`server/mineru_service.py`，复用 `scripts/start_mineru.sh` / `stop_mineru.sh`，不要在 Python 里重写启停逻辑）。MinerU 在后台启动，不能阻塞页面服务；退出路径（Ctrl+C、SIGTERM、关闭终端的 SIGHUP、atexit）都要能关掉它，`stop` 必须可以重复调用。
- **原文视图**（`server/preview.py`）：页面有文字层时用文字对齐定位分块，否则用块元数据 `regions`；Office 用 `parsing/legacy.py` 的 `convert` 转 PDF。页面图片、转换出的 PDF 和定位结果按文件内容哈希缓存在 `<数据目录>/preview/`。定位规则改动后用 `samples/` 里的 PDF、Word、PPT、Excel 核对覆盖率和未覆盖的文字是否都说得清原因。
- **本地模型**（`core/local_models.py`）：sentence-transformers 的模型一律通过 `shared_model` 加载 —— 第一次用到时才加载、进程内按模型名共享。不要在构造函数里加载：管线在建库、查看分块时也会被构建，用不到模型。精排和召回参数（`reranker`、`retrieval`）只影响查询，走 `set_retrieval` 直接生效；清单里缺这两项的老知识库由 `_with_defaults` 按升级前的行为补齐。
- **密钥不出后端**：页面只看到“是否已配置”，地址和密钥由 encoder 从环境变量读取，不写进 `kb.json`。
- **前端约定**：请求都走 `src/api.ts`；颜色只用 `styles.css` 里的令牌（浅色、深色各一套）；分块底色要保证正文对比度不低于 12:1；动画要尊重 `prefers-reduced-motion`；`defineModel` 在同一轮里只赋值一次（连续赋值会读到旧值、互相覆盖）。
- 改动后跑 `python -m unittest tests.test_server`、`cd web && npm run build`（含类型检查），再用 `learn-rag serve` 在浏览器里走一遍“上传 → 推荐 → 建库 → 查看分块”。

## 评测（`learn_rag/eval/`）

- **证据级优先**：评测集用 `evidence`（逐字摘录的原文）标注答案所在位置，不记分块 id —— 分块 id 随切分参数变化，记了就没法比较切分参数。证据是否被召回统一用 `core/text.py` 的 `coverage` 判断（只比较字母、数字和汉字）。
- **缺什么跳什么**：指标用 `needs_gold_docs`、`needs_gold_answer`、`needs_evidence`、`needs_generation` 声明依赖，缺标注或只评检索时由 runner 跳过，绝不记 0 分。新增指标时声明清楚。
- **出题要过检查**（`eval/synth.py`）：证据必须出现在原文里、问题不能依赖"本文/上文"等指代、去重；丢弃的原因要计数输出，不能悄悄丢掉。
- 页面上建的知识库用 `learn-rag evalset --kb <id>` 出题、`learn-rag eval --kb <id> --retrieval-only` 评测；改检索、切分、模型后用同一份评测集对比 `evidence_*` 指标。
- **质量回归**（`eval/regression.py`）：`--save-baseline` 只存质量指标（不含 latency），`--baseline` 比较时只看两边都有的指标、绝对容差。改了切分、检索、评测逻辑导致 `tests/baselines/sample_eval.json` 回归时，确认是预期的才用 `--save-baseline` 更新并说明原因；CI（`.github/workflows/ci.yml`）在每次推送时跑测试、示例评测回归和前端构建。
- **调参**（`eval/tuning.py`）：`learn-rag tune --kb <id> --qa <评测集>` 网格搜索融合权重、RRF 常数和候选数，`--transforms` 比较查询改写，`--apply` 写回知识库。召回结果和逐点精排分数按题缓存，模拟的"召回 → 融合 → 精排 → 展开"必须和 `RagPipeline._contexts` 一步步对应；改管线的这几步时同步改 `FusionTuner._simulate`。

## 代码风格

- 注释与 docstring 使用中文，重点解释「为什么这样设计」和容易踩的坑，与现有代码密度保持一致。
- 库代码使用 `logging.getLogger(__name__)` 输出日志，不用 `print`；`print` 只用于 CLI 与脚本面向用户的结果展示。
- 每个模块以 `from __future__ import annotations` 开头，使用 `list[str]`、`X | None` 等现代类型标注。
- 测试使用标准库 `unittest`，外部 HTTP 调用一律 mock，不依赖网络与真实 Key；可选依赖缺失时 `skipTest`。

## Git 提交规范

- 提交信息使用中文，**不超过 50 个字**，一行说清做了什么。
- 提交信息中**不要出现 "Claude Code" 字样**，也不要附加 Co-Authored-By、会话链接等署名行。
- 不要提交 `.env`、`data/`、`runs/`、`vector_store/`、`.idea/`、`.DS_Store`、`web/node_modules/`、`web/dist/` 等本地文件（已在 `.gitignore` 中）。
