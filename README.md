# Learn-RAG

一个用于**学习与实验检索增强生成（RAG）**的轻量级 Python 框架。

项目的目标不是再造一个“大而全”的 RAG 框架，而是把 RAG 的每一个环节拆成**可替换、可评测**的组件：
先用零依赖的离线实现把“建库 → 检索 → 生成 → 评测”全链路跑通，再逐个替换成真实模型，用指标量化每一步改动带来的收益。

- **跑起来**：`ask` 一条命令完成建索引 + 问答，并展示证据与各阶段耗时
- **量出来**：内置检索、生成、忠实度、延迟等十余项指标，支持 HotpotQA / CMRC2018 / BEIR 等公开数据集
- **换得动**：所有组件通过注册表 + YAML 配置装配，改实验只改配置，不改代码

---

## 分支说明（`feature/embedding-store`）

本分支基于 `feature/chunking`（父子分块），负责**切分后的向量化、存储与召回**，后续的向量化与检索优化都在本分支上进行。

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 存储与完整性 | 子块向量、正文、元数据和父块全部存入 Chroma（唯一数据来源）；BM25 与父块表从 Chroma 重建；增量更新（跳过未变化、替换修改过的、中断后续写）；拒绝混用模型；`verify` 结构检查与抽样核对向量 | 已完成 |
| 第 1 步 | BM25 编号整体成词，合同号、订单号、型号可精确检索 | 已完成 |
| 第 2 步 | 元数据过滤（`where`）作用于每一路召回，向量与 BM25 范围一致 | 已完成 |
| 第 3 步 | 接入 bge-m3：稠密向量存入 Chroma，稀疏向量存入元数据 | **后期优化路径** |
| 第 4 步 | 新增稀疏召回通道，与向量、BM25 三路 RRF 融合，同样支持 `where` | **后期优化路径** |
| 第 5、6 步 | ColBERT 精排；视语料规模评估 Milvus / Qdrant | 依赖第 3 步和评测数据，届时再评估 |

第 3、4 步是后期的优化路径，暂不实施：它们依赖 bge-m3 的部署方式（Jetson 上的服务接口，或本地用 FlagEmbedding 加载），
而且本地 Chroma 不支持稀疏向量索引，需要按“方案 A”在本项目内建稀疏倒排。当前分支的结构已为它们留好位置：
稀疏向量作为片段元数据存入 Chroma、打开时重建倒排（和 BM25 相同的方式），稀疏召回作为混合检索的一路新通道，
过滤条件按查询自动传入。具体方案、Chroma 的实测限制和取舍见
“[向量化与 Chroma 存储 → 后续改进：多种向量表示（bge-m3）](#后续改进多种向量表示bge-m3)”。

---

## 目录

- [分支说明](#分支说明featureembedding-store)
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
| 倒排索引 | `BM25Index` | 不单独落盘，打开知识库时从片段库重建，和向量索引始终是同一批片段；合同号、订单号、型号等编号整体成词，可精确命中；支持与向量索引相同的元数据过滤 |

`KnowledgeBase`（`store/knowledge_base.py`）作为离线门面，对外只暴露 `add / expand / verify / save / load / stats`，内部完成“比对文档指纹 → 切分 → 区分父子块 → 攒批向量化 → 写片段库 → 清理旧版本 → 维护 BM25”；子块进索引，父块只存储，`expand` 把命中的子块换成父块，`verify` 核对数据是否完整（见下文“向量化与 Chroma 存储”）。
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

代码分三个文件：`ingest/structure.py` 按结构划分章节、装父块和子块；`ingest/pieces.py` 切开单个超长元素；
`core/tables.py` 统一表格表示并展开合并单元格（解析层和切分层共用）。

##### 切分规则

| 规则 | 说明 |
| --- | --- |
| 章 | 文档标题之下的第一级标题算一章（PPT 的每一页就是一章）。父块不跨章，子块不跨小节 |
| 父块 | 同一章内按顺序把小节装进父块，装满换下一个；小节本身超长时按元素拆开。父块正文保留小节标题 |
| 子块 | 每个小节内按顺序装元素，装满换下一个；块首拼上“文档标题 > 章节路径”，片段脱离原文也知道自己在讲什么 |
| 不拆开的组合 | 公式、图片紧跟前一个元素（“按下式计算：”和公式，“如图 2 所示”和图片）；以冒号结尾的引导句带上后一个元素（列表、表格） |
| 超长段落 | 按句切开；句子本身超长时在逗号、顿号、冒号处断开，连分句都超长（OCR 结果、长网址）才按字数硬切。数字里的英文逗号（4,860）不算断点 |
| 句子重叠 | 默认关闭。`overlap_sentences: 1` 时，一个长段落被切成几个子块，下一块开头重复上一块的最后一句，弥补“它”“该方案”这类指代在切口处丢失；只作用于同一段落内的子块，不跨元素，表格和父块不重叠，重叠也不会让块超出预算 |
| 超长表格 | 按行切开，每块都带表题和表头，表注跟在最后一块；合并单元格先展开：跨行的值复制到每一行，多层表头合成一行（“上半年 Q1”），切出的每一行都是自包含的 |
| 代码、公式、图片 | 代码按行切，不把一行断开；公式和图片描述不切 |
| 去冗余 | 只有一个子块的父块不生成（和子块内容相同） |
| 纯文本 | 没有解析结构的文本（jsonl 语料）按空行分段后走同一套规则 |

块的大小按 token 估算（`core/text.py` 的 `count_tokens`）：中文每字约 1 个，英文每词约 1.3 个，中英文混排时块大小一致；只计正文，不计块首路径。估算值向上取整，装箱时相加不会超出预算。

##### 检索流程

召回和精排都在子块上做（短而集中，交叉编码器打分更准、不会被截断），之后 `KnowledgeBase.expand` 把子块换成父块：同一父块下命中的多个子块合并为一条，排在最靠前的子块的位置上，取够 `top_k` 个不同的上下文；命中了哪些子块记在 `debug["children"]` 中。切分器不生成父块时这一步不改变任何结果。

##### 图片、图表和表格：存储、切分与索引

| 阶段 | 做法 |
| --- | --- |
| 解析 | 图片、图表、表格截图的二进制按内容 SHA-256 存入资产库（`.cache/assets/<前两位>/<哈希>.<扩展名>`），同一张图只存一份；元素文字是图注和模型描述，没有时为 `[图片]`。原生图表（PPT / Excel / ODP / ODS）由 MinerU 转成数据表，作为表格元素 |
| 切分 | 图片不切开，和前一个元素放在同一个子块；原生图表的数据表按表格规则切分。块内每张图在元数据 `assets` 中记一条，超长表格切成几片时每片都带着同一张表格截图（同一块内只记一次） |
| 索引 | 向量和 BM25 索引的是块的文字（章节路径 + 图注 / 描述 + 前后正文），**图片本身不向量化** |
| 问答 | `ask` 在每条证据下列出页码、章节，以及块内每张图的图注和本地文件路径（在配置的 `assets_dir` 中查找）；交给大模型的只有文字 |

##### 块的元数据

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

| 字段 | 来源 | 说明 |
| --- | --- | --- |
| `title` `path` `file_type` `file_hash` `parser` `doc_id` | 文档 | 每个块都复制一份，溯源到文件，可按来源过滤 |
| `section` | 切分器 | 章节路径 |
| `kinds` | 切分器 | 块内的元素类型 |
| `page_start` / `page_end` | 切分器 | 页码范围（Office 文档没有页的概念，视解析器而定） |
| `assets` | 切分器 | 每张图一条：`asset` 资产库内相对路径；`kind` 为 `image`（图片、图表）或 `table`（表格截图）；`caption` 图注和模型描述（表格只取表题，`[图片]` 这类占位文字不算）；`page`、`bbox` 沿用解析器的页码和坐标（MinerU 为 0～1 的相对坐标）；`mime`。没有的字段不写 |
| `parent_id` | 切分器 | 只有子块有，指向父块 |

##### 持久化

- 子块（向量 + 正文 + 元数据）和父块都存在片段库里（Chroma 或 flat 索引），BM25 和父块表打开时从片段库重建，详见下一节“向量化与 Chroma 存储”。`build` 之后 `ask` 不必再传 `--docs`；`eval --reuse-index` 同样如此。
- flat 索引原样保存元数据；Chroma 的元数据只接受字符串和数字，标量字段原样存入供 `where` 过滤，完整元数据另存为一个 JSON 字段，取回时原样还原（不再把列表拼成字符串截断）。旧版本建的集合仍可读取。
- 资产库目录写成相对路径时一律按项目根目录解析（MinerU、Docling 和命令行用同一条规则），从哪个目录启动都能找到同一张图。

##### 配置

```yaml
chunker:
  type: structure
  chunk_size: 300        # 子块
  parent_size: 1200      # 父块；设为 0 关闭父块，只按小节切分
  overlap_sentences: 0   # 句子重叠，默认关闭
```

为什么默认不重叠：切口都落在元素或句子边界上，不会把一句话切成两半；被召回后交给大模型的是父块，切口两侧的内容都在里面；重叠还会让同一句话出现在两个子块里，一起被召回时白白占掉候选名额。重叠只解决“召回之前匹配不上”的问题，等评测发现子块边界导致漏召回时再打开。

`scripts/export_parsed.py` 导出的 `<文件名>.chunks.jsonl` 中，`role` 标明父块 / 子块，`tokens` 是估算的 token 数，可以直接检查切分效果（样例见 `samples/parsing_results/*/`）。

##### 分块功能清单

已实现：

| 功能 | 说明 |
| --- | --- |
| 结构感知切分 | 按标题划分章节，块首带“文档标题 > 章节路径” |
| 父子分块 | 子块召回、父块交给大模型；父块不跨章，子块不跨小节；精排后展开去重 |
| 按 token 控制大小 | 中英文混排时块大小一致 |
| 多级切分超长元素 | 段落：句子 → 逗号 / 顿号 / 冒号 → 字数；表格：按行；代码：按行 |
| 表格切分 | 每块带表题和表头，表注随最后一块；合并单元格展开、多层表头合并 |
| 内容不拆散 | 公式、图片和引导句在同一块；冒号引导句和后面的列表、表格在同一块 |
| 句子重叠（可选） | `overlap_sentences`，默认关闭 |
| 图片资产溯源 | 每张图记录路径、类型、图注、页码、坐标；问答时列出图注和本地文件 |
| 原生图表数据可检索 | PPT / Excel 图表的数据表按表格切分、索引 |
| 持久化 | 子块和父块都存在片段库（Chroma）里，元数据完整保存；BM25 和父块表从片段库重建；增量更新与完整性核对见“向量化与 Chroma 存储” |
| 纯文本兼容 | jsonl 语料按空行分段后走同一套规则 |

尚未实现：

| 功能 | 现状和影响 | 计划 |
| --- | --- | --- |
| 图片内容检索 | 图片只靠图注被检索；只有 `[图片]` 的图片和图片形式的图表，内容检索不到 | 建库时用视觉模型（PaddleOCR-VL、Qwen-VL 等）生成图片描述、从图表读出数据表，按图片哈希缓存 |
| 多模态生成 | 命中的块里有图时，交给大模型的只有文字 | 把原图一起发给支持图片输入的模型 |
| 多表示索引 / 上下文检索 | 每个子块只以原文一种形式被检索，问法和写法不一致时容易漏召回 | 为子块生成上下文说明、可回答的问题，为表格生成文字描述，作为指向同一父块的额外索引项 |
| 自适应展开（Auto-merging） | 命中子块后总是换成整个父块，可能带入无关内容 | 同一父块命中多个子块时才展开，否则只返回子块及其前后文 |
| 章 / 文档摘要索引 | 需要跨章节综合的问题（“总结全年经营情况”）只能靠零散的子块 | 按“章 → 文档”生成摘要并索引 |
| 无标题长文档的章节识别 | 解析器没识别出标题时（如扫描合同的“第一条”），只能按大小切分父块 | 把“第 X 条 / 第 X 章”等编号识别为标题 |
| 语义切分、Late Chunking | 未实现 | 暂不计划：解析结构比向量相似度更可靠；Late Chunking 需要能输出逐 token 向量的本地模型 |
| 原生解析器的图片 | 不经 MinerU / Docling 时（如 `.doc`、`.rtf` 转换后由 python-docx 解析）不提取图片，块里没有图片信息 | 需要图片时改用 MinerU 或 Docling |
| 分块效果评测集 | 还没有针对企业文档的问答标注，各种切分方案无法量化比较 | 基于 `samples/parsing/` 编写问题和出处标注，用 `run_ablation.py` 对比 |

#### 向量化与 Chroma 存储（`learn_rag/store/`）

切分后的子块交给 embedding 模型向量化，连同正文、元数据一起写入片段库；父块不向量化，只存储。
**片段库是知识库唯一的数据来源**：BM25 倒排、父块表、文档清单都是打开知识库时从片段库重建的派生数据，
不会出现“向量更新了、BM25 还是旧的”这类静默错位。

##### 存了什么、存在哪

| 位置 | 内容 |
| --- | --- |
| Chroma 集合 `<collection>` | 子块：`id`（`chunk_id`）、向量、正文（`documents`）、元数据。元数据里的标量字段（页码、章节、`doc_id` 等）单独存放供 `where` 过滤，完整元数据（含 `assets`、`kinds` 等列表）另存为一个 JSON 字段，读出时原样还原 |
| Chroma 集合 `<collection>-stored` | 父块：正文和元数据。Chroma 的每条记录都必须带向量，不传时会自动下载内置模型（约 80 MB）现场生成，维度还和我们的对不上，所以父块单独放一个集合、写入一维占位向量，这个集合从不参与检索 |
| `<index.path>/_kb_<collection>/meta.json` | 建库统计，仅供查看 |
| flat 索引（默认配置） | 数据在内存里，`save` 时写出 `vectors.npy` + `chunks.jsonl`（逐行对应）和 `stored.jsonl`（父块），每个文件先写临时文件再原子替换；读取时发现向量行数和片段数对不上会报错 |

知识库在每个片段的元数据里额外记录三项，用来保证完整性：

| 字段 | 含义 |
| --- | --- |
| `doc_fingerprint` | 所属文档版本的指纹：正文、结构元素、文档元数据、切分参数、embedding 模型的哈希 |
| `doc_chunks` | 所属文档的片段总数（子块 + 父块），用来发现写了一半的文档 |
| `embedding_model` | 生成向量的模型（只有子块有），如 `openai_compat:text-embedding-3-small` |

##### 写入流程

```
文档 ─▶ 计算指纹 ─▶ 库里已有同一版本且片段齐全？──是──▶ 跳过（不调用 embedding）
                              │否
                              ▼
        切分 ─▶ 攒批向量化 ─▶ 按 id 写入子块（带向量）─▶ 写入父块 ─▶ 删除旧版本多出来的片段 ─▶ 更新 BM25
```

写入顺序保证任何一步中断时数据都能恢复：新版本写完之前旧版本一直完整地在库里；
中断后最多多出几条旧片段，下次 `build` 时这篇文档因“指纹不一致”被识别出来，重新写入并清理。

##### 完整性保障

| 可能出现的问题 | 怎么保证 |
| --- | --- |
| 多次 `build`（不清空）后 BM25、父块和向量对不上 | BM25 和父块表从片段库重建，与向量永远是同一批片段 |
| 文档修改后，旧版本的片段残留在库里 | 按指纹识别新版本，写入后删除旧版本多出来的片段 |
| 文档没变也重复调用 embedding（费钱费时） | 指纹相同且片段齐全时跳过 |
| 建库中途被杀、网络中断 | 写入顺序保证旧版本不丢；下次 `build` 自动续写和清理，已写入的文档不再向量化 |
| embedding 服务报错 | 只跳过这一批文档（旧版本保留），计入 `failed`，其余文档照常写入；`build` 结束时以非零退出码提示 |
| 换了 embedding 模型（维度相同时向量库自己发现不了） | 子块记录模型身份（`TextEncoder.signature()`），打开时与当前 encoder 不一致直接报错；带缓存的 encoder 的缓存键也带上模型身份 |
| 坏向量（NaN、维度不一致） | 写入前检查，报错带排查提示 |
| 列表、嵌套元数据被截断或变成字符串 | 完整元数据以 JSON 保存，读出与写入完全一致 |
| 同一批里 id 重复、重复写入 | 按 `chunk_id` 覆盖写入，同一批重复的以最后一次为准 |

##### 核对数据：`verify`

`build` 结束时自动核对一次，也可以随时单独运行：

```bash
python -m learn_rag.cli verify --config configs/parsing.yaml --config configs/chroma.yaml            # 默认抽样 20 个子块
python -m learn_rag.cli verify --config configs/parsing.yaml --config configs/chroma.yaml --sample 0 # 只做结构检查
```

```
[完整性检查] 发现 1 个问题　documents=17　chunks=74　bm25_chunks=74　parents=9　抽样核对向量 20 个
  - 05_季度复盘.pptx：应有 [6] 个片段，实际 5 个，重新 build 会自动补齐
```

核对分两部分：

| 检查 | 内容 | 代价 |
| --- | --- | --- |
| 结构检查（总是做） | 每篇文档的片段是否齐全且属于同一个版本、子块的父块是否存在、有没有没人引用的父块、向量是否来自当前模型、可检索的片段数是否等于子块数 | 只读片段库 |
| 抽样核对向量（`--sample N`，默认 20；`build` 结束时抽 10 个） | 把抽到的子块正文重新编码，用这个向量去检索，子块自己应该排第一、相似度 ≥ 0.99。能发现结构检查发现不了的两类问题：向量和正文对不上（存错，或模型 / 编码方式变了但模型身份没变），以及索引查不出来（索引损坏） | 调用 N 次 embedding |

发现问题时退出码为 1，适合放进定时任务；大部分问题重新运行一次 `build` 就会自动修复。
旧版本程序建的集合（片段没有指纹）也会被指出来，同样 `build` 一次即可补齐。

##### 核对记录：样例文档的存储与检索

用 `samples/parsing/` 的样例在 Chroma 上建库（PDF、Office、图片用 MinerU 回放解析，其余用内置解析器；没有内置解析器的 5 种格式未入库，共 17 篇），
另写脚本直接读取 Chroma 的原始数据（不经过本项目的读取代码），和从源文档重新计算的结果逐条比对：

| # | 检查项 | 结果 |
| --- | --- | --- |
| 1 | 子块 id 与切分结果一致 | 通过（75 / 75） |
| 2 | 父块 id 与切分结果一致（存在 `-stored` 集合） | 通过（9 / 9） |
| 3 | 正文逐字一致 | 通过 |
| 4 | 元数据逐项一致，含 `assets` 等列表 | 通过 |
| 5 | 存储的向量与重新计算的向量一致 | 通过（最大误差 6e-8，512 维） |
| 6 | 向量已 L2 归一化 | 通过 |
| 7 | 每个子块用自己的向量检索排第一（HNSW 索引可用） | 通过（75 / 75） |
| 8 | BM25 用合同编号 `XH-2025-0386` 检索，第一条是合同 | 首次核对**未通过**（排第一的是销售数据表）；编号整体成词后通过，见“编号检索与元数据过滤” |
| 9 | 元数据过滤 `file_type=pdf` 只返回 PDF | 通过 |
| 10 | 子块展开成父块，父块包含子块正文 | 通过 |
| 11 | 完整检索流程（召回 → 精排 → 展开）用合同编号提问，第一条是合同 | 通过（向量召回和精排弥补了第 8 项） |
| 12 | `verify --sample 75`（全部子块） | 通过 |

另外验证了增量更新：原样重建时 17 篇全部跳过；修改一个 CSV 后只重写这一篇；在 Chroma 中删掉一个子块后，
`verify` 报出缺失并以退出码 1 结束，再 `build` 一次自动补齐。

##### 编号检索与元数据过滤

**编号整体成词。** 原来 BM25 把 `XH-2025-0386` 拆成 `xh`、`2025`、`0386` 三个词：销售数据的订单号（XH-1001…）和日期里
反复出现 `xh`、`2025`，合同片段又较长、受长度惩罚，结果三份销售数据都排在合同前面（核对记录第 8 项）。现在：

| | 做法 | 例子 |
| --- | --- | --- |
| 建索引 | 通用切词之外，再把编号整体作为一个词（`core/text.py` 的 `code_tokens`）；拆出的碎片也保留 | `合同编号:XH-2025-0386` → `xh`、`2025`、`0386`、`xh-2025-0386` 等 |
| 查询 | 查询里有完整编号时，用编号整体代替它的碎片——输入完整编号要的是精确匹配，碎片只是噪声 | `XH-2025-0386` → 只用 `xh-2025-0386` |
| 只记得一部分 | 查询里没有完整编号时照常按碎片匹配 | `0386` → 仍能找到合同 |

编号指含数字、并且含字母或连接符（`-` `_` `/` `.`）的片段：合同号、订单号、型号（`GPT4o`、`bge-m3`）、版本号（`v2.3.1`）、
日期（`2025-10-08`）、小数（`17.1`）。全角写法会先归一。这只改变 BM25 的切词，`index_tokens` 和 Hashing 向量不变，
而 BM25 打开时从片段库重建，所以**已有的索引不需要重建**。

**元数据过滤作用于每一路召回。** 过滤条件（`where`）是每次查询的参数，混合检索、查询改写检索会把它原样传给每一路，
向量和 BM25 两路看到的范围始终一致（原来只有向量那一路能过滤，BM25 会把范围之外的文档经 RRF 融合进结果）。

```bash
# 只在 PPT 里找；值为列表时表示任一匹配
python -m learn_rag.cli ask --config configs/parsing.yaml --config configs/chroma.yaml \
    -q "华东区第四季度营收" --where '{"file_type": "pptx"}'
python -m learn_rag.cli ask ... --where '{"file_type": ["xlsx", "csv"], "parser": "mineru"}'
```

```python
pipe.answer("华东区第四季度营收", where={"file_type": "pptx"})
pipe.retrieve_only("报销流程", where={"title": ["员工手册", "财务制度"]})
```

- 语法：`{"字段": 值}` 等值匹配，`{"字段": [值1, 值2]}` 任一匹配，多个字段同时满足。flat 索引、BM25 和 Chroma 三处语义一致。
- 只能过滤标量字段（`file_type`、`title`、`parser`、`doc_id`、`section`、`page_start` 等，以及自己加进文档元数据的部门、日期等）；
  `kinds`、`assets` 这类列表字段不能过滤。字段不存在时不匹配。
- 配置里也可以给某一路固定过滤条件（`{type: vector, where: {...}}`、`{type: bm25, where: {...}}`），和每次查询的条件合并，同名字段以查询为准。
- 过滤作用在子块上。按文档级字段（`file_type`、`title`、`doc_id` 等）过滤时，换成的父块一定在范围内；按 `section`、`page_start` 这类块级字段过滤时，父块可能还包含同一章的其他小节或页——要严格限定范围时，把父块关掉（`parent_size: 0`）或按文档级字段过滤。

##### 尚未实现

- **源文件删除后的同步**：从文档目录里删掉的文件，它的片段仍留在库里，需要清空索引重建（`index.reset: true`）。
- **改了切分代码本身**（而不是 `chunk_size` 等参数）时，指纹不会变，未变化的文档会被跳过，需要清空索引重建。
- **只有元数据变化也会重新向量化**：指纹包含文档元数据，比如用不同的路径写法指定同一个文档目录（`./docs` 和绝对路径），`path` 变了，所有文档都会重新写入；建库时请固定路径写法。
- **BM25 每次打开时重建**：片段数到几十万时启动会慢几秒到几十秒；需要时可以按片段库的版本缓存倒排表。
- Chroma 没有事务，完整性靠“写入顺序 + 指纹 + 核对”保证，而不是原子提交。

##### 后续改进：多种向量表示（bge-m3）

本分支后续在这里继续改进向量化与检索。下面是已经讨论过的方向。

**bge-m3 一次编码产出三种表示：**

| 表示 | 形式 | 擅长什么 | 每个子块的存储量（约） |
| --- | --- | --- | --- |
| 稠密向量 | 一个 1024 维向量 | 语义相近、换个说法也能找到 | 4 KB |
| 稀疏向量（词权重） | `{词: 权重}`，通常几十到一百多项 | 型号、编号、人名等关键词精确命中，相当于学出来的 BM25 | 1 KB 左右 |
| 多向量（ColBERT） | 每个 token 一个 1024 维向量 | 逐词精细比对，三者中最准 | 300 token 的块约 1.2 MB |

常见的 OpenAI 兼容 `/embeddings` 接口只返回稠密向量；要拿到另外两种，需要用 FlagEmbedding 在本地加载模型，
或者部署一个能返回三种表示的服务。

**本地 Chroma 只能原生存储和检索稠密向量。** 在当前安装的 Chroma 1.5.9 上实测：

| 测试 | 结果 |
| --- | --- |
| 本地创建稀疏向量索引 | 报错 `Sparse vector indexing is not enabled in local` |
| 本地调用混合检索接口 `collection.search()` | 报错 `Search is not implemented for Local Chroma` |
| 多向量（ColBERT） | Chroma 不支持 |

Chroma 的稀疏向量和混合检索（含 RRF 融合）只在 Chroma Cloud / 分布式版本中可用。

**方案 A（推荐）：继续用本地 Chroma，每种表示放在最合适的地方**

| 表示 | 存在哪 | 检索时怎么用 |
| --- | --- | --- |
| 稠密向量 | Chroma 的向量字段（和现在一样） | 向量召回 |
| 稀疏向量 | Chroma 的元数据（以 JSON 保存，和 `assets` 一样完整读回） | 打开知识库时在内存里建倒排，作为新的一路召回 |
| ColBERT 多向量 | 不存（10 万个子块就要 100 GB 以上） | 精排阶段对召回的 20～50 个候选现算逐词比对得分 |

- Chroma 仍是唯一的数据来源，稀疏倒排和 BM25 一样是打开时重建的派生数据，增量更新、指纹、`verify` 不用改。
- 混合检索本来就是多路 RRF 融合，加一路稀疏召回只是配置里多一行：
  `channels: [{type: vector}, {type: sparse}, {type: bm25}]`。
- ColBERT 和已有的交叉编码器精排（`api` 精排器接 `bge-reranker-v2-m3`）处在同一个位置，二选一还是串联，靠评测决定。

**方案 B：换成原生支持多种表示的向量库**

| 向量库 | 支持情况 |
| --- | --- |
| Milvus（2.4 起） | 同一集合存稠密和稀疏向量，自带混合检索和 RRF；官方集成 bge-m3 |
| Qdrant | 一条记录可存多个命名向量：稠密、稀疏、多向量（ColBERT 逐词比对），自带融合 |
| Chroma Cloud / 分布式版 | 支持稀疏向量和混合检索 |

架构上只需按片段库接口（`add` / `delete` / `chunks` / `search`）新写一个实现，并通过 `tests/test_store.py` 的通用用例，
上层不用改。适合语料到百万级或需要多副本服务时再换。

**元数据过滤要覆盖所有召回通道。** 过滤条件已经作为每次查询的参数传给每一路召回（见“编号检索与元数据过滤”），
新增的稀疏召回只要在 `retrieve` 里同样处理 `where`，就自动和其他两路一致。

**改进顺序：**

| 步骤 | 内容 | 前置条件 | 状态 |
| --- | --- | --- | --- |
| 1 | BM25 把编号整体作为一个词（修复核对记录第 8 项） | 无 | 已完成 |
| 2 | 元数据过滤作用于每一路召回（BM25 与向量一致） | 无 | 已完成 |
| 3 | 接入 bge-m3 编码器：稠密向量存入 Chroma，稀疏向量存入元数据 | 确定 bge-m3 的部署方式（Jetson 上的服务接口，或本地 FlagEmbedding） | 后期优化路径 |
| 4 | 新增稀疏召回通道，三路 RRF 融合，同样支持 `where` 过滤 | 步骤 3 | 后期优化路径 |
| 5 | ColBERT 精排，与交叉编码器精排对比 | 步骤 3、企业文档评测集 | 届时评估 |
| 6 | 视语料规模评估 Milvus / Qdrant | 评测与压测数据 | 届时评估 |

第 3、4 步实施时的要点（已在当前代码中预留，不需要改动现有接口）：

1. **编码器**：新增一个 `TextEncoder` 实现，`encode` 仍返回稠密向量；稀疏向量由同一次模型调用产出，
   通过知识库写入子块元数据（如 `sparse`，以 JSON 保存，读出与写入一致）。`signature()` 写明 `bge-m3` 及其参数，换模型时知识库会拒绝混用。
2. **稀疏召回**：新增 `sparse` 检索器，打开知识库时从片段库读出稀疏向量建倒排（与 BM25 一样是派生数据，不单独落盘）；
   `retrieve` 处理 `where`，语法与 `_match` 一致。
3. **配置**：`retriever.channels` 增加一行 `{type: sparse}`，与 `vector`、`bm25` 三路 RRF 融合。
4. **验证**：用 `test_store.py`、`test_retrieval.py` 的通用用例覆盖新通道（过滤一致性、增量更新、`verify`），
   并用 `build` 连续两次 + `verify --sample` 在样例上核对。

### 3. 在线检索与生成链路

| 环节 | 已实现组件 | 说明 |
| --- | --- | --- |
| 召回 `retriever` | `vector`、`bm25`、`hybrid`、`transformed` | 混合检索使用带权重的 **RRF** 融合；检索器可任意嵌套组合；每次查询可带元数据过滤 `where`，传给每一路召回 |
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
| `ask` | 单次问答，展示答案、证据片段（分数/来源/标题）、每条证据的出处（页码、章节、图片的图注和本地文件）与各阶段耗时；`--docs` 可直接解析文档目录，不传时打开 `build` 建好的知识库；`--where` 按元数据过滤（JSON） |
| `build` | 离线建库并持久化；可重复运行：未变化的文档跳过、修改过的文档替换旧版本、中断后续写；结束时自动核对完整性 |
| `verify` | 核对已建好的知识库是否完整（片段齐全、父块存在、模型一致），并抽样核对向量能被正确检索（`--sample N`）；有问题时退出码为 1 |
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

基于 `unittest`：`test_core.py` 覆盖文本切词、切分器、Hashing 向量、检索指标公式（Recall / MRR / NDCG / F1 / ROUGE-L）、端到端管线、API 精排、Chroma 索引、数据集加载与 `.env` 加载等容易写错的“接口契约”；`test_parsing.py` 覆盖各格式解析、MinerU / Docling / OCR 适配器（mock）、解析缓存与降级和端到端问答；`test_chunking.py` 覆盖父子分块的各条规则、长句按逗号切分与句子重叠、表格展开与按行切分、图片资产条目、知识库的父块存储、展开与持久化（含 Chroma 建库后重新加载）、问答出处展示；`test_store.py` 在 flat 和 Chroma 上分别验证片段库与知识库的数据完整性（按 id 覆盖、父块存储、删除、多次建库、文档修改、写入中断、embedding 失败、换模型、片段缺失、旧版本集合的修复）；`test_retrieval.py` 覆盖编号切词、用样例真实子块验证编号检索（含只搜编号的一部分）、过滤条件在混合检索和查询改写中逐路传递、flat 与 Chroma 上各路召回的过滤结果一致。样例文件在测试中现场生成。

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

# 离线建库并持久化（可重复运行，只处理新增和修改过的文档），之后问答不必再传 --docs
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
- 分块尚未实现的功能（图片内容检索、多模态生成、多表示索引、自适应展开、摘要索引等）见“结构感知的父子分块”一节末尾的功能清单。
- 向量化与存储的后续改进（编号检索、各通道统一过滤、bge-m3 稀疏 / 多向量）见“向量化与 Chroma 存储”一节的“后续改进”。

---

## 许可证

本项目基于 [MIT 协议](LICENSE) 开源。
