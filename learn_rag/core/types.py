"""
core.types —— 全系统唯一的数据契约层。

设计哲学（《软件设计哲学》Ousterhout）：
1. 所有跨层传递的数据结构集中定义在这里，任何一层只依赖"数据形状"，
   而不依赖其他层的实现。这样改检索器不会波及生成器。
2. 数据类保持"薄"，行为放到深模块里；但把**通用**的小工具（如 Chunk.of）
   收进来，避免调用方到处写重复样板代码。
3. 一个概念只有一种表示：检索结果统一是 ScoredChunk，
   向量检索 / BM25 / 混合检索 / 重排全部复用它，避免"格式转换税"。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

# 1. 索引侧（离线）数据结构
@dataclass
class Element:
    """
    文档解析出的最小结构单元 —— 所有解析器（原生库 / MinerU / Docling / OCR）的统一输出。

    kind 只有少数几种取值，上层按 kind 决定怎么切分，不关心它来自哪个解析器：
        heading  标题（level 表示层级 1~6）
        text     段落、列表项
        table    表格（text 为 Markdown 表格）
        image    图片（text 为图注或模型生成的描述）
        formula  公式（text 为 LaTeX）
        code     代码块
    页眉、页脚、页码等噪声由解析器直接丢弃，不进入这里。
    """

    kind: str
    text: str
    level: int = 0
    page: int | None = None
    bbox: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        if self.kind == "heading":
            return "#" * max(1, min(self.level, 6)) + " " + self.text
        if self.kind == "formula":
            return f"$$\n{self.text}\n$$"
        if self.kind == "code":
            return f"```\n{self.text}\n```"
        return self.text


@dataclass
class Document:
    """
    一篇原始文档（未切分）,doc_id 必须在语料库内唯一

    elements 是解析得到的结构（可为空，如 jsonl 语料）；有结构时 text 由它渲染而来，
    因此只认 text 的旧组件（递归切分、BM25）照常工作，认结构的组件（结构切分）能拿到更多信息。
    """

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    elements: list[Element] = field(default_factory=list)

    @classmethod
    def from_elements(cls, doc_id: str, elements: list[Element], metadata: dict[str, Any] | None = None) -> "Document":
        text = "\n\n".join(e.to_markdown() for e in elements if e.text.strip())
        return cls(doc_id=doc_id, text=text, metadata=dict(metadata or {}), elements=list(elements))

    @property
    def title(self) -> str:
        return str(self.metadata.get("title", self.doc_id))

@dataclass
class Chunk:
    """
    文档切分后的最小检索单元.
    chunk_id 由 doc_id + 序号 + 内容摘要派生，保证可复现
    同样的语料重建索引，id 不变 —— 这对评测结果的可比性很重要
    """

    chunk_id: str
    doc_id: str
    text: str
    position: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, doc: Document, text: str, position: int) -> "Chunk":
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
        return cls(
            chunk_id=f"{doc.doc_id}#{position}-{digest}",
            doc_id=doc.doc_id,
            text=text,
            position=position,
            metadata={**doc.metadata, "doc_id": doc.doc_id}
        )


# 2. 检索侧（在线）数据结构
@dataclass
class ScoredChunk:
    """
    带分数的检索结果
    score: 不强制归一化，但同一个 Retriever 内部必须保证"越大越相关"
    source: 标明结果来自哪个通道（vector / bm25 / hybrid / rerank），既方便排障，也是混合检索做融合时的依据。
    """

    chunk: Chunk
    score: float
    source: str = "unknown"
    debug: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def doc_id(self) -> str:
        return self.chunk.doc_id

@dataclass
class Answer:
    """生成层输出"""

    text: str
    # 用到的chunk_id
    citations: list[str] = field(default_factory=list)
    # 模型原始输出，便于排障
    raw: str = ""

@dataclass
class RagResult:
    """
    一次完整问答的全过程留痕 —— 评测层只依赖这一个结构
    把"答案 + 证据 + 耗时"打包在一起，是评测解耦的关键：
    评测器不需要知道内部用了几路召回、有没有重排
    """
    question: str
    answer: Answer
    contexts: list[ScoredChunk] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def context_texts(self) -> list[str]:
        return [c.text for c in self.contexts]

    @property
    def retrieved_doc_ids(self) -> list[str]:
        """按排序去重后的文档 id 列表 —— 检索指标基于文档粒度计算"""
        seen: list[str] = []
        for c in self.contexts:
            if c.doc_id not in seen:
                seen.append(c.doc_id)
        return seen


# 3. 评测侧数据结构
@dataclass
class EvalSample:
    """
    一条评测样本
    answers: 允许多个标准答案（HotpotQA / NQ 等常见做法），命中任一即可
    gold_doc_ids: 标准证据文档；没有标注的数据集留空，此时检索类指标自动跳过，只算生成类指标（见 eval/metrics.py）
    """

    sample_id: str
    question: str
    answers: list[str] = field(default_factory=list)
    gold_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return self.answers[0] if self.answers else ""

@dataclass
class EvalCase:
    """单条样本的评测明细：样本 + 系统输出 + 各指标得分"""

    sample: EvalSample
    result: RagResult
    scores: dict[str, float] = field(default_factory=dict)

class Timer:
    """
    极小的计时上下文管理器，把耗时写进 timings 字典
    用法：
        with Timer(timings, "retrieve"):
            ...
    """

    def __init__(self, sink: dict[str, float], key: str) -> None:
        self._sink = sink
        self._key = key

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._sink[self._key] = time.perf_counter() - self._t0
