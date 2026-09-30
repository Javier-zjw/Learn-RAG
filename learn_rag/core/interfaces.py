"""
core.interfaces —— 全系统的抽象接口（"窄接口 + 深实现"）。

设计要点：
1. **接口要窄**：每个抽象类原则上只有 1 个核心方法。调用方需要记住的东西越少，
   认知负担越低；实现方可以在方法背后藏任意复杂度（缓存、批处理、重试、降级）。
2. **接口要通用**：接口签名里不出现任何具体技术名词
   （没有 faiss_index、没有 openai_client、没有 hotpotqa_field），
   只出现领域概念（文本、向量、片段、分数、样本）。
   这样"换实现"只是换配置，不用改上层。
3. **接口按角色划分，而不是按技术划分**：
   BM25 与向量检索技术上天差地别，但对上层都是 Retriever，
   于是 HybridRetriever 可以用同一个接口把它们组合起来（组合优于继承）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .types import (
    Answer,
    Chunk,
    Document,
    Element,
    EvalSample,
    RagResult,
    ScoredChunk,
)

Vector = Sequence[float]

# 索引链路（离线
class DocumentSource(ABC):
    """
    数据源：把任意来源（目录、jsonl、数据库、HTTP）统一成 Document 流

    返回迭代器而不是 list，是为了让上层能流式处理超大语料而不爆内存
    """

    @abstractmethod
    def load(self) -> Iterable[Document]: ...

class DocumentParser(ABC):
    """
    文档解析器：一个文件 -> 有序的结构元素列表

    只做"读懂文件"这一件事。doc_id、元数据、缓存、失败降级都由调用方（FileSource）统一处理，
    因此接入一个新工具只需要把它的输出翻译成 Element。
    """

    @abstractmethod
    def parse(self, path: Path) -> list[Element]: ...

class Chunker(ABC):
    """
    切分器：Document -> 若干 Chunk

    接口只有一个方法，切分策略（定长 / 递归 / 语义 / 层次）全部藏在实现里
    """

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]: ...

class TextEncoder(ABC):
    """
    文本向量化

    只暴露"批量编码"一个方法：批量是性能关键路径，如果暴露单条接口，调用方迟早会写出 for 循环逐条请求的低效代码。
    单条需求由基类提供的 encode_one 兜底（默认实现，不算新接口负担）
    """

    dimension: int = 0

    # is_query 区分"查询"和"文档"：非对称模型（如 BGE/E5）需要不同前缀
    @abstractmethod
    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]: ...

    def encode_one(self, text: str, *, is_query: bool = False) -> list[float]:
        return self.encode([text], is_query=is_query)[0]

class VectorIndex(ABC):
    """
    向量索引：只管"存向量"和"按向量找最近邻"

    刻意不叫 FaissIndex/MilvusIndex —— 通用接口，实现可换
    """

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> None: ...

    # where 是可选的元数据过滤条件，实现可以忽略（返回全量近邻）
    @abstractmethod
    def search(self, vector: Vector, top_k: int, *, where: dict[str, Any] | None = None) -> list[ScoredChunk]: ...

    @abstractmethod
    def __len__(self) -> int: ...

    # 可选能力，默认不支持
    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError


# 检索链路（在线）
class Retriever(ABC):
    """
    检索器：问题 -> 排好序的候选片段

    全系统最重要的通用接口。向量检索、BM25、混合检索、带查询改写的检索器、甚至"调用远程搜索引擎"都实现它，
    因此它们可以互相嵌套、任意组合。
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int) -> list[ScoredChunk]: ...

class Reranker(ABC):
    """
    重排器：在候选集内部重新打分排序（精排）

    与 Retriever 分开，是因为二者的复杂度取舍不同：
    Retriever 面向"从百万里捞出百条"，Reranker 面向"把百条排准"
    """

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]: ...


class QueryTransformer(ABC):
    """
    查询改写：一个问题 -> 一个或多个检索用查询（HyDE、多路改写、去口语化）
    """

    @abstractmethod
    def transform(self, query: str) -> list[str]: ...


# 生成链路
class LLM(ABC):
    """
    大模型的最小接口

    只有 chat 一个方法，messages 采用 OpenAI 的 role/content 结构 —— 这是事实标准，用它可以最大化兼容性，同时不引入任何 SDK 依赖
    """

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **options: Any) -> str: ...

    def ask(self, prompt: str, system: str | None = None, **options: Any) -> str:
        msgs: list[dict[str, str]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return self.chat(msgs, **options)

class Generator(ABC):
    """
    生成器：问题 + 证据 -> 答案
    """

    @abstractmethod
    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer: ...


# 评测链路
class EvalDataset(ABC):
    """
    评测数据集的通用接口

    一个数据集要回答两个问题：
      - corpus()  : 要建索引的语料是什么（有些数据集自带语料，有些复用现成知识库）
      - samples() : 要问的问题和标准答案是什么
    通过这两个方法，HotpotQA、CMRC、自建业务集全部长一个样，
    评测器无需为任何数据集写 if-else。
    """

    name: str = "dataset"

    @abstractmethod
    def corpus(self) -> Iterable[Document]: ...

    @abstractmethod
    def samples(self) -> Iterable[EvalSample]: ...

class Metric(ABC):
    """
    指标的通用接口：给定(样本, 系统输出) -> 一组标量分数

    返回 dict 而不是单个 float，是因为很多指标天然成组产出
    （如 Precision/Recall/F1、多个 k 值），拆成多个类反而啰嗦。
    needs_gold_docs=True 的指标在缺少证据标注时会被自动跳过。
    """

    name: str = "metric"
    needs_gold_docs: bool = False
    needs_gold_answer: bool = False

    @abstractmethod
    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]: ...














