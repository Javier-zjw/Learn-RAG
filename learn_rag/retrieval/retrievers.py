"""
retrieval.retrievers —— 召回层。

四个实现全部满足同一个 Retriever 接口，因此可以像积木一样嵌套：

    HybridRetriever([
        MultiQueryRetriever(VectorRetriever(kb), llm),   # 向量 + 查询改写
        BM25Retriever(kb),                               # 关键词
    ])

这就是"通用接口"的威力：新增能力靠**组合**，而不是往一个大类里加 if。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from ..core.interfaces import LLM, QueryTransformer, Retriever
from ..core.registry import registry
from ..core.types import ScoredChunk
from ..store.knowledge_base import KnowledgeBase


@registry.register("retriever", "vector")
class VectorRetriever(Retriever):
    """
    稠密向量召回：把问题编码成向量，在向量空间找最近邻

    配置里的 where 是这一路固定的过滤条件，和每次查询传入的 where 合并（同名键以查询为准）。
    """

    def __init__(self, kb: KnowledgeBase, where: dict | None = None) -> None:
        self.kb = kb
        self.where = where

    def retrieve(self, query: str, top_k: int, *, where: dict[str, Any] | None = None) -> list[ScoredChunk]:
        vector = self.kb.encoder.encode_one(query, is_query=True)
        return self.kb.vector_index.search(vector, top_k, where=_merge(self.where, where))


@registry.register("retriever", "bm25")
class BM25Retriever(Retriever):
    """
    稀疏关键词召回。对专有名词、型号、编号、罕见词特别有效

    过滤语法和向量召回相同，配置里固定的 where 与查询传入的 where 合并。
    """

    def __init__(self, kb: KnowledgeBase, where: dict | None = None) -> None:
        self.kb = kb
        self.where = where

    def retrieve(self, query: str, top_k: int, *, where: dict[str, Any] | None = None) -> list[ScoredChunk]:
        return self.kb.bm25_index.search(query, top_k, where=_merge(self.where, where))


@registry.register("retriever", "hybrid")
class HybridRetriever(Retriever):
    """
    多路召回融合。

    默认用 RRF（Reciprocal Rank Fusion）：score = Σ w_i / (k + rank_i)。
    RRF 只看**名次**不看原始分数，因此天然免疫"向量分在 0~1、BM25 分在 0~30"
    这种量纲不可比的问题 —— 这是工程上最省心且很难被打败的融合方式。
    """

    def __init__(
            self,
            channels: Sequence[Retriever],
            weights: Sequence[float] | None = None,
            rrf_k: int = 60,
            overfetch: float = 2.0
    ) -> None:
        if not channels:
            raise ValueError("HybridRetriever 至少需要一路召回")
        self.channels = list(channels)
        self.weights = list(weights or [1.0] * len(self.channels))
        self.rrf_k = rrf_k
        self.overfetch = overfetch

    def retrieve(self, query: str, top_k: int, *, where: dict[str, Any] | None = None) -> list[ScoredChunk]:
        fetch = max(top_k, int(top_k * self.overfetch))
        fused: dict[str, float] = defaultdict(float)
        pool: dict[str, ScoredChunk] = {}
        debug: dict[str, dict[str, float]] = defaultdict(dict)

        for weight, channel in zip(self.weights, self.channels):
            for rank, hit in enumerate(channel.retrieve(query, fetch, where=where)):
                fused[hit.chunk.chunk_id] += weight / (self.rrf_k + rank + 1)
                pool.setdefault(hit.chunk.chunk_id, hit)
                debug[hit.chunk.chunk_id][hit.source] = hit.score

        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            ScoredChunk(chunk=pool[cid].chunk, score=score, source="hybrid", debug=dict(debug[cid]))
            for cid, score in ranked
        ]


@registry.register("retriever", "transformed")
class TransformedRetriever(Retriever):
    """
    给任意检索器叠加"查询改写"（装饰器模式）。

    一个问题被改写成多个查询分别召回，再按最高分合并去重。
    典型收益：口语化提问、指代不清、一句话含多个子问题的场景。
    """

    def __init__(self, inner: Retriever, transformer: QueryTransformer) -> None:
        self.inner = inner
        self.transformer = transformer

    def retrieve(self, query: str, top_k: int, *, where: dict[str, Any] | None = None) -> list[ScoredChunk]:
        best: dict[str, ScoredChunk] = {}
        for variant in self.transformer.transform(query):
            for hit in self.inner.retrieve(variant, top_k, where=where):
                prev = best.get(hit.chunk.chunk_id)
                if prev is None or hit.score > prev.score:
                    best[hit.chunk.chunk_id] = hit

        return sorted(best.values(), key=lambda h: -h.score)[:top_k]


@registry.register("query_transformer", "identity")
class IdentityTransformer(QueryTransformer):
    def transform(self, query: str) -> list[str]:
        return [query]


@registry.register("query_transformer", "multi_query")
class MultiQueryTransformer(QueryTransformer):
    """
    让 LLM 把问题改写成 n 个不同表述，缓解"用户措辞与文档措辞不一致"
    """
    PROMPT = "请把下面的问题改写成 {n} 个语义等价但用词不同的检索查询，每行一个，不要编号，不要解释。\n问题：{q}"

    def __int__(self, llm: LLM, n: int = 3) -> None:
        self.llm = llm
        self.n = n

    def transform(self, query: str) -> list[str]:
        try:
            text = self.llm.ask(self.PROMPT.format(n=self.n, q=query))
        except Exception:
            return [query]

        variants = [line.strip() for line in text.splitlines() if line.strip()]
        return [query, *variants[: self.n]]

@registry.register("query_transformer", "hyde")
class HydeTransformer(QueryTransformer):
    """
    HyDE：先让 LLM 假设性地"编"一段答案，再用这段答案去检索。

    原理：假想答案与真实文档在**表述风格**上更接近（都是陈述句、都含专业术语），
    因此比原始问句更容易在向量空间里命中正确文档。
    """

    PROMPT = "请针对下面的问题写一段大约 100 字的、像百科条目一样的假设性答案（不必保证正确）。\n问题：{q}"

    def __init__(self, llm: LLM, keep_original: bool = True) -> None:
        self.llm = llm
        self.keep_original = keep_original

    def transform(self, query: str) -> list[str]:
        try:
            doc = self.llm.ask(self.PROMPT.format(q=query)).strip()
        except Exception:
            return [query]

        return [query, doc] if self.keep_original else [doc]


def _merge(fixed: dict | None, per_query: dict | None) -> dict | None:
    """配置里固定的过滤条件和这次查询的条件合并，同名键以查询为准；都没有时返回 None（不过滤）。"""
    merged = {**(fixed or {}), **(per_query or {})}
    return merged or None
