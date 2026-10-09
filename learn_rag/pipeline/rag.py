"""
pipeline.rag —— 端到端 RAG 管线（系统级门面）。

对使用者，它只有两个方法：
    pipe.index(documents)      # 离线：建库
    pipe.answer("问题")         # 在线：问答
背后是完整的六段式流程：

    切分 → 向量化 → 索引 ──(离线)
    改写 → 召回 → 精排 → 父块展开 → 装配 → 生成 ──(在线)

这是"深类"的终极形态：接口两个方法，内部承载全部复杂度。
同时它**不自己实现任何算法**，只负责编排 —— 每一步都是可替换的组件。

from_config 把"配置 -> 对象图"的构建逻辑集中在一处；
其他地方永远不出现 `if type == "bm25"` 这类分支。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from ..core.interfaces import LLM, Generator, Reranker, Retriever, TextEncoder
from ..core.registry import registry
from ..core.types import Answer, Document, RagResult, ScoredChunk, Timer
from ..store.knowledge_base import KnowledgeBase

# 触发各实现模块的注册（import 副作用）。集中在这里，使用者只 import 本模块即可。
from .. import embedding, generation, ingest, parsing, retrieval, store

class RagPipeline:

    def __init__(
            self,
            kb: KnowledgeBase,
            retriever: Retriever,
            generator: Generator,
            reranker: Reranker | None = None,
            top_k: int = 5,
            candidate_k: int = 20
    ) -> None:
        self.kb = kb
        self.retriever = retriever
        self.generator = generator
        self.reranker = reranker
        self.top_k = top_k
        self.candidate_k = candidate_k

    def index(self, documents: Iterable[Document], *, progress: bool = False) -> dict[str, int]:
        return self.kb.add(documents, progress=progress)

    def answer(self, question: str, *, top_k: int | None = None) -> RagResult:
        """
        一次问答。返回值带齐了答案、证据、各阶段耗时 —— 评测与排障都靠它
        """
        timings: dict[str, float] = {}
        contexts = self._contexts(question, top_k or self.top_k, timings)

        with Timer(timings, "generate"):
            answer = self.generator.generate(question, contexts)

        timings["total"] = sum(timings.values())
        return RagResult(question=question, answer=answer, contexts=contexts, timings=timings)

    def retrieve_only(self, question: str, *, top_k: int | None = None) -> RagResult:
        """
        只跑检索，不调生成 —— 调检索参数时用它，能省掉 90% 的时间和费用
        """
        timings: dict[str, float] = {}
        contexts = self._contexts(question, top_k or self.top_k, timings)
        timings["total"] = sum(timings.values())
        return RagResult(question=question, answer=Answer(text=""), contexts=contexts, timings=timings)

    def _contexts(self, question: str, top_k: int, timings: dict[str, float]) -> list[ScoredChunk]:
        """
        召回 -> 精排 -> 父块展开。

        精排在子块上做：子块短而集中，交叉编码器打分更准，也不会被截断。
        精排保留全部候选的顺序再展开，因为几个子块可能属于同一个父块，展开去重后
        才能取够 top_k 个不同的上下文；没有父块时展开不改变任何东西。
        """
        with Timer(timings, "retrieve"):
            candidates: list[ScoredChunk] = self.retriever.retrieve(question, max(self.candidate_k, top_k))

        with Timer(timings, "rerank"):
            ranked = self.reranker.rerank(question, candidates, len(candidates)) if self.reranker else candidates

        with Timer(timings, "expand"):
            return self.kb.expand(ranked)[:top_k]

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "RagPipeline":
        encoder: TextEncoder = registry.build("encoder", cfg["encoder"])
        if cfg.get("encoder_cache"):
            encoder = registry.build("encoder", {"type": "cached", **cfg["encoder_cache"]}, inner=encoder)
        chunker = registry.build("chunker", cfg["chunker"])
        vector_index = registry.build("index", cfg.get("index") or {"type": "flat"})
        kb = KnowledgeBase(encoder=encoder, chunker=chunker, vector_index=vector_index)

        llm: LLM | None = registry.build("llm", cfg["llm"]) if cfg.get("llm") else None
        retriever = _build_retriever(cfg["retriever"], kb=kb, llm=llm)
        reranker = registry.build("reranker", cfg.get("reranker") or {"type": "identity"}, llm=llm)
        generator = registry.build("generator", cfg["generator"], llm=llm)

        pcfg = cfg.get("pipeline", {})
        return cls(
            kb=kb,
            retriever=retriever,
            generator=generator,
            reranker=reranker,
            top_k=pcfg.get("top_k", 5),
            candidate_k=pcfg.get("candidate_k", 20),
        )

def _build_retriever(spec: dict[str, Any], *, kb: KnowledgeBase, llm: LLM | None) -> Retriever:
    """
    递归构建检索器对象图（hybrid 的 channels、transformed 的 inner 都是检索器）。

    递归是必要的：检索器可以任意嵌套组合。
    这段代码是整个系统里唯一需要"理解组合结构"的地方，其余全部无感。
    """
    spec = dict(spec)
    kind = spec.get("type")
    if kind == "hybrid":
        channels: Sequence[dict[str, Any]] = spec.pop("channels", [])
        built = [_build_retriever(c, kb=kb, llm=llm) for c in channels]
        return registry.build("retriever", spec, channels=built)
    if kind == "transformed":
        inner = _build_retriever(spec.pop("inner"), kb=kb, llm=llm)
        transformer = registry.build("query_transformer", spec.pop("transformer"), llm=llm)
        return registry.build("retriever", spec, inner=inner, transformer=transformer)
    return registry.build("retriever", spec, kb=kb, llm=llm)
