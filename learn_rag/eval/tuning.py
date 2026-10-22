"""
eval.tuning —— 用评测集给混合召回调参：各路权重、RRF 常数 k、召回候选数，以及查询改写要不要开。

调参要在几百组参数上各评一遍，直接跑管线太慢（还要反复调精排模型）。这里利用两个事实把成本降到和评一次差不多：
  1. 每一路召回的结果和融合参数无关：每题每路只召回一次（取最大候选数需要的量），各组参数只截取前若干名；
  2. 逐个打分的精排器（交叉编码器、rerank 服务），每个候选的分数只取决于（问题, 候选）：
     每题对所有候选只打一次分，各组参数按缓存的分数排序。词重叠精排的分数掺了融合分，不能缓存，好在它很便宜。
融合用的是和线上同一个 rrf_fuse，所以模拟出来的排序和线上管线完全一致。

查询改写每个问题要调用大模型，不能这样缓存：在最好的融合参数上，用真实的检索流程分别评一遍"不改写""多查询""HyDE"。
"""

from __future__ import annotations

import itertools
import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..core.interfaces import LLM, Retriever
from ..core.types import Answer, EvalSample, RagResult, ScoredChunk
from ..pipeline.rag import RagPipeline
from ..retrieval.retrievers import HybridRetriever, TransformedRetriever, fetch_size, rrf_fuse
from .metrics import EvidenceAtK

logger = logging.getLogger(__name__)

# 默认网格。RRF 只看名次，权重整体放大缩小结果不变，所以权重组合会按比例去重
WEIGHT_STEPS = (0.0, 0.5, 1.0, 2.0)
RRF_KS = (10, 30, 60, 100)
CANDIDATE_KS = (20, 50, 100)


@dataclass
class TuneRow:
    config: dict[str, Any]          # {"weights": {...}, "rrf_k": int, "candidate_k": int}
    scores: dict[str, float]        # evidence_hit@k / evidence_recall@k / evidence_mrr@k 的平均值

    def key(self, k: int) -> tuple[float, float, float]:
        return (self.scores[f"evidence_hit@{k}"], self.scores[f"evidence_mrr@{k}"], self.scores[f"evidence_recall@{k}"])


def default_grid(channels: Sequence[str], top_k: int) -> list[dict[str, Any]]:
    """各路权重 × RRF 常数 × 候选数。权重按比例去重（1:1 和 2:2 是同一种排序），全为 0 的组合跳过。"""
    weight_sets: list[dict[str, float]] = []
    seen: set[tuple[float, ...]] = set()
    # 同一比例优先保留最大权重为 1 的写法（1:0.5 而不是 2:1），结果表更好读
    combos = sorted(itertools.product(WEIGHT_STEPS, repeat=len(channels)), key=lambda c: max(c) != 1.0)
    for combo in combos:
        top = max(combo)
        if not top:
            continue
        ratio = tuple(round(w / top, 3) for w in combo)
        if ratio not in seen:
            seen.add(ratio)
            weight_sets.append(dict(zip(channels, combo)))
    return [{"weights": w, "rrf_k": k, "candidate_k": c}
            for w in weight_sets for k in RRF_KS for c in CANDIDATE_KS if c >= top_k]


class FusionTuner:
    def __init__(self, pipeline: RagPipeline, channels: dict[str, Retriever]) -> None:
        """channels：参与调参的各路召回（名字 → 检索器），精排器和知识库沿用 pipeline 的。"""
        self.pipeline = pipeline
        self.channels = channels
        self.top_k = pipeline.top_k

    def run(self, samples: Sequence[EvalSample], grid: Sequence[dict[str, Any]] | None = None) -> list[TuneRow]:
        """评测每组参数，按 evidence_hit、evidence_mrr、evidence_recall 从高到低返回。只用标了证据的题。"""
        samples = [s for s in samples if any(s.evidence)]
        if not samples:
            raise ValueError("评测集里没有标注证据（evidence）的题，无法调参；用 learn-rag evalset 生成评测集")
        grid = list(grid or default_grid(list(self.channels), self.top_k))
        fetch = fetch_size(max(c["candidate_k"] for c in grid))
        logger.info("调参：%d 道题 × %d 组参数（每题每路召回 %d 个）", len(samples), len(grid), fetch)

        cached = [self._prepare(s, fetch) for s in samples]
        metric = EvidenceAtK(self.top_k)
        rows = []
        for config in grid:
            per_sample = []
            for sample, (lists, scores) in zip(samples, cached):
                contexts = self._simulate(sample.question, lists, scores, config)
                result = RagResult(question=sample.question, answer=Answer(""), contexts=contexts)
                per_sample.append(metric.compute(sample, result))
            means = {key: statistics.fmean(s[key] for s in per_sample) for key in per_sample[0]}
            rows.append(TuneRow(config=config, scores=means))
        rows.sort(key=lambda r: r.key(self.top_k), reverse=True)
        return rows

    def _prepare(self, sample: EvalSample, fetch: int) -> tuple[dict[str, list[ScoredChunk]], dict[str, float] | None]:
        """每路召回一次；逐个打分的精排器对所有候选打一次分，记下分数。"""
        lists = {name: channel.retrieve(sample.question, fetch) for name, channel in self.channels.items()}
        reranker = self.pipeline.reranker
        if reranker is None or not reranker.pointwise:
            return lists, None
        pool: dict[str, ScoredChunk] = {}
        for hits in lists.values():
            for hit in hits:
                pool.setdefault(hit.chunk.chunk_id, hit)
        ranked = reranker.rerank(sample.question, list(pool.values()), len(pool))
        return lists, {hit.chunk.chunk_id: hit.score for hit in ranked}

    def _simulate(self, question: str, lists: dict[str, list[ScoredChunk]], scores: dict[str, float] | None,
                  config: dict[str, Any]) -> list[ScoredChunk]:
        """
        和 RagPipeline 的"召回 → 精排 → 展开"一步步对应：每路截取融合时会取的前若干名（HybridRetriever 的 fetch_size），
        RRF 融合取 candidate_k 个，按缓存的精排分数稳定排序（和精排器内部的 sorted 一样，同分保持融合顺序），最后展开取 top_k。
        """
        names = [n for n, w in config["weights"].items() if w > 0]
        size = config["candidate_k"]
        candidates = rrf_fuse([lists[n][: fetch_size(size)] for n in names],
                              [config["weights"][n] for n in names], config["rrf_k"], size)
        reranker = self.pipeline.reranker
        if scores is not None:
            ranked = sorted(candidates, key=lambda h: -scores.get(h.chunk.chunk_id, float("-inf")))
        elif reranker is not None:
            ranked = reranker.rerank(question, candidates, len(candidates))
        else:
            ranked = candidates
        return self.pipeline.kb.expand(ranked)[: self.top_k]

    def compare_transforms(self, samples: Sequence[EvalSample], config: dict[str, Any],
                           transforms: dict[str, Any]) -> dict[str, dict[str, float]]:
        """
        在给定的融合参数上，用真实检索流程分别评测各种查询改写（"none" 表示不改写）。
        transforms：名字 → QueryTransformer（不改写时为 None）。
        """
        samples = [s for s in samples if any(s.evidence)]
        names = [n for n, w in config["weights"].items() if w > 0]
        hybrid = HybridRetriever([self.channels[n] for n in names], [config["weights"][n] for n in names],
                                 rrf_k=config["rrf_k"])
        metric = EvidenceAtK(self.top_k)
        report = {}
        for name, transformer in transforms.items():
            retriever = hybrid if transformer is None else TransformedRetriever(hybrid, transformer)
            pipe = RagPipeline(kb=self.pipeline.kb, retriever=retriever, generator=self.pipeline.generator,
                               reranker=self.pipeline.reranker, top_k=self.top_k, candidate_k=config["candidate_k"])
            per_sample = [metric.compute(s, pipe.retrieve_only(s.question)) for s in samples]
            report[name] = {key: statistics.fmean(p[key] for p in per_sample) for key in per_sample[0]}
            logger.info("查询改写 %s：%s", name, report[name])
        return report


def build_transforms(names: Sequence[str], llm: LLM | None) -> dict[str, Any]:
    """名字 → QueryTransformer；"none" 对应 None（不改写）。"""
    from ..core.registry import registry

    out: dict[str, Any] = {"none": None}
    for name in names:
        if name == "none":
            continue
        if llm is None:
            raise ValueError(f"查询改写 {name} 需要大模型：在 .env 配置 LLM_BASE_URL、LLM_API_KEY")
        out[name] = registry.build("query_transformer", {"type": name}, llm=llm)
    return out
