"""
eval.runner —— 评测执行器（评测链路的门面）。

对使用者只有一个方法：
    report = Evaluator(pipeline, metrics).run(dataset)

内部替你处理了这些"每个人都会踩一遍"的细节：
  - 先用数据集自带语料建索引（若数据集没语料就复用现有知识库）
  - 缺 gold_doc_ids 时自动跳过检索类指标，缺标准答案时跳过生成类指标
    （否则会把一堆 0 分算进平均值，得出错误结论）
  - 单条样本失败不影响整体（记录错误，继续跑完）
  - 可选并发，加速 LLM 调用密集的评测
  - 输出可复现的明细 + 汇总
"""

from __future__ import annotations

import json
import logging
import statistics
import traceback
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


from ..core.interfaces import EvalDataset, Metric
from ..core.types import Answer, EvalSample, EvalCase, RagResult
from ..pipeline.rag import RagPipeline

logger = logging.getLogger(__name__)

@dataclass
class EvalReport:
    """
    评测结果：明细 + 汇总
    """

    dataset: str
    cases: list[EvalCase] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self, top_errors: int = 3) -> str:
        lines = [
            f"# 评测报告 · {self.dataset}",
            "",
            f"- 样本数：{len(self.cases)}　失败：{len(self.errors)}",
            "",
            "| 指标 | 数值 |",
            "| --- | --- |",
        ]

        for key in sorted(self.summary):
            lines.append(f"| {key} | {self.summary[key]:.4f} |")
        if self.errors[:top_errors]:
            lines += ["", "## 失败样本（截取）", ""]
            lines += [f"- {e}" for e in self.errors[:top_errors]]

        return "\n".join(lines)

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "dataset": self.dataset,
            "summary": self.summary,
            "config": self.config,
            "errors": self.errors,
            "cases": [
                {
                    "id": c.sample.sample_id,
                    "question": c.sample.question,
                    "gold": c.sample.answers,
                    "gold_doc_ids": c.sample.gold_doc_ids,
                    "prediction": c.result.answer.text,
                    "retrieved_doc_ids": c.result.retrieved_doc_ids,
                    "citations": c.result.answer.citations,
                    "scores": c.scores,
                }
                for c in self.cases
            ],
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

class Evaluator:
    def __init__(
            self,
            pipeline: RagPipeline,
            metrics: Sequence[Metric],
            *,
            retrieval_only: bool = False,
            workers: int = 1,
    ) -> None:
        self.pipeline = pipeline
        self.metrics = list(metrics)
        self.retrieval_only = retrieval_only
        self.workers = max(1, workers)


    def run(self, dataset: EvalDataset, *, build_index: bool = True, limit: int | None = None) -> EvalReport:
        if build_index:
            docs = list(dataset.corpus())
            if docs:
                stats = self.pipeline.index(docs, progress=True)
                # 通用渲染：不同系统形态返回的统计字段不同
                # （固定管线给 chunks，wiki 给 pages/links/conflicts）。
                # 评测器不该假设任何具体键名 —— 那是对 Solver 接口的越权假设。
                logger.info("[索引] %s", "　".join(f"{k}={v}" for k, v in stats.items()))

        samples = list(dataset.samples())
        if limit:
            samples = samples[:limit]
        logger.info("[评测] 数据集=%s 样本=%d 指标=%d", dataset.name, len(samples), len(self.metrics))

        report = EvalReport(dataset=dataset.name)
        if self.workers == 1:
            cases = [self._run_one(s, report) for s in samples]
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                cases = list(pool.map(lambda s: self._run_one(s, report), samples))
        report.cases = [c for c in cases if c is not None]
        report.summary = self._aggregate(report.cases)
        return report

    def _run_one(self, sample: EvalSample, report: EvalReport) -> EvalCase | None:
        try:
            result = (
                self.pipeline.retrieve_only(sample.question)
                if self.retrieval_only
                else self.pipeline.answer(sample.question)
            )
        except Exception as exc:
            report.errors.append(f"{sample.sample_id}: {exc}\n{traceback.format_exc(limit=2)}")
            result = RagResult(question=sample.question, answer=Answer(text=""))

        scores: dict[str, float] = {}
        for metric in self.metrics:
            # 关键：标注不全时跳过对应指标，而不是记 0 分。
            # 记 0 会让"数据集没标注"看起来像"系统效果差"，是评测里最常见的自欺。
            if metric.needs_gold_docs and not sample.gold_doc_ids:
                continue
            if metric.needs_gold_answer and not any(sample.answers):
                continue
            try:
                scores.update(metric.compute(sample, result))
            except Exception as exc:
                report.errors.append(f"{sample.sample_id} · {metric.name}: {exc}")

        return EvalCase(sample=sample, result=result, scores=scores)

    @staticmethod
    def _aggregate(cases: Iterable[EvalCase]) -> dict[str, float]:
        """
        按指标名取平均；每个指标只在"实际计算过它"的样本上平均
        """
        buckets: dict[str, list[float]] = {}
        for case in cases:
            for key, value in case.scores.items():
                buckets.setdefault(key, []).append(value)
        return {k: statistics.fmean(v) for k, v in sorted(buckets.items()) if v}
