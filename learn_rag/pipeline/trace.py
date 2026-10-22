"""
pipeline.trace —— 线上查询的 trace 与统计：每次问答记一行 JSON，按天分文件；统计 P50 / P95 延迟、空召回率、错误率。

评测集回答"改了之后变好还是变差"，trace 回答"线上现在怎么样"：哪一步慢、哪些问题什么都没找到。
被标成空召回的问题是补评测集、补文档最好的素材 —— 统计里会列出最近的几条。

空召回的判断（不依赖任何模型，所以换了 embedding、精排后仍可比）：
  - 没有返回任何片段；
  - 或者返回的片段和问题几乎没有共同的实词（中文按两字词、英文按单词）：问题里的实词被召回片段覆盖的比例
    低于 min_overlap。这是启发式的 —— 纯同义改写的提问也可能被误判，所以叫"疑似"，用来发现问题而不是打分。

不引入监控系统：按天分文件的 JSONL 用标准库就能读写，过期文件自动删除；要接 Prometheus / ELK 时直接采集这些文件。
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..core.text import index_tokens
from ..core.types import RagResult

logger = logging.getLogger(__name__)

_DAY = 86400


class QueryLog:
    def __init__(self, directory: str | Path, *, retention_days: int = 30, min_overlap: float = 0.1) -> None:
        self.directory = Path(directory)
        self.retention_days = retention_days
        self.min_overlap = min_overlap
        self._lock = threading.Lock()

    def record(self, question: str, result: RagResult | None = None, *, error: str = "",
               source: str = "chat", seconds: float | None = None) -> dict[str, Any]:
        """
        记一次查询。result 为空表示这次查询出错了（error 写原因）。返回写入的记录。
        写日志失败只记警告：监控不能拖垮问答本身。
        """
        entry: dict[str, Any] = {"ts": round(time.time(), 3), "source": source, "question": question[:500]}
        if result is not None:
            query = result.extras.get("standalone_question") or question
            overlap = _overlap(query, [c.chunk.index_text for c in result.contexts])
            entry.update(
                standalone=result.extras.get("standalone_question"),
                timings={k: round(v, 4) for k, v in result.timings.items()},
                contexts=len(result.contexts),
                top=[{"chunk": c.chunk.chunk_id, "score": round(c.score, 4)} for c in result.contexts[:5]],
                overlap=round(overlap, 3),
                empty=not result.contexts or overlap < self.min_overlap,
            )
        else:
            entry.update(error=error or "未知错误", timings={"total": round(seconds or 0.0, 4)})
        try:
            self._append(entry)
        except OSError as exc:
            logger.warning("写查询日志失败（不影响问答）：%s", exc)
        return entry

    def summary(self, *, days: float = 7, recent: int = 10) -> dict[str, Any]:
        """最近 days 天的统计：查询数、错误率、空召回率、总耗时和各阶段的 P50 / P95，最慢和疑似空召回的问题。"""
        since = time.time() - days * _DAY
        entries = [e for e in self._read(since) if e["ts"] >= since]
        ok = [e for e in entries if "error" not in e]
        empty = [e for e in ok if e.get("empty")]
        stages: dict[str, list[float]] = {}
        for e in ok:
            for stage, value in e.get("timings", {}).items():
                stages.setdefault(stage, []).append(value)
        return {
            "days": days,
            "queries": len(entries),
            "errors": len(entries) - len(ok),
            "error_rate": _rate(len(entries) - len(ok), len(entries)),
            "empty_rate": _rate(len(empty), len(ok)),
            "latency": {stage: {"p50": percentile(v, 50), "p95": percentile(v, 95), "max": max(v)}
                        for stage, v in sorted(stages.items())},
            "slowest": [{"question": e["question"], "seconds": e["timings"].get("total", 0), "ts": e["ts"]}
                        for e in sorted(ok, key=lambda e: -e["timings"].get("total", 0))[:recent]],
            "recent_empty": [{"question": e["question"], "overlap": e.get("overlap"), "ts": e["ts"]}
                             for e in sorted(empty, key=lambda e: -e["ts"])[:recent]],
            "recent_errors": [{"question": e["question"], "error": e["error"], "ts": e["ts"]}
                              for e in sorted(entries, key=lambda e: -e["ts"]) if "error" in e][:recent],
        }

    # ------------------------------------------------------------------ 文件
    def _append(self, entry: dict[str, Any]) -> None:
        """一次写一整行（追加模式），多个进程同时写同一个文件也不会交错。每天第一次写时清理过期文件。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"queries-{time.strftime('%Y%m%d', time.localtime(entry['ts']))}.jsonl"
        fresh = not path.exists()
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._lock, open(path, "a", encoding="utf-8") as f:
            f.write(line)
        if fresh:
            self._expire()

    def _expire(self) -> None:
        cutoff = time.strftime("%Y%m%d", time.localtime(time.time() - self.retention_days * _DAY))
        for path in self.directory.glob("queries-*.jsonl"):
            if path.stem.split("-", 1)[1] < cutoff:
                path.unlink(missing_ok=True)

    def _read(self, since: float) -> Iterator[dict[str, Any]]:
        first = time.strftime("%Y%m%d", time.localtime(since))
        for path in sorted(self.directory.glob("queries-*.jsonl")):
            if path.stem.split("-", 1)[1] < first:
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    yield json.loads(line)
                except ValueError:           # 进程被杀时可能留下半行
                    continue


def percentile(values: list[float], p: float) -> float:
    """最近秩法的百分位数：P95 就是排序后第 ceil(0.95n) 个值，一定是真实出现过的耗时。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _terms(text: str) -> set[str]:
    """实词：中文的两字词、英文单词和数字；单个汉字太常见，不算。"""
    return {t for t in index_tokens(text) if len(t) >= 2}


def _overlap(question: str, contexts: list[str]) -> float:
    """问题里的实词有多少出现在召回的片段里（所有片段合起来算）。问题没有实词时视为完全覆盖。"""
    terms = _terms(question)
    if not terms:
        return 1.0
    found: set[str] = set()
    for text in contexts:
        found |= terms & _terms(text)
    return len(found) / len(terms)
