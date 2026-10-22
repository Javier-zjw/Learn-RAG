"""
eval.regression —— 质量回归：把这次评测的汇总指标和基线比，掉得超过容差就算回归。

用法（CI 里或者改完代码后）：
    learn-rag eval ... --save-baseline tests/baselines/xxx.json      # 认可当前效果，记成基线（提交进仓库）
    learn-rag eval ... --baseline tests/baselines/xxx.json           # 比较，有回归时退出码为 1

只比较两边都有的质量指标（召回率、MRR、证据命中率、忠实度……这些都是越大越好）；延迟指标随机器负载波动，
不参与比较。缺了的指标不比也不报错：评测集没有标注某项时 runner 本来就会跳过它（见"缺什么跳什么"），
基线和当前缺的不一样只说明评测方式变了，会在结果里列出来提醒。
容差是绝对值（默认 0.02，即 2 个百分点）：评测集只有几十道题时，一道题就是几个百分点，容差要相应放宽。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 不参与比较的指标前缀：耗时随机器负载波动
_SKIP_PREFIXES = ("latency",)


@dataclass
class Comparison:
    metric: str
    baseline: float | None
    current: float | None
    delta: float | None
    status: str                 # ok / improved / regressed / missing（只有一边有）


def load_baseline(path: str | Path) -> dict[str, float]:
    """基线文件：--save-baseline 写的 {"summary": {...}}，也接受 --out 写的完整报告。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    summary = data.get("summary", data)
    return {k: float(v) for k, v in summary.items() if isinstance(v, (int, float))}


def save_baseline(path: str | Path, summary: dict[str, float], *, note: str = "") -> None:
    """只存质量指标，排好序、保留 4 位小数：基线要提交进仓库，diff 要好读。"""
    kept = {k: round(v, 4) for k, v in sorted(summary.items()) if not k.startswith(_SKIP_PREFIXES)}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": note, "summary": kept}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compare(current: dict[str, float], baseline: dict[str, float], tolerance: float = 0.02) -> list[Comparison]:
    rows: list[Comparison] = []
    for metric in sorted(set(current) | set(baseline)):
        if metric.startswith(_SKIP_PREFIXES):
            continue
        now, before = current.get(metric), baseline.get(metric)
        if now is None or before is None:
            rows.append(Comparison(metric, before, now, None, "missing"))
            continue
        delta = now - before
        status = "regressed" if delta < -tolerance else "improved" if delta > tolerance else "ok"
        rows.append(Comparison(metric, before, now, delta, status))
    return rows


def regressions(rows: list[Comparison]) -> list[Comparison]:
    return [r for r in rows if r.status == "regressed"]


def to_markdown(rows: list[Comparison], tolerance: float) -> str:
    marks = {"ok": "", "improved": "↑ 提升", "regressed": "↓ 回归", "missing": "只有一边有"}
    lines = [f"| 指标 | 基线 | 本次 | 变化 | 结论（容差 {tolerance:g}） |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r.metric} | {_fmt(r.baseline)} | {_fmt(r.current)} | "
                     f"{'—' if r.delta is None else f'{r.delta:+.4f}'} | {marks[r.status]} |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{value:.4f}"
