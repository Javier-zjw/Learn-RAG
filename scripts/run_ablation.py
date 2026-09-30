"""
消融实验(Ablation Study):一次只改一个变量,量化每个组件值多少分。

为什么需要专门的脚本:
  手动跑十几次 `cli eval` 再肉眼比对,既容易漏记配置,也容易把两个变量一起改了
  然后得出错误结论。实验的可信度来自**纪律**,而纪律要靠工具来保证。

本脚本保证三件事:
  1. 每组实验只有一个变量不同,其余配置完全一致(从同一份 base 配置深度合并而来);
  2. 每组的完整配置都随结果一起落盘,结论可复现、可追溯;
  3. 结果以"相对第一组的增量"呈现 —— 你要看的是差值,不是绝对值。

用法:
    python scripts/run_ablation.py --suite chunk_size \
        --dataset jsonl --qa data/sample_qa.jsonl --corpus data/sample_corpus.jsonl
    python scripts/run_ablation.py --suite all --dataset hotpotqa \
        --qa data/hotpot_dev_distractor_v1.json --limit 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minirag.core.config import load_config
from minirag.eval.datasets import BeirStyleDataset, HotpotQADataset, JsonlQADataset, SquadStyleDataset
from minirag.eval.metrics import default_metrics
from minirag.eval.runner import Evaluator
from minirag.embedding.encoders import CachingEncoder
from minirag.pipeline.rag import RagPipeline

_ENCODER_CACHE: dict[str, CachingEncoder] = {}

# ---------------------------------------------------------------------------
# 实验套件：每个套件是"同一个维度上的若干取值"
# 键是这组实验的名字，值是要叠加到 base 配置上的 patch。
# 注意每个 patch 只动一个维度 —— 这是消融实验的全部纪律所在。
# ---------------------------------------------------------------------------
SUITES: dict[str, dict[str, dict[str, Any]]] = {
    # 切分粒度：太小丢上下文，太大稀释语义并引入噪声，存在最优点
    "chunk_size": {
        "chunk_150": {"chunker": {"chunk_size": 150, "chunk_overlap": 30}},
        "chunk_300": {"chunker": {"chunk_size": 300, "chunk_overlap": 60}},
        "chunk_600": {"chunker": {"chunk_size": 600, "chunk_overlap": 120}},
    },
    # 召回通道：验证"混合检索是否真的优于单通道"
    "channel": {
        "bm25_only": {"retriever": {"type": "bm25"}},
        "vector_only": {"retriever": {"type": "vector"}},
        "hybrid": {"retriever": {"type": "hybrid", "channels": [{"type": "vector"}, {"type": "bm25"}]}},
    },
    # 精排：预期是 Recall 不变而 MRR/NDCG 上升（精排改排序，不改召回）
    "rerank": {
        "no_rerank": {"reranker": {"type": "identity"}},
        "lexical": {"reranker": {"type": "lexical"}},
        # 需要 sentence-transformers，未安装时该组会被跳过并记录原因
        "cross_encoder": {"reranker": {"type": "cross_encoder", "model_name": "BAAI/bge-reranker-base"}},
    },
    # 索引实现：精确检索 vs 近似最近邻(ANN)
    # 【预期】小语料上两者指标应几乎相同；差距要在十万级以上才显现。
    # 如果小语料上 chroma 就明显低于 flat，说明参数配错了（多半是 space 或维度）。
    "index": {
        "flat_exact": {"index": {"type": "flat"}},
        "chroma_hnsw": {"index": {"type": "chroma", "path": "vector_store/ab_index",
                                  "collection": "ab_hnsw", "reset": True}},
    },
    # 距离度量：向量已 L2 归一化时三者排序等价，未归一化时差别很大。
    # 【怎么读】三行指标完全相同 → 说明你的 encoder 确实做了归一化（好事）；
    # 不同 → 去检查 encoder 的归一化逻辑。
    "space": {
        "cosine": {"index": {"type": "chroma", "path": "vector_store/ab_space",
                             "collection": "ab_cos", "space": "cosine", "reset": True}},
        "l2": {"index": {"type": "chroma", "path": "vector_store/ab_space",
                         "collection": "ab_l2", "space": "l2", "reset": True}},
        "ip": {"index": {"type": "chroma", "path": "vector_store/ab_space",
                         "collection": "ab_ip", "space": "ip", "reset": True}},
    },
    # HNSW 建图质量：ef_construction 与 M(max_neighbors)
    # 【预期】调大 → 召回升、建库变慢、内存变大。收益通常很快饱和。
    "hnsw_build": {
        "ef100_M16": {"index": {"type": "chroma", "path": "vector_store/ab_build",
                                "collection": "ab_b1", "ef_construction": 100,
                                "max_neighbors": 16, "reset": True}},
        "ef200_M32": {"index": {"type": "chroma", "path": "vector_store/ab_build",
                                "collection": "ab_b2", "ef_construction": 200,
                                "max_neighbors": 32, "reset": True}},
        "ef400_M64": {"index": {"type": "chroma", "path": "vector_store/ab_build",
                                "collection": "ab_b3", "ef_construction": 400,
                                "max_neighbors": 64, "reset": True}},
    },
    # HNSW 查询宽度：**唯一能在线调的参数**，做 recall/latency 权衡曲线就靠它
    "hnsw_search": {
        "ef_search_10": {"index": {"type": "chroma", "path": "vector_store/ab_search",
                                   "collection": "ab_s", "ef_search": 10, "reset": True}},
        "ef_search_50": {"index": {"type": "chroma", "path": "vector_store/ab_search",
                                   "collection": "ab_s2", "ef_search": 50, "reset": True}},
        "ef_search_200": {"index": {"type": "chroma", "path": "vector_store/ab_search",
                                    "collection": "ab_s3", "ef_search": 200, "reset": True}},
    },
    # top_k：Precision 与 Recall 的权衡点，同时观察延迟与忠实度
    "top_k": {
        "k3": {"pipeline": {"top_k": 3}},
        "k5": {"pipeline": {"top_k": 5}},
        "k10": {"pipeline": {"top_k": 10}},
    },
}

# 每个套件重点关注的指标（全部指标仍会落盘，这里只是让表格可读）
FOCUS = {
    "index": ["recall@k", "ndcg@k", "latency_retrieve"],
    "space": ["recall@k", "ndcg@k", "mrr@k"],
    "hnsw_build": ["recall@k", "ndcg@k", "latency_retrieve"],
    "hnsw_search": ["recall@k", "ndcg@k", "latency_retrieve"],
    "chunk_size": ["recall@k", "precision@k", "ndcg@k", "context_recall"],
    "channel": ["recall@k", "ndcg@k", "mrr@k", "latency_retrieve"],
    "rerank": ["recall@k", "mrr@k", "ndcg@k", "latency_rerank"],
    "top_k": ["recall@k", "precision@k", "faithfulness", "latency_total"],
}


def build_dataset(args: argparse.Namespace):
    if args.dataset == "jsonl":
        return JsonlQADataset(args.qa, args.corpus, limit=args.limit)
    if args.dataset == "hotpotqa":
        return HotpotQADataset(args.qa, limit=args.limit)
    if args.dataset == "squad_style":
        return SquadStyleDataset(args.qa, limit=args.limit)
    return BeirStyleDataset(args.qa, limit=args.limit)


INDEX_SUITES = {"index", "space", "hnsw_build", "hnsw_search"}


def run_variant(name: str, patch: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """跑一组实验。

    关键：**每组都重新构建 pipeline 和数据集**。
    复用同一个知识库会让 chunk_size 这类实验失效（索引还是上一组的），
    这是消融实验最常见的翻车点。
    """
    cfg = load_config(*args.config, **patch)
    try:
        pipe = RagPipeline.from_config(cfg)
    except Exception as exc:  # 缺依赖的组（如 cross_encoder）跳过而不是中断整批
        print(f"  [跳过] {name}: {exc}")
        return {"name": name, "skipped": str(exc), "config": cfg}

    # 共享编码缓存：同一 encoder 配置的各组实验复用向量。
    # 只有 encoder 或 chunker 变了，向量才需要重算；
    # index / rerank / top_k 这些维度的实验，向量完全相同。
    # FiQA 57K 篇上，不共享的话一个维度 3 组就要编码 17 万次。
    enc_key = json.dumps({"enc": cfg.get("encoder"), "chunk": cfg.get("chunker")}, sort_keys=True, default=str)
    if enc_key not in _ENCODER_CACHE:
        _ENCODER_CACHE[enc_key] = CachingEncoder(pipe.kb.encoder)
    pipe.kb.encoder = _ENCODER_CACHE[enc_key]

    dataset = build_dataset(args)
    k = cfg.get("pipeline", {}).get("top_k", 5)
    evaluator = Evaluator(pipe, default_metrics(k=k), retrieval_only=args.retrieval_only, workers=args.workers)
    report = evaluator.run(dataset)
    return {"name": name, "k": k, "summary": report.summary, "config": cfg}


def render(rows: list[dict[str, Any]], suite: str) -> str:
    """渲染对比表：绝对值 + 相对第一组的增量。

    看增量而不是绝对值，是因为绝对值受数据集难度影响很大，
    跨数据集不可比；而"混合检索比单通道高多少"是可以跨数据集讨论的。
    """
    ok_rows = [r for r in rows if "summary" in r]
    if not ok_rows:
        return "（全部实验组均被跳过）"

    focus = FOCUS.get(suite, [])
    # 指标名里带 @k，而各组 k 可能不同，这里按每组自己的 k 取值
    def value(row: dict[str, Any], metric: str) -> float | None:
        key = metric.replace("@k", f"@{row['k']}")
        return row["summary"].get(key)

    columns = [m for m in focus if any(value(r, m) is not None for r in ok_rows)]
    header = "| 实验组 | " + " | ".join(columns) + " |"
    sep = "| --- " * (len(columns) + 1) + "|"
    lines = [header, sep]
    base = ok_rows[0]
    for row in ok_rows:
        cells = []
        for metric in columns:
            v, b = value(row, metric), value(base, metric)
            if v is None:
                cells.append("-")
            elif row is base:
                cells.append(f"{v:.4f}")
            else:
                delta = v - (b or 0.0)
                cells.append(f"{v:.4f} ({delta:+.4f})")
        lines.append(f"| {row['name']} | " + " | ".join(cells) + " |")

    skipped = [r for r in rows if "skipped" in r]
    if skipped:
        lines += ["", "跳过的组：" + "；".join(f"{r['name']}（{r['skipped'][:60]}）" for r in skipped)]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", help=f"{'|'.join(SUITES)}|all")
    ap.add_argument("--config", action="append", default=None)
    ap.add_argument("--dataset", default="jsonl", choices=["jsonl", "hotpotqa", "squad_style", "beir_style"])
    ap.add_argument("--qa", required=True)
    ap.add_argument("--corpus")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--retrieval-only", action="store_true", help="只评检索，省时省钱（调检索时强烈建议开）")
    ap.add_argument("--out", default="runs/ablation")
    args = ap.parse_args()
    args.config = ["configs/default.yaml", *(args.config or [])]

    suites = list(SUITES) if args.suite == "all" else [args.suite]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = ["# 消融实验报告", ""]

    for suite in suites:
        print(f"\n{'='*60}\n实验维度：{suite}\n{'='*60}")
        # 索引类实验必须关精排：否则精排会重排召回结果，掩盖索引参数的差异
        extra = {"reranker": {"type": "identity"}} if suite in INDEX_SUITES else {}
        rows = [run_variant(name, {**extra, **patch}, args) for name, patch in SUITES[suite].items()]
        table = render(rows, suite)
        print("\n" + table)
        report_lines += [f"## 维度：{suite}", "", table, ""]
        (out_dir / f"{suite}.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告已保存：{out_dir/'report.md'}")


if __name__ == "__main__":
    main()
