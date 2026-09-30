"""
minirag.cli —— 命令行入口。

三个子命令对应 RAG 的三件事：
    ask   一次问答（含证据展示）——「跑起来」
    eval  在数据集上评测       ——「量出来」
    ls    查看有哪些可用实现   ——「换得动」

用法示例：
    python -m minirag.cli ask   --config configs/default.yaml --corpus data/sample_corpus.jsonl -q "什么是RRF?"
    python -m minirag.cli eval  --config configs/default.yaml --dataset jsonl \
        --qa data/sample_qa.jsonl --corpus data/sample_corpus.jsonl --out runs/base.json
    python -m minirag.cli eval  --config configs/hybrid_rerank.yaml --dataset hotpotqa \
        --qa data/hotpot_dev_distractor_v1.json --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core.config import load_config
from .core.registry import registry
from .eval.datasets import BeirStyleDataset, HotpotQADataset, JsonlQADataset, SquadStyleDataset
from .eval.metrics import LLMJudge, default_metrics
from .ingest.loaders import JsonlSource
from .pipeline.rag import RagPipeline


def _build_system(args: argparse.Namespace, cfg: dict):
    """按 --mode 决定构建"固定管线"还是"自主决策 agent"。

    两者都实现 Solver.answer()，所以后面的评测代码完全不用区分 —— 这正是
    通用接口的意义：新增一种系统形态，评测层零改动。
    """
    mode = getattr(args, "mode", "pipeline")
    # if mode == "agentic":
    #     return AgenticRag.from_config(cfg)
    # if mode == "wiki":
    #     return WikiSystem.from_config(cfg)
    return RagPipeline.from_config(cfg)


def _persist_dir(cfg: dict) -> str:
    """索引落盘目录：**必须按 collection 分开**。

    早期版本只用 index.path，不区分 collection。于是先建 SciFact 再建 FiQA 时，
    FiQA 的 bm25.pkl 会覆盖 SciFact 的 —— 之后复用 SciFact 加载到的是 FiQA 的
    倒排索引，指标全错，**而且不报错**。Chroma 自己按 collection 隔离了向量，
    但 BM25/元信息是我们自己存的，得自己隔离。
    """
    import os

    icfg = cfg.get("index", {}) or {}
    base = icfg.get("path") or "vector_store/default"
    collection = icfg.get("collection") or "default"
    return os.path.join(base, f"_kb_{collection}")


def _config_paths(args: argparse.Namespace) -> list[str]:
    """默认先加载 configs/default.yaml，再依次叠加用户指定的配置。"""
    return ["configs/default.yaml", *(args.config or [])]


def _build_dataset(args: argparse.Namespace):
    kind = args.dataset
    if kind == "jsonl":
        return JsonlQADataset(args.qa, args.corpus, limit=args.limit, name=args.name or "jsonl")
    if kind == "hotpotqa":
        return HotpotQADataset(args.qa, limit=args.limit)
    if kind == "squad_style":
        return SquadStyleDataset(args.qa, limit=args.limit)
    if kind == "beir_style":
        return BeirStyleDataset(args.qa, limit=args.limit)
    raise SystemExit(f"未知数据集类型：{kind}")


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = load_config(*_config_paths(args))
    pipe = _build_system(args, cfg)
    if args.corpus:
        stats = pipe.index(JsonlSource(args.corpus).load())
        print(f"[索引] {stats}")
    result = pipe.answer(args.question)
    if result.extras.get("mode") == "wiki":
        print(f"\n===== Wiki 状态 =====")
        print(f"  知识库共 {result.extras['wiki_pages_total']} 页，本次翻阅 {result.extras['pages_loaded']} 页，"
              f"上下文 {result.extras['context_chars']} 字")
    if result.extras.get("trajectory"):
        print("\n===== 决策轨迹 =====")
        for step in result.extras["trajectory"]:
            flag = "OK " if step["ok"] else "FAIL"
            print(f"  {step['step']}. [{flag}] {step['tool']}　←　{step['thought']}")
    print("\n===== 答案 =====")
    print(result.answer.text)
    print("\n===== 证据 =====")
    for i, ctx in enumerate(result.contexts, 1):
        title = ctx.chunk.metadata.get("title", ctx.doc_id)
        print(f"[{i}] ({ctx.score:.4f} · {ctx.source} · {title}) {ctx.text[:120]}...")
    print(f"\n耗时：{ {k: round(v, 3) for k, v in result.timings.items()} }")


def cmd_eval(args: argparse.Namespace) -> None:
    cfg = load_config(*_config_paths(args))
    pipe = _build_system(args, cfg)
    dataset = _build_dataset(args)

    k = cfg.get("pipeline", {}).get("top_k", 5)
    metrics = default_metrics(k=k)
    if args.judge:  # 需要一个可用的裁判模型
        metrics.append(LLMJudge(registry.build("llm", cfg["llm"])))

    from .eval.runner import Evaluator

    if args.reuse_index and hasattr(pipe, "kb"):
        persist = _persist_dir(cfg)
        pipe.kb.load(persist)
        print(f"[复用索引] {persist} → " + "　".join(f"{k}={v}" for k, v in pipe.kb.stats().items()))

    evaluator = Evaluator(pipe, metrics, retrieval_only=args.retrieval_only, workers=args.workers)
    report = evaluator.run(dataset, build_index=not args.reuse_index)
    for extra_round in range(2, args.rounds + 1):
        print(f"\n[第 {extra_round} 轮] 记忆已积累，重跑同一数据集以观察经验复用的效果")
        report = evaluator.run(dataset, build_index=False)
    report.config = cfg
    print("\n" + report.to_markdown())
    if args.out:
        report.save(args.out)
        Path(args.out).with_suffix(".md").write_text(report.to_markdown(), encoding="utf-8")
        print(f"\n明细已保存：{args.out}")


def cmd_build(args: argparse.Namespace) -> None:
    """离线建库：把语料向量化并持久化，之后 ask/eval 不再重复编码。

    这是向量检索的标准工作流，也是最容易被写错的地方：
        离线（跑一次）：文档 → 切分 → 向量化 → 落盘
        在线（每次）  ：问题 → 只编码这一条 → 查库
    每次评测都重新向量化整个语料，既慢又烧钱（远程 embedding 按 token 计费）。
    """
    import time

    cfg = load_config(*_config_paths(args))
    if cfg.get("index", {}).get("type") != "chroma":
        print("提示：index.type 不是 chroma，向量不会持久化。"
              "如需离线建库请加 --config configs/chroma.yaml")
    pipe = _build_system(args, cfg)

    docs = list(_build_dataset(args).corpus()) if (args.dataset and args.qa) \
        else list(JsonlSource(args.corpus).load())

    t0 = time.perf_counter()
    stats = pipe.index(docs, progress=True)
    elapsed = time.perf_counter() - t0

    # 关键：**两路索引都要落盘**。
    # 只持久化向量库的话，复用时 BM25 是空的，混合检索会静默退化成纯向量 ——
    # 指标莫名其妙下降，却不会有任何报错。这类"静默降级"最难排查。
    persist = _persist_dir(cfg)
    if hasattr(pipe, "kb"):
        pipe.kb.save(persist)

    print("\n[建库完成] " + "　".join(f"{k}={v}" for k, v in stats.items()))
    print(f"  耗时 {elapsed:.1f}s，平均 {elapsed / max(stats.get('chunks', 1), 1) * 1000:.2f} ms/片段")
    icfg = cfg.get("index", {})
    if icfg.get("type") == "chroma":
        print(f"  已持久化到 {icfg.get('path', 'vector_store/chroma')}"
              f"（collection={icfg.get('collection', 'minirag_default')}）")
        print(f"  BM25 倒排索引同样已落盘到 {persist}")
    print("  后续评测加 --reuse-index 即可跳过向量化")


def cmd_ls(_: argparse.Namespace) -> None:
    from . import eval as _eval  # noqa: F401  触发指标/数据集注册

    for ns in ["source", "chunker", "encoder", "index", "retriever", "query_transformer", "reranker", "generator", "llm", "dataset", "metric"]:
        print(f"{ns:20s} {registry.options(ns)}")


def main() -> None:
    parser = argparse.ArgumentParser("minirag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="单次问答")
    p_ask.add_argument("--config", action="append", default=None, help="可多次指定，后者覆盖前者")
    p_ask.add_argument("--corpus", help="jsonl 语料，用于现场建索引（已 build 过则不用传）")
    p_ask.add_argument("-q", "--question", required=True)
    p_ask.add_argument("--mode", default="pipeline", choices=["pipeline", "agentic", "wiki"],
                       help="pipeline=传统固定管线，agentic=自主决策 agent，wiki=知识编译（LLM Wiki）")
    p_ask.set_defaults(func=cmd_ask)

    p_eval = sub.add_parser("eval", help="数据集评测")
    p_eval.add_argument("--config", action="append", default=None, help="可多次指定，后者覆盖前者")
    p_eval.add_argument("--dataset", default="jsonl", choices=["jsonl", "hotpotqa", "squad_style", "beir_style"])
    p_eval.add_argument("--qa", required=True, help="问答文件（或 beir 目录）")
    p_eval.add_argument("--corpus", help="语料文件（jsonl 数据集可选）")
    p_eval.add_argument("--name", help="报告里显示的数据集名")
    p_eval.add_argument("--limit", type=int)
    p_eval.add_argument("--workers", type=int, default=1)
    p_eval.add_argument("--retrieval-only", action="store_true", help="只评检索，不调生成")
    p_eval.add_argument("--judge", action="store_true", help="启用 LLM 裁判指标")
    p_eval.add_argument("--mode", default="pipeline", choices=["pipeline", "agentic", "wiki"],
                        help="pipeline=传统固定管线，agentic=自主决策 agent，wiki=知识编译（LLM Wiki）")
    p_eval.add_argument("--rounds", type=int, default=1,
                        help="重复跑几遍（跑两遍可观察长期记忆带来的变化）")
    p_eval.add_argument("--reuse-index", action="store_true",
                        help="复用已建好的持久化索引，跳过向量化（配合 build 子命令使用）")
    p_eval.add_argument("--out", help="评测明细输出路径 .json")
    p_eval.set_defaults(func=cmd_eval)

    p_build = sub.add_parser("build", help="离线建库并持久化（只需跑一次）")
    p_build.add_argument("--config", action="append", default=None)
    p_build.add_argument("--corpus", help="jsonl 语料")
    p_build.add_argument("--dataset", choices=["jsonl", "hotpotqa", "squad_style", "beir_style"],
                         help="也可以直接用数据集自带的语料")
    p_build.add_argument("--qa", help="数据集问答文件（配合 --dataset）")
    p_build.add_argument("--limit", type=int)
    p_build.add_argument("--name")
    p_build.add_argument("--mode", default="pipeline", choices=["pipeline", "agentic", "wiki"])
    p_build.set_defaults(func=cmd_build)

    p_ls = sub.add_parser("ls", help="列出可用组件")
    p_ls.set_defaults(func=cmd_ls)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
