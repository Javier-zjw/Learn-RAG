"""
learn_rag.cli —— 命令行入口。

主要子命令：
    build   离线建库并持久化   ——「存下来」
    verify  核对知识库是否完整 ——「查得清」
    ask     一次问答（含证据展示）——「跑起来」
    eval    在数据集上评测       ——「量出来」
    ls      查看有哪些可用实现   ——「换得动」
    serve   启动可视化建库页面   ——「点得到」

用法示例：
    python -m learn_rag.cli ask   --config configs/default.yaml --corpus data/sample_corpus.jsonl -q "什么是RRF?"
    python -m learn_rag.cli eval  --config configs/default.yaml --dataset jsonl \
        --qa data/sample_qa.jsonl --corpus data/sample_corpus.jsonl --out runs/base.json
    python -m learn_rag.cli eval  --dataset hotpotqa \
        --qa data/hotpot_dev_distractor_v1.json --limit 200
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from .core.config import load_config
from .core.registry import registry
from .eval.datasets import BeirStyleDataset, HotpotQADataset, JsonlQADataset, SquadStyleDataset
from .eval.metrics import LLMJudge, default_metrics
from .ingest.loaders import JsonlSource
from .parsing.assets import configured_assets_dirs, locate_asset
from .parsing.source import FileSource
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


def _load_documents(args: argparse.Namespace, cfg: dict):
    """--docs 走文档解析（PDF / Office / 网页 / 图片），--corpus 读 jsonl 语料。"""
    if getattr(args, "docs", None):
        return FileSource(args.docs, **(cfg.get("parsing") or {})).load()
    return JsonlSource(args.corpus).load()


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = load_config(*_config_paths(args))
    pipe = _build_system(args, cfg)
    if args.corpus or args.docs:
        stats = pipe.index(_load_documents(args, cfg))
        print(f"[索引] {stats}")
    else:
        _load_index(pipe, cfg)
    result = pipe.answer(args.question, where=_parse_where(args.where))
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
    asset_dirs = _asset_dirs(cfg)
    for i, ctx in enumerate(result.contexts, 1):
        title = ctx.chunk.metadata.get("title", ctx.doc_id)
        print(f"[{i}] ({ctx.score:.4f} · {ctx.source} · {title}) {ctx.text[:120]}...")
        for line in _provenance(ctx.chunk.metadata, asset_dirs):
            print(f"     {line}")
    print(f"\n耗时：{ {k: round(v, 3) for k, v in result.timings.items()} }")


def _parse_where(text: str | None) -> dict | None:
    """--where 的 JSON 过滤条件，如 '{"file_type": "pdf"}' 或 '{"title": ["文档1", "文档2"]}'。"""
    if not text:
        return None
    try:
        where = json.loads(text)
    except ValueError as exc:
        raise SystemExit(f"--where 不是合法的 JSON：{exc}\n  示例：--where '{{\"file_type\": \"pdf\"}}'") from None
    if not isinstance(where, dict):
        raise SystemExit("--where 必须是 JSON 对象，如 '{\"file_type\": \"pdf\"}'")
    return where or None


def _load_index(pipe: RagPipeline, cfg: dict) -> None:
    """
    没有现场建索引时，打开 build 建好的知识库：flat 索引从落盘目录读文件，Chroma 本身就是持久化的。
    BM25 倒排和父块表都从片段库重建，不会和向量对不上。
    """
    persist = _persist_dir(cfg)
    if (Path(persist) / "meta.json").exists():
        pipe.kb.load(persist)
    if not len(pipe.kb.vector_index):
        raise SystemExit(f"索引为空：没有找到 {persist}。先运行 build 建库，或用 --docs / --corpus 现场建索引")
    print(f"[复用索引] {persist} → " + "　".join(f"{k}={v}" for k, v in pipe.kb.stats().items()))


_ASSET_KINDS = {"image": "图片", "table": "表格截图"}


def _asset_dirs(cfg: dict) -> list[str]:
    return configured_assets_dirs(cfg.get("parsing"))


def _provenance(meta: dict, asset_dirs: list[str]) -> list[str]:
    """证据的出处：页码、章节，以及块里每张图的图注和本地文件路径。"""
    lines = []
    start, end = meta.get("page_start"), meta.get("page_end")
    pages = f"第 {start} 页" if start == end else f"第 {start}-{end} 页"
    where = "　".join(x for x in (pages if start else "", meta.get("section", "")) if x)
    if where:
        lines.append(f"出处：{where}")
    for entry in meta.get("assets") or []:
        kind = _ASSET_KINDS.get(entry.get("kind"), entry.get("kind", "资产"))
        page = f"第 {entry['page']} 页 " if entry.get("page") else ""
        caption = f"「{entry['caption']}」" if entry.get("caption") else "（无图注）"
        found = locate_asset(entry["asset"], asset_dirs)
        lines.append(f"{kind}：{page}{caption} → {found or entry['asset'] + '（资产库中未找到）'}")
    return lines


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
        else list(_load_documents(args, cfg))

    t0 = time.perf_counter()
    stats = pipe.index(docs, progress=True)
    elapsed = time.perf_counter() - t0

    # 片段库是唯一的数据来源（Chroma 写入即落盘，flat 在 save 时写文件）；
    # BM25 和父块表打开时从片段库重建，不会出现"向量更新了、BM25 还是旧的"这种静默错位
    persist = _persist_dir(cfg)
    pipe.kb.save(persist)

    print("\n[建库完成] " + "　".join(f"{k}={v}" for k, v in stats.items()))
    print(f"  写入 {stats['documents']} 篇（共 {stats['chunks']} 个子块、{stats['parents']} 个父块），"
          f"{stats['skipped']} 篇未变化已跳过，{stats['failed']} 篇失败")
    per_chunk = f"，平均 {elapsed / stats['chunks'] * 1000:.2f} ms/片段" if stats["chunks"] else ""
    print(f"  耗时 {elapsed:.1f}s{per_chunk}")
    icfg = cfg.get("index", {})
    if icfg.get("type") == "chroma":
        print(f"  已持久化到 {icfg.get('path', 'vector_store/chroma')}"
              f"（collection={icfg.get('collection', 'learn_rag_default')}）")
    _print_verify(pipe.kb.verify(sample=10))
    print("  后续 ask 不必再传 --docs，评测加 --reuse-index 即可跳过向量化")
    if stats["failed"]:
        raise SystemExit(f"有 {stats['failed']} 篇文档写入失败，原因见上方日志；修复后重新运行 build 会续写，已写入的不会重复向量化")


def cmd_verify(args: argparse.Namespace) -> None:
    """核对已建好的知识库是否完整：文档片段是否齐全、父块是否存在、向量是否来自当前模型，并抽样核对向量能否被正确检索。"""
    cfg = load_config(*_config_paths(args))
    pipe = _build_system(args, cfg)
    persist = _persist_dir(cfg)
    if (Path(persist) / "meta.json").exists():
        pipe.kb.load(persist)
    report = pipe.kb.verify(sample=args.sample)
    _print_verify(report)
    if not report["ok"]:
        raise SystemExit(1)


def _print_verify(report: dict) -> None:
    counts = "　".join(f"{k}={report[k]}" for k in ("documents", "chunks", "bm25_chunks", "parents"))
    if report.get("sampled"):
        counts += f"　抽样核对向量 {report['sampled']} 个"
    if report["ok"]:
        print(f"[完整性检查] 通过　{counts}")
        return
    print(f"[完整性检查] 发现 {len(report['problems'])} 个问题　{counts}")
    for problem in report["problems"][:20]:
        print(f"  - {problem}")


def cmd_serve(args: argparse.Namespace) -> None:
    """
    一条命令启动全部：需要时先构建前端，后台启动本机 MinerU（有 .mineru.env 时），再启动页面服务。
    Ctrl+C 或关闭终端时一并关闭 MinerU。参数、模型、解析器都在页面上选，这里不需要任何参数。
    """
    try:
        import uvicorn

        from .server.app import create_app
        from .server.frontend import ensure_built
        from .server.mineru_service import MinerUService
    except ImportError as exc:
        raise SystemExit(f"启动页面服务需要 fastapi 和 uvicorn：pip install -e \".[web]\"（{exc}）") from None
    import atexit
    import signal

    print(f"[前端] {ensure_built()}")
    mineru = MinerUService()
    # 正常退出时由服务的生命周期关闭 MinerU；atexit 兜底异常退出，stop 可以重复调用
    atexit.register(mineru.stop)
    # 直接关掉终端窗口时进程收到的是 SIGHUP，默认会立即退出、来不及关 MinerU；转成 SIGTERM 走正常退出流程
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda *_: os.kill(os.getpid(), signal.SIGTERM))
    app = create_app(args.data, mineru=mineru)
    print(f"[页面服务] http://{args.host}:{args.port}　数据目录 {args.data}　按 Ctrl+C 退出（MinerU 会一并关闭）")
    if mineru.env_file.is_file():
        print("[MinerU] 后台启动中，就绪前提交的 PDF 建库任务会先等它；状态见页面“模型与环境”")
    else:
        print("[MinerU] 未找到 .mineru.env，不启动；PDF 用 PyMuPDF 解析")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def cmd_ls(_: argparse.Namespace) -> None:
    from . import eval as _eval  # noqa: F401  触发指标/数据集注册

    for ns in ["source", "parser", "chunker", "encoder", "index", "retriever", "query_transformer", "reranker", "generator", "llm", "dataset", "metric"]:
        print(f"{ns:20s} {registry.options(ns)}")


def main() -> None:
    parser = argparse.ArgumentParser("learn-rag")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="日志级别（默认 INFO）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="单次问答")
    p_ask.add_argument("--config", action="append", default=None, help="可多次指定，后者覆盖前者")
    p_ask.add_argument("--corpus", help="jsonl 语料，用于现场建索引（已 build 过则不用传）")
    p_ask.add_argument("--docs", help="文档目录或文件（PDF/Word/PPT/Excel/网页/图片），解析后现场建索引")
    p_ask.add_argument("-q", "--question", required=True)
    p_ask.add_argument("--where", help='元数据过滤（JSON），作用于每一路召回，如 \'{"file_type": "pdf"}\'；值为列表时表示任一匹配')
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
    p_build.add_argument("--docs", help="文档目录或文件（PDF/Word/PPT/Excel/网页/图片）")
    p_build.add_argument("--dataset", choices=["jsonl", "hotpotqa", "squad_style", "beir_style"],
                         help="也可以直接用数据集自带的语料")
    p_build.add_argument("--qa", help="数据集问答文件（配合 --dataset）")
    p_build.add_argument("--limit", type=int)
    p_build.add_argument("--name")
    p_build.add_argument("--mode", default="pipeline", choices=["pipeline", "agentic", "wiki"])
    p_build.set_defaults(func=cmd_build)

    p_verify = sub.add_parser("verify", help="核对已建好的知识库是否完整")
    p_verify.add_argument("--config", action="append", default=None)
    p_verify.add_argument("--sample", type=int, default=20,
                          help="抽样核对多少个子块的向量（会调用同样次数的 embedding），0 表示只做结构检查")
    p_verify.add_argument("--mode", default="pipeline", choices=["pipeline", "agentic", "wiki"])
    p_verify.set_defaults(func=cmd_verify)

    p_serve = sub.add_parser("serve", help="启动可视化建库页面")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--data", default="data/web", help="知识库和上传文件的存放目录")
    p_serve.set_defaults(func=cmd_serve)

    p_ls = sub.add_parser("ls", help="列出可用组件")
    p_ls.set_defaults(func=cmd_ls)

    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args.func(args)


if __name__ == "__main__":
    main()
