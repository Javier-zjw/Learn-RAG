"""
把文档的解析结果导出成可读文件，用来检查解析效果（MinerU / 内置解析器）。

每个文档生成三个文件（默认输出到 runs/parsed/，runs/ 已在 .gitignore 中）：
    <文件名>.md              解析结果渲染成的 Markdown：看标题层级、表格、段落顺序
    <文件名>.elements.json   每个结构元素的类型、层级、页码、坐标、附加信息：看解析细节
    <文件名>.chunks.jsonl    结构切分后的块和元数据：看最终进入知识库的内容
另外在输出目录生成：
    summary.md               每个文件用了哪个解析器、耗时、各类元素数量；没导出的文件也会列出
    export.log               完整日志（解析器失败、降级的原因都在这里）

用法：
    python scripts/export_parsed.py path/to/docs
    python scripts/export_parsed.py path/to/docs --mineru            # 所有 MinerU 支持的格式都优先交给 MinerU
    python scripts/export_parsed.py samples/parsing --mineru --no-cache \\
        --raw samples/parsing_results/mineru_raw --out samples/parsing_results/mineru

--raw 会把 MinerU 的原始结果 zip（含 middle_json.json 和图片）另存一份，用于核对适配器有没有漏掉信息；
--replay 读取这些已保存的 zip 而不重新运行 MinerU，修改适配器后用它在几秒内重新生成结果：
    python scripts/export_parsed.py samples/parsing --mineru --no-cache \\
        --replay samples/parsing_results/mineru_raw --out samples/parsing_results/mineru
解析结果默认有缓存（.cache/parsed/），--no-cache 强制重新解析。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learn_rag.core.config import load_config
from learn_rag.core.registry import registry
from learn_rag.ingest import structure  # noqa: F401  触发结构切分器注册
from learn_rag.parsing.source import FileSource

# --mineru：MinerU 4.x 支持的格式都优先交给它，失败时退回内置解析器（没有内置解析器的格式只用 MinerU）
_MINERU_FIRST: dict[str, list[str]] = {
    ".pdf": ["mineru", "pdf"],
    ".docx": ["mineru", "docx"], ".pptx": ["mineru", "pptx"], ".xlsx": ["mineru", "xlsx"],
    ".doc": ["mineru", "libreoffice"], ".ppt": ["mineru", "libreoffice"],
    ".xls": ["mineru", "libreoffice"], ".rtf": ["mineru", "libreoffice"],
    ".csv": ["mineru", "csv"], ".tsv": ["mineru", "tsv"],
    ".html": ["mineru", "html"], ".htm": ["mineru", "html"],
    ".png": ["mineru", "vlm_ocr"], ".jpg": ["mineru", "vlm_ocr"], ".jpeg": ["mineru", "vlm_ocr"],
    ".webp": ["mineru", "vlm_ocr"],
    ".odt": ["mineru"], ".ods": ["mineru"], ".odp": ["mineru"],
    ".epub": ["mineru"], ".mhtml": ["mineru"], ".mht": ["mineru"], ".ofd": ["mineru"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description="导出文档解析结果")
    ap.add_argument("docs", help="文档目录或单个文件")
    ap.add_argument("--config", action="append", default=None,
                    help="叠加在 configs/default.yaml 之上的配置，可多次指定（默认 configs/parsing.yaml）")
    ap.add_argument("--out", default="runs/parsed", help="输出目录")
    ap.add_argument("--mineru", action="store_true", help="所有 MinerU 支持的格式都优先使用 MinerU 解析")
    ap.add_argument("--no-cache", action="store_true", help="不读也不写解析缓存，强制重新解析")
    ap.add_argument("--raw", help="另存 MinerU 原始结果 zip 的目录")
    ap.add_argument("--replay", help="直接读取该目录里已保存的 MinerU 结果 zip，不重新运行 MinerU（修改适配器后用来快速验证）")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _setup_logging(out / "export.log")

    # load_config 会静默跳过不存在的文件，相对路径按项目根解析，避免在别的目录运行时丢掉 MinerU 路由
    configs = ["configs/default.yaml", *(args.config or ["configs/parsing.yaml"])]
    cfg = load_config(*(c if Path(c).exists() else ROOT / c for c in configs))
    parsing = dict(cfg.get("parsing") or {})
    if args.mineru:
        parsing["parsers"] = {**(parsing.get("parsers") or {}), **_MINERU_FIRST}
    if args.no_cache:
        parsing["cache_dir"] = None
    if args.raw or args.replay:
        options = {k: dict(v) for k, v in (parsing.get("options") or {}).items()}
        if args.raw:
            options.setdefault("mineru", {})["raw_dir"] = str(Path(args.raw).resolve())
        if args.replay:
            options.setdefault("mineru", {})["replay_dir"] = str(Path(args.replay).resolve())
        parsing["options"] = options
    chunker = registry.build("chunker", cfg["chunker"])

    rows: list[dict] = []
    docs = iter(FileSource(args.docs, **parsing).load())
    while True:
        started = time.perf_counter()
        doc = next(docs, None)        # 解析发生在取下一个文档时，所以这样计时
        if doc is None:
            break
        elapsed = time.perf_counter() - started
        name = Path(doc.doc_id).with_suffix("").as_posix().replace("/", "__")
        (out / f"{name}.md").write_text(doc.text, encoding="utf-8")
        elements = [asdict(e) for e in doc.elements]
        (out / f"{name}.elements.json").write_text(json.dumps(elements, ensure_ascii=False, indent=2), encoding="utf-8")
        chunks = chunker.split(doc)
        with (out / f"{name}.chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps({"id": c.chunk_id, "text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")
        kinds = Counter(e.kind for e in doc.elements)
        rows.append({"file": doc.doc_id, "parser": doc.metadata["parser"], "seconds": elapsed,
                     "kinds": kinds, "chunks": len(chunks)})
        logging.info("导出 %s：解析器=%s 耗时=%.1fs 元素=%d 块=%d",
                     doc.doc_id, doc.metadata["parser"], elapsed, len(doc.elements), len(chunks))

    _write_summary(out / "summary.md", Path(args.docs), rows, args)
    print(f"\n共导出 {len(rows)} 个文档 → {out.resolve()}（汇总见 summary.md，日志见 export.log）")


def _setup_logging(log_file: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)


def _write_summary(path: Path, docs: Path, rows: list[dict], args: argparse.Namespace) -> None:
    kinds = ["heading", "text", "table", "image", "formula", "code"]
    lines = [
        "# 解析结果汇总",
        "",
        f"- 输入：`{docs}`　参数：mineru={args.mineru} no_cache={args.no_cache} raw={args.raw or '-'} "
        f"replay={args.replay or '-'}",
        "",
        "| 文件 | 解析器 | 耗时(秒) | " + " | ".join(kinds) + " | 块数 |",
        "| --- | --- | --- | " + " | ".join("---" for _ in kinds) + " | --- |",
    ]
    for r in rows:
        counts = " | ".join(str(r["kinds"].get(k, 0)) for k in kinds)
        lines.append(f"| {r['file']} | {r['parser']} | {r['seconds']:.1f} | {counts} | {r['chunks']} |")

    exported = {r["file"] for r in rows}
    if docs.is_dir():
        missing = [p.relative_to(docs).as_posix() for p in sorted(docs.rglob("*"))
                   if p.is_file() and not p.name.startswith(".") and p.relative_to(docs).as_posix() not in exported]
    else:
        missing = [] if docs.name in exported else [docs.name]
    if missing:
        lines += ["", "## 未导出的文件（原因见 export.log）", ""] + [f"- {m}" for m in missing]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
