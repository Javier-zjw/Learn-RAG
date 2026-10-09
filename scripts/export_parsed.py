"""
把文档的解析结果导出成可读文件，用来检查解析效果（MinerU / 内置解析器）。

每个文档生成三个文件（默认输出到 runs/parsed/，runs/ 已在 .gitignore 中）：
    <文件名>.md              解析结果渲染成的 Markdown：看标题层级、表格、段落顺序
    <文件名>.elements.json   每个结构元素的类型、层级、页码、坐标、附加信息：看解析细节
    <文件名>.chunks.jsonl    结构切分后的块和元数据：看最终进入知识库的内容

用法：
    python scripts/export_parsed.py path/to/docs
    python scripts/export_parsed.py path/to/docs --mineru          # Word / PPT / Excel 也优先交给 MinerU
    python scripts/export_parsed.py a.pdf --out runs/parsed_v2

终端会打印每个文档实际使用的解析器：显示 mineru 说明 MinerU 生效；
显示其他解析器说明 MinerU 失败后已降级，失败原因在上方的 WARNING 日志里。
解析结果有缓存（.cache/parsed/），改了 MinerU 配置想重新解析时先删掉对应的 mineru-*.json。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from learn_rag.core.config import load_config
from learn_rag.core.registry import registry
from learn_rag.ingest import structure  # noqa: F401  触发结构切分器注册
from learn_rag.parsing.source import FileSource

# --mineru 时改为"MinerU 优先、失败退回内置解析器"的格式；PDF 在 parsing.yaml 里已经是这样
_MINERU_FIRST = {".pdf": ["mineru", "pdf"], ".docx": ["mineru", "docx"],
                 ".pptx": ["mineru", "pptx"], ".xlsx": ["mineru", "xlsx"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="导出文档解析结果")
    ap.add_argument("docs", help="文档目录或单个文件")
    ap.add_argument("--config", action="append", default=None,
                    help="叠加在 configs/default.yaml 之上的配置，可多次指定（默认 configs/parsing.yaml）")
    ap.add_argument("--out", default="runs/parsed", help="输出目录")
    ap.add_argument("--mineru", action="store_true", help="Word / PPT / Excel 也优先使用 MinerU 解析")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # load_config 会静默跳过不存在的文件，相对路径按项目根解析，避免在别的目录运行时丢掉 MinerU 路由
    configs = ["configs/default.yaml", *(args.config or ["configs/parsing.yaml"])]
    cfg = load_config(*(c if Path(c).exists() else ROOT / c for c in configs))
    parsing = dict(cfg.get("parsing") or {})
    if args.mineru:
        parsing["parsers"] = {**(parsing.get("parsers") or {}), **_MINERU_FIRST}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chunker = registry.build("chunker", cfg["chunker"])

    count = 0
    for doc in FileSource(args.docs, **parsing).load():
        name = Path(doc.doc_id).with_suffix("").as_posix().replace("/", "__")
        (out / f"{name}.md").write_text(doc.text, encoding="utf-8")
        elements = [asdict(e) for e in doc.elements]
        (out / f"{name}.elements.json").write_text(json.dumps(elements, ensure_ascii=False, indent=2), encoding="utf-8")
        chunks = chunker.split(doc)
        with (out / f"{name}.chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps({"id": c.chunk_id, "text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")
        print(f"{doc.doc_id}: 解析器={doc.metadata['parser']}  元素={len(doc.elements)}  块={len(chunks)}")
        count += 1

    print(f"\n共导出 {count} 个文档 → {out.resolve()}")


if __name__ == "__main__":
    main()
