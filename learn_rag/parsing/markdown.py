"""
parsing.markdown —— Markdown 与 Element 之间的互转。

Markdown 是各种解析工具的"通用语"：纯文本文件本身是 Markdown，VLM OCR 输出 Markdown，
MinerU / Docling 也都能导出 Markdown。把"Markdown -> Element"写成一个函数，
所有这些来源就共享同一套结构识别逻辑，不必各写一份。
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.interfaces import DocumentParser
from ..core.registry import registry
from ..core.types import Element

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_IMAGE = re.compile(r"^!\[([^\]]*)\]\(([^)]*)\)\s*$")


def table_to_markdown(rows: list[list[str]]) -> str:
    """
    二维单元格 -> Markdown 表格，第一行视为表头。

    单元格里的换行和竖线会破坏表格结构，统一转义；各行列数不齐时补空，
    这样下游按行切分表格时总能拿到完整表头。
    """
    rows = [[_clean_cell(c) for c in row] for row in rows if any(str(c or "").strip() for c in row)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def _clean_cell(value: object) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.split()).replace("|", "\\|")


def markdown_to_elements(text: str, *, page: int | None = None) -> list[Element]:
    """
    按行扫描 Markdown，识别标题、表格、代码块、公式块、图片和段落。

    只认最常见的块级语法：解析工具产出的 Markdown 都很规整，
    完整实现 CommonMark 在这里没有收益。
    """
    elements: list[Element] = []
    lines = text.splitlines()
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            elements.append(Element("text", "\n".join(paragraph).strip(), page=page))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            end = _find(lines, i + 1, lambda s: s.startswith("```"))
            elements.append(Element("code", "\n".join(lines[i + 1:end]), page=page))
            i = end + 1
            continue

        if stripped.startswith("$$"):
            flush_paragraph()
            if stripped != "$$" and stripped.endswith("$$"):   # 单行 $$ ... $$
                elements.append(Element("formula", stripped.strip("$").strip(), page=page))
                i += 1
                continue
            end = _find(lines, i + 1, lambda s: s.startswith("$$"))
            elements.append(Element("formula", "\n".join(lines[i + 1:end]).strip(), page=page))
            i = end + 1
            continue

        if stripped.startswith("|"):
            flush_paragraph()
            end = _find(lines, i, lambda s: not s.startswith("|"))
            elements.append(Element("table", "\n".join(l.strip() for l in lines[i:end]), page=page))
            i = end
            continue

        if stripped.lower().startswith("<table"):
            # OCR 模型常把复杂表格（合并单元格）输出成 HTML，原样保留，信息最全
            flush_paragraph()
            end = _find(lines, i, lambda s: "</table>" in s.lower())
            elements.append(Element("table", "\n".join(lines[i:end + 1]).strip(), page=page))
            i = end + 1
            continue

        heading = _HEADING.match(stripped)
        if heading:
            flush_paragraph()
            elements.append(Element("heading", heading.group(2), level=len(heading.group(1)), page=page))
            i += 1
            continue

        image = _IMAGE.match(stripped)
        if image:
            flush_paragraph()
            elements.append(Element("image", image.group(1), page=page, extra={"src": image.group(2)}))
            i += 1
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph()
    return elements


def _find(lines: list[str], start: int, predicate) -> int:
    """从 start 开始找第一个满足条件的行号（按去空白后的内容判断），找不到返回末尾。"""
    for j in range(start, len(lines)):
        if predicate(lines[j].strip()):
            return j
    return len(lines)


@registry.register("parser", "markdown")
class MarkdownParser(DocumentParser):
    """纯文本与 Markdown 文件（.txt / .md）。纯文本没有标题语法时，会得到一串段落元素。"""

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def parse(self, path: Path) -> list[Element]:
        return markdown_to_elements(path.read_text(encoding=self.encoding, errors="ignore"))
