"""
parsing.markdown —— Markdown 与 Element 之间的互转，以及表格的统一表示。

Markdown 是各种解析工具的"通用语"：纯文本文件本身是 Markdown，VLM OCR 输出 Markdown，
MinerU / Docling 也都能导出 Markdown。把"Markdown -> Element"写成一个函数，
所有这些来源就共享同一套结构识别逻辑，不必各写一份。

表格同理：各工具输出的表格有 Markdown、带缩进的 HTML、带 <p> 标签的 HTML 等多种写法，
normalize_table 把它们统一成两种：没有合并单元格的用 Markdown，有合并单元格的用紧凑 HTML。
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
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


# 只有日期没有时间的单元格，经 Excel / 解析工具转出来常带一个零点时间，去掉它
_MIDNIGHT = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]00:00:00$")


def _clean_cell(value: object) -> str:
    return _cell_text(value).replace("|", "\\|")


def _cell_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _MIDNIGHT.sub(r"\1", " ".join(text.split()))


def normalize_table(table: str) -> str:
    """
    统一表格的表示：
      - 没有合并单元格的 HTML 表格转成 Markdown，结构切分器可以按行切分超长表格；
      - 有合并单元格时保留 HTML（Markdown 表达不了跨行跨列），但只保留结构，
        去掉缩进、换行和 <p>、<strong> 等排版标签，避免空白撑大切分块。
    不是 HTML 表格的文本（如已经是 Markdown）原样返回。
    """
    rows = _HtmlTable.parse(table) if "<table" in table.lower() else []
    if not rows:
        return table.strip()
    if all(rowspan == colspan == 1 for row in rows for _, rowspan, colspan in row):
        return table_to_markdown([[text for text, _, _ in row] for row in rows])
    lines = []
    for row in rows:
        cells = []
        for text, rowspan, colspan in row:
            attrs = (f' rowspan="{rowspan}"' if rowspan > 1 else "") + (f' colspan="{colspan}"' if colspan > 1 else "")
            cells.append(f"<td{attrs}>{html.escape(_cell_text(text), quote=False)}</td>")
        lines.append("<tr>" + "".join(cells) + "</tr>")
    return "<table>" + "".join(lines) + "</table>"


class _HtmlTable(HTMLParser):
    """读出 HTML 表格的单元格：[[(文字, rowspan, colspan), ...], ...]。只看第一层表格，嵌套表格的文字并入外层单元格。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self._cell: list[str] | None = None
        self._span = (1, 1)
        self._depth = 0

    @classmethod
    def parse(cls, table: str) -> list[list[tuple[str, int, int]]]:
        parser = cls()
        parser.feed(table)
        parser.close()
        return [row for row in parser.rows if row]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._depth += 1
        elif self._depth == 1 and tag == "tr":
            self.rows.append([])
        elif self._depth == 1 and tag in ("td", "th"):
            values = dict(attrs)
            self._cell, self._span = [], (_span(values.get("rowspan")), _span(values.get("colspan")))
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._depth -= 1
        elif self._depth == 1 and tag in ("td", "th") and self._cell is not None:
            if not self.rows:
                self.rows.append([])
            self.rows[-1].append(("".join(self._cell), *self._span))
            self._cell = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def _span(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


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
            # OCR 模型常把复杂表格（合并单元格）输出成 HTML，统一表示后保留
            flush_paragraph()
            end = _find(lines, i, lambda s: "</table>" in s.lower())
            elements.append(Element("table", normalize_table("\n".join(lines[i:end + 1])), page=page))
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
