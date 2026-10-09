"""
core.tables —— 表格的统一表示。

表格元素（Element.kind == "table"）的正文只有两种写法：没有合并单元格的用 Markdown，
有合并单元格的用紧凑 HTML。解析层用 normalize_table 把各工具的输出统一成这两种，
切分层用 flatten_table 把合并单元格展开，才能按行切开超长表格。
两边共用同一份 HTML 读取逻辑，所以放在 core，而不是让切分层去引用解析层的内部实现。
"""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser


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


def flatten_table(table: str) -> str:
    """
    把带合并单元格的 HTML 表格展开成 Markdown，切分超长表格时用：
      - 跨行跨列的值复制到它覆盖的每个格子，切出来的每一行都是自包含的（"华东"不会只留在第一行）；
      - 多行表头合成一行，上下层用空格连接，如"上半年 Q1"。
    表头行数取第一行单元格的最大 rowspan：解析工具输出的 HTML 一律用 <td>，没有 <th> 可依赖，
    而多层表头的左上角单元格（"区域"）总是纵向合并到表头的最后一行。
    不是 HTML 表格的文本原样返回。
    """
    rows = _HtmlTable.parse(table) if "<table" in table.lower() else []
    if not rows:
        return table.strip()
    grid = _expand(rows)
    head = min(max(rowspan for _, rowspan, _ in rows[0]), len(grid) - 1) or 1
    header = []
    for col in range(len(grid[0])):
        names: list[str] = []
        for row in grid[:head]:
            name = _cell_text(row[col])
            if name and name not in names:
                names.append(name)
        header.append(" ".join(names))
    return table_to_markdown([header] + grid[head:])


def _expand(rows: list[list[tuple[str, int, int]]]) -> list[list[str]]:
    """按 rowspan / colspan 把单元格铺进二维网格。被上方单元格占住的格子跳过，网格补齐成矩形。"""
    grid: dict[tuple[int, int], str] = {}
    for r, row in enumerate(rows):
        c = 0
        for text, rowspan, colspan in row:
            while (r, c) in grid:
                c += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r + dr, c + dc)] = text
            c += colspan
    height = max(r for r, _ in grid) + 1
    width = max(c for _, c in grid) + 1
    return [[grid.get((r, c), "") for c in range(width)] for r in range(height)]


# 只有日期没有时间的单元格，经 Excel / 解析工具转出来常带一个零点时间，去掉它
_MIDNIGHT = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T]00:00:00$")


def _clean_cell(value: object) -> str:
    return _cell_text(value).replace("|", "\\|")


def _cell_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _MIDNIGHT.sub(r"\1", " ".join(text.split()))


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
