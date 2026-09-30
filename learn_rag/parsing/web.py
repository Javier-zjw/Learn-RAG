"""
parsing.web —— 网页（.html / .htm）解析。

只用标准库 html.parser：网页本身就带结构（h1~h6、p、table），直接读标签即可。
导航栏、页眉页脚、脚本样式这类与正文无关的区域整块跳过，避免噪声进入知识库。
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from ..core.interfaces import DocumentParser
from ..core.registry import registry
from ..core.types import Element
from .markdown import table_to_markdown

_SKIP = {"script", "style", "noscript", "template", "nav", "header", "footer", "aside", "form"}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_BR = "\x00"   # <br> 的占位符，与源码中无意义的换行区分开
_BLOCKS = {"p", "div", "li", "blockquote", "section", "article", "dt", "dd", "figcaption", "caption"}


@registry.register("parser", "html")
class HtmlParser(DocumentParser):
    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def parse(self, path: Path) -> list[Element]:
        collector = _Collector()
        collector.feed(path.read_text(encoding=self.encoding, errors="ignore"))
        collector.close()
        return collector.elements


class _Collector(HTMLParser):
    """
    边读标签边产出 Element。

    状态只有三类：是否在要跳过的区域里、当前文本块是什么类型、是否在表格里。
    表格内部的其他标签一律忽略，文字全部归入当前单元格（嵌套表格也并入外层单元格）。
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[Element] = []
        self._buf: list[str] = []
        self._kind, self._level = "text", 0
        self._skip_depth = 0
        self._table_depth = 0
        self._rows: list[list[str]] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip_depth += 1
        if self._skip_depth:
            return

        if tag == "table":
            self._table_depth += 1
            if self._table_depth == 1:
                self._flush()
                self._rows = []
            return
        if self._table_depth:
            if self._table_depth == 1 and tag == "tr":
                self._rows.append([])
            elif self._table_depth == 1 and tag in ("td", "th"):
                self._cell = []
            return

        if tag in _HEADINGS:
            self._flush()
            self._kind, self._level = "heading", _HEADINGS[tag]
        elif tag == "pre":
            self._flush()
            self._kind = "code"
        elif tag in _BLOCKS:
            self._flush()
        elif tag == "br":
            self._buf.append(_BR)
        elif tag == "img":
            alt = (dict(attrs).get("alt") or "").strip()
            if alt:
                self._flush()
                self.elements.append(Element("image", alt, extra={"src": dict(attrs).get("src") or ""}))

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return

        if self._table_depth:
            if self._table_depth == 1 and tag in ("td", "th") and self._cell is not None:
                if not self._rows:
                    self._rows.append([])
                self._rows[-1].append("".join(self._cell))
                self._cell = None
            elif tag == "table":
                self._table_depth -= 1
                if self._table_depth == 0:
                    markdown = table_to_markdown(self._rows)
                    if markdown:
                        self.elements.append(Element("table", markdown))
            return

        if tag in _HEADINGS or tag == "pre" or tag in _BLOCKS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._table_depth:
            if self._cell is not None:
                self._cell.append(data)
            return
        self._buf.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        text = "".join(self._buf)
        self._buf.clear()
        if self._kind == "code":
            text = text.replace(_BR, "\n")
        else:
            # 网页源码里的换行和缩进没有语义，压成单个空格；只有 <br> 才是真正的换行
            text = "\n".join(" ".join(part.split()) for part in text.split(_BR))
        text = text.strip()
        if text:
            self.elements.append(Element(self._kind, text, level=self._level))
        self._kind, self._level = "text", 0
