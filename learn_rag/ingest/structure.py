"""
ingest.structure —— 结构感知切分：利用解析得到的 Element 切分，而不是在纯文本上猜边界。

规则只有四条：
  1. 以标题为边界：一个块只包含同一章节的内容，块首拼上"文档标题 > 章节路径"，
     片段脱离原文后依然知道自己在讲什么；
  2. 同一章节内的段落依次装入块，装满 chunk_size 就换下一块；单个超长段落用递归切分；
  3. 表格单独成块、不与正文混排；超长的 Markdown 表格按行切分，每块都重复表头；
  4. 每个块记录章节路径、页码范围和包含的元素类型，回答时可以溯源到页。

没有结构的文档（如 jsonl 语料）自动退回递归切分，调用方无需区分。
"""

from __future__ import annotations

from ..core.interfaces import Chunker
from ..core.registry import registry
from ..core.types import Chunk, Document, Element
from .chunkers import RecursiveChunker


@registry.register("chunker", "structure")
class StructureChunker(Chunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self._recursive = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, document: Document) -> list[Chunk]:
        if not document.elements:
            return self._recursive.split(document)

        chunks: list[Chunk] = []
        for path, elements in _sections(document.elements):
            trail = path if path and path[0] == document.title else [document.title, *path]
            header = "[" + " > ".join(trail) + "]\n"
            for group in self._pack(elements):
                body = "\n\n".join(e.to_markdown() for e in group)
                chunk = Chunk.of(document, header + body, len(chunks))
                chunk.metadata.update(_describe(path, group))
                chunks.append(chunk)
        return chunks

    def _pack(self, elements: list[Element]) -> list[list[Element]]:
        """把一个章节的元素装进若干块。表格总是单独成块；其他元素按长度依次装箱。"""
        groups: list[list[Element]] = []
        current: list[Element] = []
        size = 0

        def close() -> None:
            nonlocal current, size
            if current:
                groups.append(current)
            current, size = [], 0

        for element in elements:
            if element.kind == "table":
                close()
                groups.extend([piece] for piece in self._split_table(element))
                continue
            for piece in self._split_long(element):
                length = len(piece.to_markdown())
                if current and size + length > self.chunk_size:
                    close()
                current.append(piece)
                size += length + 2
        close()
        return groups

    def _split_long(self, element: Element) -> list[Element]:
        """超长的段落按递归规则切开；公式、代码、图片描述保持完整，切开反而会破坏含义。"""
        if element.kind != "text" or len(element.text) <= self.chunk_size:
            return [element]
        pieces = self._recursive._split_text(element.text)
        return [Element("text", p.strip(), page=element.page, extra=dict(element.extra)) for p in pieces if p.strip()]

    def _split_table(self, table: Element) -> list[Element]:
        """
        Markdown 表格超长时按行切分，每块都带上表头（前两行），这样每一块单独看都是完整的表格。
        HTML 表格（含合并单元格）无法安全地按行切，保持整体。
        """
        lines = table.text.splitlines()
        if len(table.text) <= self.chunk_size or len(lines) < 3 or not lines[0].startswith("|"):
            return [table]
        head, rows = lines[:2], lines[2:]
        budget = self.chunk_size - sum(len(h) + 1 for h in head)
        pieces: list[list[str]] = [[]]
        used = 0
        for row in rows:
            if pieces[-1] and used + len(row) + 1 > budget:
                pieces.append([])
                used = 0
            pieces[-1].append(row)
            used += len(row) + 1
        return [
            Element("table", "\n".join(head + piece), page=table.page, extra={**table.extra, "table_part": i + 1})
            for i, piece in enumerate(pieces)
        ]


def _sections(elements: list[Element]) -> list[tuple[list[str], list[Element]]]:
    """按标题把元素分组，返回 [(标题路径, 该章节的非标题元素)]。只有标题没有内容的章节不产生块。"""
    sections: list[tuple[list[str], list[Element]]] = []
    path: list[str] = []
    body: list[Element] = []
    for element in elements:
        if element.kind == "heading":
            if body:
                sections.append((path, body))
            level = max(element.level, 1)
            path = path[: level - 1] + [element.text]
            body = []
        elif element.text.strip():
            body.append(element)
    if body:
        sections.append((path, body))
    return sections


def _describe(path: list[str], group: list[Element]) -> dict[str, object]:
    """块的溯源信息：章节路径、页码范围、元素类型。"""
    meta: dict[str, object] = {"section": " > ".join(path), "kinds": sorted({e.kind for e in group})}
    pages = [p for e in group for p in (e.page, e.extra.get("page_end")) if p]
    if pages:
        meta["page_start"], meta["page_end"] = min(pages), max(pages)
    # 资产库里的相对路径跟着块走：图片、图表和表格截图都能按 ID 取回原图
    assets = [e.extra["asset"] for e in group if e.extra.get("asset")]
    if assets:
        meta["assets"] = assets
    return meta
