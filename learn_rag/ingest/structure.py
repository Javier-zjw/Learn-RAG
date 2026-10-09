"""
ingest.structure —— 结构感知的父子分块。

检索和生成对块大小的要求是矛盾的：块小，向量表达集中，召回才准；块大，交给大模型的上下文才完整。
父子分块把两件事拆开：
  子块（chunk_size，默认 300 token）进入向量索引和 BM25，负责"找得准"；
  父块（parent_size，默认 1200 token）只存不索引，命中子块后换成它所在的父块交给大模型，负责"看得全"。

切分沿着文档结构走，规则如下：
  1. 章：文档标题之下的第一级标题算一章（PPT 的每一页就是一章）。父块不跨章，子块不跨小节；
  2. 父块：同一章内按顺序把小节装进父块，装满 parent_size 换下一个，一个小节本身超长就按元素拆开；
  3. 子块：在每个小节内按顺序装元素，装满 chunk_size 换下一个，块首拼上"文档标题 > 章节路径"，
     片段脱离原文后依然知道自己在讲什么；
  4. 不能拆开的组合：公式和图片紧跟前一个元素（"按下式计算："和公式、"如图 2 所示"和图片），
     以冒号结尾的引导句紧跟后一个元素（"部署前需要准备以下环境："和后面的列表、表格）；
  5. 一个元素自己就放不下时交给 pieces.split_element：段落按句、表格按行（每片带表题和表头）、代码按行切；
  6. 只有一个子块的父块不生成：它和子块内容相同，展开没有意义，还要多存一份。

子块的 metadata.parent_id 指向父块。父块和子块都从 split() 返回，由知识库决定谁进索引、谁只存储。
每个块都记录章节路径、页码范围、元素类型和图片资产，回答时可以溯源到页、取回原图。
没有解析结构的纯文本（如 jsonl 语料）按空行分段后走同一套规则，调用方无需区分。
"""

from __future__ import annotations

import re

from ..core.interfaces import Chunker
from ..core.registry import registry
from ..core.types import Chunk, Document, Element
from .pieces import element_size, split_element

# 一个小节：(标题路径, 正文元素, 所在章的序号)
Section = tuple[list[str], list[Element], int]


@registry.register("chunker", "structure")
class StructureChunker(Chunker):
    def __init__(self, chunk_size: int = 300, parent_size: int = 1200) -> None:
        if parent_size and parent_size <= chunk_size:
            raise ValueError("parent_size 必须大于 chunk_size；设为 0 表示不生成父块")
        self.chunk_size = chunk_size
        self.parent_size = parent_size

    def split(self, document: Document) -> list[Chunk]:
        elements = document.elements or _paragraphs(document.text)
        chunks: list[Chunk] = []
        for group in self._parent_groups(_sections(elements)):
            children = [(path, piece) for path, body in group for piece in _pack(body, self.chunk_size)]
            parent_id = None
            if self.parent_size and len(children) > 1:
                path, text = _render_parent(group)
                parent = _make_chunk(document, path, text, [e for _, body in group for e in body], len(chunks))
                chunks.append(parent)
                parent_id = parent.chunk_id
            for path, piece in children:
                text = "\n\n".join(e.to_markdown() for e in piece)
                child = _make_chunk(document, path, text, piece, len(chunks))
                if parent_id:
                    child.metadata["parent_id"] = parent_id
                chunks.append(child)
        return chunks

    def _parent_groups(self, sections: list[Section]) -> list[list[tuple[list[str], list[Element]]]]:
        """把小节分成若干组，每组对应一个父块：同一章内按顺序装箱，超长小节先按元素拆成几段。"""
        if not self.parent_size:
            return [[(path, body)] for path, body, _ in sections]
        groups: list[list[tuple[list[str], list[Element]]]] = []
        last_chapter, used = None, 0
        for path, body, chapter in sections:
            for piece in _pack(body, self.parent_size):
                size = sum(element_size(e) for e in piece)
                if groups and chapter == last_chapter and used + size <= self.parent_size:
                    groups[-1].append((path, piece))
                    used += size
                else:
                    groups.append([(path, piece)])
                    last_chapter, used = chapter, size
        return groups


def _pack(elements: list[Element], budget: int) -> list[list[Element]]:
    """按顺序把元素装进不超过 budget 的若干组。不能拆开的组合整体装入，整体放不下时才拆开，单个元素超长时切片。"""
    groups: list[list[Element]] = []
    current: list[Element] = []
    used = 0
    for unit in _units(elements):
        size = sum(element_size(e) for e in unit)
        parts = [unit] if size <= budget else [[piece] for e in unit for piece in split_element(e, budget)]
        for part in parts:
            size = sum(element_size(e) for e in part)
            if current and used + size > budget:
                groups.append(current)
                current, used = [], 0
            current.extend(part)
            used += size
    if current:
        groups.append(current)
    return groups


def _units(elements: list[Element]) -> list[list[Element]]:
    """把必须放在一起的相邻元素合成一个单元：公式和图片跟着前一个元素，冒号结尾的引导句带上后一个元素。"""
    units: list[list[Element]] = []
    for element in elements:
        if units and (element.kind in ("formula", "image") or _leads_in(units[-1][-1])):
            units[-1].append(element)
        else:
            units.append([element])
    return units


def _leads_in(element: Element) -> bool:
    return element.kind == "text" and element.text.rstrip().endswith(("：", ":"))


def _sections(elements: list[Element]) -> list[Section]:
    """
    按标题把元素分组，返回 [(标题路径, 正文元素, 章序号)]。只有标题没有内容的小节不产生块，标题留在下级小节的路径里。

    章的层级：取最高一级标题；但如果最高一级只出现一次且在最前面，它是文档标题（MinerU 的 doc_title、
    PPT 的封面标题），章从下一级算起。
    """
    levels = [max(e.level, 1) for e in elements if e.kind == "heading"]
    top = min(levels, default=1)
    if levels and levels[0] == top and levels.count(top) == 1 and len(set(levels)) > 1:
        top = min(level for level in levels if level > top)

    sections: list[Section] = []
    path: list[str] = []
    body: list[Element] = []
    chapter = 0
    for element in elements:
        if element.kind == "heading":
            if body:
                sections.append((path, body, chapter))
            level = max(element.level, 1)
            if level <= top:
                chapter += 1
            path = path[: level - 1] + [element.text]
            body = []
        elif element.text.strip():
            body.append(element)
    if body:
        sections.append((path, body, chapter))
    return sections


def _render_parent(group: list[tuple[list[str], list[Element]]]) -> tuple[list[str], str]:
    """父块的正文：块首路径取各小节的公共前缀，更深的小节标题渲染成 Markdown 标题留在正文里，保留原文层次。"""
    common = _common_prefix([path for path, _ in group])
    lines: list[str] = []
    previous = common
    for path, body in group:
        shared = max(len(_common_prefix([path, previous])), len(common))
        lines.extend("#" * (depth + 1) + " " + path[depth] for depth in range(shared, len(path)))
        lines.extend(e.to_markdown() for e in body)
        previous = path
    return common, "\n\n".join(lines)


def _common_prefix(paths: list[list[str]]) -> list[str]:
    prefix = list(paths[0])
    for path in paths[1:]:
        n = 0
        while n < min(len(prefix), len(path)) and prefix[n] == path[n]:
            n += 1
        prefix = prefix[:n]
    return prefix


def _make_chunk(document: Document, path: list[str], body: str, elements: list[Element], position: int) -> Chunk:
    trail = path if path and path[0] == document.title else [document.title, *path]
    chunk = Chunk.of(document, "[" + " > ".join(trail) + "]\n" + body, position)
    chunk.metadata.update(_describe(path, elements))
    return chunk


def _paragraphs(text: str) -> list[Element]:
    """没有结构的纯文本按空行分段，交给同一套切分规则。"""
    return [Element("text", p.strip()) for p in re.split(r"\n\s*\n", text) if p.strip()]


def _describe(path: list[str], group: list[Element]) -> dict[str, object]:
    """块的溯源信息：章节路径、页码范围、元素类型、图片资产。"""
    meta: dict[str, object] = {"section": " > ".join(path), "kinds": sorted({e.kind for e in group})}
    pages = [p for e in group for p in (e.page, e.extra.get("page_end")) if p]
    if pages:
        meta["page_start"], meta["page_end"] = min(pages), max(pages)
    # 资产库里的相对路径跟着块走：图片、图表和表格截图都能按 ID 取回原图
    assets = [e.extra["asset"] for e in group if e.extra.get("asset")]
    if assets:
        meta["assets"] = assets
    return meta
