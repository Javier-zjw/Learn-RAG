"""
server.chunk_view —— 把一篇文档的片段整理成分块查看页需要的样子。

页面按顺序显示子块，用父块把相邻的子块框在一起，点开一个块看它的元数据。这里负责：
  - 按"被引用即父块"的规则区分父块和子块（和知识库同一条规则，不依赖切分器类型）；
  - 去掉块首的"[文档标题 > 章节路径]"：查看页上章节标题已经单独显示，块里再重复只会干扰阅读；
  - 打开句子重叠时，算出每个子块开头有多少字和上一块重复，页面用斜纹标出来。
"""

from __future__ import annotations

from typing import Any

from ..core.text import count_tokens
from ..core.types import Chunk

_MIN_OVERLAP = 6   # 少于这么多字的"重叠"多半是巧合（比如都以句号结尾），不标


def document_view(chunks: list[Chunk], *, overlap: bool = False) -> dict[str, Any]:
    parent_ids = {c.metadata.get("parent_id") for c in chunks} - {None, ""}
    children = [c for c in chunks if c.chunk_id not in parent_ids]
    parents = [c for c in chunks if c.chunk_id in parent_ids]

    items = []
    previous: tuple[str, str] | None = None     # (章节, 正文)
    for chunk in children:
        body = _body(chunk)
        section = str(chunk.metadata.get("section", ""))
        shared = _overlap(previous[1], body) if overlap and previous and previous[0] == section else 0
        items.append({**_describe(chunk, body), "parent_id": chunk.metadata.get("parent_id"), "overlap": shared})
        previous = (section, body)

    members: dict[str, list[str]] = {}
    for chunk in children:
        if chunk.metadata.get("parent_id"):
            members.setdefault(chunk.metadata["parent_id"], []).append(chunk.chunk_id)
    parent_items = [{**_describe(p, _body(p)), "children": members.get(p.chunk_id, [])} for p in parents]

    meta = chunks[0].metadata if chunks else {}
    tokens = [item["tokens"] for item in items]
    pages = [c.metadata.get("page_end") for c in chunks if c.metadata.get("page_end")]
    return {
        "doc_id": chunks[0].doc_id if chunks else "",
        "title": meta.get("title", ""),
        "file_type": meta.get("file_type", ""),
        "parser": meta.get("parser", ""),
        "pages": max(pages) if pages else None,
        "stats": {
            "children": len(items), "parents": len(parent_items), "tokens": sum(tokens),
            "avg_tokens": round(sum(tokens) / len(tokens)) if tokens else 0,
            "max_tokens": max(tokens, default=0),
        },
        "children": items,
        "parents": parent_items,
    }


def _describe(chunk: Chunk, body: str) -> dict[str, Any]:
    meta = chunk.metadata
    return {
        "id": chunk.chunk_id,
        "position": chunk.position,
        "body": body,
        "section": meta.get("section", ""),
        "tokens": count_tokens(body),
        "chars": len(body),
        "kinds": meta.get("kinds", []),
        "page_start": meta.get("page_start"),
        "page_end": meta.get("page_end"),
        "assets": meta.get("assets", []),
        "metadata": meta,
    }


def _body(chunk: Chunk) -> str:
    """结构切分的块首有一行"[标题 > 章节]"，其他切分器没有；有 section 元数据才说明是结构切分。"""
    text = chunk.text
    if "section" in chunk.metadata and text.startswith("["):
        first, _, rest = text.partition("\n")
        if first.endswith("]"):
            return rest
    return text


def _overlap(previous: str, current: str) -> int:
    """current 开头和 previous 结尾重复的最长长度。"""
    for n in range(min(len(previous), len(current)), _MIN_OVERLAP - 1, -1):
        if previous.endswith(current[:n]):
            return n
    return 0
