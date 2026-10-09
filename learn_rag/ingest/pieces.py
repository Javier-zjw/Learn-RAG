"""
ingest.pieces —— 把一个超长元素切成若干不超过预算的片段，并且每一片单独看都是完整的。

结构切分器装箱时遇到一个元素自己就放不下，才会调用这里。不同类型的元素，"完整"的含义不同：
  text     按句子切，句子本身超长才按字数硬切；
  table    按行切，每片都带上表题和表头，表注跟在最后一片；合并单元格先展开成普通网格，
           否则"华东"只出现在跨行单元格的第一行，后面几片就不知道是哪个区域的数据；
  code     按行切，不会把一行代码从中间断开；
  formula、image、heading 不切：半个公式、半段图片描述都没有意义，宁可超出预算。
"""

from __future__ import annotations

import re

from ..core.tables import flatten_table
from ..core.text import count_tokens
from ..core.types import Element

# 在句末标点、分号和换行之后断句；英文句号要求后面跟空格，避免切开 3.14 和 config.yaml
_SENTENCE_END = re.compile(r"(?<=[。！？；!?;\n])|(?<=\. )")
_HTML_TABLE = re.compile(r"<table.*?</table>", re.S | re.I)


def element_size(element: Element) -> int:
    """元素渲染成 Markdown 后的 token 数，装箱时用它计算大小。"""
    return count_tokens(element.to_markdown())


def split_element(element: Element, budget: int) -> list[Element]:
    """把超过 budget 的元素切成若干片，不超过的原样返回。切出的片段保留原元素的页码和附加信息，并记录序号 part。"""
    if element.kind in ("formula", "image", "heading") or element_size(element) <= budget:
        return [element]
    if element.kind == "table":
        texts = _split_table(element.text, budget)
    elif element.kind == "code":
        texts = _pack(element.text.splitlines(keepends=True), budget)
    else:
        sentences = [s for s in _SENTENCE_END.split(element.text) if s]
        texts = _pack([part for s in sentences for part in _cut(s, budget)], budget)
    if len(texts) <= 1:
        return [element]
    return [
        Element(element.kind, text, element.level, element.page, element.bbox, {**element.extra, "part": i + 1})
        for i, text in enumerate(texts)
    ]


def _split_table(text: str, budget: int) -> list[str]:
    """
    表格按行切分。表格前面的行（表题）和表头、分隔线是每一片共用的"头"，
    表格后面的行（表注）跟在最后一片。HTML 表格先展开成 Markdown，切出来的每一行都是自包含的。
    """
    match = _HTML_TABLE.search(text)
    if match:
        text = text[:match.start()] + flatten_table(match.group()) + text[match.end():]
    lines = text.strip().splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("|")), None)
    if start is None or len(lines) < start + 3:
        return [text]
    end = next((i for i in range(start, len(lines)) if not lines[i].startswith("|")), len(lines))
    head, rows, tail = lines[:start + 2], lines[start + 2:end], lines[end:]
    room = max(budget - count_tokens("\n".join(head)), 1)
    groups = _pack([row + "\n" for row in rows], room)
    pieces = ["\n".join(head) + "\n" + group for group in groups]
    if tail:
        pieces[-1] += "\n" + "\n".join(tail)
    return pieces


def _pack(units: list[str], budget: int) -> list[str]:
    """按顺序把小段文字拼成不超过 budget 的片段。单个小段超出预算时独占一片，不在这里切开它。"""
    pieces: list[str] = []
    current, used = "", 0
    for unit in units:
        size = count_tokens(unit)
        if current.strip() and used + size > budget:
            pieces.append(current)
            current, used = "", 0
        current += unit
        used += size
    pieces.append(current)
    return [p.strip() for p in pieces if p.strip()]


def _cut(text: str, budget: int) -> list[str]:
    """没有标点可断的超长句子（OCR 结果、长网址）只能按字数硬切，字数按 token 比例换算。"""
    tokens = count_tokens(text)
    if tokens <= budget:
        return [text]
    step = max(1, len(text) * budget // tokens)
    return [text[i:i + step] for i in range(0, len(text), step)]
