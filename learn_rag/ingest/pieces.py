"""
ingest.pieces —— 把一个超长元素切成若干不超过预算的片段，并且每一片单独看都是完整的。

结构切分器装箱时遇到一个元素自己就放不下，才会调用这里。不同类型的元素，"完整"的含义不同：
  text     按句子切；句子本身超长时在逗号、顿号、冒号处断开，连这样的分句都超长才按字数硬切。
           可选句子重叠：下一片开头重复上一片的最后几句，弥补"它""该方案"这类指代在切口处丢失；
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
# 超长句子在逗号、顿号、冒号之后断开；英文逗号同样要求后面跟空格，避免切开 4,860 这样的数字
_CLAUSE_END = re.compile(r"(?<=[，、：])|(?<=, )")
_HTML_TABLE = re.compile(r"<table.*?</table>", re.S | re.I)


def element_size(element: Element) -> int:
    """元素渲染成 Markdown 后的 token 数，装箱时用它计算大小。"""
    return count_tokens(element.to_markdown())


def split_element(element: Element, budget: int, overlap: int = 0) -> list[Element]:
    """
    把超过 budget 的元素切成若干片，不超过的原样返回。切出的片段保留原元素的页码和附加信息，并记录序号 part。

    overlap 是段落相邻两片之间重复的句子数，只作用于段落：表格每片已经重复了表头，代码重复几行没有意义。
    """
    if element.kind in ("formula", "image", "heading") or element_size(element) <= budget:
        return [element]
    if element.kind == "table":
        texts = _split_table(element.text, budget)
    elif element.kind == "code":
        texts = _pack(element.text.splitlines(keepends=True), budget)
    else:
        texts = _pack([part for s in _sentences(element.text) for part in _fit(s, budget)], budget, overlap)
    if len(texts) <= 1:
        return [element]
    # 表格每片都重复了表头，在原文里找不到对应的连续一段，不记比例
    spans = _spans(element.text, texts) if element.kind != "table" else [None] * len(texts)
    pieces = []
    for i, (text, span) in enumerate(zip(texts, spans)):
        extra = {**element.extra, "part": i + 1}
        if span:
            extra["span"] = span
        pieces.append(Element(element.kind, text, element.level, element.page, element.bbox, extra))
    return pieces


def _spans(whole: str, texts: list[str]) -> list[list[float] | None]:
    """
    每片在原元素里所占的比例 [起, 止]。扫描件只有整段的坐标，查看页按这个比例截出每片大致占的几行，
    相邻的块不会画成同一个框。找不到对应位置的片（被重新拼接过的）记 None，退回整段坐标。
    """
    spans: list[list[float] | None] = []
    pos = 0
    for text in texts:
        start = whole.find(text[:30], pos)
        if start < 0 or not whole:
            spans.append(None)
            continue
        spans.append([round(start / len(whole), 4), round(min(1.0, (start + len(text)) / len(whole)), 4)])
        pos = start + 1
    return spans


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


def _pack(units: list[str], budget: int, overlap: int = 0) -> list[str]:
    """
    按顺序把小段文字拼成不超过 budget 的片段。单个小段超出预算时独占一片，不在这里切开它。

    overlap > 0 时，新片段先带上前一片的最后 overlap 段；带上之后放不下新的一段，就从最早的一段开始少带，
    保证重叠永远不会让片段超出预算。
    """
    pieces: list[list[str]] = []
    current: list[str] = []
    for unit in units:
        size = count_tokens(unit)
        if current and _size(current) + size > budget:
            pieces.append(current)
            current = current[-overlap:] if overlap else []
            while current and _size(current) + size > budget:
                current.pop(0)
        current.append(unit)
    pieces.append(current)
    texts = ["".join(piece).strip() for piece in pieces]
    return [t for t in texts if t]


def _size(units: list[str]) -> int:
    return sum(count_tokens(u) for u in units)


def _sentences(text: str) -> list[str]:
    """断句。只有空白的片段（段内空行）并进前一句，否则句子重叠时可能只重复了一个换行。"""
    sentences: list[str] = []
    for part in _SENTENCE_END.split(text):
        if sentences and not part.strip():
            sentences[-1] += part
        elif part:
            sentences.append(part)
    return sentences


def _fit(sentence: str, budget: int) -> list[str]:
    """超长句子先在逗号、顿号、冒号处断成分句，分句仍然超长才按字数硬切。"""
    if count_tokens(sentence) <= budget:
        return [sentence]
    clauses = [c for c in _CLAUSE_END.split(sentence) if c]
    return [part for clause in clauses for part in _cut(clause, budget)]


def _cut(text: str, budget: int) -> list[str]:
    """没有标点可断的超长文字（OCR 结果、长网址）只能按字数硬切，字数按 token 比例换算。"""
    tokens = count_tokens(text)
    if tokens <= budget:
        return [text]
    step = max(1, len(text) * budget // tokens)
    return [text[i:i + step] for i in range(0, len(text), step)]
