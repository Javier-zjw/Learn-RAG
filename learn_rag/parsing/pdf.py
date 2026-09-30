"""
parsing.pdf —— PDF 解析（PyMuPDF）。

对调用方只有一个 parse(path)，内部处理了 PDF 最常见的几类问题：
  - 标题识别：PDF 里没有"标题"这个概念，只有字号。比正文明显大的短文本块视为标题，
    按字号从大到小分配层级；
  - 表格还原：用 PyMuPDF 的表格检测取出单元格，转成 Markdown 表格，并从正文中去掉表格区域的文字；
  - 去噪：页面上下边缘、并且在多数页面上重复出现的文字（页眉页脚、页码）直接丢弃；
  - 跨页段落：上一页最后一段没有以句末标点结束时，与下一页第一段合并；
  - 扫描页：文字层太少的页面渲染成图片交给 OCR（需要配置 parsing.ocr），否则跳过并告警。

版面复杂（多栏、复杂表格、公式多）的文档，换成 MinerU 解析器效果更好。
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from ..core.interfaces import DocumentParser
from ..core.registry import registry
from ..core.types import Element
from .markdown import table_to_markdown

logger = logging.getLogger(__name__)

_MARGIN = 0.07             # 页面上下各 7% 的区域视为页眉页脚候选区
_HEADING_RATIO = 1.15      # 字号超过正文 15% 才可能是标题
_HEADING_MAX_CHARS = 80    # 标题通常很短
_SENTENCE_END = tuple("。！？；：.!?;:")
_CJK = re.compile(r"[一-鿿]")


@registry.register("parser", "pdf")
class PdfParser(DocumentParser):
    def __init__(self, ocr: DocumentParser | None = None, min_chars_per_page: int = 30, dpi: int = 200) -> None:
        """
        ocr: 可选的 OCR 解析器（需要有 recognize 方法，如 VlmOcrParser），由 FileSource 按配置注入
        min_chars_per_page: 文字层少于这个字数的页面视为扫描页
        """
        self.ocr = ocr
        self.min_chars_per_page = min_chars_per_page
        self.dpi = dpi

    def parse(self, path: Path) -> list[Element]:
        import pymupdf

        elements: list[Element] = []
        page_count = 0
        with pymupdf.open(str(path)) as pdf:
            for number, page in enumerate(pdf, start=1):
                page_count += 1
                page_elements = _read_page(page, number)
                if sum(len(e.text) for e in page_elements) < self.min_chars_per_page:
                    page_elements = self._ocr_page(page, number, path)
                elements.extend(page_elements)

        elements = _drop_repeated_margins(elements, page_count)
        _assign_headings(elements)
        elements = _merge_across_pages(elements)
        for element in elements:           # 临时字段只在本模块内使用，不外泄
            element.extra.pop("size", None)
            element.extra.pop("margin", None)
        return elements

    def _ocr_page(self, page, number: int, path: Path) -> list[Element]:
        if self.ocr is None or not hasattr(self.ocr, "recognize"):
            logger.warning("%s 第 %d 页几乎没有文字层（可能是扫描页），未配置 OCR，已跳过", path.name, number)
            return []
        png = page.get_pixmap(dpi=self.dpi).tobytes("png")
        return self.ocr.recognize(png, page=number)


def _read_page(page, number: int) -> list[Element]:
    """读取一页的文字块和表格，按页面上的先后位置排好。字号和是否在页边暂存在 extra 里。"""
    tables = page.find_tables().tables
    table_boxes = [t.bbox for t in tables]
    height = page.rect.height

    items: list[tuple[float, Element]] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:          # 1 为图片块，暂不处理
            continue
        bbox = list(block["bbox"])
        if any(_inside(bbox, box) for box in table_boxes):
            continue
        text, size = _block_text(block)
        if not text:
            continue
        margin = bbox[3] < height * _MARGIN or bbox[1] > height * (1 - _MARGIN)
        element = Element("text", text, page=number, bbox=bbox, extra={"size": size, "margin": margin})
        items.append((bbox[1], element))

    for table in tables:
        markdown = table_to_markdown(table.extract())
        if markdown:
            items.append((table.bbox[1], Element("table", markdown, page=number, bbox=list(table.bbox))))

    items.sort(key=lambda item: item[0])
    return [element for _, element in items]


def _block_text(block: dict) -> tuple[str, float]:
    """拼出一个文字块的文本，并返回其中字数最多的字号。PDF 的换行只是排版折行，中文直接相连，西文补空格。"""
    lines: list[str] = []
    sizes: Counter[float] = Counter()
    for line in block.get("lines", []):
        spans = line.get("spans", [])
        text = "".join(s.get("text", "") for s in spans).strip()
        if text:
            lines.append(text)
        for span in spans:
            sizes[round(span.get("size", 0), 1)] += len(span.get("text", "").strip())
    merged = ""
    for text in lines:
        if merged and not (_CJK.search(merged[-1]) or _CJK.search(text[0])):
            merged += " "
        merged += text
    size = sizes.most_common(1)[0][0] if sizes else 0.0
    return merged.strip(), size


def _inside(bbox: list[float], box) -> bool:
    """文字块的中心点落在表格区域内，就认为它属于表格。"""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _drop_repeated_margins(elements: list[Element], page_count: int) -> list[Element]:
    """
    页边区域里，去掉数字后内容相同、并出现在一半以上页面的文字块 —— 页眉、页脚和页码。
    页数太少时无法判断"重复"，只去掉页边的纯页码。
    """
    def key(e: Element) -> str:
        return re.sub(r"\d+", "#", e.text)

    pages_by_key: dict[str, set[int]] = {}
    for e in elements:
        if e.extra.get("margin"):
            pages_by_key.setdefault(key(e), set()).add(e.page or 0)

    threshold = max(2, page_count // 2)
    repeated = {k for k, pages in pages_by_key.items() if len(pages) >= threshold}

    def is_noise(e: Element) -> bool:
        if not e.extra.get("margin"):
            return False
        return key(e) in repeated or _is_page_number(e.text)

    return [e for e in elements if not is_noise(e)]


def _is_page_number(text: str) -> bool:
    """只由数字和"第 页 共 Page of / -"这类页码用字组成的短文本，例如 "3"、"- 3 -"、"第 3 页"、"Page 3 of 10"。"""
    if len(text) > 15 or not re.search(r"\d", text):
        return False
    return re.sub(r"(?i)page|of|[\d\s/\-—第页共.]", "", text) == ""


def _assign_headings(elements: list[Element]) -> None:
    """正文字号取全文出现最多的字号；明显更大的短文本块是标题，字号越大层级越高（最多三级）。"""
    weights: Counter[float] = Counter()
    for e in elements:
        if e.kind == "text" and "size" in e.extra:
            weights[e.extra["size"]] += len(e.text)
    if not weights:
        return
    body = weights.most_common(1)[0][0]

    candidates = [
        e for e in elements
        if e.kind == "text" and e.extra.get("size", 0) >= body * _HEADING_RATIO and len(e.text) <= _HEADING_MAX_CHARS
    ]
    sizes = sorted({e.extra["size"] for e in candidates}, reverse=True)
    for e in candidates:
        e.kind, e.level = "heading", min(sizes.index(e.extra["size"]) + 1, 3)


def _merge_across_pages(elements: list[Element]) -> list[Element]:
    """上一页最后一段没说完（不以句末标点结尾），就把下一页的第一段接上去。"""
    merged: list[Element] = []
    for e in elements:
        prev = merged[-1] if merged else None
        if (
            prev is not None
            and prev.kind == e.kind == "text"
            and prev.page is not None and e.page == prev.page + 1
            and not prev.text.endswith(_SENTENCE_END)
        ):
            prev.text += ("" if _CJK.search(prev.text[-1]) else " ") + e.text
            prev.extra["page_end"] = e.page
            continue
        merged.append(e)
    return merged
