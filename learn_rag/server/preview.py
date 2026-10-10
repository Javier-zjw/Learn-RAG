"""
server.preview —— 原文预览：把文档渲染成页面图片，并找出每个分块在页面上的位置。

查看页在原文页面上框出分块，用户才能对照原文判断分块是否完整、切口是否合理。定位有两个来源：
  1. 文字层对齐（优先）：页面有文字层时（原生 PDF、Office 转出的 PDF、带文字层的 MinerU 输入），
     把分块正文和页面上的字符序列对齐，得到每一行的精确矩形。不依赖解析器给不给坐标，旧知识库也能用；
  2. 版面坐标（兜底）：扫描页没有文字层，用分块元数据 regions 里的元素坐标（MinerU 给出），精确到版面块。
对齐时只比较字母、数字和汉字：PDF 文字层里的换行、连字符、全角半角、Markdown 的表格竖线和井号
都会让逐字比较失败，去掉之后两边才对得上。

同时算出页面上"没有进入任何分块"的文字（页眉页脚是解析层故意丢弃的，其余多半是解析遗漏），
页面上标出来，一眼看出分块是否完整。

Word、PPT、Excel 先用 LibreOffice 转成 PDF 再渲染；Markdown、CSV、网页等没有版面，只能按文本查看。
页面图片、转换出的 PDF、定位结果都按"文件内容哈希"缓存，多个知识库共用，同一组分块只算一次。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from ..parsing.legacy import convert

# 直接能用 PyMuPDF 打开的格式（图片会被当成一页）
_DIRECT = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
# 要先用 LibreOffice 转成 PDF 的格式
_OFFICE = {".doc", ".docx", ".rtf", ".odt", ".ppt", ".pptx", ".odp", ".xls", ".xlsx", ".ods"}
# 表格默认按打印分页，宽表的右侧几列会被挪到别的页上，一行被拆开就和解析出的行对不上；每个工作表导出成一整页
_SHEETS = {".xls", ".xlsx", ".ods"}
_SHEET_PDF = 'pdf:calc_pdf_Export:{"SinglePageSheets":{"type":"boolean","value":"true"}}'
_SEGMENT = 12          # 对齐时每段的字数：太短容易错配到别处的相同文字，太长遇到一个错字整段都对不上
_MIN_CLUSTER = 24      # 一个分块里孤立的一小段匹配（少于两段）多半是别处的重复文字，丢掉
_MIN_GAP = 6           # 少于这么多字的未覆盖文字不标（页码、孤立的标点数字）
_TEXT_RATIO = 0.5      # 正文一半以上能在文字层对上，才用文字层的位置
_PAGE_WIDTH = 1400     # 页面图片的目标宽度（像素）
_NEAR = 400           # 拆半后只在上一处匹配之后这么多字以内找
# 只认真正的 HTML 标签（合并单元格的表格是 HTML）；"<服务器IP>" 这类尖括号里的正文要保留
_HTML_TAG = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*)?/?>")


class PreviewUnavailable(Exception):
    """这个文件不能按原文预览（格式没有版面、缺少依赖或转换失败），消息直接显示在页面上。"""


class Preview:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def layout(self, file: Path, digest: str, chunks: list[dict[str, Any]], headings: list[str]) -> dict[str, Any]:
        """
        chunks 是按顺序排列的子块 [{id, text, regions}]，headings 是文档里的章节标题。返回：
          pages     每页的宽高（PDF 点）
          regions   {子块 id: [{page, bbox}]}，bbox 是 [0,1] 归一化坐标，左上角为原点
          methods   {子块 id: text | layout | none}，位置来自文字层、版面坐标，还是没能定位
          uncovered 没有进入任何分块的文字的位置
          coverage  页面文字被分块覆盖的比例；整份文档都没有文字层（扫描件）时为 None
        """
        key = hashlib.sha1(json.dumps([[c["id"] for c in chunks], headings], ensure_ascii=False).encode()).hexdigest()
        cached = self.cache_dir / digest / f"layout-{key[:16]}.json"
        if cached.is_file():
            return json.loads(cached.read_text(encoding="utf-8"))
        with self._lock(digest):
            pdf = self._pdf(file, digest)
            fitz = _pymupdf()
            with fitz.open(pdf) as doc:
                pages, stream, boxes = _read_text(doc)
            result = _locate(pages, stream, boxes, chunks, headings)
            _write_atomic(cached, json.dumps(result, ensure_ascii=False).encode("utf-8"))
        return result

    def page_image(self, file: Path, digest: str, page: int) -> Path:
        """第 page 页（从 1 开始）的 PNG，渲染一次后缓存。"""
        target = self.cache_dir / digest / f"page-{page}.png"
        if target.is_file():
            return target
        with self._lock(digest):
            pdf = self._pdf(file, digest)
            fitz = _pymupdf()
            with fitz.open(pdf) as doc:
                if not 1 <= page <= doc.page_count:
                    raise LookupError(f"没有第 {page} 页（共 {doc.page_count} 页）")
                rect = doc[page - 1].rect
                zoom = min(3.0, max(1.0, _PAGE_WIDTH / rect.width))
                png = doc[page - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False).tobytes("png")
            _write_atomic(target, png)
        return target

    def _pdf(self, file: Path, digest: str) -> Path:
        """能直接渲染的文件原样返回，Office 文件转成 PDF 并缓存。"""
        suffix = file.suffix.lower()
        if suffix in _DIRECT:
            return file
        if suffix not in _OFFICE:
            raise PreviewUnavailable(f"{suffix or '这种'} 文件没有页面版面，只能按文本查看")
        target = self.cache_dir / digest / "document.pdf"
        if not target.is_file():
            with tempfile.TemporaryDirectory(prefix="learn-rag-preview-") as out:
                try:
                    converted = convert(file, _SHEET_PDF if suffix in _SHEETS else "pdf", Path(out), timeout=300)
                except RuntimeError as exc:
                    raise PreviewUnavailable(f"Word、PPT、Excel 要先转成 PDF 才能显示原文：{exc}") from None
                _write_atomic(target, converted.read_bytes())
        return target

    def _lock(self, digest: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(digest, threading.Lock())


def _pymupdf():
    try:
        import pymupdf
    except ImportError:
        raise PreviewUnavailable('原文预览需要 PyMuPDF：pip install -e ".[parsing]"') from None
    return pymupdf


def _normalize(text: str) -> str:
    """只留字母、数字和汉字，统一全角半角和大小写。"""
    text = unicodedata.normalize("NFKC", _HTML_TAG.sub(" ", text)).lower()
    return "".join(ch for ch in text if ch.isalnum())


def _read_text(doc: Any) -> tuple[list[dict[str, float]], str, list[tuple[int, int, float, float, float, float]]]:
    """
    全文的字符序列（已规范化）和每个字符的位置 (页码, 行号, x0, y0, x1, y1)，坐标归一化到 [0,1]。
    行号在整份文档里唯一，同一行的字符最后会合成一个矩形。
    """
    pages: list[dict[str, float]] = []
    chars: list[str] = []
    boxes: list[tuple[int, int, float, float, float, float]] = []
    line_id = 0
    for number, page in enumerate(doc, start=1):
        width, height = page.rect.width, page.rect.height
        pages.append({"width": round(width, 2), "height": round(height, 2)})
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                line_id += 1
                for span in line["spans"]:
                    for char in span["chars"]:
                        x0, y0, x1, y1 = char["bbox"]
                        for ch in _normalize(char["c"]):      # 连字 ﬁ 规范化后是两个字符，位置相同
                            chars.append(ch)
                            boxes.append((number, line_id, x0 / width, y0 / height, x1 / width, y1 / height))
    return pages, "".join(chars), boxes


def _locate(pages: list[dict[str, float]], stream: str, boxes: list[tuple], chunks: list[dict[str, Any]],
            headings: list[str]) -> dict[str, Any]:
    covered = bytearray(len(stream))
    regions: dict[str, list[dict[str, Any]]] = {}
    methods: dict[str, str] = {}
    position = 0
    for chunk in chunks:
        spans, ratio = _match(stream, _normalize(chunk["text"]), position)
        # 文字层对上一半以上用文字层；对不上（公式多、扫描页）有版面坐标就用坐标；都没有时能对上几行算几行
        if spans and (ratio >= _TEXT_RATIO or not chunk.get("regions")):
            regions[chunk["id"]] = _rects(boxes, spans)
            methods[chunk["id"]] = "text"
            position = spans[-1][1]
            for start, end in spans:
                covered[start:end] = b"\x01" * (end - start)
        elif chunk.get("regions"):
            regions[chunk["id"]] = chunk["regions"]
            methods[chunk["id"]] = "layout"
            _cover_inside(covered, boxes, chunk["regions"])
        else:
            regions[chunk["id"]] = []
            methods[chunk["id"]] = "none"
    # 章节标题不在子块正文里（它在块首路径和父块里），也算覆盖，否则每个标题都会被标成遗漏
    line_sizes = Counter(box[1] for box in boxes)
    for heading in headings:
        span = _find_heading(stream, boxes, line_sizes, _normalize(heading))
        if span:
            covered[span[0]:span[1]] = b"\x01" * (span[1] - span[0])
    gaps = _gaps(covered)
    return {
        "pages": pages,
        "regions": regions,
        "methods": methods,
        "uncovered": _rects(boxes, gaps),
        "coverage": round(sum(covered) / len(stream), 4) if stream else None,
    }


def _match(stream: str, target: str, start: int, *, cluster: bool = True) -> tuple[list[list[int]], float]:
    """
    在 stream 里找 target，返回匹配上的区间和 target 被对上的比例。

    target 按 _SEGMENT 个字一段分别查找，优先在上一段之后找（分块按原文顺序排列），找不到再全文找
    （双栏版面里解析顺序和文字层顺序可能不同）。一段对不上时（表格里打印格式和单元格的值不同，如 5% 和 0.05）
    拆成两半只在附近找，避免短字符串错配到远处。每处匹配再向前后逐字延伸，段与段的接缝和小差异不会漏字。
    """
    if len(target) < 4:
        return [], 0.0
    # 最后一段不足 _SEGMENT 个字时改用结尾的 _SEGMENT 个字（和前一段有重叠），块尾的几个字才不会漏掉
    offsets = list(range(0, max(len(target) - _SEGMENT, 0) + 1, _SEGMENT))
    if len(target) > _SEGMENT and len(target) % _SEGMENT:
        offsets.append(len(target) - _SEGMENT)
    hits: list[tuple[int, int, int]] = []          # (stream 起点, target 起点, 长度)
    position = start
    for offset in offsets:
        segment = target[offset:offset + _SEGMENT]
        index = stream.find(segment, position)
        if index < 0:
            index = stream.find(segment)
        if index >= 0:
            hits.append((index, offset, len(segment)))
            position = index + len(segment)
            continue
        half = len(segment) // 2
        if half < 4:
            continue
        for part_offset in (offset, offset + half):
            part = target[part_offset:part_offset + half]
            index = stream.find(part, position, position + _NEAR)
            if index >= 0:
                hits.append((index, part_offset, len(part)))
                position = index + len(part)

    matched = bytearray(len(target))
    spans: list[list[int]] = []
    for index, offset, length in hits:
        a, b, ta, tb = index, index + length, offset, offset + length
        while b < len(stream) and tb < len(target) and stream[b] == target[tb]:
            b, tb = b + 1, tb + 1
        while a > 0 and ta > 0 and stream[a - 1] == target[ta - 1]:
            a, ta = a - 1, ta - 1
        matched[ta:tb] = b"\x01" * (tb - ta)
        spans.append([a, b])
    merged: list[list[int]] = []
    for span in sorted(spans):
        if merged and span[0] - merged[-1][1] <= 2 * _SEGMENT:
            merged[-1][1] = max(merged[-1][1], span[1])
        else:
            merged.append(span)
    if cluster and len(merged) > 1:
        merged = [span for span in merged if span[1] - span[0] >= _MIN_CLUSTER] or merged
    return merged, sum(matched) / len(target)


def _find_heading(stream: str, boxes: list[tuple], line_sizes: Counter, target: str) -> list[int] | None:
    """
    标题在原文里的位置。优先取独占一行的那一处：文档标题常常也出现在页眉里（"星河科技 · 数据平台产品手册"），
    取第一处会把页眉当成标题，真正的标题反而被标成遗漏。没有独占一行的（标题折行了）取第一处。
    """
    if len(target) < 2:
        return None
    first = None
    index = stream.find(target)
    while index >= 0:
        line = boxes[index][1]
        if boxes[index + len(target) - 1][1] == line and line_sizes[line] == len(target):
            return [index, index + len(target)]
        if first is None:
            first = [index, index + len(target)]
        index = stream.find(target, index + 1)
    return first


def _cover_inside(covered: bytearray, boxes: list[tuple], regions: list[dict[str, Any]]) -> None:
    """按版面坐标定位的块：坐标框里的文字都算覆盖，否则框里的公式、表格文字会被误标成遗漏。"""
    for index, (page, _, x0, y0, x1, y1) in enumerate(boxes):
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for region in regions:
            bx0, by0, bx1, by1 = region["bbox"]
            if region["page"] == page and bx0 <= cx <= bx1 and by0 <= cy <= by1:
                covered[index] = 1
                break


def _gaps(covered: bytearray) -> list[list[int]]:
    """连续 _MIN_GAP 个字以上没被覆盖的区间。"""
    gaps: list[list[int]] = []
    start = None
    for index, flag in enumerate(covered):
        if not flag and start is None:
            start = index
        elif flag and start is not None:
            if index - start >= _MIN_GAP:
                gaps.append([start, index])
            start = None
    if start is not None and len(covered) - start >= _MIN_GAP:
        gaps.append([start, len(covered)])
    return gaps


def _rects(boxes: list[tuple], spans: list[list[int]]) -> list[dict[str, Any]]:
    """把字符区间合成每行一个矩形。"""
    lines: dict[tuple[int, int], list[float]] = {}
    for start, end in spans:
        for page, line, x0, y0, x1, y1 in boxes[start:end]:
            box = lines.get((page, line))
            if box is None:
                lines[(page, line)] = [x0, y0, x1, y1]
            else:
                box[0], box[1], box[2], box[3] = min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1)
    return [{"page": page, "bbox": [round(v, 4) for v in box]} for (page, _), box in lines.items()]


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
