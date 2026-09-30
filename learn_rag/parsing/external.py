"""
parsing.external —— 高精度外部解析工具的适配器：MinerU、Docling。

两个工具都比 PyMuPDF 重得多（需要下载模型，最好有 GPU），但在多栏版面、复杂表格、
公式和扫描件上明显更准。适配器只做一件事：把工具的输出翻译成 Element。
其中的翻译函数（content_list_to_elements / docling_to_elements）不依赖工具本身，可以单独测试。

  MinerU  : 调用它的命令行（mineru -p 文件 -o 目录），读取输出的 *_content_list.json。
            走命令行而不是内部 Python API，是因为命令行是它最稳定的对外接口。
  Docling : 调用 DocumentConverter，遍历 DoclingDocument 中的元素。
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..core.interfaces import DocumentParser
from ..core.registry import registry
from ..core.types import Element


@registry.register("parser", "mineru")
class MinerUParser(DocumentParser):
    """
    backend: pipeline（传统版面模型流水线，CPU 可跑）或 vlm-transformers / vlm-vllm-engine 等 VLM 后端
    lang:    OCR 语言，中文文档用 ch
    """

    def __init__(self, backend: str = "pipeline", lang: str = "ch", command: str = "mineru", timeout: int = 1800) -> None:
        self.backend = backend
        self.lang = lang
        self.command = command
        self.timeout = timeout

    def parse(self, path: Path) -> list[Element]:
        with tempfile.TemporaryDirectory() as out:
            cmd = [self.command, "-p", str(path), "-o", out, "-b", self.backend, "-l", self.lang]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=self.timeout)
            except FileNotFoundError:
                raise RuntimeError(f"找不到 MinerU 命令 '{self.command}'，请先安装：pip install -U \"mineru[core]\"") from None
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"MinerU 解析失败：{(exc.stderr or '').strip()[-500:]}") from None

            outputs = sorted(Path(out).rglob("*_content_list.json"))
            if not outputs:
                raise RuntimeError(f"MinerU 没有生成 content_list.json，输出目录：{out}")
            return content_list_to_elements(json.loads(outputs[0].read_text(encoding="utf-8")))


# MinerU content_list 中与正文无关的块
_MINERU_NOISE = {"header", "footer", "page_number", "page_footnote", "aside_text", "discarded"}


def content_list_to_elements(items: list[dict[str, Any]]) -> list[Element]:
    elements: list[Element] = []
    for item in items:
        kind = item.get("type", "text")
        if kind in _MINERU_NOISE:
            continue
        page = item["page_idx"] + 1 if "page_idx" in item else None
        common = {"page": page, "bbox": item.get("bbox")}

        if kind == "table":
            # 表格正文是 HTML（能表达合并单元格），标题和脚注拼在前后，切分时不会和表格分开
            parts = [*item.get("table_caption", []), item.get("table_body", ""), *item.get("table_footnote", [])]
            text = "\n".join(p for p in parts if p)
            if text:
                elements.append(Element("table", text, extra={"img_path": item.get("img_path", "")}, **common))
        elif kind == "image":
            caption = " ".join(item.get("image_caption", []) + item.get("image_footnote", []))
            if caption:
                elements.append(Element("image", caption, extra={"img_path": item.get("img_path", "")}, **common))
        elif kind == "equation":
            text = item.get("text", "").strip().strip("$").strip()
            if text:
                elements.append(Element("formula", text, **common))
        elif kind == "code":
            text = item.get("code_body", "") or item.get("text", "")
            if text:
                elements.append(Element("code", text, **common))
        elif kind == "list":
            text = "\n".join(item.get("list_items", [])) or item.get("text", "")
            if text.strip():
                elements.append(Element("text", text.strip(), **common))
        else:
            text = (item.get("text") or "").strip()
            if not text:
                continue
            level = int(item.get("text_level") or 0)
            elements.append(Element("heading", text, level=level, **common) if level else Element("text", text, **common))
    return elements


@registry.register("parser", "docling")
class DoclingParser(DocumentParser):
    """Docling 支持 PDF、Word、PPT、Excel、HTML、图片，统一输出 DoclingDocument。转换器初始化要加载模型，只建一次。"""

    def __init__(self) -> None:
        self._converter = None

    def parse(self, path: Path) -> list[Element]:
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError:
                raise RuntimeError("未安装 Docling：pip install docling") from None
            self._converter = DocumentConverter()
        return docling_to_elements(self._converter.convert(str(path)).document)


# 页眉页脚是噪声；图注会随表格、图片一起输出，单独出现会重复。其余未特殊处理的标签都按普通文本处理
_DOCLING_SKIP = {"page_header", "page_footer", "caption"}


def docling_to_elements(doc: Any) -> list[Element]:
    elements: list[Element] = []
    for item, _depth in doc.iterate_items():
        label = str(getattr(item.label, "value", item.label))
        if label in _DOCLING_SKIP:
            continue
        common = _docling_position(item)

        if label == "title":
            elements.append(Element("heading", item.text, level=1, **common))
        elif label == "section_header":
            elements.append(Element("heading", item.text, level=min(getattr(item, "level", 1) + 1, 6), **common))
        elif label == "table":
            caption = item.caption_text(doc) if hasattr(item, "caption_text") else ""
            text = "\n".join(p for p in (caption, item.export_to_markdown(doc=doc)) if p)
            if text:
                elements.append(Element("table", text, **common))
        elif label == "picture":
            caption = item.caption_text(doc) if hasattr(item, "caption_text") else ""
            if caption:
                elements.append(Element("image", caption, **common))
        elif label == "formula":
            if getattr(item, "text", ""):
                elements.append(Element("formula", item.text, **common))
        elif label == "code":
            if getattr(item, "text", ""):
                elements.append(Element("code", item.text, **common))
        elif getattr(item, "text", "").strip():
            elements.append(Element("text", item.text.strip(), **common))
    return elements


def _docling_position(item: Any) -> dict[str, Any]:
    prov = getattr(item, "prov", None)
    if not prov:
        return {}
    box = prov[0].bbox
    return {"page": prov[0].page_no, "bbox": [box.l, box.t, box.r, box.b]}
