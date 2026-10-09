"""
parsing.external —— 高精度外部解析工具的适配器：MinerU、Docling。

两个工具都比 PyMuPDF 重得多（需要下载模型，最好有 GPU），但在多栏版面、复杂表格、
公式和扫描件上明显更准。适配器只做一件事：把工具的输出翻译成 Element。
其中的翻译函数（mineru_middle_to_elements / docling_to_elements）不依赖工具本身，可以单独测试。

  MinerU  : 调用 mineru-kit parse --format zip，读取 zip 内的 middle_json.json。
            走命令行而不是内部 Python API，是因为命令行是它最稳定的对外接口。
  Docling : 调用 DocumentConverter，遍历 DoclingDocument 中的元素。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile
from typing import Any

from ..core.interfaces import DocumentParser
from ..core.registry import registry
from ..core.types import Element
from .assets import mime_for, save_asset

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@registry.register("parser", "mineru")
class MinerUParser(DocumentParser):
    """
    MinerU 4.x 适配器。standard 档位适合复杂版面、表格、公式和扫描件；
    CPU 部署通过独立 VLM 服务隔离模型加载，避免每次导入都在进程内复制大模型。
    """

    def __init__(
            self,
            tier: str = "standard",
            command: str = "mineru-kit",
            timeout: int = 1800,
            assets_dir: str | None = None,
            models_dir: str | None = None,
            vlm_server_url: str = "http://127.0.0.1:30000",
            ocr_mode: str = "auto",
            image_analysis: bool = True,
    ) -> None:
        if tier not in {"flash", "basic", "standard", "advanced"}:
            raise ValueError(f"不支持的 MinerU 解析档位：{tier}")
        self.tier = tier
        self.command = command
        self.timeout = timeout
        self.vlm_server_url = vlm_server_url.rstrip("/")
        self.ocr_mode = ocr_mode
        self.image_analysis = image_analysis
        # 相对资产目录按项目根解析，避免从不同目录启动 CLI 时把图片写到别处。
        if assets_dir is None:
            self.assets_dir = None
        else:
            path = Path(assets_dir)
            self.assets_dir = path if path.is_absolute() else _PROJECT_ROOT / path
        # 模型权重默认放在项目内的 mineru_model_weight/（ModelScope 缓存根目录），
        # 不依赖运行机器的用户目录；需要共享缓存时在配置里改 models_dir 即可
        if models_dir is None:
            self.models_dir = _PROJECT_ROOT / "mineru_model_weight"
        else:
            path = Path(models_dir)
            self.models_dir = path if path.is_absolute() else _PROJECT_ROOT / path
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def parse(self, path: Path) -> list[Element]:
        with tempfile.TemporaryDirectory() as out:
            zip_path = Path(out) / f"{path.stem}.zip"
            cmd = [
                self.command, "parse", str(path),
                "-o", str(zip_path), "--format", "zip", "--tier", self.tier,
                "--ocr-mode", self.ocr_mode,
            ]
            if not self.image_analysis:
                cmd.append("--disable-image-analysis")
            # 权重来源和缓存位置由解析器显式控制：换机器、换用户目录都不影响模型加载
            env = dict(os.environ)
            env["MINERU_HOME"] = str(self.models_dir.resolve())
            env["MINERU_MODEL_SOURCE"] = "modelscope"
            env["MINERU_MODEL_SMALL_BACKEND"] = "onnx"
            env["MINERU_MODEL_VLM_ENGINE"] = "llama-cpp"
            if self.vlm_server_url:
                env["MINERU_MODEL_VLM_SERVER_URL"] = self.vlm_server_url
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=self.timeout, env=env)
            except FileNotFoundError:
                raise RuntimeError(f"找不到 MinerU 命令 '{self.command}'，请先安装：pip install -U \"mineru>=4.0,<5\"") from None
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"MinerU 解析失败：{(exc.stderr or '').strip()[-500:]}") from None
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"MinerU 解析超时（{self.timeout} 秒）：{exc}") from None

            if not zip_path.is_file():
                raise RuntimeError(f"MinerU 没有生成结果 zip：{zip_path}")
            try:
                with ZipFile(zip_path) as archive:
                    if "middle_json.json" not in archive.namelist():
                        raise RuntimeError("MinerU 结果 zip 中缺少 middle_json.json")
                    middle = json.loads(archive.read("middle_json.json"))
                    elements = mineru_middle_to_elements(middle)
                    _persist_mineru_assets(elements, archive, self.assets_dir)
                    return elements
            except (OSError, ValueError, KeyError) as exc:
                raise RuntimeError(f"读取 MinerU 结果 zip 失败：{exc}") from None


def _persist_mineru_assets(elements: list[Element], archive: ZipFile, assets_dir: Path | None) -> None:
    """把 zip 中的图片、图表和表格截图复制到内容寻址资产库。

    解析 zip 是临时文件，不能把其中的相对路径直接交给上层；这里在读完后立即
    转成稳定的资产路径。单个资产读取失败只降级为文本元素，不让整份文档失败。
    """
    for element in elements:
        image_path = element.extra.pop("image_path", None)
        if not image_path:
            continue
        if not _safe_zip_member(image_path) or image_path not in archive.namelist():
            logger.warning("MinerU 资产 %s 不存在或路径不安全，仅保留文本：%s", image_path, element.text[:80])
            continue
        try:
            data = archive.read(image_path)
        except OSError as exc:
            logger.warning("读取 MinerU 资产 %s 失败，仅保留文本：%s", image_path, exc)
            continue
        if assets_dir is not None:
            suffix = "." + image_path.rsplit(".", 1)[-1] if "." in image_path else ""
            element.extra.update(save_asset(data, assets_dir, ext=suffix.lstrip(".") or "bin", mime=mime_for(suffix)))


def _safe_zip_member(name: str) -> bool:
    """只接受 zip 内的 POSIX 相对路径，阻止路径穿越进入资产读取逻辑。"""
    return bool(name) and not name.startswith("/") and ".." not in name.split("/")


# MinerU middle JSON 中与正文无关的页面临近信息；正文脚注仍可能有业务价值，单独保留。
_MINERU_NOISE = {"header", "footer", "page_number", "page_footnote", "aside_text", "discarded"}


def mineru_middle_to_elements(middle: dict[str, Any]) -> list[Element]:
    """把 MinerU 4.x 的 docvortex.middle 协议转换成项目统一 Element。"""
    elements: list[Element] = []
    last_by_type: dict[str | None, int] = {}
    for page in middle.get("pages", []):
        page_no = page.get("page_idx", 0) + 1 if "page_idx" in page else None
        for block in page.get("blocks", []):
            element = _mineru_block_to_element(block, page_no)
            if element is None:
                continue
            _merge_continued_element(elements, element, last_by_type)
    return elements


def _mineru_block_to_element(block: dict[str, Any], page: int | None) -> Element | None:
    kind = block.get("type", "text")
    if kind in _MINERU_NOISE:
        return None

    common = {"page": page, "bbox": block.get("bbox")}
    if kind == "doc_title":
        return _heading(block, 1, common)
    if kind == "paragraph_title":
        return _heading(block, int(block.get("level") or 2), common)
    if kind in {"image", "chart"}:
        return _mineru_visual(block, kind, common)
    if kind == "table":
        element = _mineru_table(block, common)
        if element is not None:
            element.extra["mineru_type"] = "table"
            if block.get("continues_prev"):
                element.extra["continues_prev"] = True
        return element
    if kind == "equation":
        text = str(block.get("content", "")).strip()
        if text:
            return Element("formula", text, **common)
        return None
    if kind == "code":
        body = _body_text(block, {"code_body", "algorithm_body"})
        captions = _annotations(block, {"code_caption"}, "caption")
        footnotes = _annotations(block, {"code_footnote"}, "footnote")
        text = "\n".join(part for part in [*captions, body, *footnotes] if part)
        if text:
            return Element("code", text, **common)
        return None
    if kind == "list":
        items = [_inline_text(child) for child in block.get("content", [])]
        text = "\n".join(item for item in items if item)
        if text:
            return Element("text", text, extra={"list": True, "mineru_type": "list"}, **common)
        return None

    text = _inline_text(block)
    if text:
        extra = {"mineru_type": kind}
        if block.get("continues_prev"):
            extra["continues_prev"] = True
        return Element("text", text, extra=extra, **common)
    return None


def _heading(block: dict[str, Any], level: int, common: dict[str, Any]) -> Element | None:
    text = _inline_text(block)
    if not text:
        return None
    return Element("heading", text, level=max(1, min(level, 6)), **common)


def _mineru_visual(block: dict[str, Any], mineru_type: str, common: dict[str, Any]) -> Element:
    body = _child(block, {"image_body", "chart_body"})
    captions = _annotations(block, {"image_caption", "chart_caption"}, "caption")
    footnotes = _annotations(block, {"image_footnote", "chart_footnote"}, "footnote")
    text = "\n".join(part for part in [*captions, *footnotes] if part) or ("[图片]" if mineru_type == "image" else "[图表]")
    extra: dict[str, Any] = {"mineru_type": mineru_type}
    image_path = body.get("image_path") if body else None
    if image_path:
        extra["image_path"] = image_path
    return Element("image", text, extra=extra, **common)


def _mineru_table(block: dict[str, Any], common: dict[str, Any]) -> Element | None:
    body = _child(block, {"table_body"})
    body_text = str(body.get("content", "")) if body else ""
    captions = _annotations(block, {"table_caption"}, "caption")
    footnotes = _annotations(block, {"table_footnote"}, "footnote")
    text = "\n".join(part for part in [*captions, body_text, *footnotes] if part)
    if not text:
        return None
    extra: dict[str, Any] = {}
    if body and body.get("image_path"):
        extra["image_path"] = body["image_path"]
    return Element("table", text, extra=extra, **common)


def _merge_continued_element(elements: list[Element], element: Element, last_by_type: dict[str | None, int]) -> None:
    """合并 MinerU 标记的跨页续块，避免一个段落被切成两个检索片段。"""
    continued = element.extra.pop("continues_prev", None)
    index = last_by_type.get(element.extra.get("mineru_type")) if continued else None
    if index is not None and 0 <= index < len(elements):
        previous = elements[index]
        # 只合并同类且页码不回退的块；版面模型可能把图片插在两段文字之间，不能只看列表末尾。
        if previous.kind == element.kind and previous.page and element.page and element.page >= previous.page:
            separator = "" if element.kind == "text" else "\n"
            previous.text += separator + element.text
            if element.page > previous.page:
                previous.extra["page_end"] = element.page
            return

    elements.append(element)
    last_by_type[element.extra.get("mineru_type")] = len(elements) - 1


def _child(block: dict[str, Any], types: set[str]) -> dict[str, Any] | None:
    for child in block.get("content", []):
        if isinstance(child, dict) and child.get("type") in types:
            return child
    return None


def _body_text(block: dict[str, Any], types: set[str]) -> str:
    child = _child(block, types)
    return str(child.get("content", "")).strip() if child else ""


def _annotations(block: dict[str, Any], types: set[str], suffix: str) -> list[str]:
    result: list[str] = []
    for child in block.get("content", []):
        if isinstance(child, dict) and child.get("type") in types and str(child.get("type", "")).endswith(suffix):
            text = _inline_text(child)
            if text:
                result.append(text)
    return result


def _inline_text(block: dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for span in content:
        if not isinstance(span, dict):
            continue
        span_type = str(span.get("type", ""))
        if span_type == "equation_inline":
            value = str(span.get("content", "")).strip()
            parts.append(f"${value}$" if value else "")
        elif span_type == "hyperlink":
            parts.append(_inline_text(span) or str(span.get("url", "")))
        else:
            parts.append(_inline_text(span))
    return "".join(parts).strip()


@registry.register("parser", "docling")
class DoclingParser(DocumentParser):
    """Docling 支持 PDF、Word、PPT、Excel、HTML、图片，统一输出 DoclingDocument。转换器初始化要加载模型，只建一次。"""

    def __init__(self, assets_dir: str | None = None) -> None:
        self._converter = None
        self.assets_dir = assets_dir

    def parse(self, path: Path) -> list[Element]:
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
            except ImportError:
                raise RuntimeError("未安装 Docling：pip install docling") from None
            self._converter = DocumentConverter()
        return docling_to_elements(self._converter.convert(str(path)).document, assets_dir=self.assets_dir)


# 页眉页脚是噪声；图注会随表格、图片一起输出，单独出现会重复。其余未特殊处理的标签都按普通文本处理
_DOCLING_SKIP = {"page_header", "page_footer", "caption"}


def docling_to_elements(doc: Any, *, assets_dir: str | None = None) -> list[Element]:
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
                extra = _save_docling_image(item, assets_dir) if assets_dir else {}
                elements.append(Element("image", caption, extra=extra, **common))
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


def _save_docling_image(item: Any, assets_dir: str) -> dict[str, Any]:
    """把 Docling 的图片落到资产库；图取不到就只保留图注，图片存储不能拖垮解析。

    Docling 各版本里图片的载体不完全一样（PIL 对象 / 本地路径 / data URI），
    这里统一走 pil_image 属性，取不到或抛异常都静默降级为"只有图注"。
    """
    try:
        ref = getattr(item, "image", None)
        if ref is None:
            return {}
        pil = getattr(ref, "pil_image", None)
        if pil is None:
            return {}
        import io

        mime = getattr(ref, "mimetype", "") or "image/png"
        fmt = "JPEG" if "jpeg" in mime or "jpg" in mime else "PNG"
        buf = io.BytesIO()
        pil.save(buf, format=fmt)
        return save_asset(buf.getvalue(), assets_dir, ext=fmt.lower(), mime=mime)
    except Exception as exc:
        logger.warning("保存 Docling 图片失败，仅保留图注：%s", exc)
        return {}
