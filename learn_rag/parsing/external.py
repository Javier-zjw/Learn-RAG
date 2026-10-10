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
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile
from typing import Any

from ..core.interfaces import DocumentParser
from ..core.registry import registry
from ..core.types import Element
from .assets import mime_for, resolve_assets_dir, save_asset
from .markdown import normalize_table

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# MinerU 4.x 中可以选择质量档位（flash/basic/standard/advanced）的输入格式
_TIERED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".jp2"}
# .mineru.env 没写 MINERU_PYTHON_ENV 时的默认环境，和 scripts/start_mineru.sh 保持一致
_DEFAULT_MINERU_PYTHON_ENV = "/opt/anaconda3/envs/langchain_env"


def read_mineru_env(env_file: str | None = ".mineru.env") -> dict[str, str]:
    """读取 .mineru.env（相对路径按项目根解析），文件不存在时返回空字典。"""
    if not env_file:
        return {}
    path = Path(env_file) if Path(env_file).is_absolute() else _PROJECT_ROOT / env_file
    return _read_env_file(path) if path.is_file() else {}


def find_mineru_command(command: str = "mineru-kit", env: dict[str, str] | None = None) -> str | None:
    """
    找到 MinerU 命令的完整路径：先在 PATH 里找，再到 MINERU_PYTHON_ENV/bin 下找。
    MinerU 依赖很重，常装在单独的 Python 环境里，不在当前环境的 PATH 上；启动脚本用的也是 MINERU_PYTHON_ENV。
    """
    env = env if env is not None else {**os.environ, **read_mineru_env()}
    found = shutil.which(command, path=env.get("PATH"))
    if found:
        return found
    candidate = Path(env.get("MINERU_PYTHON_ENV") or _DEFAULT_MINERU_PYTHON_ENV) / "bin" / command
    return str(candidate) if candidate.is_file() and os.access(candidate, os.X_OK) else None


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
            raw_dir: str | None = None,
            env_file: str | None = ".mineru.env",
            replay_dir: str | None = None,
    ) -> None:
        """
        raw_dir:  设置后把 MinerU 的原始结果 zip（含 middle_json.json 和图片）另存一份，用于排查解析问题
        replay_dir: 设置后优先读取该目录里已保存的结果 zip（即 raw_dir 存下的文件），不再调用 MinerU。
                  修改适配器后用它在已有的 MinerU 输出上快速验证，不必重新跑一遍耗时的解析
        env_file: MinerU 运行环境文件，与 scripts/start_mineru.sh 读的是同一份，保证两边的模型目录、
                  VLM 服务地址一致；相对路径按项目根解析，设为 None 时不读取
        """
        if tier not in {"flash", "basic", "standard", "advanced"}:
            raise ValueError(f"不支持的 MinerU 解析档位：{tier}")
        self.tier = tier
        self.command = command
        self.timeout = timeout
        self.vlm_server_url = vlm_server_url.rstrip("/")
        self.ocr_mode = ocr_mode
        self.image_analysis = image_analysis
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.replay_dir = Path(replay_dir) if replay_dir else None
        self.assets_dir = resolve_assets_dir(assets_dir) if assets_dir else None
        # 模型权重默认放在项目内的 mineru_model_weight/（ModelScope 缓存根目录），
        # 不依赖运行机器的用户目录；.mineru.env 里设置了 MINERU_HOME 时以它为准
        default_home = _PROJECT_ROOT / "mineru_model_weight" if models_dir is None else Path(models_dir)
        if not default_home.is_absolute():
            default_home = _PROJECT_ROOT / default_home
        self.env = self._runtime_env(env_file, default_home)
        self.command = find_mineru_command(command, self.env) or command
        self.models_dir = Path(self.env["MINERU_HOME"])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _runtime_env(self, env_file: str | None, default_home: Path) -> dict[str, str]:
        """
        MinerU 子进程的环境变量，优先级：.mineru.env > 当前 shell 的环境变量 > 解析器默认值。

        .mineru.env 的优先级最高，是为了和 start_mineru.sh（用 source 加载同一个文件）保持一致：
        启动服务和解析文档用的永远是同一套模型目录和 VLM 地址。实际生效的值会打印到日志里。
        """
        env = dict(os.environ)
        source = "未找到 .mineru.env，使用 shell 环境变量和默认值"
        values = read_mineru_env(env_file)
        if values:
            env.update(values)
            source = f"已加载 {env_file}"
        defaults = {
            "MINERU_HOME": str(default_home.resolve()),
            "MINERU_MODEL_SOURCE": "modelscope",
            "MINERU_MODEL_SMALL_BACKEND": "onnx",
            "MINERU_MODEL_VLM_ENGINE": "llama-cpp",
        }
        if self.vlm_server_url:
            defaults["MINERU_MODEL_VLM_SERVER_URL"] = self.vlm_server_url
        for key, value in defaults.items():
            env.setdefault(key, value)
        logger.info("MinerU 运行环境：%s；MINERU_HOME=%s；VLM 服务=%s",
                    source, env["MINERU_HOME"], env.get("MINERU_MODEL_VLM_SERVER_URL", "未设置"))
        return env

    def parse(self, path: Path) -> list[Element]:
        recorded = self.replay_dir / f"{path.name}.zip" if self.replay_dir else None
        if recorded is not None and recorded.is_file():
            return self._read_result(recorded)
        with tempfile.TemporaryDirectory() as out:
            zip_path = Path(out) / f"{path.stem}.zip"
            cmd = [self.command, "parse", str(path), "-o", str(zip_path), "--format", "zip"]
            # 只有 PDF 和图片区分质量档位；Office、HTML、CSV 等固定走 flash（直接读文件结构），
            # 对它们显式传 --tier 时 MinerU 会直接报错，所以这些参数只在可分档的格式上传
            if path.suffix.lower() in _TIERED_SUFFIXES:
                cmd += ["--tier", self.tier, "--ocr-mode", self.ocr_mode]
                if not self.image_analysis:
                    cmd.append("--disable-image-analysis")
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=self.timeout, env=self.env)
            except FileNotFoundError:
                raise RuntimeError(f"找不到 MinerU 命令 '{self.command}'，请先安装：pip install -U \"mineru>=4.0,<5\"") from None
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"MinerU 解析失败：{(exc.stderr or '').strip()[-500:]}") from None
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"MinerU 解析超时（{self.timeout} 秒）：{exc}") from None

            if not zip_path.is_file():
                raise RuntimeError(f"MinerU 没有生成结果 zip：{zip_path}")
            if self.raw_dir is not None:
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(zip_path, self.raw_dir / f"{path.name}.zip")   # 用完整文件名，避免同名不同格式互相覆盖
            return self._read_result(zip_path)

    def _read_result(self, zip_path: Path) -> list[Element]:
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


def _read_env_file(path: Path) -> dict[str, str]:
    """按 shell 的写法读取 KEY=VALUE：支持 export 前缀、引号、注释行和行尾注释，展开 $VAR / ~。"""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.removeprefix("export ").split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        values[key.strip()] = os.path.expanduser(os.path.expandvars(value))
    return values


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


# 与正文无关的页面元素，直接丢弃。page_footnote 不在其中：PDF 的页脚注释和 PPT 的演讲者备注
# 在 MinerU 里都是 page_footnote，属于正文内容，保留为带 footnote 标记的段落。
_MINERU_NOISE = {"header", "footer", "page_number", "aside_text", "discarded"}
# image_body 的内容是图片描述；PPT 等文档里它有时只是图片的原始文件名（如 "image.png"），没有信息量
_BARE_FILENAME = re.compile(r"^[\w\-. ]+\.(png|jpe?g|gif|bmp|svg|emf|wmf|tiff?|webp)$", re.I)
_SENTENCE_END = tuple("。！？；：.!?;:")
_SOFT_BREAK = tuple("，、,")
_COLUMN_TOLERANCE = 0.03     # bbox 是 [0,1] 归一化坐标；左边界相差不到 3% 视为同一栏
_CJK = re.compile(r"[　-〿一-鿿＀-￯]")


def mineru_middle_to_elements(middle: dict[str, Any]) -> list[Element]:
    """把 MinerU 4.x 的 docvortex.middle 协议转换成项目统一 Element。"""
    elements: list[Element] = []
    for page in middle.get("pages", []):
        page_no = page.get("page_idx", 0) + 1 if "page_idx" in page else None
        for block in page.get("blocks", []):
            converted = _mineru_block_to_elements(block, page_no)
            if converted and block.get("continues_prev") and block.get("type") == "table":
                if elements and elements[-1].kind == "table":
                    _append_table_rows(elements[-1], converted.pop(0))
            elements.extend(converted)
    return _join_wrapped_text(elements)


def _mineru_block_to_elements(block: dict[str, Any], page: int | None) -> list[Element]:
    kind = block.get("type", "text")
    if kind in _MINERU_NOISE:
        return []

    common = {"page": page, "bbox": block.get("bbox")}
    if kind in {"image", "chart"}:
        return _mineru_visual(block, kind, common)
    if kind == "table":
        return _mineru_table(block, common)
    if kind == "equation":
        text = str(block.get("content", "")).strip()
        return [Element("formula", text, **common)] if text else []
    if kind == "code":
        body = _child(block, {"code_body", "algorithm_body"})
        parts = [*_texts(block, {"code_caption"}), str(body.get("content", "")).strip() if body else "",
                 *_texts(block, {"code_footnote"})]
        text = "\n".join(p for p in parts if p)
        return [Element("code", text, **common)] if text else []

    # index 是目录，结构和列表一样
    text = "\n".join(_list_lines(block)) if kind in {"list", "index"} else _inline_text(block)
    if not text:
        return []
    if kind == "doc_title":
        return [Element("heading", text, level=1, **common)]
    if kind == "paragraph_title":
        return [Element("heading", text, level=max(1, min(int(block.get("level") or 2), 6)), **common)]
    extra: dict[str, Any] = {"mineru_type": kind}
    if kind == "page_footnote":
        extra["footnote"] = True
    return [Element("text", text, extra=extra, **common)]


def _mineru_visual(block: dict[str, Any], mineru_type: str, common: dict[str, Any]) -> list[Element]:
    """
    图片和图表：
      - 原生图表（PPT / Excel 里的图表）的 chart_body 是图表数据转成的表格，作为表格元素输出，数据能被检索到；
      - 其余情况输出图片元素，文字是图题加图片描述，没有任何文字时用 [图片] / [图表] 占位，保证原图仍能被引用；
      - 图注（footnote）单独输出为段落：MinerU 有时会把紧跟在图后面的正文段落识别成图注。
    """
    body = _child(block, {"image_body", "chart_body"}) or {}
    content = str(body.get("content") or "").strip()
    captions = _texts(block, {"image_caption", "chart_caption"})
    extra: dict[str, Any] = {"mineru_type": mineru_type}
    if body.get("image_path"):
        extra["image_path"] = body["image_path"]

    if mineru_type == "chart" and content:
        element = Element("table", "\n".join([*captions, normalize_table(content)]), extra=extra, **common)
    else:
        description = "" if _BARE_FILENAME.match(content) else content
        text = "\n".join(p for p in [*captions, description] if p) or ("[图片]" if mineru_type == "image" else "[图表]")
        element = Element("image", text, extra=extra, **common)

    footnotes = [
        Element("text", text, page=common["page"], bbox=child.get("bbox"), extra={"mineru_type": child["type"]})
        for child in _children(block, {"image_footnote", "chart_footnote"})
        if (text := _inline_text(child))
    ]
    return [element, *footnotes]


def _mineru_table(block: dict[str, Any], common: dict[str, Any]) -> list[Element]:
    """表题、表格、表注拼在一起：表注通常是"注：数据未经审计"这类说明，和表格分开就失去了上下文。"""
    body = _child(block, {"table_body"}) or {}
    parts = [*_texts(block, {"table_caption"}), normalize_table(str(body.get("content") or "")),
             *_texts(block, {"table_footnote"})]
    text = "\n".join(p for p in parts if p)
    if not text:
        return []
    extra: dict[str, Any] = {"mineru_type": "table"}
    if body.get("image_path"):
        extra["image_path"] = body["image_path"]
    return [Element("table", text, extra=extra, **common)]


def _append_table_rows(table: Element, continued: Element) -> None:
    """
    跨页续表：Markdown 表格只追加数据行，其他情况直接拼接。
    续表转 Markdown 时第一行会被当成表头、后面跟一行分隔线：分隔线总是去掉；
    第一行只有和原表头相同（续页重复了表头）时才去掉，否则它是数据行，要保留。
    """
    lines = continued.text.splitlines()
    header = next((line for line in table.text.splitlines() if line.startswith("|")), None)
    if header and len(lines) >= 2 and lines[0].startswith("|") and set(lines[1]) <= set("|- "):
        lines = ([] if lines[0] == header else [lines[0]]) + lines[2:]
    table.text += "\n" + "\n".join(lines)
    if continued.page and table.page and continued.page > table.page:
        table.extra["page_end"] = continued.page


def _join_wrapped_text(elements: list[Element]) -> list[Element]:
    """
    把版面上被折行拆开的段落接回去（同页的上下两块，或上一页末尾和下一页开头）。

    MinerU 按版面块输出，同一段话常被拆成两块。它的 continues_prev 标记既会漏标也会误标
    （样例里真正跨页的句子没有标，扫描件里印章反而和下一页的条款标成了一段），
    所以正文只看版面几何，两个条件同时满足才合并：
      1. 上一块不以句末标点结尾；
      2. 上一块以逗号、顿号结尾，或者它写满了所在栏的宽度 —— 说明是排版折行，而不是一段话结束了。
    只处理带坐标的正文块（PDF、图片）；Office 等流式文档本来就按段落输出。
    """
    out: list[Element] = []
    for element in elements:
        target = next((e for e in reversed(out) if not e.extra.get("footnote")), None)
        if target is not None and _continues(target, element, elements):
            separator = "" if _CJK.match(target.text[-1]) or _CJK.match(element.text[0]) else " "
            target.text += separator + element.text
            if element.page and target.page and element.page > target.page:
                target.extra["page_end"] = element.page
            continue
        out.append(element)
    return out


def _continues(prev: Element, cur: Element, elements: list[Element]) -> bool:
    if not (_is_body_text(prev) and _is_body_text(cur)):
        return False
    same_page_below = cur.page == prev.page and cur.bbox[1] >= prev.bbox[1]
    if not (same_page_below or cur.page == prev.page + 1):
        return False
    text = prev.text.rstrip()
    if text.endswith(_SENTENCE_END):
        return False
    return text.endswith(_SOFT_BREAK) or _fills_column(prev, elements)


def _is_body_text(element: Element) -> bool:
    return (element.kind == "text" and bool(element.bbox) and element.page is not None
            and not element.extra.get("footnote") and element.extra.get("mineru_type") not in {"list", "index"})


def _fills_column(element: Element, elements: list[Element]) -> bool:
    """所在栏 = 同一页上左边界相近的正文块；至少两块才算一栏，右边界达到栏宽才算写满。"""
    peers = [e for e in elements if _is_body_text(e) and e.page == element.page
             and abs(e.bbox[0] - element.bbox[0]) <= _COLUMN_TOLERANCE]
    return len(peers) >= 2 and element.bbox[2] >= max(e.bbox[2] for e in peers) - _COLUMN_TOLERANCE


def _list_lines(block: dict[str, Any], depth: int = 0) -> list[str]:
    """列表展开成每项一行，嵌套的子列表每深一层缩进两个空格。"""
    lines: list[str] = []
    for child in block.get("content", []):
        if not isinstance(child, dict):
            continue
        if child.get("type") in {"list", "index"}:
            lines += _list_lines(child, depth + 1)
        elif text := _inline_text(child):
            lines.append("  " * depth + text)
    return lines


def _child(block: dict[str, Any], types: set[str]) -> dict[str, Any] | None:
    children = _children(block, types)
    return children[0] if children else None


def _children(block: dict[str, Any], types: set[str]) -> list[dict[str, Any]]:
    return [c for c in block.get("content", []) if isinstance(c, dict) and c.get("type") in types]


def _texts(block: dict[str, Any], types: set[str]) -> list[str]:
    return [text for child in _children(block, types) if (text := _inline_text(child))]


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
        self.assets_dir = str(resolve_assets_dir(assets_dir)) if assets_dir else None

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
