"""
parsing.source —— 文件数据源：一个目录（或单个文件） -> 带结构的 Document 流。

对上层仍然只是 DocumentSource.load()。解析过程中所有"每个人都要处理一遍"的事都在这里：
  - 按后缀选解析器，可在配置里覆盖，也可以配"首选 + 备选"列表，首选失败自动降级；
  - 解析结果按"文件内容哈希 + 解析器 + 参数"缓存成 JSON：OCR 和版面模型很慢，
    同一个文件只解析一次，之后调整切分策略、重建索引都直接读缓存；
  - 单个文件解析失败只记日志并跳过，不影响整批。

配置示例（configs/parsing.yaml）：
    parsing:
      cache_dir: .cache/parsed
      parsers:                    # 按后缀覆盖默认解析器；列表表示依次尝试
        .pdf: [mineru, pdf]
      options:                    # 各解析器的构造参数
        mineru: {backend: pipeline}
      ocr: {type: vlm_ocr}        # 可选：扫描页和图片的 OCR
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..core.interfaces import DocumentParser, DocumentSource
from ..core.registry import registry
from ..core.types import Document, Element

logger = logging.getLogger(__name__)

DEFAULT_PARSERS: dict[str, str] = {
    ".md": "markdown", ".markdown": "markdown", ".txt": "markdown",
    ".html": "html", ".htm": "html",
    ".docx": "docx", ".pptx": "pptx", ".xlsx": "xlsx",
    ".pdf": "pdf",
    ".png": "vlm_ocr", ".jpg": "vlm_ocr", ".jpeg": "vlm_ocr", ".webp": "vlm_ocr",
}


@registry.register("source", "files")
class FileSource(DocumentSource):
    def __init__(
            self,
            path: str,
            *,
            parsers: dict[str, str | list[str]] | None = None,
            options: dict[str, dict[str, Any]] | None = None,
            ocr: dict[str, Any] | None = None,
            cache_dir: str | None = ".cache/parsed",
    ) -> None:
        self.root = Path(path)
        routes = {**DEFAULT_PARSERS, **(parsers or {})}
        self.routes = {suffix.lower(): [names] if isinstance(names, str) else list(names) for suffix, names in routes.items()}
        self.options = dict(options or {})
        # OCR 的配置同时就是该 OCR 解析器（处理图片文件）的配置
        self.ocr_name: str | None = None
        if ocr:
            ocr = dict(ocr)
            self.ocr_name = ocr.pop("type", "vlm_ocr")
            self.options.setdefault(self.ocr_name, ocr)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._parsers: dict[str, DocumentParser] = {}

    def load(self) -> Iterable[Document]:
        files = [self.root] if self.root.is_file() else sorted(p for p in self.root.rglob("*") if p.is_file())
        for file in files:
            names = self.routes.get(file.suffix.lower())
            if not names:
                continue
            result = self._parse(file, names)
            if result is None:
                continue
            elements, parser_name, digest = result
            if not elements:
                logger.warning("%s 没有解析出任何内容，已跳过", file)
                continue
            doc_id = file.name if self.root.is_file() else file.relative_to(self.root).as_posix()
            metadata = {
                "title": file.stem,
                "path": str(file),
                "file_type": file.suffix.lower().lstrip("."),
                "file_hash": digest,
                "parser": parser_name,
            }
            yield Document.from_elements(doc_id, elements, metadata)

    def _parse(self, file: Path, names: list[str]) -> tuple[list[Element], str, str] | None:
        """依次尝试各个解析器，返回 (元素, 实际使用的解析器, 文件哈希)；全部失败返回 None。"""
        content = file.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        for name in names:
            cache_file = self._cache_file(digest, name)
            if cache_file is not None and cache_file.exists():
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                return [Element(**item) for item in data], name, digest
            try:
                elements = self._parser(name).parse(file)
            except Exception as exc:
                logger.warning("解析器 %s 处理 %s 失败：%s", name, file, exc)
                continue
            if cache_file is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                payload = json.dumps([asdict(e) for e in elements], ensure_ascii=False)
                cache_file.write_text(payload, encoding="utf-8")
            return elements, name, digest
        logger.error("%s 的所有解析器（%s）都失败了，已跳过", file, ", ".join(names))
        return None

    def _parser(self, name: str) -> DocumentParser:
        """按名字构建解析器并复用（部分解析器初始化要加载模型）。配置了 OCR 时注入给需要它的解析器（如 pdf）。"""
        if name not in self._parsers:
            spec = {"type": name, **self.options.get(name, {})}
            self._parsers[name] = registry.build("parser", spec, ocr=self._ocr_for(name))
        return self._parsers[name]

    def _ocr_for(self, name: str) -> DocumentParser | None:
        if self.ocr_name is None or name == self.ocr_name:
            return None
        try:
            return self._parser(self.ocr_name)
        except Exception as exc:   # OCR 配错不应该连带文字层 PDF 也解析不了
            logger.warning("OCR 解析器 %s 初始化失败，扫描页将被跳过：%s", self.ocr_name, exc)
            return None

    def _cache_file(self, digest: str, name: str) -> Path | None:
        """缓存键包含解析器参数和 OCR 配置：改了参数（比如换 OCR 模型）就会重新解析，不会读到旧结果。"""
        if self.cache_dir is None:
            return None
        ocr_options = self.options.get(self.ocr_name) if self.ocr_name else None
        options = json.dumps([self.options.get(name, {}), self.ocr_name, ocr_options], sort_keys=True, ensure_ascii=False)
        key = hashlib.sha256(f"{digest}|{name}|{options}".encode("utf-8")).hexdigest()[:32]
        return self.cache_dir / f"{name}-{key}.json"
