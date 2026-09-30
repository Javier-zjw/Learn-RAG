"""
ingest.loaders —— 各类数据源到 Document 的适配。

对上层而言只有一个 DocumentSource.load()；PDF、目录、jsonl、内存列表
的差异全部消化在实现内部。新增一种数据源 = 新增一个类 + 一行注册，
上层与配置文件之外的代码一律不动。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..core.interfaces import DocumentSource
from ..core.registry import registry
from ..core.types import Document


@registry.register("source", "memory")
class MemorySource(DocumentSource):
    """
    内存文档列表 —— 单元测试与"数据集自带语料"场景使用
    """

    def __init__(self, documents: Sequence[Document]) -> None:
        self._documents = list(documents)

    def load(self) -> Iterable[Document]:
        return iter(self._documents)


@registry.register("source", "jsonl")
class JsonlSource(DocumentSource):
    """
    每行一个 JSON 对象的语料文件。

    字段名可配置，因此同一个类能吃下不同数据集的字段命名习惯（通用接口：不为某个数据集写死字段）
    """

    def __init__(
            self,
            path: str,
            *,
            id_field: str = "id",
            text_field: str = "text",
            title_field: str | None = "title",
    ) -> None:
        self.path = Path(path)
        self.id_field = id_field
        self.text_field = text_field
        self.title_field = title_field

    def load(self) -> Iterable[Document]:
        with self.path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row: dict[str, Any] = json.loads(line)
                text = row.get(self.text_field, "")
                if not text:
                    continue
                meta = {k: v for k, v in row.items() if k not in {self.id_field, self.title_field}}
                if self.title_field and self.text_field in row:
                    meta["title"] = row[self.title_field]
                yield Document(doc_id=str(row.get(self.id_field, lineno)), text=text, metadata=meta)

@registry.register("source", "directory")
class DirectorySource(DocumentSource):
    """
    递归读取目录下的纯文本类文件（.txt/.md）。

    真实项目里的 PDF/Word 解析属于"另一个深模块"（OCR、版面还原），
    这里留出扩展点：只要新写一个 DocumentSource 即可接入，上层无感。
    """

    def __init__(
            self,
            path: str,
            *,
            patterns: Sequence[str] = ("*.txt", "*.md"),
            encoding: str = "utf-8"
    ) -> None:
        self.root = Path(path)
        self.patterns = tuple(patterns)
        self.encoding = encoding

    def load(self) -> Iterable[Document]:
        for pattern in self.patterns:
            for file in sorted(self.root.rglob(pattern)):
                text = file.read_text(encoding=self.encoding, errors="ignore")
                if text.strip():
                    yield Document(
                        doc_id=str(file.relative_to(self.root)),
                        text=text,
                        metadata={"title": file.stem, "path": str(file)}
                    )
