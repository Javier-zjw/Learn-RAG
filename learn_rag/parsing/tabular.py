"""
parsing.tabular —— CSV / TSV 解析。

CSV 是企业里最常见的导出格式（报表、台账、日志），标准库 csv 就能稳定读取，
不引入任何新依赖。输出形状与 Excel 保持一致：文件名作为一级标题 + 一张表格，
后续由结构切分器按行切分并重复表头，两类表格数据的处理方式完全统一。

中文企业环境里 GBK 编码的 CSV 非常常见（Excel 在中文 Windows 上的默认导出编码），
读取时按 utf-8-sig → gbk → latin-1 依次尝试：前两种覆盖绝大多数真实文件，
latin-1 兜底保证西文文件不因编码问题中断整批导入。
"""

from __future__ import annotations

import csv
import logging
from itertools import islice
from pathlib import Path

from ..core.interfaces import DocumentParser
from ..core.registry import registry
from ..core.types import Element
from .markdown import table_to_markdown

logger = logging.getLogger(__name__)

_ENCODINGS = ("utf-8-sig", "gbk", "latin-1")


class _DelimitedParser(DocumentParser):
    """CSV / TSV 共用实现，只有分隔符不同。"""

    def __init__(self, delimiter: str, max_rows: int = 200_000) -> None:
        self.delimiter = delimiter
        self.max_rows = max_rows

    def parse(self, path: Path) -> list[Element]:
        rows, truncated = self._read(path)
        if truncated:
            logger.warning("%s 超过 %d 行，只保留前 %d 行", path.name, self.max_rows, self.max_rows)
        markdown = table_to_markdown(rows)
        if not markdown:
            return []
        return [
            Element("heading", path.stem, level=1),
            Element("table", markdown, extra={"sheet": path.stem}),
        ]

    def _read(self, path: Path) -> tuple[list[list[str]], bool]:
        """返回 (行数据, 是否因超过 max_rows 被截断)。"""
        last_error: UnicodeDecodeError | None = None
        for encoding in _ENCODINGS:
            try:
                with path.open(newline="", encoding=encoding) as fh:
                    reader = csv.reader(fh, delimiter=self.delimiter)
                    rows = [["" if v is None else str(v) for v in row] for row in islice(reader, self.max_rows)]
                    # 再读一行判断是否被截断，不把整个大文件读进内存
                    truncated = next(reader, None) is not None
                    return rows, truncated
            except UnicodeDecodeError as exc:
                last_error = exc
        raise RuntimeError(f"无法解码 {path.name}：{last_error}")


@registry.register("parser", "csv")
class CsvParser(_DelimitedParser):
    """CSV（默认逗号分隔，欧洲常用的分号可通过 parsing.options.csv.delimiter 覆盖）。"""

    def __init__(self, delimiter: str = ",", max_rows: int = 200_000) -> None:
        super().__init__(delimiter=delimiter, max_rows=max_rows)


@registry.register("parser", "tsv")
class TsvParser(_DelimitedParser):
    """TSV（制表符分隔）。"""

    def __init__(self, delimiter: str = "\t", max_rows: int = 200_000) -> None:
        super().__init__(delimiter=delimiter, max_rows=max_rows)
