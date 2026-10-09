"""
parsing 层：各种文件 -> 统一的 Element 列表 -> 带结构的 Document。import 本包即完成注册

  markdown  Markdown / 纯文本，以及各解析器共用的 Markdown <-> Element 转换
  web       网页
  office    Word / PowerPoint / Excel
  tabular   CSV / TSV
  legacy    旧版 Office（.doc / .xls / .ppt / .rtf），经 LibreOffice 转换后复用 office
  pdf       PDF（文字层 + 扫描页交给 OCR）
  ocr       视觉语言模型 OCR（图片、扫描页）
  external  MinerU / Docling 适配器
  source    FileSource：按后缀路由、缓存、失败降级
"""

from .external import DoclingParser, MinerUParser
from .legacy import LibreOfficeParser
from .markdown import MarkdownParser, markdown_to_elements, table_to_markdown
from .ocr import VlmOcrParser
from .office import DocxParser, PptxParser, XlsxParser
from .pdf import PdfParser
from .source import FileSource
from .tabular import CsvParser, TsvParser
from .web import HtmlParser

__all__ = [
    "CsvParser",
    "DoclingParser",
    "DocxParser",
    "FileSource",
    "HtmlParser",
    "LibreOfficeParser",
    "MarkdownParser",
    "MinerUParser",
    "PdfParser",
    "PptxParser",
    "TsvParser",
    "VlmOcrParser",
    "XlsxParser",
    "markdown_to_elements",
    "table_to_markdown",
]
