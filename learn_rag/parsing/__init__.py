"""
parsing 层：各种文件 -> 统一的 Element 列表 -> 带结构的 Document。import 本包即完成注册

  markdown  Markdown / 纯文本，以及各解析器共用的 Markdown <-> Element 转换
  web       网页
  office    Word / PowerPoint / Excel
  pdf       PDF（文字层 + 扫描页交给 OCR）
  ocr       视觉语言模型 OCR（图片、扫描页）
  external  MinerU / Docling 适配器
  source    FileSource：按后缀路由、缓存、失败降级
"""

from .external import DoclingParser, MinerUParser
from .markdown import MarkdownParser, markdown_to_elements, table_to_markdown
from .ocr import VlmOcrParser
from .office import DocxParser, PptxParser, XlsxParser
from .pdf import PdfParser
from .source import FileSource
from .web import HtmlParser

__all__ = [
    "DoclingParser",
    "DocxParser",
    "FileSource",
    "HtmlParser",
    "MarkdownParser",
    "MinerUParser",
    "PdfParser",
    "PptxParser",
    "VlmOcrParser",
    "XlsxParser",
    "markdown_to_elements",
    "table_to_markdown",
]
