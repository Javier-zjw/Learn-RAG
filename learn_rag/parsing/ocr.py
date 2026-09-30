"""
parsing.ocr —— 用视觉语言模型（VLM）做 OCR：图片 -> Markdown -> Element。

当前效果最好的文档 OCR 都是小型专用 VLM（PaddleOCR-VL、MinerU2.5、GLM-OCR 等），
它们可以用 vLLM 部署成 OpenAI 兼容服务，云厂商也提供同样协议的接口。
因此这里只依赖 OpenAI 兼容的多模态对话协议，换模型只改配置：
    OCR_BASE_URL / OCR_API_KEY / OCR_MODEL（未设置 URL 和 Key 时沿用 LLM_* 的配置）

同一个实例有两种用法：
    parse(path)            解析图片文件（.png / .jpg）
    recognize(png_bytes)   识别一页渲染好的图片，供 PDF 解析器处理扫描页
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

from ..core.interfaces import DocumentParser
from ..core.registry import registry
from ..core.types import Element
from ..generation.llms import OpenAICompatLLM
from .markdown import markdown_to_elements

DEFAULT_PROMPT = (
    "请把这张文档图片完整转写为 Markdown：\n"
    "- 标题用 #、## 等表示层级；\n"
    "- 表格用 Markdown 表格，含合并单元格时改用 HTML <table>；\n"
    "- 公式用 $$ LaTeX $$；\n"
    "- 图片或图表用一句话描述其内容，写成 ![描述]()；\n"
    "- 忽略页眉、页脚和页码。\n"
    "只输出转写结果，不要任何解释。"
)

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
_WRAPPER = re.compile(r"^```(?:markdown|md)?\s*\n(.*)\n```\s*$", re.S)


@registry.register("parser", "vlm_ocr")
class VlmOcrParser(DocumentParser):
    def __init__(
            self,
            model: str | None = None,
            base_url: str | None = None,
            api_key: str | None = None,
            prompt: str = DEFAULT_PROMPT,
            max_tokens: int = 4096,
            timeout: int = 180,
    ) -> None:
        model = model or os.getenv("OCR_MODEL", "")
        if not model:
            raise ValueError("未配置 OCR 模型：请设置环境变量 OCR_MODEL，或在 parsing.ocr.model 中指定")
        self.prompt = prompt
        self.llm = OpenAICompatLLM(
            model=model,
            base_url=base_url or os.getenv("OCR_BASE_URL"),
            api_key=api_key or os.getenv("OCR_API_KEY"),
            max_tokens=max_tokens,
            timeout=timeout,
        )

    def parse(self, path: Path) -> list[Element]:
        mime = _MIME.get(path.suffix.lower(), "image/png")
        return self.recognize(path.read_bytes(), mime=mime)

    def recognize(self, image: bytes, *, page: int | None = None, mime: str = "image/png") -> list[Element]:
        url = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": self.prompt},
            ],
        }]
        markdown = self.llm.chat(messages).strip()
        # 有些模型会把整段输出包在 ```markdown 代码块里，剥掉外壳再解析
        wrapped = _WRAPPER.match(markdown)
        if wrapped:
            markdown = wrapped.group(1)
        elements = markdown_to_elements(markdown, page=page)
        for element in elements:
            element.extra["ocr"] = True
        return elements
