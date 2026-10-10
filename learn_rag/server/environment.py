"""
server.environment —— 当前机器上能用什么：embedding 模型、解析工具、可选依赖、各项密钥是否配置。

前端据此把不能用的选项置灰并说明原因，推荐规则也只从能用的选项里挑。
只检查"装没装、配没配"，不发网络请求：打开页面不应该因为某个服务连不上而变慢。
密钥只报告是否已配置，绝不把值返回给前端。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from typing import Any

from ..core.config import load_env

# 页面上展示配置状态的环境变量（只报是否已配置）
_KEYS = {
    "EMBEDDING_API_KEY": "Embedding 服务密钥",
    "EMBEDDING_BASE_URL": "Embedding 服务地址",
    "EMBEDDING_MODEL": "Embedding 模型名",
    "LLM_API_KEY": "大模型密钥",
    "RERANK_API_KEY": "精排服务密钥",
    "OCR_BASE_URL": "OCR 服务地址",
}

# 可选依赖：(导入名, 用途, 安装命令)
_PACKAGES = [
    ("chromadb", "Chroma 持久化向量库", 'pip install -e ".[chroma]"'),
    ("pymupdf", "PDF 原生解析（PyMuPDF）", 'pip install -e ".[parsing]"'),
    ("docx", "Word 解析（python-docx）", 'pip install -e ".[parsing]"'),
    ("pptx", "PPT 解析（python-pptx）", 'pip install -e ".[parsing]"'),
    ("openpyxl", "Excel 解析（openpyxl）", 'pip install -e ".[parsing]"'),
    ("docling", "Docling 高精度解析", 'pip install -e ".[docling]"'),
]


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def has_mineru() -> bool:
    return shutil.which("mineru-kit") is not None


def has_docling() -> bool:
    return _installed("docling")


def has_chroma() -> bool:
    return _installed("chromadb")


def has_ocr() -> bool:
    """OCR 解析器未配置 OCR_* 时沿用 LLM_* 的服务。"""
    load_env()
    return bool(os.getenv("OCR_BASE_URL") or os.getenv("LLM_BASE_URL"))


def encoders() -> list[dict[str, Any]]:
    """可选的 embedding 模型。本地模型暂未接入，固定置灰。"""
    load_env()
    configured = bool(os.getenv("EMBEDDING_API_KEY") or os.getenv("EMBEDDING_BASE_URL"))
    return [
        {
            "type": "hashing", "label": "Hashing（离线演示）", "model": "hashing", "available": True,
            "note": "特征哈希，不理解语义，只用于跑通流程",
        },
        {
            "type": "openai_compat", "label": "OpenAI 兼容 API",
            "model": os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small",
            "available": configured,
            "note": "调用 .env 里配置的 embedding 服务" if configured
            else "未配置 EMBEDDING_API_KEY / EMBEDDING_BASE_URL（.env）",
        },
        {
            "type": "sentence_transformers", "label": "本地模型（sentence-transformers）", "model": "",
            "available": False, "note": "本地模型尚未接入",
        },
    ]


def parser_choices() -> dict[str, list[dict[str, Any]]]:
    """高级参数里可以切换解析器的格式：每个选项是一条"首选 → 备选"的解析器链。"""
    mineru, docling = has_mineru(), has_docling()
    return {
        ".pdf": [
            {"value": ["pdf"], "label": "PyMuPDF（快，适合有文字层的 PDF）", "available": True},
            {"value": ["mineru", "pdf"], "label": "MinerU → PyMuPDF（版面、表格、扫描件更准）",
             "available": mineru, "note": "" if mineru else "未找到 mineru-kit 命令"},
            {"value": ["docling", "pdf"], "label": "Docling → PyMuPDF",
             "available": docling, "note": "" if docling else "未安装 docling"},
        ],
        ".docx": [
            {"value": ["docx"], "label": "python-docx（快）", "available": True},
            {"value": ["docling", "docx"], "label": "Docling → python-docx（结构还原更完整）",
             "available": docling, "note": "" if docling else "未安装 docling"},
        ],
    }


def default_settings() -> dict[str, Any]:
    """新建知识库时表单的初始值：和 configs/parsing.yaml 的切分参数一致，模型和解析器取当前能用的。"""
    available = [e for e in encoders() if e["available"]]
    encoder = next((e for e in available if e["type"] != "hashing"), available[0])
    return {
        "chunker": {"type": "structure", "chunk_size": 300, "parent_size": 1200, "overlap_sentences": 0},
        "encoder": {"type": encoder["type"], "model": encoder["model"]},
        "index": {"type": "chroma" if has_chroma() else "flat", "space": "cosine",
                  "ef_construction": 200, "max_neighbors": 32, "ef_search": 100},
        "parsers": {
            ".pdf": ["mineru", "pdf"] if has_mineru() else ["pdf"],
            ".docx": ["docling", "docx"] if has_docling() else ["docx"],
        },
        "ocr": False,
    }


def report() -> dict[str, Any]:
    """"模型与环境"页面的全部内容。"""
    load_env()
    packages = [{"name": label, "installed": _installed(module), "install": cmd} for module, label, cmd in _PACKAGES]
    packages += [
        {"name": "MinerU 命令行（mineru-kit）", "installed": has_mineru(), "install": 'pip install -e ".[mineru]"'},
        {"name": "LibreOffice（.doc/.xls/.ppt 转换）", "installed": bool(shutil.which("soffice")),
         "install": "安装系统软件包 libreoffice"},
    ]
    return {
        "encoders": encoders(),
        "keys": [{"name": name, "label": label, "configured": bool(os.getenv(name))} for name, label in _KEYS.items()],
        "packages": packages,
        "parsers": parser_choices(),
        "ocr": has_ocr(),
        "defaults": default_settings(),
    }
