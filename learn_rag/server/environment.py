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
from ..parsing.external import find_mineru_command

# 页面上展示配置状态的环境变量（只报是否已配置）
_KEYS = {
    "EMBEDDING_API_KEY": "Embedding 服务密钥",
    "EMBEDDING_BASE_URL": "Embedding 服务地址",
    "EMBEDDING_MODEL": "Embedding 模型名",
    "LLM_API_KEY": "大模型密钥",
    "RERANK_API_KEY": "精排服务密钥",
    "RERANK_BASE_URL": "精排服务地址",
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
    ("sentence_transformers", "本地 BGE-M3 / bge-reranker（sentence-transformers）", 'pip install -e ".[local]"'),
    ("FlagEmbedding", "BGE-M3 稠密 + 稀疏（FlagEmbedding）", 'pip install -e ".[bge-m3]"'),
]

# 混合召回的默认融合方式：向量和 BM25 等权、RRF 常数 60（论文推荐值），不改写查询。
# 稀疏通道（bge-m3 的稀疏向量）只有 embedding 选 BGE-M3（FlagEmbedding）时才能开
DEFAULT_FUSION = {"weights": {"vector": 1.0, "bm25": 1.0, "sparse": 0.0}, "rrf_k": 60, "query_transform": "none"}

# 本地模型的默认名称：HuggingFace 上的名字（首次使用自动下载），也可以在 .env 里改成已下载好的本地目录
BGE_M3 = "BAAI/bge-m3"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def has_mineru() -> bool:
    """PATH 上或 .mineru.env 的 MINERU_PYTHON_ENV 里有 mineru-kit。"""
    return find_mineru_command() is not None


def has_docling() -> bool:
    return _installed("docling")


def has_chroma() -> bool:
    return _installed("chromadb")


def has_llm() -> bool:
    """配置了大模型服务（查询改写、多轮对话问题补全、出题都要用）。"""
    load_env()
    return bool(os.getenv("LLM_API_KEY") or os.getenv("LLM_BASE_URL"))


def query_transforms() -> list[dict[str, Any]]:
    """可选的查询改写。改写要调用大模型，每次查询多一次（HyDE）或几次（多查询）调用。"""
    llm = has_llm()
    note = "" if llm else "未配置大模型（.env 中的 LLM_BASE_URL / LLM_API_KEY）"
    return [
        {"type": "none", "label": "不改写", "available": True, "note": "直接用原问题检索"},
        {"type": "multi_query", "label": "多查询改写", "available": llm,
         "note": note or "大模型把问题改写成 3 种说法分别召回再合并，缓解用户措辞和文档措辞不一致"},
        {"type": "hyde", "label": "HyDE", "available": llm,
         "note": note or "大模型先写一段假设答案，用它和原问题一起检索，口语化、很短的问题受益最大"},
    ]


def has_ocr() -> bool:
    """OCR 解析器未配置 OCR_* 时沿用 LLM_* 的服务。"""
    load_env()
    return bool(os.getenv("OCR_BASE_URL") or os.getenv("LLM_BASE_URL"))


def has_local_models() -> bool:
    return _installed("sentence_transformers")


def has_flag_embedding() -> bool:
    return _installed("FlagEmbedding")


def encoders() -> list[dict[str, Any]]:
    """可选的 embedding 模型。"""
    load_env()
    configured = bool(os.getenv("EMBEDDING_API_KEY") or os.getenv("EMBEDDING_BASE_URL"))
    local = has_local_models()
    flag = has_flag_embedding()
    return [
        {
            "type": "hashing", "label": "Hashing（离线演示）", "model": "hashing", "available": True,
            "note": "特征哈希，不理解语义，只用于跑通流程",
        },
        {
            "type": "openai_compat", "label": "OpenAI 兼容 API",
            "model": os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small",
            "available": configured,
            "note": "调用 .env 里配置的 embedding 服务（bge-m3 部署成服务后选这个，模型名填服务里的名字）" if configured
            else "未配置 EMBEDDING_API_KEY / EMBEDDING_BASE_URL（.env）",
        },
        {
            "type": "sentence_transformers", "label": "本地 BGE-M3",
            "model": os.getenv("LOCAL_EMBEDDING_MODEL") or BGE_M3,
            "available": local,
            "note": "在本机运行，中英文都强，1024 维；首次使用自动下载约 2.2 GB，"
                    "也可以在 .env 的 LOCAL_EMBEDDING_MODEL 填已下载好的目录" if local
            else '未安装 sentence-transformers：pip install -e ".[local]"',
        },
        {
            "type": "bge_m3", "label": "BGE-M3（稠密 + 稀疏）",
            "model": os.getenv("LOCAL_EMBEDDING_MODEL") or BGE_M3,
            "available": flag,
            "note": "本机运行，建库时同时生成稀疏向量，可开第三路“稀疏召回”；首次使用自动下载约 2.2 GB" if flag
            else '未安装 FlagEmbedding：pip install -e ".[bge-m3]"',
        },
    ]


def rerankers() -> list[dict[str, Any]]:
    """
    可选的精排模型。精排在召回的几十个候选上逐个打分，只影响查询，不影响存进去的向量，改了不用重建。
    """
    load_env()
    api = bool(os.getenv("RERANK_API_KEY") or os.getenv("RERANK_BASE_URL"))
    local = has_local_models()
    return [
        {"type": "lexical", "label": "词重叠（离线演示）", "model": "", "available": True,
         "note": "只看查询词在片段里出现了多少，不理解语义，只用于跑通流程"},
        {"type": "api", "label": "bge-reranker 服务", "model": os.getenv("RERANK_MODEL") or BGE_RERANKER,
         "available": api,
         "note": "调用 .env 里配置的 rerank 服务（SiliconFlow、Xinference、TEI 等），失败时退回召回顺序" if api
         else "未配置 RERANK_API_KEY / RERANK_BASE_URL（.env）"},
        {"type": "cross_encoder", "label": "本地 bge-reranker-v2-m3", "model": os.getenv("LOCAL_RERANK_MODEL") or BGE_RERANKER,
         "available": local,
         "note": "在本机运行，首次查询时加载（首次使用自动下载约 2.2 GB），CPU 上每次查询要几秒" if local
         else '未安装 sentence-transformers：pip install -e ".[local]"'},
        {"type": "identity", "label": "不精排", "model": "", "available": True,
         "note": "直接用召回的顺序，做对照实验时用：量出精排到底提升了多少"},
    ]


def parser_choices() -> dict[str, list[dict[str, Any]]]:
    """高级参数里可以切换解析器的格式：每个选项是一条"首选 → 备选"的解析器链。"""
    mineru, docling = has_mineru(), has_docling()
    return {
        ".pdf": [
            {"value": ["pdf"], "label": "PyMuPDF（快，适合有文字层的 PDF）", "available": True},
            {"value": ["mineru", "pdf"], "label": "MinerU → PyMuPDF（版面、表格、扫描件更准）",
             "available": mineru, "note": "" if mineru else "未找到 mineru-kit（PATH 和 MINERU_PYTHON_ENV 下都没有）"},
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
    # 默认只选已经配置好的服务，不默认选本地模型：本地模型首次使用要下载几个 GB，应该由用户自己决定
    available = [e for e in encoders() if e["available"] and e["type"] not in ("sentence_transformers", "bge_m3")]
    encoder = next((e for e in available if e["type"] != "hashing"), available[0])
    ranker = next(r for r in rerankers() if r["type"] in ("api", "lexical") and r["available"])
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
        "reranker": {"type": ranker["type"], "model": ranker["model"]},
        # 有真正的精排时召回 50 个候选给它挑；词重叠精排判别力有限，候选多了反而引入噪声
        "retrieval": {"candidate_k": 50 if ranker["type"] == "api" else 20, "top_k": 5, **DEFAULT_FUSION},
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
        "rerankers": rerankers(),
        "query_transforms": query_transforms(),
        "llm": has_llm(),
        "keys": [{"name": name, "label": label, "configured": bool(os.getenv(name))} for name, label in _KEYS.items()],
        "packages": packages,
        "parsers": parser_choices(),
        "ocr": has_ocr(),
        "defaults": default_settings(),
    }
