"""
core.local_models —— 进程内共享的本地模型（sentence-transformers 的 SentenceTransformer / CrossEncoder）。

本地 embedding 和本地精排都在这里加载：第一次用到时才加载，同一个模型在进程里只加载一份。
页面服务里每个知识库各有一条管线，bge-m3、bge-reranker-v2-m3 各有 2 GB，按知识库各加载一份内存很快就不够了。
sentence-transformers 是可选依赖，没装时报错并给出安装命令。
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_SHARED: dict[tuple[str, str, str | None], object] = {}
_SHARED_LOCK = threading.Lock()


def shared_model(kind: str, name: str, device: str | None = None):
    """
    进程内共享的本地模型。kind：sentence_transformer（向量模型）、cross_encoder（精排模型）、
    bge_m3（FlagEmbedding 的 BGE-M3，一次前向同时给出稠密和稀疏向量）。
    页面服务里每个知识库各有一条管线，按模型名共享后，用同一个模型的知识库只占一份内存。
    """
    key = (kind, name, device)
    with _SHARED_LOCK:
        if key not in _SHARED:
            logger.info("加载本地模型 %s（首次使用会从 HuggingFace 下载）……", name)
            _SHARED[key] = _LOADERS[kind](name, device)
        return _SHARED[key]


def _sentence_transformers():
    try:
        import sentence_transformers   # 延迟导入：不装也能用别的实现
    except ImportError:
        raise RuntimeError('本地模型需要 sentence-transformers：pip install -e ".[local]"') from None
    return sentence_transformers


def _bge_m3(name: str, device: str | None):
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        raise RuntimeError('BGE-M3（稠密 + 稀疏）需要 FlagEmbedding：pip install -e ".[bge-m3]"') from None
    # use_fp16 只在 GPU 上有意义，CPU 上开了反而报错或变慢
    return BGEM3FlagModel(name, use_fp16=bool(device and device.startswith("cuda")))


_LOADERS = {
    "sentence_transformer": lambda name, device: _sentence_transformers().SentenceTransformer(name, device=device),
    "cross_encoder": lambda name, device: _sentence_transformers().CrossEncoder(name, device=device),
    "bge_m3": _bge_m3,
}
