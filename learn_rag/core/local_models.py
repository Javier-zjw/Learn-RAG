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
    进程内共享的本地模型（sentence-transformers 的 SentenceTransformer / CrossEncoder）。
    页面服务里每个知识库各有一条管线，按模型名共享后，用同一个 bge-m3 的知识库只占一份内存。
    """
    key = (kind, name, device)
    with _SHARED_LOCK:
        if key not in _SHARED:
            try:
                import sentence_transformers   # 延迟导入：不装也能用别的实现
            except ImportError:
                raise RuntimeError('本地模型需要 sentence-transformers：pip install -e ".[local]"') from None
            loader = sentence_transformers.CrossEncoder if kind == "cross_encoder" else sentence_transformers.SentenceTransformer
            logger.info("加载本地模型 %s（首次使用会从 HuggingFace 下载）……", name)
            _SHARED[key] = loader(name, device=device)
        return _SHARED[key]
