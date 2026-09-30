"""
core 层：数据契约 + 抽象接口 + 注册表 + 配置。不依赖任何上层模块
"""

from .config import load_config, load_env
from .interfaces import (
    Chunker,
    DocumentSource,
    EvalDataset,
    Generator,
    LLM,
    Metric,
    QueryTransformer,
    Reranker,
    Retriever,
    TextEncoder,
    VectorIndex
)
from .registry import registry
from .types import (
    Answer,
    Chunk,
    Document,
    EvalCase,
    EvalSample,
    RagResult,
    ScoredChunk,
    Timer
)

__all__ = [
    "load_config",
    "load_env",
    "registry",
    "Chunker",
    "DocumentSource",
    "EvalDataset",
    "Generator",
    "LLM",
    "Metric",
    "QueryTransformer",
    "Reranker",
    "Retriever",
    "TextEncoder",
    "VectorIndex",
    "Answer",
    "Chunk",
    "Document",
    "EvalCase",
    "EvalSample",
    "RagResult",
    "ScoredChunk",
    "Timer"
]