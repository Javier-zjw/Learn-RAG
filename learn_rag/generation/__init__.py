"""
generation 层：LLM 客户端 + 生成策略
"""

from .generators import ExtractiveGenerator, StuffGenerator, pack_contexts
from .llms import EchoLLM, OpenAICompatLLM

__all__ = [
    "ExtractiveGenerator",
    "StuffGenerator",
    "pack_contexts",
    "EchoLLM",
    "OpenAICompatLLM",
]