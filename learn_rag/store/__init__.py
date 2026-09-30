"""store 层：向量索引、倒排索引、知识库门面。"""

from .indexes import BM25Index, FlatVectorIndex
from .knowledge_base import KnowledgeBase

__all__ = ["BM25Index", "FlatVectorIndex", "KnowledgeBase"]

# ---------------------------------------------------------------------------
# Chroma 是可选依赖：没装也不能影响 flat 索引。
#
# 早期版本是这样写的：
#     try:
#         from .chroma_index import ChromaVectorIndex
#     except ImportError:
#         pass
# 问题是**把失败原因吞掉了**。chromadb 没装、chroma_index.py 不存在、
# 或者手敲时某个 import 写错，最后都只得到一句
#     KeyError: "[index] 未注册的实现 'chroma'，可选：['flat']"
# 完全看不出是哪一种。
#
# 现在的做法：导入失败时注册一个**占位工厂**。不用 chroma 时毫无影响；
# 真用到时，占位工厂把导入失败的真实原因原样报出来。
# 这是"把异常推迟到真正需要时再抛，但抛的时候必须带上完整上下文"。
# ---------------------------------------------------------------------------
try:
    from .chroma_index import ChromaVectorIndex  # noqa: F401

    __all__.append("ChromaVectorIndex")
except ImportError as _exc:  # pragma: no cover
    from ..core.registry import registry as _registry

    _REASON = f"{type(_exc).__name__}: {_exc}"
    _PKG = __name__.rsplit(".", 1)[0]  # 自动适配包名（minirag / learn_rag 都行）

    @_registry.register("index", "chroma")
    def _chroma_unavailable(**_kwargs):
        hint = [
            "Chroma 索引不可用 —— 导入 store/chroma_index.py 时失败：",
            f"  {_REASON}",
            "",
            "  排查方向：",
        ]
        if "chromadb" in _REASON:
            hint.append("  · 没装 chromadb → pip install chromadb")
            hint.append("    注意要装到 PyCharm 实际使用的那个解释器里（Settings → Project → Python Interpreter）")
        elif "chroma_index" in _REASON:
            hint.append(f"  · {_PKG}/store/chroma_index.py 文件不存在，需要先创建它")
        else:
            hint.append("  · chroma_index.py 里有 import 写错了，运行下面这行看完整报错：")
            hint.append(f"      python -c \"import {_PKG}.store.chroma_index\"")
        raise ImportError("\n".join(hint))