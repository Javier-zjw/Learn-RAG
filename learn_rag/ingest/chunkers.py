"""
ingest.chunkers —— 切分策略。

切分是 RAG 里最被低估的一环：切得太碎丢上下文，切得太大稀释语义、还会把噪声塞进 prompt。
这里给三种典型策略，接口完全一致，方便做 A/B 实验（改配置即可，看指标怎么变）。

基类 BaseChunker 承担了所有实现共有的脏活：
过滤空片段、生成可复现 id、写入 position/父文档元信息。
子类只需实现"怎么把长文本切成字符串列表"这一件事 —— 这就是"深类"的体现：
子类的接口和职责都很小，重复逻辑被基类吸收。
"""

from __future__ import annotations

import re
from abc import abstractmethod
from collections.abc import Sequence

from ..core.interfaces import Chunker
from ..core.registry import registry
from ..core.types import Chunk, Document


class BaseChunker(Chunker):
    def __init__(
            self,
            chunk_size: int = 400,
            chunk_overlap: int = 80,
            min_chars: int = 1
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chars = min_chars

    def split(self, document: Document) -> list[Chunk]:
        pieces = [p.strip() for p in self._split_text(document.text)]
        pieces = [p for p in pieces if len(p) >= self.min_chars]
        return [Chunk.of(document, text, i) for i, text in enumerate(pieces)]

    @abstractmethod
    def _split_text(self, text: str) -> list[str]: ...

    # 供子类复用：把过长的片段按定长窗口二次切分
    def _window(self, text: str) -> list[str]:
        step = self.chunk_size - self.chunk_overlap
        return [text[i: i + self.chunk_size] for i in range(0, max(len(text), 1), step)]

@registry.register("chunker", "fixed")
class FixedSizeChunker(BaseChunker):
    """
    定长滑窗切分：最简单、最可复现，做基线用
    """

    def _split_text(self, text: str) -> list[str]:
        return self._window(text)

@registry.register("chunker", "recursive")
class RecursiveChunker(BaseChunker):
    """
    递归切分（推荐默认）

    思路：优先按"语义边界强"的分隔符切（空行 > 换行 > 句号 > 逗号），
    只有当片段仍然超长时才降级到更弱的分隔符，最后才硬切。
    再把相邻小片段贪心合并到接近 chunk_size，避免产生大量碎片。
    """

    DEFAULT_SEPARATORS = ("\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "; ", "，", ", ", " ")

    def __init__(self, separators: Sequence[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.separators = tuple(separators or self.DEFAULT_SEPARATORS)

    def _split_text(self, text: str) -> list[str]:
        units = self._recursive(text, 0)
        return self._merge(units)

    def _recursive(self, text: str, depth: int) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        if depth >= len(self.separators):
            return self._window(text)

        sep = self.separators[depth]
        parts = [p + sep for p in text.split(sep)]
        parts[-1] = parts[-1][: -len(sep)] if parts[-1].endswith(sep) else parts[-1]
        out: list[str] = []
        for part in parts:
            out.extend(self._recursive(part, depth + 1) if len(part) > self.chunk_size else [part])
        return out

    def _merge(self, units: Sequence[str]) -> list[str]:
        merged: list[str] = []
        buf = ""
        for unit in units:
            if len(buf) + len(unit) <= self.chunk_size:
                buf += unit
                continue
            if buf:
                merged.append(buf)

            tail = buf[-self.chunk_overlap:] if self.chunk_overlap else ""
            buf = tail + unit

        if buf:
            merged.append(buf)
        return merged

@registry.register("chunker", "markdown")
class MarkdownChunker(BaseChunker):
    """
    结构感知切分：按 Markdown 标题分段，并把标题路径写进片段正文。

    把 "第3章 > 3.2 计费规则" 这样的标题链接回正文，
    可以显著改善"片段脱离上下文后语义丢失"的问题（也利于命中率）。
    """

    _HEAD = re.compile(r"^(#{1,6})\s+(.*)$", re.M)

    def _split_text(self, text: str) -> list[str]:
        matches = list(self._HEAD.finditer(text))
        if not matches:
            return RecursiveChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)._split_text(text)

        sections: list[str] = []
        path: list[str] = []
        for i, m in enumerate(matches):
            level, title = len(m.group(1)), m.group(2).strip()
            path = path[: level - 1] + [title]
            body = text[m.end() : matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
            if not body:
                continue
            prefix = " > ".join(path)
            for piece in RecursiveChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )._split_text(body):
                sections.append(f"[{prefix}]\n{piece}")

        return sections
