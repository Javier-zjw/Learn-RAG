"""
store.knowledge_base —— 知识库（离线链路的门面 / Facade）。

这是本项目最典型的"深类"：
  对外接口只有 add(documents) / save(path) / load(path) / stats()
  对内却完成了：切分 -> 批量向量化 -> 写稠密索引 -> 写倒排索引 -> 统计。

上层（检索器、pipeline、评测器）永远不需要知道"先切分还是先向量化""批大小多少""BM25 和向量索引怎么保持 id 对齐"。
一旦这些细节泄漏到上层，每加一个功能都要改好几处 —— 那就是浅类的代价。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..core.interfaces import Chunker, TextEncoder
from ..core.types import Chunk, Document
from .indexes import BM25Index, FlatVectorIndex

class KnowledgeBase:
    def __init__(
            self,
            encoder: TextEncoder,
            chunker: Chunker,
            vector_index: FlatVectorIndex | None = None,
            bm25_index: BM25Index | None = None,
            batch_size: int = 64
    ) -> None:
        self.encoder = encoder
        self.chunker = chunker
        self.vector_index = vector_index if vector_index is not None else FlatVectorIndex()
        self.bm25_index = bm25_index if bm25_index is not None else BM25Index()
        self.batch_size = batch_size
        self._doc_count = 0
        self._chunks_by_id: dict[str, Chunk] = {}

    def add(self, documents: Iterable[Document], *, progress: bool = False) -> dict[str, int]:
        """
        把文档写入知识库，返回本次写入统计。
        采用"攒批"写入：向量化是最贵的一步，逐条调用会让远程 API 慢十倍以上
        """
        buffer: list[Chunk] = []
        added = {"documents": 0, "chunks": 0}
        for doc in documents:
            chunks = self.chunker.split(doc)
            buffer.extend(chunks)
            added["documents"] += 1
            added["chunks"] += len(chunks)
            if len(buffer) >= self.batch_size:
                self._flush(buffer)
                buffer = []
            if progress and added["documents"] % 200 == 0:
                print(f"  已处理 {added['documents']} 篇文档 / {added['chunks']} 个片段")

        if buffer:
            self._flush(buffer)
        self._doc_count += added["documents"]
        return added

    def _flush(self, chunks: list[Chunk]) -> None:
        vectors = self.encoder.encode([c.text for c in chunks], is_query=False)
        self.vector_index.add(chunks, vectors)
        self.bm25_index.add(chunks)
        for c in chunks:
            self._chunks_by_id[c.chunk_id] = c

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks_by_id.get(chunk_id)

    def stats(self) -> dict[str, int]:
        return {
            "documents": self._doc_count,
            "chunks": len(self.vector_index),
            "bm25_chunks": len(self.bm25_index),
            "dimension": self.encoder.dimension
        }

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.vector_index.save(path)
        self.bm25_index.save(path)
        (p / "meta.json").write_text(json.dumps(self.stats(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        self.vector_index.load(path)
        self.bm25_index.load(path)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        self._doc_count = meta.get("documents", 0)