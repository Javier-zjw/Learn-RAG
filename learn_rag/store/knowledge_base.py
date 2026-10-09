"""
store.knowledge_base —— 知识库（离线链路的门面 / Facade）。

这是本项目最典型的"深类"：
  对外接口只有 add(documents) / expand(hits) / save(path) / load(path) / stats()
  对内却完成了：切分 -> 区分父子块 -> 批量向量化 -> 写稠密索引 -> 写倒排索引 -> 统计。

父子分块的约定也藏在这里：被其他块的 metadata.parent_id 引用的块是父块，只存储、不进索引；
expand 把命中的子块换成父块。没有父块的切分器（递归、定长）不受任何影响。

上层（检索器、pipeline、评测器）永远不需要知道"先切分还是先向量化""批大小多少""BM25 和向量索引怎么保持 id 对齐"。
一旦这些细节泄漏到上层，每加一个功能都要改好几处 —— 那就是浅类的代价。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from ..core.interfaces import Chunker, TextEncoder
from ..core.types import Chunk, Document, ScoredChunk
from .indexes import BM25Index, FlatVectorIndex

logger = logging.getLogger(__name__)

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
        self._parents: dict[str, Chunk] = {}

    def add(self, documents: Iterable[Document], *, progress: bool = False) -> dict[str, int]:
        """
        把文档写入知识库，返回本次写入统计。
        采用"攒批"写入：向量化是最贵的一步，逐条调用会让远程 API 慢十倍以上
        """
        buffer: list[Chunk] = []
        added = {"documents": 0, "chunks": 0, "parents": 0}
        for doc in documents:
            chunks = self.chunker.split(doc)
            parent_ids = {c.metadata["parent_id"] for c in chunks if c.metadata.get("parent_id")}
            for chunk in chunks:
                if chunk.chunk_id in parent_ids:
                    self._parents[chunk.chunk_id] = chunk
                else:
                    buffer.append(chunk)
            added["documents"] += 1
            added["parents"] += len(parent_ids)
            added["chunks"] += len(chunks) - len(parent_ids)
            if len(buffer) >= self.batch_size:
                self._flush(buffer)
                buffer = []
            if progress and added["documents"] % 200 == 0:
                logger.info("已处理 %d 篇文档 / %d 个片段", added["documents"], added["chunks"])

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
        return self._chunks_by_id.get(chunk_id) or self._parents.get(chunk_id)

    def expand(self, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """
        把命中的子块换成所在的父块，交给大模型完整的上下文。

        同一个父块下命中多个子块时只保留一份，排在最靠前的子块的位置上，用它的分数；
        命中了哪些子块记在 debug["children"] 里，排障时能看出是哪几句话把父块召回的。
        没有父块的片段原样保留，所以对任何切分器调用它都是安全的。
        """
        expanded: list[ScoredChunk] = []
        seen: dict[str, ScoredChunk] = {}
        for hit in hits:
            parent = self._parents.get(hit.chunk.metadata.get("parent_id", ""))
            if parent is None:
                expanded.append(hit)
                continue
            if parent.chunk_id in seen:
                seen[parent.chunk_id].debug["children"].append(hit.chunk.chunk_id)
                continue
            merged = ScoredChunk(chunk=parent, score=hit.score, source=hit.source,
                                 debug={**hit.debug, "children": [hit.chunk.chunk_id]})
            seen[parent.chunk_id] = merged
            expanded.append(merged)
        return expanded

    def stats(self) -> dict[str, int]:
        return {
            "documents": self._doc_count,
            "chunks": len(self.vector_index),
            "bm25_chunks": len(self.bm25_index),
            "parents": len(self._parents),
            "dimension": self.encoder.dimension
        }

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.vector_index.save(path)
        self.bm25_index.save(path)
        # 父块不进索引，单独落盘；少了它，复用索引时命中的子块就展开不了
        with (p / "parents.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self._parents.values():
                fh.write(json.dumps(chunk.__dict__, ensure_ascii=False) + "\n")
        (p / "meta.json").write_text(json.dumps(self.stats(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        p = Path(path)
        self.vector_index.load(path)
        self.bm25_index.load(path)
        meta = json.loads((p / "meta.json").read_text(encoding="utf-8"))
        self._doc_count = meta.get("documents", 0)
        parents = p / "parents.jsonl"
        lines = parents.read_text(encoding="utf-8").splitlines() if parents.exists() else []
        self._parents = {c.chunk_id: c for c in (Chunk(**json.loads(line)) for line in lines if line.strip())}
