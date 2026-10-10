"""
store.knowledge_base —— 知识库（离线链路的门面 / Facade）。

这是本项目最典型的"深类"：
  对外接口只有 add(documents) / expand(hits) / verify() / save(path) / load(path) / stats()
  对内却完成了：切分 -> 区分父子块 -> 批量向量化 -> 写片段库 -> 维护倒排索引 -> 增量更新 -> 完整性核对。

上层（检索器、pipeline、评测器）永远不需要知道"先切分还是先向量化""批大小多少""BM25 和向量索引怎么保持 id 对齐"。
一旦这些细节泄漏到上层，每加一个功能都要改好几处 —— 那就是浅类的代价。

数据完整性靠下面几条约定，全部藏在这个类里：
1. 片段库（VectorIndex，如 Chroma）是唯一的数据来源：子块的向量、正文、元数据和父块都存在里面。
   BM25 倒排、父块表和文档清单是派生数据，打开时从片段库重建，不会和它对不上。
2. 子块和父块：父块由切分器决定（被子块的 parent_id 引用的块），写入时不生成向量、只存储；
   子块带向量，并在元数据里记下 embedding 模型（embedding_model），重建派生数据时据此区分两者。
3. 增量更新：每个片段记下所属文档的指纹（正文、元数据、切分参数、模型的哈希）和该文档的片段总数。
   同一篇文档再次写入时，指纹相同且片段齐全就跳过，不再调用 embedding；否则先按 id 覆盖写入新版本，
   再删掉旧版本多出来的片段。中途失败时旧版本还在，最多多出几条旧片段，下次写入时因指纹不一致被清理。
4. 模型一致：片段库里的向量和当前 encoder 来自不同模型时直接报错，不让两种向量混在一起。
5. 失败就近处理：一批 embedding 调用失败，只跳过这批文档（旧版本保留）并计入 failed，不中断整批。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from ..core.interfaces import Chunker, TextEncoder, VectorIndex
from ..core.types import Chunk, Document, ScoredChunk
from .indexes import BM25Index, FlatVectorIndex

logger = logging.getLogger(__name__)

# 知识库写进片段元数据的字段
_FINGERPRINT = "doc_fingerprint"   # 所属文档的指纹
_DOC_CHUNKS = "doc_chunks"         # 所属文档的片段总数（子块 + 父块），用来发现写了一半的文档
_MODEL = "embedding_model"         # 生成向量的模型，只有子块有


class KnowledgeBase:
    def __init__(
            self,
            encoder: TextEncoder,
            chunker: Chunker,
            vector_index: VectorIndex | None = None,
            bm25_index: BM25Index | None = None,
            batch_size: int = 64
    ) -> None:
        self.encoder = encoder
        self.chunker = chunker
        self.vector_index = vector_index if vector_index is not None else FlatVectorIndex()
        self._bm25 = bm25_index if bm25_index is not None else BM25Index()
        self.batch_size = batch_size
        # 以下都是派生数据，由 _sync 从片段库重建
        self._docs: dict[str, dict[str, Chunk]] = {}     # doc_id -> {chunk_id: 片段}
        self._parents: dict[str, Chunk] = {}
        self._models: set[str] = set()
        self._synced = False
        self._lock = threading.Lock()

    @property
    def bm25_index(self) -> BM25Index:
        self._sync()
        self._check_model()
        return self._bm25

    def add(self, documents: Iterable[Document], *, progress: bool = False) -> dict[str, int]:
        """
        把文档写入知识库，返回本次写入统计：
          documents 写入（新增或更新）的文档数；skipped 未变化而跳过的文档数；failed 写入失败的文档数；
          chunks / parents 写入的子块 / 父块数。
        采用"攒批"写入：向量化是最贵的一步，逐条调用会让远程 API 慢十倍以上
        """
        self._sync()
        self._check_model()
        stats = {"documents": 0, "chunks": 0, "parents": 0, "skipped": 0, "failed": 0}
        pending: list[tuple[str, list[Chunk]]] = []
        size = 0
        for doc in documents:
            fingerprint = self._fingerprint(doc)
            if self._is_current(doc.doc_id, fingerprint):
                stats["skipped"] += 1
                continue
            chunks = self._prepare(doc, fingerprint)
            pending.append((doc.doc_id, chunks))
            size += sum(1 for c in chunks if _MODEL in c.metadata)
            if size >= self.batch_size:
                self._flush(pending, stats)
                pending, size = [], 0
            if progress and (stats["documents"] + stats["skipped"]) % 200 == 0:
                logger.info("已写入 %d 篇、跳过 %d 篇未变化的文档", stats["documents"], stats["skipped"])
        if pending:
            self._flush(pending, stats)
        if stats["failed"]:
            logger.error("%d 篇文档写入失败（原因见上方日志），它们已入库的旧版本保持不变，重新运行即可续写", stats["failed"])
        return stats

    def _prepare(self, doc: Document, fingerprint: str) -> list[Chunk]:
        """切分并给每个片段打上文档指纹、片段总数；子块再记下 embedding 模型。"""
        chunks = self.chunker.split(doc)
        parent_ids = {c.metadata["parent_id"] for c in chunks if c.metadata.get("parent_id")}
        model = self.encoder.signature()
        for chunk in chunks:
            chunk.metadata[_FINGERPRINT] = fingerprint
            chunk.metadata[_DOC_CHUNKS] = len(chunks)
            if chunk.chunk_id not in parent_ids:
                chunk.metadata[_MODEL] = model
        return chunks

    def _flush(self, pending: list[tuple[str, list[Chunk]]], stats: dict[str, int]) -> None:
        children = [c for _, chunks in pending for c in chunks if _MODEL in c.metadata]
        try:
            vectors = self.encoder.encode([c.text for c in children], is_query=False)
        except Exception as exc:
            doc_ids = [doc_id for doc_id, _ in pending]
            logger.error("embedding 失败，跳过 %d 篇文档 %s：%s", len(doc_ids), doc_ids[:5], exc)
            stats["failed"] += len(pending)
            return
        if len(vectors) != len(children):
            logger.error("encoder 返回 %d 条向量，但提交了 %d 段文本，跳过这一批 %d 篇文档",
                         len(vectors), len(children), len(pending))
            stats["failed"] += len(pending)
            return
        by_id = {c.chunk_id: v for c, v in zip(children, vectors)}
        for doc_id, chunks in pending:
            try:
                self._write(doc_id, chunks, by_id)
            except ValueError as exc:   # 坏向量、维度不一致：这篇不写，旧版本保留
                logger.error("写入文档 %s 失败，已入库的旧版本保持不变：%s", doc_id, exc)
                stats["failed"] += 1
                continue
            children_count = sum(1 for c in chunks if _MODEL in c.metadata)
            stats["documents"] += 1
            stats["chunks"] += children_count
            stats["parents"] += len(chunks) - children_count

    def _write(self, doc_id: str, chunks: list[Chunk], vectors: dict[str, list[float]]) -> None:
        """
        写入一篇文档的新版本。顺序保证中途失败时数据仍可恢复：
        先写子块（带向量）、再写父块、最后删旧版本多出来的片段 —— 任何一步失败，旧版本都还完整地在库里。
        """
        children = [c for c in chunks if _MODEL in c.metadata]
        parents = [c for c in chunks if _MODEL not in c.metadata]
        self.vector_index.add(children, [vectors[c.chunk_id] for c in children])
        self.vector_index.add(parents)
        old = self._docs.get(doc_id, {})
        new_ids = {c.chunk_id for c in chunks}
        self.vector_index.delete([cid for cid in old if cid not in new_ids])

        self._bm25.remove(list(old))
        self._bm25.add(children)
        for cid in old:
            self._parents.pop(cid, None)
        self._parents.update((p.chunk_id, p) for p in parents)
        self._docs[doc_id] = {c.chunk_id: c for c in chunks}
        if children:
            self._models.add(self.encoder.signature())

    def _is_current(self, doc_id: str, fingerprint: str) -> bool:
        """库里已有这篇文档的同一版本，并且片段齐全（没有写了一半的情况）。"""
        stored = self._docs.get(doc_id)
        if not stored:
            return False
        expected = {c.metadata.get(_DOC_CHUNKS) for c in stored.values()}
        fingerprints = {c.metadata.get(_FINGERPRINT) for c in stored.values()}
        return fingerprints == {fingerprint} and expected == {len(stored)}

    def _fingerprint(self, doc: Document) -> str:
        """
        文档版本的指纹：正文、结构元素、元数据、切分参数、embedding 模型，任何一项变了都要重新写入。
        注意它只看切分器的参数，改了切分代码本身（参数不变）时，要清空索引重建。
        """
        parts = [
            doc.text,
            json.dumps([asdict(e) for e in doc.elements], ensure_ascii=False, sort_keys=True, default=str),
            json.dumps(doc.metadata, ensure_ascii=False, sort_keys=True, default=str),
            type(self.chunker).__name__ + json.dumps(vars(self.chunker), sort_keys=True, default=str),
            self.encoder.signature(),
        ]
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]

    def _sync(self) -> None:
        """
        从片段库重建派生数据：文档清单、父块表、BM25 倒排。只在打开和 load 之后做一次。

        先在局部变量里建好再一次性替换，并且加锁：读片段库中途出错不会留下半截状态，
        评测多线程同时发起第一次查询时也只会重建一次。
        """
        with self._lock:
            if self._synced:
                return
            docs: dict[str, dict[str, Chunk]] = {}
            parents: dict[str, Chunk] = {}
            models: set[str] = set()
            children: list[Chunk] = []
            for chunk in self.vector_index.chunks():
                docs.setdefault(chunk.doc_id, {})[chunk.chunk_id] = chunk
                model = chunk.metadata.get(_MODEL)
                if model is None:
                    parents[chunk.chunk_id] = chunk
                else:
                    children.append(chunk)
                    models.add(model)
            bm25 = BM25Index(self._bm25.k1, self._bm25.b)
            bm25.add(children)
            self._docs, self._parents, self._models, self._bm25 = docs, parents, models, bm25
            self._synced = True

    def _check_model(self) -> None:
        current = self.encoder.signature()
        other = self._models - {current}
        if other:
            raise ValueError(
                f"片段库里的向量来自 {sorted(other)}，当前 encoder 是 {current}。"
                "两个模型的向量不可比，混在一起检索结果会莫名变差。"
                "请换回原来的模型，或者清空索引后重建（Chroma 设 index.reset: true 或换一个 collection）")

    def expand(self, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        """
        把命中的子块换成所在的父块，交给大模型完整的上下文。

        同一个父块下命中多个子块时只保留一份，排在最靠前的子块的位置上，用它的分数；
        命中了哪些子块记在 debug["children"] 里，排障时能看出是哪几句话把父块召回的。
        没有父块的片段原样保留，所以对任何切分器调用它都是安全的。
        """
        self._sync()
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

    def get(self, chunk_id: str) -> Chunk | None:
        self._sync()
        return self._docs.get(chunk_id.rsplit("#", 1)[0], {}).get(chunk_id)

    def stats(self) -> dict[str, int]:
        self._sync()
        self._check_model()
        return {
            "documents": len(self._docs),
            "chunks": len(self.vector_index),
            "bm25_chunks": len(self._bm25),
            "parents": len(self._parents),
            "dimension": self.encoder.dimension
        }

    def verify(self) -> dict[str, object]:
        """
        全量核对片段库，返回 {"ok": 是否完整, "problems": [问题描述], 以及 stats() 的各项}。

        重新从片段库读取再核对，查的是落盘的数据而不是内存里的状态。检查：
        每篇文档的片段是否齐全且属于同一个版本、子块的父块是否存在、有没有没人引用的父块、
        向量是否来自当前模型、可检索的片段数是否等于子块数。
        """
        self._synced = False
        self._sync()
        problems: list[str] = []
        legacy = [doc_id for doc_id, chunks in self._docs.items()
                  if any(_FINGERPRINT not in c.metadata for c in chunks.values())]
        if legacy:
            # 旧版本程序写入的片段没有指纹和模型标记，下面几项检查对它们没有意义，只报这一条
            problems.append(f"{len(legacy)} 篇文档是旧版本程序写入的（缺少文档指纹），例如 {legacy[:3]}；"
                            "重新 build 一次会自动补齐，或清空索引后重建")
        else:
            problems += self._check_structure()
        other = self._models - {self.encoder.signature()}
        if other:
            problems.append(f"片段库里有其他模型生成的向量 {sorted(other)}，当前 encoder 是 {self.encoder.signature()}")
        stats = {
            "documents": len(self._docs), "chunks": len(self.vector_index),
            "bm25_chunks": len(self._bm25), "parents": len(self._parents),
        }
        return {"ok": not problems, "problems": problems, **stats}

    def _check_structure(self) -> list[str]:
        """每篇文档片段齐全且属于同一个版本、子块的父块都在、没有孤立的父块、可检索片段数等于子块数。"""
        problems: list[str] = []
        for doc_id, chunks in self._docs.items():
            fingerprints = {c.metadata.get(_FINGERPRINT) for c in chunks.values()}
            expected = {c.metadata.get(_DOC_CHUNKS) for c in chunks.values()}
            if len(fingerprints) > 1:
                problems.append(f"{doc_id}：混有 {len(fingerprints)} 个版本的片段（上次写入中断），重新 build 会自动清理")
            elif expected != {len(chunks)}:
                problems.append(f"{doc_id}：应有 {sorted(expected)} 个片段，实际 {len(chunks)} 个，重新 build 会自动补齐")
        children = [c for chunks in self._docs.values() for c in chunks.values() if _MODEL in c.metadata]
        referenced = {c.metadata.get("parent_id") for c in children}
        missing = [c.chunk_id for c in children
                   if c.metadata.get("parent_id") and c.metadata["parent_id"] not in self._parents]
        if missing:
            problems.append(f"{len(missing)} 个子块的父块不存在，例如 {missing[:3]}")
        orphans = [pid for pid in self._parents if pid not in referenced]
        if orphans:
            problems.append(f"{len(orphans)} 个父块没有子块引用，例如 {orphans[:3]}")
        if len(self.vector_index) != len(children):
            problems.append(f"可检索的片段 {len(self.vector_index)} 个，但子块有 {len(children)} 个")
        return problems

    def save(self, path: str) -> None:
        """片段库自己负责落盘（Chroma 写入即落盘，flat 在这里写文件）；BM25 和父块表不落盘，打开时重建。"""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.vector_index.save(path)
        (p / "meta.json").write_text(json.dumps(self.stats(), ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        self.vector_index.load(path)
        self._synced = False
