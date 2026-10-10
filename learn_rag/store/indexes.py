"""
store.indexes —— 索引（存储层）。

两个索引对应 RAG 的两条经典召回通道：
  FlatVectorIndex : 稠密向量（语义近似），擅长"换个说法也能找到"
  BM25Index       : 稀疏关键词（词频统计），擅长"专有名词、编号、罕见词精确命中"
二者的缺点恰好互补，所以工业界几乎都用混合检索（见 retrieval/hybrid.py）。

BM25 这里手写实现（不到 60 行），是为了让你真正看懂
tf 饱和(k1)、长度惩罚(b)、idf 这三件事，而不是黑箱调库。
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..core.interfaces import VectorIndex
from ..core.registry import registry
from ..core.text import index_tokens
from ..core.types import Chunk, ScoredChunk


@registry.register("index", "flat")
class FlatVectorIndex(VectorIndex):
    """暴力检索的稠密索引（内积 = 余弦，因为向量都已 L2 归一化）。

    十万级片段以内它足够快且零依赖；要上百万时把这个类换成 faiss/hnswlib
    的实现即可，上层一行不用改 —— 这正是窄接口的价值。

    数据全在内存里，save 时一次写出：vectors.npy 与 chunks.jsonl 逐行对应（可检索的片段），
    stored.jsonl 存只存储不检索的片段（父块）。
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._rows: dict[str, int] = {}          # chunk_id -> 行号，用于按 id 覆盖和删除
        self._stored: dict[str, Chunk] = {}

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]] | None = None) -> None:
        if not chunks:
            return
        if vectors is None:
            for chunk in chunks:
                self._stored[chunk.chunk_id] = chunk
            return
        dimension = self._matrix.shape[1] if self._matrix is not None and self._matrix.size else None
        arr = check_vectors(chunks, vectors, dimension)
        # 同一批里重复的 id 以最后一次为准；已存在的 id 覆盖原行，重复建库、续跑都不会产生重复片段
        latest = {chunk.chunk_id: (chunk, row) for chunk, row in zip(chunks, arr)}
        fresh: list[tuple[Chunk, np.ndarray]] = []
        for chunk_id, (chunk, row) in latest.items():
            i = self._rows.get(chunk_id)
            if i is None:
                fresh.append((chunk, row))
            else:
                self._chunks[i] = chunk
                self._matrix[i] = row
        if fresh:
            block = np.stack([row for _, row in fresh])
            self._matrix = block if self._matrix is None or not self._matrix.size else np.vstack([self._matrix, block])
            for chunk, _ in fresh:
                self._rows[chunk.chunk_id] = len(self._chunks)
                self._chunks.append(chunk)

    def delete(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            self._stored.pop(chunk_id, None)
        drop = {self._rows[c] for c in chunk_ids if c in self._rows}
        if not drop:
            return
        keep = [i for i in range(len(self._chunks)) if i not in drop]
        self._chunks = [self._chunks[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else None
        self._rows = {chunk.chunk_id: i for i, chunk in enumerate(self._chunks)}

    def chunks(self) -> Iterable[Chunk]:
        return [*self._chunks, *self._stored.values()]

    def search(self, vector: Sequence[float], top_k: int, *, where: dict[str, Any] | None = None) -> list[ScoredChunk]:
        if self._matrix is None or not self._chunks:
            return []
        query = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(query).all():
            raise ValueError("查询向量含 NaN/Inf，检查 encoder 输出")
        # numpy 2.x 在 macOS 上走 Accelerate 做 float32 矩阵乘法时，会冒出
        # "overflow / invalid value encountered in matmul" 的**假警告**，结果本身是对的。
        # 这里屏蔽警告，但**不盲信**：算完检查结果，真有非有限值就用 float64 重算一次。
        # 因为写入时已经保证矩阵全是有限值，这一步只是对 BLAS 实现差异的兜底。
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scores = self._matrix @ query
        if not np.isfinite(scores).all():
            scores = (self._matrix.astype(np.float64) @ query.astype(np.float64)).astype(np.float32)
        order = np.argsort(-scores)
        hits: list[ScoredChunk] = []
        for idx in order:
            chunk = self._chunks[int(idx)]
            if where and not _match(chunk.metadata, where):
                continue  # 元数据过滤：按时间/来源/作者等收窄检索范围
            hits.append(ScoredChunk(chunk=chunk, score=float(scores[idx]), source="vector"))
            if len(hits) >= top_k:
                break
        return hits

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: str) -> None:
        """每个文件先写临时文件再原子替换：进程中途被杀也不会留下写了一半的文件。"""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        matrix = self._matrix if self._matrix is not None else np.zeros((0, 0), np.float32)
        _atomic_write(p / "vectors.npy", lambda fh: np.save(fh, matrix))
        _atomic_write(p / "chunks.jsonl", lambda fh: _write_jsonl(fh, self._chunks))
        _atomic_write(p / "stored.jsonl", lambda fh: _write_jsonl(fh, self._stored.values()))

    def load(self, path: str) -> None:
        p = Path(path)
        matrix = np.load(p / "vectors.npy")
        chunks = _read_jsonl(p / "chunks.jsonl")
        if matrix.shape[0] != len(chunks):
            raise ValueError(f"{p} 中向量 {matrix.shape[0]} 行、片段 {len(chunks)} 条，两者对不上，"
                             "索引文件不完整（可能上次保存时被中断），请重新 build")
        self._chunks = chunks
        self._matrix = matrix if matrix.size else None
        self._rows = {chunk.chunk_id: i for i, chunk in enumerate(chunks)}
        stored = p / "stored.jsonl"
        self._stored = {c.chunk_id: c for c in _read_jsonl(stored)} if stored.exists() else {}


def check_vectors(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], dimension: int | None) -> np.ndarray:
    """
    写入前检查向量：坏向量一旦进了索引，之后每次查询都会被污染，而且很难定位是哪条。
    dimension 是索引里已有向量的维度（空索引为 None）。
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != len(chunks):
        raise ValueError(f"向量形状 {arr.shape} 与片段数 {len(chunks)} 不匹配")
    if dimension is not None and arr.shape[1] != dimension:
        raise ValueError(
            f"向量维度 {arr.shape[1]} 与已有索引维度 {dimension} 不一致 —— "
            "换了 embedding 模型？请清空索引后重建（Chroma 设 index.reset: true 或换一个 collection）")
    bad = ~np.isfinite(arr).all(axis=1)
    if bad.any():
        ids = [chunks[i].chunk_id for i in np.where(bad)[0][:5]]
        raise ValueError(f"有 {int(bad.sum())} 条向量含 NaN/Inf（前几条：{ids}），检查 encoder 输出")
    return arr


def _atomic_write(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:
        write(fh)
    os.replace(tmp, path)


def _write_jsonl(fh, chunks) -> None:
    for chunk in chunks:
        fh.write((json.dumps(chunk.__dict__, ensure_ascii=False) + "\n").encode("utf-8"))


def _read_jsonl(path: Path) -> list[Chunk]:
    return [Chunk(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _match(metadata: dict[str, Any], where: dict[str, Any]) -> bool:
    """极简元数据过滤：等值或 in 列表。够用且可预测。"""
    for key, expected in where.items():
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class BM25Index:
    """Okapi BM25 倒排索引。

    公式：score(q,d) = Σ_t idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1-b+b·|d|/avgdl))
      - idf 抑制"的、是、the"这类高频无信息词
      - k1 让词频收益饱和（一个词出现 20 次不该比出现 5 次强 4 倍）
      - b  惩罚长文档（长文档天然更容易含有查询词）

    它不单独落盘：知识库打开时从片段库重建，和向量索引永远是同一批片段。
    删除只把位置标记为空（文档更新时每篇都要删旧片段，每次都重建倒排表太慢），
    空位超过一半时再整体压缩一次。
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._chunks: list[Chunk | None] = []
        self._freqs: list[dict[str, int]] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._slots: dict[str, int] = {}         # chunk_id -> 位置
        self._total_length = 0

    def add(self, chunks: Sequence[Chunk]) -> None:
        """按 chunk_id 写入，已存在的先删掉再加，同一个片段不会被统计两次。"""
        self.remove([c.chunk_id for c in chunks])
        for chunk in chunks:
            slot = len(self._chunks)
            tokens = index_tokens(chunk.text)
            freq: dict[str, int] = defaultdict(int)
            for tok in tokens:
                freq[tok] += 1
            for tok in freq:
                self._postings[tok].append(slot)
            self._chunks.append(chunk)
            self._freqs.append(dict(freq))
            self._lengths.append(len(tokens))
            self._slots[chunk.chunk_id] = slot
            self._total_length += len(tokens)

    def remove(self, chunk_ids: Sequence[str]) -> None:
        for chunk_id in chunk_ids:
            slot = self._slots.pop(chunk_id, None)
            if slot is None:
                continue
            self._total_length -= self._lengths[slot]
            self._chunks[slot], self._freqs[slot], self._lengths[slot] = None, {}, 0
        if len(self._chunks) > 2 * len(self._slots) + 64:
            self._compact()

    def _compact(self) -> None:
        """去掉空位、重建倒排表。用保存的词频重建，不需要重新切词。"""
        keep = [i for i, chunk in enumerate(self._chunks) if chunk is not None]
        self._chunks = [self._chunks[i] for i in keep]
        self._freqs = [self._freqs[i] for i in keep]
        self._lengths = [self._lengths[i] for i in keep]
        self._slots = {chunk.chunk_id: i for i, chunk in enumerate(self._chunks)}
        self._postings = defaultdict(list)
        for slot, freq in enumerate(self._freqs):
            for tok in freq:
                self._postings[tok].append(slot)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        n = len(self._slots)
        if not n:
            return []
        avgdl = self._total_length / n
        scores: dict[int, float] = defaultdict(float)
        for term in set(index_tokens(query)):
            posting = [slot for slot in self._postings.get(term, ()) if self._chunks[slot] is not None]
            if not posting:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for slot in posting:
                tf = self._freqs[slot][term]
                denom = tf + self.k1 * (1 - self.b + self.b * self._lengths[slot] / max(avgdl, 1e-9))
                scores[slot] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(s), source="bm25") for i, s in ranked]

    def __len__(self) -> int:
        return len(self._slots)
