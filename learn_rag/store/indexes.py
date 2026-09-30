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
import pickle
from collections import defaultdict
from collections.abc import Sequence
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
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        arr = np.asarray(vectors, dtype=np.float32)
        # 写入时就把关：坏向量一旦进了矩阵，之后每次查询都会被污染，而且很难定位是哪条
        if arr.ndim != 2 or arr.shape[0] != len(chunks):
            raise ValueError(f"向量形状 {arr.shape} 与片段数 {len(chunks)} 不匹配")
        if self._matrix is not None and arr.shape[1] != self._matrix.shape[1]:
            raise ValueError(
                f"向量维度 {arr.shape[1]} 与已有索引维度 {self._matrix.shape[1]} 不一致 —— "
                "换了 embedding 模型？请清空缓存与索引后重建")
        bad = ~np.isfinite(arr).all(axis=1)
        if bad.any():
            ids = [chunks[i].chunk_id for i in np.where(bad)[0][:5]]
            raise ValueError(f"有 {int(bad.sum())} 条向量含 NaN/Inf（前几条：{ids}），检查 encoder 输出")
        self._matrix = arr if self._matrix is None else np.vstack([self._matrix, arr])
        self._chunks.extend(chunks)

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
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "vectors.npy", self._matrix if self._matrix is not None else np.zeros((0, 0), np.float32))
        with (p / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for c in self._chunks:
                fh.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    def load(self, path: str) -> None:
        p = Path(path)
        self._matrix = np.load(p / "vectors.npy")
        self._chunks = [Chunk(**json.loads(line)) for line in (p / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]


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
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._chunks: list[Chunk] = []
        self._freqs: list[dict[str, int]] = []
        self._lengths: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._avgdl = 0.0

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            doc_id = len(self._chunks)
            tokens = index_tokens(chunk.text)
            freq: dict[str, int] = defaultdict(int)
            for tok in tokens:
                freq[tok] += 1
            for tok in freq:
                self._postings[tok].append(doc_id)
            self._chunks.append(chunk)
            self._freqs.append(dict(freq))
            self._lengths.append(len(tokens))
        self._avgdl = sum(self._lengths) / max(len(self._lengths), 1)

    def search(self, query: str, top_k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        n = len(self._chunks)
        scores: dict[int, float] = defaultdict(float)
        for term in set(index_tokens(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for doc_id in posting:
                tf = self._freqs[doc_id][term]
                denom = tf + self.k1 * (1 - self.b + self.b * self._lengths[doc_id] / max(self._avgdl, 1e-9))
                scores[doc_id] += idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [ScoredChunk(chunk=self._chunks[i], score=float(s), source="bm25") for i, s in ranked]

    def __len__(self) -> int:
        return len(self._chunks)

    def save(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with (p / "bm25.pkl").open("wb") as fh:
            pickle.dump(
                {
                    "chunks": self._chunks,
                    "freqs": self._freqs,
                    "lengths": self._lengths,
                    "postings": dict(self._postings),
                    "avgdl": self._avgdl,
                },
                fh,
            )

    def load(self, path: str) -> None:
        with (Path(path) / "bm25.pkl").open("rb") as fh:
            state = pickle.load(fh)
        self._chunks, self._freqs = state["chunks"], state["freqs"]
        self._lengths, self._avgdl = state["lengths"], state["avgdl"]
        self._postings = defaultdict(list, state["postings"])