"""
store.chroma_index —— Chroma 向量索引（持久化 + HNSW 可调）。

【为什么要有它】
FlatVectorIndex 是暴力检索：每次查询都算一遍全量内积。十万片段以内够用，
再大就必须上近似最近邻（ANN）。Chroma 内部用 HNSW 图索引，把复杂度从
O(N) 降到约 O(log N)，代价是**牺牲一点召回率**换速度 —— 这个取舍值多少，
正是本类存在的意义：它让你能把"精确 vs 近似"做成一组可量化的对照实验。

【它只实现了一个接口】
VectorIndex 的 add / search / __len__ / save / load。
因此上层（KnowledgeBase、检索器、管线、评测）一行都不用改，
配置里把 index.type 从 flat 换成 chroma 就行 —— 这是窄接口的红利。

【离线建库、在线只编码 query】
这是向量检索的标准工作流，也是很多人会写错的地方：
    离线（一次）：文档 → 切分 → **向量化** → 写入 Chroma（落盘）
    在线（每次）：问题 → **只编码这一条** → 查库
Chroma 用 PersistentClient 落盘，进程重启后 load() 直接复用，
不需要重新向量化 —— 省下的是真金白银（远程 embedding 按 token 计费）。

【关键参数（都会显著影响指标，值得逐个做消融）】
  space            距离度量：cosine / l2 / ip
                   向量已 L2 归一化时三者排序等价，但未归一化时差别很大
  ef_construction  建图质量。越大图越好、建库越慢。默认 100，调到 200~400 常见
  max_neighbors    HNSW 的 M：每个节点的连接数。越大召回越高、内存越大
  ef_search        查询时的搜索宽度。**唯一可以在线调的参数**，
                   调大 → 召回升、延迟升。做 recall/latency 权衡曲线就靠它
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..core.interfaces import VectorIndex
from ..core.registry import registry
from ..core.types import Chunk, ScoredChunk


@registry.register("index", "chroma")
class ChromaVectorIndex(VectorIndex):
    """Chroma 持久化向量索引。

    注意：本类**不使用** Chroma 自带的 embedding function。
    向量一律由本项目的 TextEncoder 生成后传入，原因有二：
      1. 保证"索引时"和"查询时"用的是同一个模型、同一套前缀；
      2. 换 embedding 模型时不必换向量库，两件事互相独立。
    把编码职责留在 encoder 层，是接口分工清晰的体现。
    """

    def __init__(
        self,
        path: str = "vector_store/chroma",
        collection: str = "learn_rag_default",
        space: str = "cosine",
        ef_construction: int = 200,
        max_neighbors: int = 32,
        ef_search: int = 100,
        batch_size: int = 2000,
        reset: bool = False,
    ) -> None:
        import chromadb  # 延迟导入：不装 chromadb 也能用 flat 索引

        self.path, self.collection_name = path, _safe_name(collection)
        self.space, self.ef_search = space, ef_search
        self.batch_size = batch_size

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=path)
        if reset:
            # 换了 embedding 模型或维度必须重建，否则新旧向量混在一起，
            # 查询结果会莫名其妙地差，而且**不报错** —— 极难排查的坑
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass

        self._collection = self._client.get_or_create_collection(
            self.collection_name,
            configuration={
                "hnsw": {
                    "space": space,
                    "ef_construction": ef_construction,
                    "max_neighbors": max_neighbors,
                    "ef_search": ef_search,
                }
            },
        )
        # Chroma 只存 metadata（标量），chunk 的完整结构另存一份，
        # 保证取回的 Chunk 和写入时完全一致
        self._meta_path = Path(path) / f"{self.collection_name}_chunks.jsonl"

    # ------------------------------------------------------------------
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if not chunks:
            return
        # 单次写入有上限（实测 5461），超了会报错，这里主动分批
        for i in range(0, len(chunks), self.batch_size):
            batch, vecs = chunks[i : i + self.batch_size], vectors[i : i + self.batch_size]
            self._collection.upsert(  # upsert 而不是 add：重复建库时幂等，不会报 id 冲突
                ids=[c.chunk_id for c in batch],
                embeddings=[list(map(float, v)) for v in vecs],
                documents=[c.text for c in batch],
                metadatas=[_flatten_meta(c) for c in batch],
            )
        with self._meta_path.open("a", encoding="utf-8") as fh:
            for c in chunks:
                fh.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    def search(
        self, vector: Sequence[float], top_k: int, *, where: dict[str, Any] | None = None
    ) -> list[ScoredChunk]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(map(float, vector))],
            n_results=min(top_k, self._collection.count()),
            where=_to_chroma_where(where),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        hits: list[ScoredChunk] = []
        for cid, text, raw, dist in zip(ids, docs, metas, dists):
            raw = dict(raw or {})
            hits.append(
                ScoredChunk(
                    chunk=Chunk(
                        chunk_id=cid,
                        doc_id=str(raw.get("doc_id", cid.split("#")[0])),
                        text=text or "",
                        position=int(raw.get("position", 0)),
                        metadata=_restore_meta(raw),
                    ),
                    score=_to_score(dist, self.space),
                    source="vector",
                    debug={"distance": float(dist), "space": self.space},
                )
            )
        return hits

    def __len__(self) -> int:
        return int(self._collection.count())

    # ------------------------------------------------------------------
    # Chroma 本来就是持久化的，save/load 只需保证接口一致
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Chroma 写入即落盘，这里只记录一份元信息便于人工核对。"""
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "chroma_meta.json").write_text(
            json.dumps(
                {"path": self.path, "collection": self.collection_name,
                 "space": self.space, "count": len(self)},
                ensure_ascii=False, indent=2),
            encoding="utf-8")

    def load(self, path: str) -> None:
        """无需动作：PersistentClient 在 __init__ 时已经把数据挂上了。

        故意留空而不是抛异常 —— 上层的"建库/加载"流程因此不需要为
        Chroma 写特例分支。让异常情况消失，而不是把它抛给调用方。
        """
        return None

    def set_ef_search(self, ef_search: int) -> None:
        """在线调整搜索宽度 —— 做 recall/latency 权衡曲线时用。

        注意它只影响查询，不影响已建好的图，所以可以随时改、立刻生效。
        """
        self.ef_search = ef_search
        try:
            self._collection.modify(configuration={"hnsw": {"ef_search": ef_search}})
        except Exception:
            pass  # 旧版本不支持在线修改时静默跳过，不影响主流程


# ---------------------------------------------------------------------------
def _to_score(distance: float, space: str) -> float:
    """把"距离（越小越好）"翻译成本项目统一的"分数（越大越好）"。

    Chroma 三种度量返回的都是距离：
      cosine → 1 - 余弦相似度，范围 [0, 2]
      ip     → 1 - 内积（向量归一化后等价于 cosine）
      l2     → 欧氏距离平方，范围 [0, ∞)
    统一到"越大越相关"是 ScoredChunk 的契约，必须在适配器里做掉，
    不能让上层去判断"这个分数到底是越大好还是越小好"。
    """
    d = float(distance)
    if space in {"cosine", "ip"}:
        return 1.0 - d
    return 1.0 / (1.0 + d)   # l2：单调递减映射到 (0, 1]


# 完整元数据序列化后存放的字段名
_FULL_META = "_metadata_json"


def _flatten_meta(chunk: Chunk) -> dict[str, Any]:
    """
    Chroma 的 metadata 只接受 str/int/float/bool，而块的元数据里有列表（kinds）和字典列表（assets）。

    早期做法是把列表拼成逗号分隔的字符串并截断到 500 字：一个块里图片一多，后面的资产路径就被从中间截断，
    取回后列表也变成了字符串。现在分两份存：标量字段原样放进去，供 where 过滤；完整元数据序列化成
    一个 JSON 字段，取回时原样还原，写进去什么、读出来就是什么。
    """
    out: dict[str, Any] = {"doc_id": chunk.doc_id, "position": chunk.position}
    for key, value in (chunk.metadata or {}).items():
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
    out[_FULL_META] = json.dumps(chunk.metadata or {}, ensure_ascii=False)
    return out


def _restore_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    """还原写入时的完整元数据。旧版本建的集合没有 JSON 字段，退回拍平后的标量字段，不需要重建索引。"""
    meta = dict(meta or {})
    full = meta.pop(_FULL_META, None)
    if full is None:
        return meta
    try:
        return json.loads(full)
    except ValueError:
        return meta


def _to_chroma_where(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """把本项目的过滤语法翻译成 Chroma 的操作符语法。

    本项目：{"year": 2024} 或 {"year": [2023, 2024]}
    Chroma：{"year": {"$eq": 2024}} / {"year": {"$in": [...]}}，多条件要 $and
    翻译放在适配器里，上层因此不需要知道用的是哪个向量库。
    """
    if not where:
        return None
    clauses = []
    for key, value in where.items():
        if isinstance(value, (list, tuple, set)):
            clauses.append({key: {"$in": list(value)}})
        else:
            clauses.append({key: {"$eq": value}})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _safe_name(name: str) -> str:
    """Chroma 要求集合名 3~512 字符、只含 [a-zA-Z0-9._-] 且首尾为字母数字。"""
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", name).strip("._-")
    if len(cleaned) < 3:
        cleaned = f"col_{cleaned}"
    return cleaned[:512]