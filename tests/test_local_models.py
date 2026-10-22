"""
本地 BGE-M3 和本地 bge-reranker 的接入测试。

真实模型各有 2 GB 多，测试里用一个同接口的假 sentence_transformers 模块代替，检验的是"接线"：
页面参数正确翻译成配置、模型第一次用到时才加载、多个知识库共用一份模型、建库不加载精排模型。
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.core import local_models
from learn_rag.core.text import index_tokens
from learn_rag.embedding.encoders import HashingEncoder


class _FakeSentenceTransformer:
    loads: list[str] = []

    def __init__(self, name, device=None):
        _FakeSentenceTransformer.loads.append(name)
        self._hashing = HashingEncoder(dimension=32)

    def get_sentence_embedding_dimension(self):
        return 32

    def encode(self, texts, batch_size=32, normalize_embeddings=True):
        import numpy as np

        return np.array(self._hashing.encode(list(texts)))


class _FakeCrossEncoder:
    loads: list[str] = []

    def __init__(self, name, device=None):
        _FakeCrossEncoder.loads.append(name)
        self.max_length = 512

    def predict(self, pairs, batch_size=16):
        # 和问题共有的词越多分越高：足够检验"精排确实改变了顺序"
        return [len(set(index_tokens(q)) & set(index_tokens(t))) for q, t in pairs]


class _FakeBGEM3:
    """FlagEmbedding.BGEM3FlagModel 的替身：稠密向量用 hashing，稀疏向量用"词 → 出现比例"。"""

    loads: list[str] = []
    calls = 0

    def __init__(self, name, use_fp16=False):
        _FakeBGEM3.loads.append(name)
        self._hashing = HashingEncoder(dimension=48)

    def encode(self, texts, batch_size=16, max_length=1024, return_dense=True, return_sparse=False,
               return_colbert_vecs=False):
        import numpy as np

        _FakeBGEM3.calls += 1
        sparse = []
        for text in texts:
            tokens = index_tokens(text)
            sparse.append({t: np.float16(tokens.count(t) / len(tokens)) for t in set(tokens)} if tokens else {})
        return {"dense_vecs": np.array(self._hashing.encode(list(texts))), "lexical_weights": sparse}


_FLAG = types.ModuleType("FlagEmbedding")
_FLAG.BGEM3FlagModel = _FakeBGEM3

_FAKE = types.ModuleType("sentence_transformers")
_FAKE.SentenceTransformer = _FakeSentenceTransformer
_FAKE.CrossEncoder = _FakeCrossEncoder


class TestLocalModels(unittest.TestCase):
    def setUp(self) -> None:
        from learn_rag.server.library import Library

        _FakeSentenceTransformer.loads, _FakeCrossEncoder.loads = [], []
        local_models._SHARED.clear()
        patches = [patch.dict(sys.modules, {"sentence_transformers": _FAKE}),
                   patch("learn_rag.server.environment.has_local_models", return_value=True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(local_models._SHARED.clear)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.library = Library(self.tmp, worker=True)

    def _create(self, name: str) -> str:
        text = "# 运维手册\n\n## 巡检\n\n" + "每日检查磁盘、内存和网络延迟，异常时上报值班负责人。" * 3 + \
               "\n\n## 备份\n\n" + "数据库每晚两点全量备份，保留三十天，备份失败自动告警。" * 3
        upload_id, _ = self.library.staging.save(None, f"{name}.md", io.BytesIO(text.encode("utf-8")))
        settings = {"encoder": {"type": "sentence_transformers"},           # 不填模型名：用默认的 BAAI/bge-m3
                    "reranker": {"type": "cross_encoder"},
                    "retrieval": {"candidate_k": 50, "top_k": 3},
                    "index": {"type": "flat"}}
        created = self.library.create(name, settings, upload_id)
        job = self.library.jobs.wait(created["job"]["id"])
        self.assertEqual(job.status, "done", job.error)
        self.assertTrue(job.verify["ok"], job.verify["problems"])
        return created["kb"]["id"]

    def test_bge_m3_and_reranker_are_wired_and_shared(self):
        first, second = self._create("一号库"), self._create("二号库")
        self.assertEqual(_FakeSentenceTransformer.loads, ["BAAI/bge-m3"])   # 两个知识库共用一份 bge-m3
        self.assertEqual(_FakeCrossEncoder.loads, [])                        # 建库用不到精排，不加载

        info = self.library.info(first)
        self.assertEqual(info["settings"]["encoder"], {"type": "sentence_transformers", "model": "BAAI/bge-m3"})
        self.assertEqual(info["settings"]["reranker"], {"type": "cross_encoder", "model": "BAAI/bge-reranker-v2-m3"})
        pipe = self.library.pipeline(first)
        self.assertIn("BAAI/bge-m3", pipe.kb.encoder.signature())

        result = pipe.retrieve_only("数据库备份保留多少天")
        self.assertEqual(_FakeCrossEncoder.loads, ["BAAI/bge-reranker-v2-m3"])   # 第一次查询时才加载
        self.assertLessEqual(len(result.contexts), 3)
        self.assertIn("备份", result.contexts[0].text)
        self.library.pipeline(second).retrieve_only("巡检内容")
        self.assertEqual(len(_FakeCrossEncoder.loads), 1)                         # 第二个知识库也共用

    def test_viewing_chunks_does_not_load_models(self):
        kb_id = self._create("查看")
        local_models._SHARED.clear()
        _FakeSentenceTransformer.loads = []
        from learn_rag.server.library import Library

        reopened = Library(self.tmp)                     # 模拟服务重启后打开分块查看页
        doc = reopened.document(kb_id, "查看.md")
        self.assertTrue(doc["children"])
        self.assertEqual(_FakeSentenceTransformer.loads, [])


def _has_chroma() -> bool:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_chroma(), "未安装 chromadb")
class TestSparseChannel(unittest.TestCase):
    """BGE-M3 稠密 + 稀疏：稀疏向量随片段存进 Chroma，稀疏倒排从片段库重建，作为第三路召回。"""

    def setUp(self) -> None:
        from learn_rag.server.library import Library

        _FakeBGEM3.loads, _FakeBGEM3.calls = [], 0
        local_models._SHARED.clear()
        for p in (patch.dict(sys.modules, {"FlagEmbedding": _FLAG}),
                  patch("learn_rag.server.environment.has_flag_embedding", return_value=True)):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(local_models._SHARED.clear)
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.library = Library(self.tmp, worker=True)
        docs = {
            "运维.md": "# 运维手册\n\n## 备份\n\n" + "数据库每晚两点全量备份，保留三十天，备份失败自动告警。" * 3,
            "财务.md": "# 报销制度\n\n## 时限\n\n" + "差旅报销需在出差结束后三十天内提交，逾期需部门负责人说明。" * 3,
        }
        upload_id = None
        for name, text in docs.items():
            upload_id, _ = self.library.staging.save(upload_id, name, io.BytesIO(text.encode("utf-8")))
        settings = {"encoder": {"type": "bge_m3"}, "index": {"type": "chroma"},
                    "retrieval": {"weights": {"vector": 1, "bm25": 1, "sparse": 1}}}
        created = self.library.create("三路召回", settings, upload_id)
        job = self.library.jobs.wait(created["job"]["id"])
        self.assertEqual(job.status, "done", job.error)
        self.assertTrue(job.verify["ok"], job.verify["problems"])
        self.kb = created["kb"]["id"]

    def test_sparse_vectors_are_stored_and_searchable(self):
        pipe = self.library.pipeline(self.kb)
        children = [c for d in pipe.kb.documents() for c in pipe.kb.chunks_of(d) if "embedding_model" in c.metadata]
        self.assertTrue(children)
        self.assertTrue(all(isinstance(c.metadata.get("sparse"), dict) and c.metadata["sparse"] for c in children))
        self.assertEqual(len(pipe.retriever.channels), 3)                     # 向量 + BM25 + 稀疏

        from learn_rag.retrieval.retrievers import SparseRetriever

        sparse = SparseRetriever(pipe.kb)
        self.assertEqual(sparse.retrieve("数据库备份保留多久", 1)[0].doc_id, "运维.md")
        self.assertEqual({h.doc_id for h in sparse.retrieve("三十天", 5)}, {"运维.md", "财务.md"})   # 两篇都有
        self.assertEqual({h.doc_id for h in sparse.retrieve("三十天", 5, where={"doc_id": "财务.md"})}, {"财务.md"})
        calls = _FakeBGEM3.calls
        pipe.retrieve_only("差旅报销时限")                                     # 向量和稀疏两路共用一次模型调用
        self.assertEqual(_FakeBGEM3.calls, calls + 1)
        self.assertEqual(_FakeBGEM3.loads, ["BAAI/bge-m3"])

    def test_sparse_index_follows_the_store(self):
        from learn_rag.server.library import Library

        self.library.jobs.wait(self.library.remove_document(self.kb, "财务.md").id)
        reopened = Library(self.tmp)                                          # 换一个进程：稀疏倒排从 Chroma 重建
        kb = reopened.pipeline(self.kb).kb
        self.assertEqual(len(kb.sparse_index), len([c for c in kb.chunks_of("运维.md") if "embedding_model" in c.metadata]))
        self.assertEqual({h.doc_id for h in kb.sparse_index.search({"报": 1.0, "销": 1.0}, 5)}, set())
        self.assertTrue(kb.verify()["ok"])


if __name__ == "__main__":
    unittest.main()
