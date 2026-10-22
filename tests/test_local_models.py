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
        self.library = Library(self.tmp)

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


if __name__ == "__main__":
    unittest.main()
