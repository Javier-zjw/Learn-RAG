"""
最小测试集：用标准库 unittest 编写，pytest 也能直接跑。

测试选取原则（也是《软件设计哲学》的建议）：
优先测"接口契约"和"容易算错的公式"，而不是追求行覆盖率。
下面每个用例都对应一个你手敲时最容易写错的地方。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.core.text import index_tokens, normalize, tokenize
from learn_rag.core.types import Answer, Document, EvalSample, RagResult, ScoredChunk
from learn_rag.eval.metrics import MRRAtK, NDCGAtK, RecallAtK, RougeL, TokenF1
from learn_rag.embedding.encoders import HashingEncoder
from learn_rag.ingest.chunkers import RecursiveChunker
from learn_rag.pipeline.rag import RagPipeline
from learn_rag.core.config import load_config


def _result(doc_ids: list[str], answer: str = "", contexts: str = "") -> RagResult:
    chunks = [
        ScoredChunk(
            chunk=__import__("learn_rag.core.types", fromlist=["Chunk"]).Chunk(
                chunk_id=f"c{i}", doc_id=d, text=contexts or d
            ),
            score=1.0 - i * 0.1,
        )
        for i, d in enumerate(doc_ids)
    ]
    return RagResult(question="q", answer=Answer(text=answer), contexts=chunks)


class TestText(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("　Hello, World！ "), "hello world")

    def test_tokenize_mixed(self):
        self.assertEqual(tokenize("RAG 很好 2024"), ["rag", "很", "好", "2024"])

    def test_index_tokens_has_bigram(self):
        self.assertIn("机器", index_tokens("机器学习"))


class TestChunker(unittest.TestCase):
    def test_size_and_overlap(self):
        doc = Document(doc_id="d", text="句子一。" * 200)
        chunks = RecursiveChunker(chunk_size=100, chunk_overlap=20).split(doc)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(c.text) <= 130 for c in chunks))
        # id 必须可复现
        again = RecursiveChunker(chunk_size=100, chunk_overlap=20).split(doc)
        self.assertEqual([c.chunk_id for c in chunks], [c.chunk_id for c in again])


class TestEncoder(unittest.TestCase):
    def test_normalized_and_deterministic(self):
        enc = HashingEncoder(dimension=64)
        v1, v2 = enc.encode(["检索增强生成", "检索增强生成"])
        self.assertEqual(v1, v2)
        self.assertAlmostEqual(sum(x * x for x in v1), 1.0, places=5)

    def test_similar_texts_closer(self):
        enc = HashingEncoder(dimension=256)
        a, b, c = enc.encode(["向量检索很重要", "向量检索非常重要", "今天天气不错"])
        dot = lambda x, y: sum(i * j for i, j in zip(x, y))
        self.assertGreater(dot(a, b), dot(a, c))


class TestMetrics(unittest.TestCase):
    def test_recall(self):
        sample = EvalSample("1", "q", ["a"], gold_doc_ids=["d1", "d2"])
        self.assertAlmostEqual(RecallAtK(3).compute(sample, _result(["d1", "x", "y"]))["recall@3"], 0.5)

    def test_mrr_position(self):
        sample = EvalSample("1", "q", ["a"], gold_doc_ids=["d1"])
        self.assertAlmostEqual(MRRAtK(5).compute(sample, _result(["x", "d1"]))["mrr@5"], 0.5)

    def test_ndcg_perfect_is_one(self):
        sample = EvalSample("1", "q", ["a"], gold_doc_ids=["d1", "d2"])
        self.assertAlmostEqual(NDCGAtK(5).compute(sample, _result(["d1", "d2", "x"]))["ndcg@5"], 1.0)

    def test_ndcg_penalizes_bad_order(self):
        sample = EvalSample("1", "q", ["a"], gold_doc_ids=["d1", "d2"])
        good = NDCGAtK(5).compute(sample, _result(["d1", "d2", "x"]))["ndcg@5"]
        bad = NDCGAtK(5).compute(sample, _result(["x", "d1", "d2"]))["ndcg@5"]
        self.assertLess(bad, good)

    def test_token_f1_partial(self):
        sample = EvalSample("1", "q", ["北京 是 首都"])
        score = TokenF1().compute(sample, _result(["d"], answer="北京是中国的首都"))
        self.assertGreater(score["token_f1"], 0.5)

    def test_rouge_l_cares_about_order(self):
        sample = EvalSample("1", "q", ["abc def ghi"])
        same = RougeL().compute(sample, _result(["d"], answer="abc def ghi"))["rouge_l"]
        shuffled = RougeL().compute(sample, _result(["d"], answer="ghi def abc"))["rouge_l"]
        self.assertGreater(same, shuffled)


class TestPipeline(unittest.TestCase):
    def test_end_to_end_offline(self):
        cfg = load_config()  # 纯默认配置：全离线
        pipe = RagPipeline.from_config(cfg)
        pipe.index([Document(doc_id="d1", text="RRF 的常数 k 通常取 60。", metadata={"title": "RRF"})])
        result = pipe.answer("RRF 的常数 k 取多少？")
        self.assertIn("60", result.answer.text)
        self.assertEqual(result.retrieved_doc_ids[0], "d1")
        self.assertIn("total", result.timings)



class TestApiReranker(unittest.TestCase):
    """API 精排：协议解析与失败降级。

    用 mock 而不是真发请求 —— 单元测试不该依赖网络和 Key。
    """

    def _cands(self):
        from learn_rag.core.types import Chunk, ScoredChunk

        return [ScoredChunk(Chunk(f"c{i}", "d", t), 0.5) for i, t in enumerate(
            ["今天天气不错。", "RRF常数k取60。", "BM25参数b惩罚长文档。"])]

    def test_parses_results(self):
        import json as _json
        import urllib.request
        from unittest.mock import patch

        from learn_rag.retrieval.rerankers import ApiReranker

        body = {"results": [{"index": 1, "relevance_score": 0.95},
                            {"index": 2, "relevance_score": 0.40}]}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return _json.dumps(body).encode()

        with patch.object(urllib.request, "urlopen", return_value=FakeResp()):
            out = ApiReranker(api_key="fake").rerank("RRF常数", self._cands(), 2)
        self.assertEqual(out[0].text, "RRF常数k取60。")
        self.assertAlmostEqual(out[0].score, 0.95)
        self.assertEqual(out[0].source, "rerank:api")

    def test_dashscope_native_protocol(self):
        """通义原生接口：路径、请求体和响应结构都与标准协议不同。"""
        import json as _json
        import urllib.request
        from unittest.mock import patch

        from learn_rag.retrieval.rerankers import ApiReranker

        body = {"output": {"results": [{"index": 2, "relevance_score": 0.8}]}}
        sent = {}

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): pass
            def read(self): return _json.dumps(body).encode()

        def fake_urlopen(req, timeout):
            sent["url"], sent["body"] = req.full_url, _json.loads(req.data)
            return FakeResp()

        reranker = ApiReranker(api_key="fake", base_url="https://dashscope.aliyuncs.com/api/v1")
        with patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            out = reranker.rerank("BM25", self._cands(), 1)
        self.assertTrue(sent["url"].endswith("/services/rerank/text-rerank/text-rerank"))
        self.assertEqual(sent["body"]["input"]["query"], "BM25")
        self.assertEqual(out[0].chunk.chunk_id, "c2")

    def test_falls_back_on_failure(self):
        """精排挂了不能让整条链路挂掉 —— 退回召回顺序，降级但可用。"""
        import urllib.request
        from unittest.mock import patch

        from learn_rag.retrieval.rerankers import ApiReranker

        with patch.object(urllib.request, "urlopen", side_effect=RuntimeError("refused")):
            out = ApiReranker(api_key="fake", max_retries=1).rerank("x", self._cands(), 2)
        self.assertEqual([h.chunk.chunk_id for h in out], ["c0", "c1"])


class TestChromaIndex(unittest.TestCase):
    """Chroma 适配器：距离→分数的翻译、元数据过滤、持久化。

    装了 chromadb 才跑 —— 它是可选依赖。
    """

    @classmethod
    def setUpClass(cls):
        try:
            import chromadb  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("未安装 chromadb")

    def _build(self, tmp, space="cosine"):
        from learn_rag.core.types import Chunk
        from learn_rag.embedding.encoders import HashingEncoder
        from learn_rag.store.chroma_index import ChromaVectorIndex

        enc = HashingEncoder(dimension=128)
        chunks = [
            Chunk("c1", "d1", "混合检索使用RRF算法，常数k取60。", metadata={"year": 2025}),
            Chunk("c2", "d2", "BM25的参数b惩罚长文档。", metadata={"year": 2024}),
            Chunk("c3", "d3", "今天天气很好。", metadata={"year": 2023}),
        ]
        idx = ChromaVectorIndex(path=tmp, collection="unit_test_coll", space=space, reset=True)
        idx.add(chunks, enc.encode([c.text for c in chunks]))
        return idx, enc

    def test_search_returns_scores_high_is_better(self):
        """距离必须被翻译成"越大越相关"的分数 —— 这是 ScoredChunk 的契约。"""
        with tempfile.TemporaryDirectory() as tmp:
            idx, enc = self._build(tmp)
            hits = idx.search(enc.encode_one("RRF常数是多少", is_query=True), 3)
            self.assertEqual(hits[0].chunk.chunk_id, "c1")
            self.assertGreater(hits[0].score, hits[-1].score)

    def test_metadata_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            idx, enc = self._build(tmp)
            hits = idx.search(enc.encode_one("参数"), 3, where={"year": 2024})
            self.assertEqual([h.chunk.chunk_id for h in hits], ["c2"])
            hits_in = idx.search(enc.encode_one("参数"), 3, where={"year": [2024, 2025]})
            self.assertEqual(set(h.chunk.chunk_id for h in hits_in), {"c1", "c2"})

    def test_persistence_across_instances(self):
        """重新打开必须能读到数据 —— 这是"离线建库、在线复用"的前提。"""
        from learn_rag.store.chroma_index import ChromaVectorIndex

        with tempfile.TemporaryDirectory() as tmp:
            idx, _ = self._build(tmp)
            self.assertEqual(len(idx), 3)
            reopened = ChromaVectorIndex(path=tmp, collection="unit_test_coll")
            self.assertEqual(len(reopened), 3)

    def test_full_metadata_survives_reopen(self):
        """列表和字典列表要原样取回：早期拼成字符串截断到 500 字，图片一多资产路径就被截断。"""
        from learn_rag.core.types import Chunk
        from learn_rag.embedding.encoders import HashingEncoder
        from learn_rag.store.chroma_index import ChromaVectorIndex

        enc = HashingEncoder(dimension=128)
        assets = [{"asset": f"ab/{'0' * 64}{i}.png", "kind": "image", "caption": f"图 {i}", "page": 2,
                   "bbox": [0.1, 0.2, 0.3, 0.4]} for i in range(12)]
        meta = {"doc_id": "d1", "kinds": ["image", "text"], "assets": assets, "page_start": 2, "parent_id": "d1#0-x"}
        chunk = Chunk("d1#1-y", "d1", "图 1 区域营收", position=1, metadata=meta)
        with tempfile.TemporaryDirectory() as tmp:
            ChromaVectorIndex(path=tmp, collection="unit_meta", reset=True).add([chunk], enc.encode([chunk.text]))
            reopened = ChromaVectorIndex(path=tmp, collection="unit_meta")
            hit = reopened.search(enc.encode_one("营收"), 1, where={"page_start": 2})[0]
        self.assertEqual(hit.chunk.metadata, meta)
        self.assertEqual((hit.chunk.chunk_id, hit.chunk.position), ("d1#1-y", 1))

    def test_old_collections_without_json_field_still_load(self):
        from learn_rag.store.chroma_index import _restore_meta

        self.assertEqual(_restore_meta({"doc_id": "d", "year": 2024}), {"doc_id": "d", "year": 2024})
        self.assertEqual(_restore_meta({"doc_id": "d", "_metadata_json": "{坏的"}), {"doc_id": "d"})

    def test_normalized_vectors_make_spaces_equivalent(self):
        """向量已 L2 归一化时，三种度量给出相同排序。

        这条不成立就说明 encoder 的归一化坏了 —— 比 Chroma 本身更值得警惕。
        """
        orders = []
        for space in ["cosine", "l2", "ip"]:
            with tempfile.TemporaryDirectory() as tmp:
                idx, enc = self._build(tmp, space=space)
                hits = idx.search(enc.encode_one("RRF常数", is_query=True), 3)
                orders.append([h.chunk.chunk_id for h in hits])
        self.assertEqual(orders[0], orders[1])
        self.assertEqual(orders[1], orders[2])


class TestDatasetPlumbing(unittest.TestCase):
    """本轮修掉的三个"静默出错"问题的回归测试。"""

    def test_env_expansion_in_config(self):
        """yaml 里的 ${VAR} 要展开，数字要转成 int。"""
        import os

        from learn_rag.core.config import _expand_env

        os.environ["_T_DIM"] = "1024"
        os.environ["_T_MODEL"] = "bge-m3"
        out = _expand_env({"m": "${_T_MODEL}", "d": "${_T_DIM}", "x": "${_T_NONE:-fallback}"})
        self.assertEqual(out, {"m": "bge-m3", "d": 1024, "x": "fallback"})

    def test_persist_dir_isolated_by_collection(self):
        """不同 collection 的 BM25 必须落在不同目录，否则会互相覆盖。"""
        from learn_rag.cli import _persist_dir

        a = _persist_dir({"index": {"path": "vs", "collection": "scifact"}})
        b = _persist_dir({"index": {"path": "vs", "collection": "fiqa"}})
        self.assertNotEqual(a, b)

    def test_squad_doc_ids_unique_with_duplicate_titles(self):
        """同名文章不能共用 doc_id，否则检索到另一篇也被算作命中。"""
        import json as _json

        from learn_rag.eval.datasets import SquadStyleDataset

        data = {"data": [
            {"id": "A", "title": "同名", "paragraphs": [{"context": "甲", "qas": []}]},
            {"id": "B", "title": "同名", "paragraphs": [{"context": "乙", "qas": []}]},
        ]}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
            _json.dump(data, fh, ensure_ascii=False)
        ids = [d.doc_id for d in SquadStyleDataset(fh.name).corpus()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_caching_encoder_avoids_recompute(self):
        """共享缓存：第二次编码同样的文本不应再调用底层模型。"""
        from learn_rag.embedding.encoders import CachingEncoder, HashingEncoder

        calls = []

        class Counting(HashingEncoder):
            def encode(self, texts, *, is_query=False):
                calls.append(len(texts))
                return super().encode(texts, is_query=is_query)

        enc = CachingEncoder(Counting(dimension=32))
        enc.encode(["a", "b", "c"])
        enc.encode(["a", "b", "c"])
        self.assertEqual(sum(calls), 3)


class TestDotenv(unittest.TestCase):
    """.env 加载：优先级与幂等性。"""

    def setUp(self):
        try:
            import dotenv  # noqa: F401
        except ImportError:
            self.skipTest("未安装 python-dotenv")
        from learn_rag.core import config

        config._ENV_LOADED = False

    def test_loads_file(self):
        import os

        from learn_rag.core.config import load_env

        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("_MINIRAG_T1=from_file\n", encoding="utf-8")
            os.environ.pop("_MINIRAG_T1", None)
            self.assertEqual(load_env(env), str(env))
            self.assertEqual(os.environ["_MINIRAG_T1"], "from_file")

    def test_shell_wins_over_file(self):
        """shell 里已有的变量不能被 .env 覆盖 —— 否则临时 export 做对照实验会失效。"""
        import os

        from learn_rag.core.config import load_env

        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text("_MINIRAG_T2=from_file\n", encoding="utf-8")
            os.environ["_MINIRAG_T2"] = "from_shell"
            load_env(env)
            self.assertEqual(os.environ["_MINIRAG_T2"], "from_shell")


if __name__ == "__main__":
    unittest.main(verbosity=2)