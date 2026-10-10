"""
召回层的测试：编号类关键词检索，以及元数据过滤在各路召回之间保持一致。

过滤一致性在 flat 和 Chroma 两种片段库上各跑一遍：向量那一路和 BM25 那一路必须看到同样的范围，
否则范围之外的文档会经 RRF 融合混进结果。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.core.interfaces import QueryTransformer, Retriever
from learn_rag.core.text import code_tokens
from learn_rag.core.types import Chunk, Document, Element
from learn_rag.embedding.encoders import HashingEncoder
from learn_rag.ingest.structure import StructureChunker
from learn_rag.retrieval.retrievers import BM25Retriever, HybridRetriever, TransformedRetriever, VectorRetriever
from learn_rag.store.indexes import BM25Index, FlatVectorIndex
from learn_rag.store.knowledge_base import KnowledgeBase


class TestCodeTokens(unittest.TestCase):
    def test_keeps_codes_whole(self):
        self.assertEqual(code_tokens("合同编号:XH-2025-0386"), ["xh-2025-0386"])
        self.assertEqual(code_tokens("订单 XH-1001，下单日期 2025-10-08"), ["xh-1001", "2025-10-08"])
        self.assertEqual(code_tokens("型号 GPT4o 与 bge-m3、v2.3.1"), ["gpt4o", "bge-m3", "v2.3.1"])
        self.assertEqual(code_tokens("ＸＨ－２０２５－０３８６"), ["xh-2025-0386"])   # 全角写法归一

    def test_skips_plain_numbers_and_words(self):
        """纯数字、纯单词、没有数字的文件名已由通用切词处理，不重复产出。"""
        self.assertEqual(code_tokens("2025 年营收 4,860 万元，见 config.yaml"), [])


class TestBM25(unittest.TestCase):
    @staticmethod
    def _sample_index() -> BM25Index:
        """用样例文档真实切出来的子块（samples/parsing_results/mineru/*.chunks.jsonl）建 BM25。"""
        children = []
        for path in sorted(Path(__file__).resolve().parents[1].glob("samples/parsing_results/mineru/*.chunks.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["role"] == "child":
                    children.append(Chunk(row["id"], row["metadata"]["doc_id"], row["text"], metadata=row["metadata"]))
        index = BM25Index()
        index.add(children)
        return index

    def test_code_query_finds_the_document_with_that_code(self):
        """
        销售数据反复出现 XH-xxxx 订单号和 2025 年的日期，合同里只出现一次合同编号 XH-2025-0386。
        按 xh / 2025 / 0386 三个碎片打分时三份销售数据都排在合同前面；按编号整体匹配时合同排第一。
        """
        index = self._sample_index()
        self.assertEqual(index.search("XH-2025-0386", 1)[0].chunk.doc_id, "02_扫描件_技术服务合同.pdf")
        self.assertEqual(index.search("合同编号 XH-2025-0386 的服务期限", 1)[0].chunk.doc_id, "02_扫描件_技术服务合同.pdf")

    def test_partial_and_other_codes_still_match(self):
        index = self._sample_index()
        self.assertEqual(index.search("0386", 1)[0].chunk.doc_id, "02_扫描件_技术服务合同.pdf")   # 只记得编号的一部分
        self.assertIn("销售数据", index.search("XH-1002", 1)[0].chunk.doc_id)                        # 订单号

    def test_where_filters_before_taking_top_k(self):
        index = BM25Index()
        index.add([Chunk(f"a#{i}", "a", "报销审批流程 " * 5, metadata={"file_type": "pdf"}) for i in range(5)]
                  + [Chunk(f"b#{i}", "b", "报销审批", metadata={"file_type": "docx"}) for i in range(3)])
        hits = index.search("报销审批流程", 3, where={"file_type": "docx"})
        self.assertEqual(sorted(h.chunk.chunk_id for h in hits), ["b#0", "b#1", "b#2"])
        hits = index.search("报销审批流程", 10, where={"file_type": ["docx", "xlsx"]})
        self.assertEqual(len(hits), 3)
        self.assertEqual(index.search("报销审批流程", 3, where={"missing_key": "x"}), [])


class _SpyRetriever(Retriever):
    """记录收到的过滤条件。"""

    def __init__(self) -> None:
        self.seen: list = []

    def retrieve(self, query, top_k, *, where=None):
        self.seen.append(where)
        return []


class _TwoVariants(QueryTransformer):
    def transform(self, query):
        return [query, query + "？"]


class TestWherePropagation(unittest.TestCase):
    def test_hybrid_and_transformed_pass_where_to_every_channel(self):
        a, b, c = _SpyRetriever(), _SpyRetriever(), _SpyRetriever()
        retriever = HybridRetriever([TransformedRetriever(a, _TwoVariants()), b, c])
        retriever.retrieve("问题", 5, where={"file_type": "pdf"})
        self.assertEqual(a.seen + b.seen + c.seen, [{"file_type": "pdf"}] * 4)

    def test_configured_where_merges_with_query_where(self):
        kb = KnowledgeBase(HashingEncoder(dimension=64), StructureChunker())
        docs = [Document(f"d{i}", "报销审批流程说明。", {"title": f"文档{i}", "dept": dept, "year": year})
                for i, (dept, year) in enumerate([("财务部", 2024), ("财务部", 2025), ("法务部", 2025)])]
        kb.add(docs)
        for retriever in (VectorRetriever(kb, where={"dept": "财务部"}), BM25Retriever(kb, where={"dept": "财务部"})):
            self.assertEqual({h.doc_id for h in retriever.retrieve("报销审批", 5)}, {"d0", "d1"})
            self.assertEqual({h.doc_id for h in retriever.retrieve("报销审批", 5, where={"year": 2025})}, {"d1"})
            self.assertEqual({h.doc_id for h in retriever.retrieve("报销审批", 5, where={"dept": "法务部"})}, {"d2"})


def _has_chroma() -> bool:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return True


class _FilterConsistencyCases:
    """同一个过滤条件下，向量、BM25、混合召回和完整管线返回的都只有范围内的文档。"""

    def open_index(self, tmp: str):
        raise NotImplementedError

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        chunker = StructureChunker(chunk_size=40, parent_size=200)
        self.kb = KnowledgeBase(HashingEncoder(dimension=64), chunker, vector_index=self.open_index(self.tmp))
        docs = []
        for i, file_type in enumerate(["pdf", "docx", "pdf", "xlsx"]):
            elements = [Element("heading", "报销制度", level=1)]
            elements += [Element("text", f"第{j}条：差旅报销需在 30 天内提交审批，编号 BX-{i}{j}。" * 2) for j in range(3)]
            docs.append(Document.from_elements(f"doc{i}.{file_type}", elements, {"title": f"制度{i}", "file_type": file_type}))
        self.kb.add(docs)

    def test_every_channel_respects_the_filter(self):
        where = {"file_type": "pdf"}
        vector, bm25 = VectorRetriever(self.kb), BM25Retriever(self.kb)
        hybrid = HybridRetriever([vector, bm25])
        for retriever in (vector, bm25, hybrid):
            hits = retriever.retrieve("差旅报销审批", 10, where=where)
            self.assertTrue(hits)
            self.assertEqual({h.chunk.metadata["file_type"] for h in hits}, {"pdf"}, type(retriever).__name__)

    def test_pipeline_filters_and_expands_within_range(self):
        from learn_rag.core.config import load_config
        from learn_rag.pipeline.rag import RagPipeline

        pipe = RagPipeline.from_config(load_config("configs/default.yaml"))
        pipe.kb = self.kb
        pipe.retriever = HybridRetriever([VectorRetriever(self.kb), BM25Retriever(self.kb)])
        result = pipe.retrieve_only("差旅报销审批", top_k=5, where={"file_type": ["docx", "xlsx"]})
        self.assertTrue(result.contexts)
        self.assertEqual({c.doc_id for c in result.contexts} - {"doc1.docx", "doc3.xlsx"}, set())
        self.assertTrue(all(c.chunk.metadata["file_type"] in ("docx", "xlsx") for c in result.contexts))


class TestFilterConsistencyFlat(_FilterConsistencyCases, unittest.TestCase):
    def open_index(self, tmp: str):
        return FlatVectorIndex()


@unittest.skipUnless(_has_chroma(), "未安装 chromadb")
class TestFilterConsistencyChroma(_FilterConsistencyCases, unittest.TestCase):
    def open_index(self, tmp: str):
        from learn_rag.store.chroma_index import ChromaVectorIndex

        return ChromaVectorIndex(path=tmp, collection="filter_test", reset=True)


class TestCliWhere(unittest.TestCase):
    def test_parse_where(self):
        from learn_rag.cli import _parse_where

        self.assertEqual(_parse_where('{"file_type": "pdf", "year": [2024, 2025]}'), {"file_type": "pdf", "year": [2024, 2025]})
        self.assertIsNone(_parse_where(None))
        self.assertIsNone(_parse_where("{}"))
        for bad in ("file_type=pdf", '["pdf"]'):
            with self.assertRaises(SystemExit):
                _parse_where(bad)


if __name__ == "__main__":
    unittest.main()
