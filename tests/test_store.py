"""
片段库与知识库的数据完整性测试。

每个场景都在 flat 和 Chroma 两种片段库上跑（Chroma 是可选依赖，没装时跳过）：
按 id 覆盖写入、只存储的父块、删除、落盘后重新打开；知识库多次建库、文档修改、
写入中断、embedding 失败、换模型、片段缺失时，数据都要保持一致或能被发现并自动修复。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.core.types import Chunk, Document, Element
from learn_rag.embedding.encoders import CachingEncoder, HashingEncoder
from learn_rag.ingest.structure import StructureChunker
from learn_rag.store.indexes import FlatVectorIndex
from learn_rag.store.knowledge_base import KnowledgeBase


def _has_chroma() -> bool:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return False
    return True


def _doc(i: int, note: str = "") -> Document:
    """一篇两章的文档：每章的段落超过子块大小，会生成父块。"""
    elements = []
    for chapter in range(2):
        elements.append(Element("heading", f"第{chapter + 1}章", level=1))
        elements += [Element("text", f"文档{i}第{chapter + 1}章第{j}段讲述编号{i}{chapter}{j}的事项{note}。" * 3)
                     for j in range(3)]
    return Document.from_elements(f"doc{i}.pdf", elements, {"title": f"文档{i}"})


class _CountingEncoder(HashingEncoder):
    """记录 encode 被调用了几次、编码了多少段文本；fail=True 时模拟服务故障。"""

    def __init__(self, **kwargs) -> None:
        super().__init__(dimension=64, **kwargs)
        self.texts = 0
        self.fail = False

    def encode(self, texts, *, is_query=False):
        if self.fail and not is_query:
            raise RuntimeError("embedding 服务不可用")
        if not is_query:
            self.texts += len(texts)
        return super().encode(texts, is_query=is_query)


class _StoreCases:
    """片段库的契约，flat 和 Chroma 各跑一遍。子类提供 open_index(tmp, reset)。"""

    def open_index(self, tmp: str, reset: bool = False):
        raise NotImplementedError

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _reopen(self, index):
        """模拟换一个进程重新打开：flat 需要先落盘再读，Chroma 本身持久化。"""
        index.save(self.tmp + "/saved")
        fresh = self.open_index(self.tmp)
        fresh.load(self.tmp + "/saved")
        return fresh

    def test_upsert_replaces_by_id(self):
        index = self.open_index(self.tmp, reset=True)
        enc = HashingEncoder(dimension=32)
        index.add([Chunk("d#0", "d", "旧内容"), Chunk("d#1", "d", "另一段")], enc.encode(["旧内容", "另一段"]))
        index.add([Chunk("d#0", "d", "新内容")], enc.encode(["新内容"]))
        index = self._reopen(index)
        self.assertEqual(len(index), 2)
        texts = {c.chunk_id: c.text for c in index.chunks()}
        self.assertEqual(texts, {"d#0": "新内容", "d#1": "另一段"})

    def test_duplicate_ids_in_one_batch_keep_last(self):
        index = self.open_index(self.tmp, reset=True)
        enc = HashingEncoder(dimension=32)
        index.add([Chunk("d#0", "d", "第一次"), Chunk("d#0", "d", "第二次")], enc.encode(["第一次", "第二次"]))
        self.assertEqual([c.text for c in index.chunks()], ["第二次"])

    def test_stored_chunks_are_kept_but_not_searchable(self):
        index = self.open_index(self.tmp, reset=True)
        enc = HashingEncoder(dimension=32)
        index.add([Chunk("d#1", "d", "子块正文", metadata={"parent_id": "d#0"})], enc.encode(["子块正文"]))
        index.add([Chunk("d#0", "d", "父块正文", metadata={"kinds": ["text"], "page_start": 1})])
        index = self._reopen(index)
        self.assertEqual(len(index), 1)
        hits = index.search(enc.encode_one("父块正文"), 5)
        self.assertEqual([h.chunk.chunk_id for h in hits], ["d#1"])
        stored = {c.chunk_id: c for c in index.chunks()}
        self.assertEqual(stored["d#0"].metadata, {"kinds": ["text"], "page_start": 1})

    def test_delete_removes_both_kinds(self):
        index = self.open_index(self.tmp, reset=True)
        enc = HashingEncoder(dimension=32)
        index.add([Chunk("d#1", "d", "子块一"), Chunk("d#2", "d", "子块二")], enc.encode(["子块一", "子块二"]))
        index.add([Chunk("d#0", "d", "父块")])
        index.delete(["d#0", "d#1", "不存在的id"])
        index = self._reopen(index)
        self.assertEqual([c.chunk_id for c in index.chunks()], ["d#2"])
        self.assertEqual(len(index), 1)

    def test_rejects_bad_vectors(self):
        index = self.open_index(self.tmp, reset=True)
        index.add([Chunk("d#0", "d", "a")], [[1.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "维度"):
            index.add([Chunk("d#1", "d", "b")], [[1.0, 0.0, 0.0]])
        with self.assertRaisesRegex(ValueError, "NaN"):
            index.add([Chunk("d#2", "d", "c")], [[float("nan"), 0.0]])
        self.assertEqual(len(index), 1)


class TestFlatStore(_StoreCases, unittest.TestCase):
    def open_index(self, tmp: str, reset: bool = False):
        return FlatVectorIndex()

    def test_detects_truncated_files(self):
        index = FlatVectorIndex()
        index.add([Chunk("d#0", "d", "a"), Chunk("d#1", "d", "b")], [[1.0, 0.0], [0.0, 1.0]])
        index.save(self.tmp)
        lines = (Path(self.tmp) / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        (Path(self.tmp) / "chunks.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "对不上"):
            FlatVectorIndex().load(self.tmp)


@unittest.skipUnless(_has_chroma(), "未安装 chromadb")
class TestChromaStore(_StoreCases, unittest.TestCase):
    def open_index(self, tmp: str, reset: bool = False):
        from learn_rag.store.chroma_index import ChromaVectorIndex

        return ChromaVectorIndex(path=tmp + "/chroma", collection="store_test", reset=reset)

    def test_reading_all_chunks_pages_through_both_collections(self):
        index = self.open_index(self.tmp, reset=True)
        enc = HashingEncoder(dimension=32)
        children = [Chunk(f"d#{i}", "d", f"子块{i}") for i in range(5)]
        index.add(children, enc.encode([c.text for c in children]))
        index.add([Chunk(f"d#p{i}", "d", f"父块{i}") for i in range(3)])
        with patch("learn_rag.store.chroma_index._PAGE", 2):
            ids = sorted(c.chunk_id for c in index.chunks())
        self.assertEqual(len(ids), 8)

    def test_reset_clears_stored_chunks_too(self):
        index = self.open_index(self.tmp, reset=True)
        index.add([Chunk("d#0", "d", "父块")])
        self.assertEqual(list(self.open_index(self.tmp, reset=True).chunks()), [])


class _KnowledgeBaseCases:
    """知识库的完整性契约，flat 和 Chroma 各跑一遍。子类提供 open_index(tmp, reset)。"""

    chunker = StructureChunker(chunk_size=40, parent_size=200)

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.encoder = _CountingEncoder()

    def open_index(self, tmp: str, reset: bool = False):
        raise NotImplementedError

    def _kb(self, encoder=None) -> KnowledgeBase:
        """打开知识库：每次都是新的对象，模拟另起一个进程（flat 从落盘目录读）。"""
        kb = KnowledgeBase(encoder or self.encoder, self.chunker, vector_index=self.open_index(self.tmp))
        if (Path(self.tmp) / "kb" / "meta.json").exists():
            kb.load(self.tmp + "/kb")
        return kb

    def _build(self, docs, encoder=None) -> dict:
        kb = self._kb(encoder)
        stats = kb.add(docs)
        kb.save(self.tmp + "/kb")
        return stats

    def test_separate_builds_stay_consistent(self):
        self._build([_doc(1)])
        self._build([_doc(2)])
        kb = self._kb()
        report = kb.verify()
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["documents"], 2)
        self.assertEqual(report["bm25_chunks"], report["chunks"])
        self.assertTrue(kb.bm25_index.search("编号100", 1))      # 第一次建库的文档在 BM25 里
        hit = kb.bm25_index.search("编号210", 1)[0]               # 第二次建库的文档也在，且能展开成父块
        self.assertNotEqual(kb.expand([hit])[0].chunk.chunk_id, hit.chunk.chunk_id)

    def test_unchanged_documents_are_skipped_without_embedding(self):
        self._build([_doc(1), _doc(2)])
        embedded = self.encoder.texts
        stats = self._build([_doc(1), _doc(2)])
        self.assertEqual((stats["skipped"], stats["documents"]), (2, 0))
        self.assertEqual(self.encoder.texts, embedded)

    def test_modified_document_replaces_old_version(self):
        self._build([_doc(1), _doc(2)])
        stats = self._build([_doc(2, note="（已修订）")])
        self.assertEqual(stats["documents"], 1)
        kb = self._kb()
        self.assertTrue(kb.verify()["ok"])
        chunks = [c for c in kb.vector_index.chunks() if c.doc_id == "doc2.pdf"]
        self.assertEqual(len(chunks), len(self.chunker.split(_doc(2, note="（已修订）"))))
        self.assertTrue(all("已修订" in c.text for c in chunks))
        self.assertEqual(kb.verify()["documents"], 2)

    def _interrupt_update(self) -> None:
        """第一版入库后更新文档：新版本已写入、旧版本还没删时进程被杀。"""
        self._build([_doc(1)])
        kb = self._kb()
        with patch.object(kb.vector_index, "delete", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                kb.add([_doc(1, note="（新版）")])

    def test_missing_chunk_is_detected_and_repaired(self):
        self._build([_doc(1)])
        kb = self._kb()
        child = next(c for c in kb.vector_index.chunks() if "parent_id" in c.metadata)
        kb.vector_index.delete([child.chunk_id])
        kb.save(self.tmp + "/kb")
        report = self._kb().verify()
        self.assertFalse(report["ok"])
        self.assertTrue(any("实际" in p for p in report["problems"]))
        stats = self._build([_doc(1)])
        self.assertEqual(stats["documents"], 1)                 # 不再被当成"未变化"跳过
        self.assertTrue(self._kb().verify()["ok"])

    def test_embedding_failure_keeps_old_version(self):
        self._build([_doc(1)])
        before = sorted(c.chunk_id for c in self._kb().vector_index.chunks())
        self.encoder.fail = True
        stats = self._build([_doc(1, note="（新版）"), _doc(2)])
        self.assertEqual((stats["failed"], stats["documents"]), (2, 0))
        kb = self._kb()
        self.assertEqual(sorted(c.chunk_id for c in kb.vector_index.chunks()), before)
        self.assertTrue(kb.verify()["ok"])
        self.encoder.fail = False
        stats = self._build([_doc(1, note="（新版）"), _doc(2)])  # 服务恢复后重新运行即可续写
        self.assertEqual(stats["documents"], 2)
        self.assertTrue(self._kb().verify()["ok"])

    def test_refuses_to_mix_embedding_models(self):
        self._build([_doc(1)])
        other = HashingEncoder(dimension=64, ngram=2)            # 维度相同、模型不同：向量库自己发现不了
        kb = self._kb(encoder=other)
        with self.assertRaisesRegex(ValueError, "两个模型的向量不可比"):
            kb.add([_doc(2)])
        report = kb.verify()
        self.assertFalse(report["ok"])
        self.assertIn("其他模型", report["problems"][0])

    def test_sampled_vectors_are_retrievable(self):
        self._build([_doc(1), _doc(2)])
        report = self._kb().verify(sample=100)
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["sampled"], 100)

    def test_detects_vector_stored_for_wrong_text(self):
        """结构完全正常、只有某个子块的向量存成了别的正文的向量：只有抽样核对能发现。"""
        self._build([_doc(1)])
        kb = self._kb()
        a, b = [c for c in kb.vector_index.chunks() if "embedding_model" in c.metadata][:2]
        kb.vector_index.add([a], self.encoder.encode([b.text]))
        kb.save(self.tmp + "/kb")
        self.assertTrue(self._kb().verify()["ok"])
        report = self._kb().verify(sample=1000)
        self.assertFalse(report["ok"])
        self.assertTrue(any(a.chunk_id in p for p in report["problems"]))

    def test_detects_encoder_change_hidden_behind_same_signature(self):
        self._build([_doc(1)])

        class Drifted(HashingEncoder):            # 模型身份没变，输出却变了（比如服务端悄悄换了模型）
            def encode(self, texts, *, is_query=False):
                return [list(reversed(v)) for v in super().encode(texts, is_query=is_query)]

        report = self._kb(encoder=Drifted(dimension=64)).verify(sample=3)
        self.assertFalse(report["ok"])
        self.assertTrue(any("相似度只有" in p or "前 5 名里没有" in p for p in report["problems"]))

    def test_metadata_survives_storage(self):
        self._build([_doc(1)])
        expected = {c.chunk_id: c for c in self.chunker.split(_doc(1))}
        stored = {c.chunk_id: c for c in self._kb().vector_index.chunks()}
        self.assertEqual(set(stored), set(expected))
        for chunk_id, chunk in expected.items():
            meta = stored[chunk_id].metadata
            self.assertEqual({k: meta[k] for k in chunk.metadata}, chunk.metadata)
            self.assertEqual(stored[chunk_id].text, chunk.text)
            self.assertEqual(meta["doc_chunks"], len(expected))
            self.assertEqual("embedding_model" in meta, "parent_id" in chunk.metadata)


class TestFlatKnowledgeBase(_KnowledgeBaseCases, unittest.TestCase):
    def open_index(self, tmp: str, reset: bool = False):
        return FlatVectorIndex()

    def test_interrupted_update_leaves_saved_index_untouched(self):
        """flat 只在 save 时一次写出文件：中途被杀，落盘的仍是完整的旧版本。"""
        self._interrupt_update()
        kb = self._kb()
        self.assertTrue(kb.verify()["ok"])
        self.assertFalse(any("新版" in c.text for c in kb.vector_index.chunks()))


@unittest.skipUnless(_has_chroma(), "未安装 chromadb")
class TestChromaKnowledgeBase(_KnowledgeBaseCases, unittest.TestCase):
    def open_index(self, tmp: str, reset: bool = False):
        from learn_rag.store.chroma_index import ChromaVectorIndex

        return ChromaVectorIndex(path=tmp + "/chroma", collection="kb_integrity", reset=reset)

    def test_interrupted_update_is_detected_and_repaired(self):
        """Chroma 写入即落盘：中途被杀会留下新旧两个版本，核对能发现，重新 build 会清理。"""
        self._interrupt_update()
        report = self._kb().verify()
        self.assertFalse(report["ok"])
        self.assertIn("混有 2 个版本", report["problems"][0])
        stats = self._build([_doc(1, note="（新版）")])
        self.assertEqual(stats["documents"], 1)
        kb = self._kb()
        self.assertTrue(kb.verify()["ok"])
        self.assertTrue(all("新版" in c.text for c in kb.vector_index.chunks()))


@unittest.skipUnless(_has_chroma(), "未安装 chromadb")
class TestLegacyChromaCollection(unittest.TestCase):
    def test_old_collection_is_flagged_and_repaired_by_rebuild(self):
        """旧版本程序建的集合：片段没有指纹，父块不在 Chroma 里。核对时指出来，再 build 一次即补齐。"""
        import chromadb
        from learn_rag.store.chroma_index import ChromaVectorIndex

        chunker = StructureChunker(chunk_size=40, parent_size=200)
        with tempfile.TemporaryDirectory() as tmp:
            old = chromadb.PersistentClient(path=tmp).get_or_create_collection("legacy_kb")
            children = [c for c in chunker.split(_doc(1)) if "parent_id" in c.metadata]
            old.upsert(ids=[c.chunk_id for c in children], embeddings=HashingEncoder(dimension=64).encode([c.text for c in children]),
                       documents=[c.text for c in children], metadatas=[{"doc_id": c.doc_id, "position": c.position} for c in children])

            def kb():
                return KnowledgeBase(HashingEncoder(dimension=64), chunker,
                                     vector_index=ChromaVectorIndex(path=tmp, collection="legacy_kb"))

            report = kb().verify()
            self.assertEqual(len(report["problems"]), 1)
            self.assertIn("旧版本程序写入", report["problems"][0])
            self.assertEqual(kb().add([_doc(1)])["documents"], 1)
            self.assertTrue(kb().verify()["ok"])


class TestEncoderIdentity(unittest.TestCase):
    def test_signatures_distinguish_models(self):
        self.assertNotEqual(HashingEncoder(dimension=64).signature(), HashingEncoder(dimension=64, ngram=2).signature())
        self.assertEqual(CachingEncoder(HashingEncoder()).signature(), HashingEncoder().signature())

    def test_cache_is_keyed_by_model(self):
        """同一个缓存文件给两个模型用时，不能把 A 模型的向量返回给 B。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = tmp + "/cache.json"
            first = CachingEncoder(HashingEncoder(dimension=64), cache_path=path)
            vec_a = first.encode(["同一段文本"])[0]
            first.flush()
            second = CachingEncoder(HashingEncoder(dimension=64, ngram=2), cache_path=path)
            self.assertNotEqual(second.encode(["同一段文本"])[0], vec_a)


if __name__ == "__main__":
    unittest.main()
