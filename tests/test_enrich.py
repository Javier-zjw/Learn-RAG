"""
第 4 阶段（建库增强）的测试：背景和表格描述进索引但不改正文、假设问题和摘要作为视图命中后换回所属片段、
结果缓存、失败不中断建库并在下次重试、存进 Chroma 后完整读回。大模型一律用假的实现。
"""

from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.core.interfaces import LLM
from learn_rag.core.types import Document
from learn_rag.embedding.encoders import HashingEncoder
from learn_rag.ingest.enrich import LLMEnricher
from learn_rag.ingest.structure import StructureChunker
from learn_rag.parsing.markdown import markdown_to_elements
from learn_rag.store.knowledge_base import KnowledgeBase

_MANUAL = """# 运维手册

## 数据库备份

每晚两点对业务库执行全量备份，备份写入对象存储。备份失败时值班人员会收到短信告警，需要在一小时内处理。

## 保留策略

保留期为三十天，到期后自动删除，删除前不再另行通知。

## 资源配额

| 环境 | CPU | 内存 |
| --- | --- | --- |
| 生产 | 32 核 | 128 GB |
| 测试 | 8 核 | 32 GB |
"""


class _FakeLLM(LLM):
    """
    按提示词里的要求输出：背景写明片段属于哪一节（"保留策略"那一节补上"数据库备份文件"），
    表格给出描述，假设问题按要求的个数生成。fail 里的片段关键词命中时抛异常，模拟服务失败。
    """

    model = "fake-llm"

    def __init__(self, fail: str = "") -> None:
        self.calls = 0
        self.fail = fail
        self._lock = threading.Lock()

    def chat(self, messages, **options):
        with self._lock:
            self.calls += 1
        prompt = messages[-1]["content"]
        if prompt.startswith("用两三句话概括"):
            return "这一节规定了数据库备份的时间、保留期限和资源配额。"
        chunk = prompt.split("片段：\n", 1)[1].split("\n\n要求", 1)[0]
        if self.fail and self.fail in chunk:
            raise TimeoutError("服务超时")
        out = {"context": "运维手册中关于数据库备份文件保留期限的规定" if "保留期" in chunk else "运维手册的一部分"}
        if '"description"' in prompt:
            out["description"] = "各环境的 CPU 和内存配额，生产环境 32 核 128 GB"
        match = re.search(r'"questions"：(\d+) 个', prompt)
        if match:
            out["questions"] = [f"{chunk[-12:]}的问题{i}" for i in range(int(match.group(1)))]
        return "```json\n" + json.dumps(out, ensure_ascii=False) + "\n```"


def _document(text: str = _MANUAL) -> Document:
    return Document.from_elements("运维手册.md", markdown_to_elements(text), {"title": "运维手册"})


class TestEnrichment(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _kb(self, llm: LLM, **options) -> KnowledgeBase:
        enricher = LLMEnricher(llm, cache_dir=str(Path(self.tmp) / "cache"), workers=2, **options)
        return KnowledgeBase(HashingEncoder(dimension=256), StructureChunker(chunk_size=60, parent_size=0),
                             enricher=enricher)

    def test_context_and_table_description_are_indexed_not_shown(self):
        kb = self._kb(_FakeLLM())
        kb.add([_document()])
        chunks = {c.metadata["section"].split(" > ")[-1]: c for c in kb.chunks_of("运维手册.md")}
        retention, quota = chunks["保留策略"], chunks["资源配额"]
        self.assertNotIn("数据库备份文件", retention.text)                    # 正文原样
        self.assertIn("数据库备份文件", retention.index_text)
        self.assertEqual(quota.metadata["enrich"]["description"], "各环境的 CPU 和内存配额，生产环境 32 核 128 GB")
        self.assertNotIn("description", retention.metadata["enrich"])          # 没有表格的块不写描述
        hits = kb.bm25_index.search("数据库备份文件的保留期限", 1)               # 只有背景里有"期限"
        self.assertEqual(hits[0].chunk.chunk_id, retention.chunk_id)
        self.assertTrue(kb.verify(sample=3)["ok"])

    def test_question_views_resolve_to_their_chunk(self):
        kb = self._kb(_FakeLLM(), questions=2)
        stats = kb.add([_document()])
        children = kb.chunks_of("运维手册.md")
        self.assertEqual(stats["views"], 2 * len(children))
        self.assertEqual(kb.stats()["chunks"], len(children))                  # 视图不算子块
        self.assertEqual(len(kb.bm25_index), len(children))                    # 视图不进 BM25
        target = next(c for c in children if "保留期" in c.text)
        question = target.metadata["enrich"]["questions"][1]
        hits = kb.vector_index.search(kb.encoder.encode_one(question, is_query=True), 3)
        self.assertEqual(hits[0].chunk.metadata["view_of"], target.chunk_id)
        self.assertEqual(hits[0].chunk.text, target.text)                      # 视图存的是所属片段的正文
        expanded = kb.expand(hits)
        self.assertEqual(expanded[0].chunk.chunk_id, target.chunk_id)
        self.assertEqual(expanded[0].debug["views"][0], question)                # 最先命中的问题排在前面
        self.assertEqual(len({h.chunk.chunk_id for h in expanded}), len(expanded))   # 同一片段只留一份
        report = kb.verify(sample=20)
        self.assertTrue(report["ok"], report["problems"])
        self.assertEqual(report["views"], 2 * len(children))

    def test_summaries_point_to_parents(self):
        enricher = LLMEnricher(_FakeLLM(), context=False, describe=False, summaries=True,
                               cache_dir=str(Path(self.tmp) / "cache"))
        kb = KnowledgeBase(HashingEncoder(dimension=256), StructureChunker(chunk_size=40, parent_size=400),
                           enricher=enricher)
        kb.add([_document()])
        parents = [c for c in kb.chunks_of("运维手册.md") if c.chunk_id in kb._parents]
        self.assertTrue(parents)
        self.assertTrue(all(p.metadata["enrich"]["summary"] for p in parents))
        summary = parents[0].metadata["enrich"]["summary"]
        hits = kb.vector_index.search(kb.encoder.encode_one(summary, is_query=True), 1)
        self.assertEqual(kb.expand(hits)[0].chunk.chunk_id, parents[0].chunk_id)
        self.assertTrue(kb.verify()["ok"])

    def test_results_are_cached_and_unchanged_documents_skipped(self):
        llm = _FakeLLM()
        kb = self._kb(llm, questions=1)
        kb.add([_document()])
        first = llm.calls
        self.assertEqual(kb.add([_document()])["skipped"], 1)                 # 指纹没变：不切分、不调用
        fresh = self._kb(llm, questions=1)                                     # 换一个空库：全部读缓存
        fresh.add([_document()])
        self.assertEqual(llm.calls, first)
        more = self._kb(llm, questions=2)                                      # 参数变了：重新调用
        more.add([_document()])
        self.assertGreater(llm.calls, first)

    def test_failures_are_recorded_and_retried(self):
        llm = _FakeLLM(fail="保留期")
        kb = self._kb(llm)
        stats = kb.add([_document()])
        self.assertEqual(stats["documents"], 1)                               # 建库照常完成
        failed = [c for c in kb.chunks_of("运维手册.md") if c.metadata["enrich"].get("failed")]
        self.assertEqual(len(failed), 1)
        self.assertNotIn("context", failed[0].metadata["enrich"])
        report = kb.verify()
        self.assertFalse(report["ok"])
        self.assertIn("增强没有完成", report["problems"][0])

        llm.fail = ""
        calls = llm.calls
        self.assertEqual(kb.add([_document()])["documents"], 1)               # 有失败的文档不算"未变化"
        self.assertEqual(llm.calls, calls + 1)                                 # 只重试失败的那一次
        self.assertTrue(kb.verify()["ok"])

    def test_images_are_described_by_the_vision_model(self):
        assets = Path(self.tmp) / "assets"
        (assets / "ab").mkdir(parents=True)
        (assets / "ab" / "chart.png").write_bytes(b"\x89PNG fake")

        class _Vision(LLM):
            model = "fake-vlm"
            seen: list = []

            def chat(self, messages, **options):
                _Vision.seen.append(messages[0]["content"][0]["image_url"]["url"][:22])
                return "柱状图，展示近四个季度的备份成功率，均在 99% 以上"

        text = "# 运维手册\n\n## 监控\n\n备份成功率如下图所示。\n\n![备份成功率](ab/chart.png)\n"
        elements = markdown_to_elements(text)
        for element in elements:
            if element.kind == "image":
                element.extra["asset"] = "ab/chart.png"
        enricher = LLMEnricher(_FakeLLM(), vlm=_Vision(), cache_dir=str(Path(self.tmp) / "cache"),
                               assets_dirs=[str(assets)])
        kb = KnowledgeBase(HashingEncoder(dimension=256), StructureChunker(chunk_size=200, parent_size=0),
                           enricher=enricher)
        kb.add([Document.from_elements("运维手册.md", elements, {"title": "运维手册"})])
        chunk = next(c for c in kb.chunks_of("运维手册.md") if c.metadata.get("assets"))
        self.assertIn("季度的备份成功率", chunk.metadata["enrich"]["description"])
        self.assertIn("备份成功率", chunk.index_text)
        self.assertEqual(_Vision.seen, ["data:image/png;base64,"])
        self.assertIn("vlm=fake-vlm", enricher.signature())

    def test_enrichment_survives_chroma(self):
        try:
            from learn_rag.store.chroma_index import ChromaVectorIndex
        except ImportError:
            self.skipTest("未安装 chromadb")
        path = str(Path(self.tmp) / "chroma")

        def kb() -> KnowledgeBase:
            enricher = LLMEnricher(_FakeLLM(), questions=1, cache_dir=str(Path(self.tmp) / "cache"))
            return KnowledgeBase(HashingEncoder(dimension=256), StructureChunker(chunk_size=60, parent_size=0),
                                 vector_index=ChromaVectorIndex(path=path, collection="enrich"), enricher=enricher)

        kb().add([_document()])
        reopened = kb()                                                         # 换个进程：从 Chroma 读回
        retention = next(c for c in reopened.chunks_of("运维手册.md") if "保留期" in c.text)
        self.assertIn("数据库备份文件", retention.index_text)
        self.assertEqual(reopened.bm25_index.search("保留期限", 1)[0].chunk.chunk_id, retention.chunk_id)
        self.assertTrue(reopened.verify(sample=10)["ok"])
        self.assertEqual(reopened.add([_document()])["skipped"], 1)


class TestBuildFromConfig(unittest.TestCase):
    def test_enrich_requires_a_real_llm(self):
        from learn_rag.core.config import load_config
        from learn_rag.pipeline.rag import RagPipeline

        cfg = load_config("configs/default.yaml")
        cfg["enrich"] = {"context": True}
        with self.assertRaises(ValueError) as ctx:
            RagPipeline.from_config(cfg)
        self.assertIn("LLM_API_KEY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
