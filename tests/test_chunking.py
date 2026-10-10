"""
结构切分与父子分块的测试。

重点测切分的"契约"：块不超过预算、父块不跨章、子块不跨小节、表格每块都带表题和表头、
公式和图片不与引导句分开、知识库只索引子块并能把命中展开成父块。
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.core.tables import flatten_table, table_to_markdown
from learn_rag.core.text import count_tokens
from learn_rag.core.types import Chunk, Document, Element, ScoredChunk
from learn_rag.embedding.encoders import HashingEncoder
from learn_rag.ingest.pieces import split_element
from learn_rag.ingest.structure import StructureChunker
from learn_rag.store.knowledge_base import KnowledgeBase

_MERGED = (
    '<table><tr><td rowspan="2">区域</td><td colspan="2">上半年</td><td rowspan="2">合计</td></tr>'
    "<tr><td>Q1</td><td>Q2</td></tr>"
    '<tr><td rowspan="2">华东</td><td>320</td><td>360</td><td>680</td></tr>'
    "<tr><td>330</td><td>370</td><td>700</td></tr></table>"
)


def _doc(elements: list[Element]) -> Document:
    return Document.from_elements("manual.pdf", elements, {"title": "员工手册"})


def _body(chunk) -> str:
    """去掉块首的标题路径行。"""
    return chunk.text.split("\n", 1)[1]


class TestCountTokens(unittest.TestCase):
    def test_chinese_and_english_are_comparable(self):
        self.assertEqual(count_tokens("全年营收"), 4)
        self.assertEqual(count_tokens("revenue grew fast"), 4)
        self.assertEqual(count_tokens("4860"), 2)
        self.assertEqual(count_tokens(""), 0)


class TestFlattenTable(unittest.TestCase):
    def test_merged_header_and_body_are_expanded(self):
        lines = flatten_table(_MERGED).splitlines()
        self.assertEqual(lines[0], "| 区域 | 上半年 Q1 | 上半年 Q2 | 合计 |")
        # 跨行的"华东"复制到它覆盖的每一行，切开后每行仍然知道是哪个区域
        self.assertEqual(lines[2:], ["| 华东 | 320 | 360 | 680 |", "| 华东 | 330 | 370 | 700 |"])

    def test_markdown_is_unchanged(self):
        table = table_to_markdown([["a", "b"], ["1", "2"]])
        self.assertEqual(flatten_table(table), table)


class TestSplitElement(unittest.TestCase):
    def test_paragraph_split_at_sentences(self):
        text = "".join(f"第{i}句话讲一件事。" for i in range(40))
        pieces = split_element(Element("text", text, page=3), 30)
        self.assertGreater(len(pieces), 5)
        self.assertTrue(all(p.text.endswith("。") for p in pieces))
        self.assertTrue(all(count_tokens(p.text) <= 30 for p in pieces))
        self.assertEqual("".join(p.text for p in pieces), text)
        self.assertEqual({p.page for p in pieces}, {3})

    def test_sentence_without_punctuation_is_hard_cut(self):
        pieces = split_element(Element("text", "字" * 100), 30)
        self.assertEqual("".join(p.text for p in pieces), "字" * 100)
        self.assertTrue(all(count_tokens(p.text) <= 30 for p in pieces))

    def test_long_sentence_split_at_commas(self):
        """只有逗号没有句号的长句在逗号处断开，不按字数硬切；数字里的英文逗号不算断点。"""
        text = "，".join(f"第{i}项指标为4,860万元" for i in range(20)) + "。"
        pieces = split_element(Element("text", text), 40)
        self.assertGreater(len(pieces), 2)
        self.assertTrue(all(p.text.endswith(("，", "。")) for p in pieces))
        self.assertTrue(all(count_tokens(p.text) <= 40 for p in pieces))
        self.assertEqual("".join(p.text for p in pieces), text)

    def test_sentence_overlap(self):
        sentences = [f"第{i}句话讲一件事。" for i in range(40)]
        text = "".join(sentences)
        plain = split_element(Element("text", text), 30)
        pieces = split_element(Element("text", text), 30, overlap=1)
        self.assertGreater(len(pieces), len(plain))
        self.assertTrue(all(count_tokens(p.text) <= 30 for p in pieces))
        for prev, nxt in zip(pieces, pieces[1:]):
            last = prev.text[prev.text.rindex("。", 0, len(prev.text) - 1) + 1:]
            self.assertTrue(nxt.text.startswith(last))
        # 去掉重叠后，内容和不重叠时完全一致
        self.assertEqual("".join(p.text for p in plain), text)

    def test_overlap_skips_blank_lines_and_tables(self):
        text = "\n\n".join(f"第{i}句话讲一件事。" for i in range(40))
        pieces = split_element(Element("text", text), 30, overlap=1)
        self.assertTrue(all(p.text.startswith("第") for p in pieces))
        rows = [["型号", "价格"]] + [[f"M{i}", str(i)] for i in range(30)]
        table = Element("table", table_to_markdown(rows))
        self.assertEqual(split_element(table, 60, overlap=2), split_element(table, 60))

    def test_markdown_table_keeps_caption_header_and_note(self):
        rows = [["型号", "价格"]] + [[f"M{i}", str(i)] for i in range(30)]
        text = "表 1 报价单\n" + table_to_markdown(rows) + "\n注：价格含税。"
        pieces = split_element(Element("table", text), 60)
        self.assertGreater(len(pieces), 2)
        for p in pieces:
            self.assertEqual(p.text.splitlines()[:3], ["表 1 报价单", "| 型号 | 价格 |", "| --- | --- |"])
        self.assertTrue(pieces[-1].text.endswith("注：价格含税。"))
        self.assertEqual(sum(p.text.count("| M") for p in pieces), 30)
        self.assertEqual([p.extra["part"] for p in pieces], list(range(1, len(pieces) + 1)))

    def test_merged_html_table_is_flattened_before_split(self):
        rows = "".join(f"<tr><td>R{i}</td><td>{i}</td><td>{i}</td><td>{i}</td></tr>" for i in range(40))
        html = _MERGED.replace("</table>", rows + "</table>")
        pieces = split_element(Element("table", "表 2 区域营收\n" + html), 80)
        self.assertGreater(len(pieces), 1)
        for p in pieces:
            self.assertEqual(p.text.splitlines()[:2], ["表 2 区域营收", "| 区域 | 上半年 Q1 | 上半年 Q2 | 合计 |"])

    def test_small_html_table_is_kept(self):
        self.assertEqual(split_element(Element("table", _MERGED), 500)[0].text, _MERGED)

    def test_code_split_by_lines_and_formula_kept(self):
        code = "\n".join(f"print('line number {i}')" for i in range(50))
        pieces = split_element(Element("code", code), 40)
        self.assertGreater(len(pieces), 1)
        self.assertEqual("\n".join(p.text for p in pieces), code)
        formula = Element("formula", r"\sum_{i=1}^{n} x_i " * 50)
        self.assertEqual(split_element(formula, 10), [formula])


class TestStructureChunker(unittest.TestCase):
    def test_sections_paths_and_pages(self):
        doc = _doc([
            Element("heading", "报销", level=1, page=1),
            Element("text", "差旅标准如下。", page=1),
            Element("heading", "住宿", level=2, page=2),
            Element("text", "一线城市 600 元。", page=2, extra={"page_end": 3}),
            Element("heading", "餐饮", level=2, page=3),   # 只有标题没有内容，不产生块
            Element("heading", "补贴", level=2, page=3),
            Element("text", "每天 120 元。", page=4),
        ])
        chunks = StructureChunker().split(doc)
        self.assertEqual([c.text.splitlines()[0] for c in chunks],
                         ["[员工手册 > 报销]", "[员工手册 > 报销 > 住宿]", "[员工手册 > 报销 > 补贴]"])
        self.assertEqual((chunks[1].metadata["page_start"], chunks[1].metadata["page_end"]), (2, 3))
        self.assertEqual(chunks[1].metadata["section"], "报销 > 住宿")
        # 每章只有一个子块：父块和子块内容相同，不生成
        self.assertFalse(any("parent_id" in c.metadata for c in chunks))
        self.assertEqual(len({c.chunk_id for c in chunks}), 3)

    def test_children_point_to_parent_of_their_chapter(self):
        para = "这一段介绍安装前的准备工作和注意事项。" * 3
        doc = _doc([
            Element("heading", "产品手册", level=1),
            Element("heading", "1 安装", level=2),
            Element("heading", "1.1 环境", level=3),
            Element("text", para),
            Element("text", para),
            Element("heading", "1.2 步骤", level=3),
            Element("text", para),
            Element("heading", "2 计费", level=2),
            Element("text", "基础版每月 999 元。"),
        ])
        chunks = StructureChunker(chunk_size=80, parent_size=400).split(doc)
        parents = [c for c in chunks if any(x.metadata.get("parent_id") == c.chunk_id for x in chunks)]
        self.assertEqual(len(parents), 1)
        parent = parents[0]
        children = [c for c in chunks if c.metadata.get("parent_id") == parent.chunk_id]
        self.assertEqual(len(children), 3)
        # 父块：块首是两个小节的公共路径，小节标题留在正文里
        self.assertTrue(parent.text.startswith("[员工手册 > 产品手册 > 1 安装]\n### 1.1 环境\n\n"))
        self.assertIn("### 1.2 步骤", parent.text)
        # 子块不跨小节，块首是完整路径
        self.assertEqual(children[-1].text.splitlines()[0], "[员工手册 > 产品手册 > 1 安装 > 1.2 步骤]")
        # 第 2 章不并进第 1 章的父块，它只有一个子块，也不生成父块
        last = chunks[-1]
        self.assertEqual(last.metadata["section"], "产品手册 > 2 计费")
        self.assertNotIn("parent_id", last.metadata)

    def test_each_slide_is_a_chapter(self):
        """PPT：封面标题是文档标题，每页一个二级标题，各页不合并进同一个父块。"""
        elements = [Element("heading", "季度复盘", level=1)]
        for i in range(3):
            elements += [Element("heading", f"第{i}页", level=2)] + [Element("text", f"第{i}页的要点。" * 6)] * 2
        chunks = StructureChunker(chunk_size=40, parent_size=1000).split(_doc(elements))
        for chunk in chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                parent = next(c for c in chunks if c.chunk_id == parent_id)
                self.assertEqual(parent.metadata["section"], chunk.metadata["section"])

    def test_long_section_becomes_several_parents(self):
        doc = _doc([Element("heading", "章", level=1)] + [Element("text", "字" * 90 + "。") for _ in range(12)])
        chunks = StructureChunker(chunk_size=100, parent_size=300).split(doc)
        parent_ids = [c.metadata["parent_id"] for c in chunks if "parent_id" in c.metadata]
        parents = [c for c in chunks if c.chunk_id in set(parent_ids)]
        self.assertEqual(len(parents), 4)
        self.assertTrue(all(count_tokens(_body(p)) <= 310 for p in parents))

    def test_children_respect_budget(self):
        text = "".join(f"第{i}条规定适用于全体员工。" for i in range(60))
        rows = [["型号", "价格"]] + [[f"M{i}", str(i)] for i in range(60)]
        doc = _doc([Element("heading", "制度", level=1), Element("text", text),
                    Element("table", table_to_markdown(rows))])
        chunks = StructureChunker(chunk_size=120, parent_size=600).split(doc)
        children = [c for c in chunks if "parent_id" in c.metadata]
        self.assertGreater(len(children), 5)
        self.assertTrue(all(count_tokens(_body(c)) <= 120 for c in children))
        tables = [c for c in children if c.metadata["kinds"] == ["table"]]
        self.assertTrue(all("| 型号 | 价格 |" in c.text for c in tables))
        self.assertEqual(sum(c.text.count("| M") for c in tables), 60)

    def test_formula_and_image_stay_with_lead_in(self):
        filler = "其他说明文字。" * 10
        doc = _doc([
            Element("heading", "分析", level=1),
            Element("text", filler),
            Element("text", "同比增长率按下式计算："),
            Element("formula", r"g = \frac{x_t - x_{t-1}}{x_{t-1}}"),
            Element("text", filler),
            Element("text", "单据按图 2 所示的四个环节流转。"),
            Element("image", "图 2 报销审批流程", extra={"asset": "ab/flow.png"}),
            Element("text", "部署前需要准备以下环境："),
            Element("text", "- 内存不少于 16 GB"),
        ])
        chunks = StructureChunker(chunk_size=60, parent_size=400).split(doc)
        children = [c for c in chunks if "parent_id" in c.metadata]
        formula = next(c for c in children if "frac" in c.text)
        self.assertIn("按下式计算：", formula.text)
        image = next(c for c in children if "图 2 报销审批流程" in c.text)
        self.assertIn("四个环节流转", image.text)
        self.assertEqual(image.metadata["assets"], [{"asset": "ab/flow.png", "kind": "image", "caption": "图 2 报销审批流程"}])
        listing = next(c for c in children if "16 GB" in c.text)
        self.assertIn("以下环境：", listing.text)

    def test_sentence_overlap_only_between_children(self):
        text = "".join(f"第{i}条规定适用于全体员工。" for i in range(40))
        doc = _doc([Element("heading", "制度", level=1), Element("text", text)])
        plain = StructureChunker(chunk_size=60, parent_size=600).split(doc)
        chunks = StructureChunker(chunk_size=60, parent_size=600, overlap_sentences=1).split(doc)
        children = [c for c in chunks if "parent_id" in c.metadata]
        for prev, nxt in zip(children, children[1:]):
            self.assertTrue(_body(nxt).startswith(_body(prev).rsplit("。", 2)[-2] + "。"))
        # 父块不重叠：和关闭重叠时完全一样
        parents = lambda cs: [c.text for c in cs if "parent_id" not in c.metadata]
        self.assertEqual(parents(chunks), parents(plain))
        with self.assertRaises(ValueError):
            StructureChunker(overlap_sentences=-1)

    def test_parent_size_zero_disables_parents(self):
        doc = _doc([Element("heading", "章", level=1)] + [Element("text", "字" * 40) for _ in range(5)])
        chunks = StructureChunker(chunk_size=100, parent_size=0).split(doc)
        self.assertEqual([_body(c).count("字" * 40) for c in chunks], [2, 2, 1])
        self.assertFalse(any("parent_id" in c.metadata for c in chunks))

    def test_plain_text_uses_paragraphs(self):
        text = "\n\n".join("第{}段。".format(i) + "内容" * 30 for i in range(6))
        chunks = StructureChunker(chunk_size=80, parent_size=300).split(Document("d", text, {"title": "纯文本"}))
        children = [c for c in chunks if "parent_id" in c.metadata]
        self.assertGreater(len(children), 3)
        self.assertTrue(all(c.text.startswith("[纯文本]\n") for c in chunks))

    def test_rejects_parent_smaller_than_child(self):
        with self.assertRaises(ValueError):
            StructureChunker(chunk_size=300, parent_size=200)


class TestParentExpansion(unittest.TestCase):
    @staticmethod
    def _policy() -> Document:
        para = "差旅住宿按城市分级报销，一线城市每晚上限六百元，其他城市四百元。"
        return _doc([
            Element("heading", "报销制度", level=1),
            Element("heading", "住宿", level=2),
            Element("text", para * 2),
            Element("text", "超标部分需要部门负责人审批。"),
            Element("text", para * 2),
        ])

    def _kb(self) -> tuple[KnowledgeBase, dict]:
        kb = KnowledgeBase(HashingEncoder(dimension=128), StructureChunker(chunk_size=60, parent_size=400))
        stats = kb.add([self._policy()])
        return kb, stats

    def test_only_children_are_indexed(self):
        kb, stats = self._kb()
        self.assertEqual(stats["parents"], 1)
        self.assertEqual(len(kb.vector_index), stats["chunks"])
        self.assertEqual(len(kb.bm25_index), stats["chunks"])
        indexed = [c.chunk for c in kb.bm25_index.search("审批", 10)]
        self.assertTrue(indexed and all("parent_id" in c.metadata for c in indexed))

    def test_expand_dedupes_children_of_same_parent(self):
        kb, _ = self._kb()
        hits = kb.bm25_index.search("城市 审批", 10)
        self.assertGreater(len(hits), 1)
        expanded = kb.expand(hits)
        self.assertEqual(len(expanded), 1)
        self.assertIn("超标部分需要部门负责人审批", expanded[0].text)
        self.assertEqual(expanded[0].score, hits[0].score)
        self.assertEqual(expanded[0].debug["children"], [h.chunk.chunk_id for h in hits])

    def test_hits_without_parent_are_kept(self):
        kb, _ = self._kb()
        orphan = ScoredChunk(chunk=Chunk("x#0", "x", "没有父块的片段"), score=1.0)
        self.assertIs(kb.expand([orphan])[0], orphan)

    def test_parents_survive_save_and_load(self):
        kb, _ = self._kb()
        with tempfile.TemporaryDirectory() as tmp:
            kb.save(tmp)
            fresh = KnowledgeBase(HashingEncoder(dimension=128), StructureChunker())
            fresh.load(tmp)
            expanded = fresh.expand(fresh.bm25_index.search("审批", 3))
        self.assertEqual(fresh.stats()["parents"], 1)
        self.assertIn("超标部分", expanded[0].text)
        self.assertTrue(expanded[0].text.startswith("[员工手册 > 报销制度 > 住宿]"))

    def test_pipeline_returns_parents(self):
        from learn_rag.core.config import load_config
        from learn_rag.pipeline.rag import RagPipeline

        cfg = load_config("configs/default.yaml")
        cfg["chunker"] = {"type": "structure", "chunk_size": 60, "parent_size": 400}
        pipe = RagPipeline.from_config(cfg)
        pipe.index([self._policy()])
        result = pipe.retrieve_only("超标部分谁审批？", top_k=3)
        self.assertEqual(len(result.contexts), 1)
        self.assertIn("一线城市每晚上限六百元", result.contexts[0].text)
        self.assertIn("expand", result.timings)


class TestAssetEntries(unittest.TestCase):
    """块元数据里的每张图：路径、类型、图注、页码、坐标，回答时能说清"见第几页图几"并取回原图。"""

    def test_entries_carry_caption_page_and_bbox(self):
        doc = _doc([
            Element("heading", "分析", level=1),
            Element("text", "各区域营收如图 1 所示。", page=2),
            Element("image", "2025年各区域季度营收\n图 1 2025 年各区域季度营收", page=2, bbox=[0.2, 0.5, 0.8, 0.7],
                    extra={"asset": "9f/chart.jpg", "mime": "image/jpeg"}),
            Element("image", "[图片]", page=3, extra={"asset": "ae/trend.png"}),
            Element("table", "表 1 价格\n| 版本 | 月费 |\n| --- | --- |\n| 基础版 | 999 |\n注：含税。", page=3,
                    extra={"asset": "01/table.jpg"}),
        ])
        assets = StructureChunker().split(doc)[0].metadata["assets"]
        self.assertEqual(assets, [
            {"asset": "9f/chart.jpg", "kind": "image", "caption": "2025年各区域季度营收 图 1 2025 年各区域季度营收",
             "page": 2, "bbox": [0.2, 0.5, 0.8, 0.7], "mime": "image/jpeg"},
            {"asset": "ae/trend.png", "kind": "image", "page": 3},          # 占位文字不算图注
            {"asset": "01/table.jpg", "kind": "table", "caption": "表 1 价格", "page": 3},   # 表格只取表题
        ])

    def test_split_table_records_its_screenshot_once_per_chunk(self):
        rows = [["型号", "价格"]] + [[f"M{i}", str(i)] for i in range(60)]
        table = Element("table", "表 2 报价\n" + table_to_markdown(rows), page=5, extra={"asset": "01/t.jpg"})
        chunks = StructureChunker(chunk_size=80, parent_size=400).split(_doc([Element("heading", "价格", level=1), table]))
        self.assertGreater(len(chunks), 3)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["assets"], [{"asset": "01/t.jpg", "kind": "table", "caption": "表 2 报价", "page": 5}])

    def test_chroma_knowledge_base_keeps_assets_and_parents(self):
        """Chroma 建库 -> 落盘 -> 另起一个知识库加载 -> 召回子块并展开成父块，资产信息一路不丢。"""
        try:
            from learn_rag.store.chroma_index import ChromaVectorIndex
        except ImportError:
            self.skipTest("未安装 chromadb")
        para = "平台内置的报表模块可以直接生成区域营收对比图，数据每天凌晨更新一次。"
        doc = _doc([
            Element("heading", "数据分析", level=1),
            Element("text", para * 2, page=1),
            Element("text", "各区域季度营收如图 1 所示。", page=2),
            Element("image", "图 1 各区域季度营收对比", page=2, extra={"asset": "d9/chart.png"}),
            Element("text", para * 2, page=2),
        ])
        chunker = StructureChunker(chunk_size=60, parent_size=400)
        with tempfile.TemporaryDirectory() as tmp:
            kb = KnowledgeBase(HashingEncoder(dimension=128), chunker,
                               vector_index=ChromaVectorIndex(path=tmp, collection="kb_assets", reset=True))
            kb.add([doc])
            kb.save(tmp + "/kb")
            fresh = KnowledgeBase(HashingEncoder(dimension=128), chunker,
                                  vector_index=ChromaVectorIndex(path=tmp, collection="kb_assets"))
            fresh.load(tmp + "/kb")
            query = fresh.encoder.encode_one("各区域季度营收对比图", is_query=True)
            hits = fresh.vector_index.search(query, 3)
            expanded = fresh.expand(hits)
        child = next(h for h in hits if "图 1" in h.text)
        self.assertEqual(child.chunk.metadata["assets"], [{"asset": "d9/chart.png", "kind": "image",
                                                            "caption": "图 1 各区域季度营收对比", "page": 2}])
        self.assertEqual(len(expanded), 1)
        self.assertEqual(expanded[0].chunk.metadata["assets"], child.chunk.metadata["assets"])
        self.assertEqual((expanded[0].chunk.metadata["page_start"], expanded[0].chunk.metadata["page_end"]), (1, 2))


class TestProvenance(unittest.TestCase):
    """命令行展示的出处：页码、章节和每张图的本地文件。"""

    def test_lists_pages_and_resolves_asset_files(self):
        from learn_rag.cli import _provenance

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "9f").mkdir()
            (Path(tmp) / "9f" / "chart.jpg").write_bytes(b"jpg")
            meta = {"page_start": 2, "page_end": 3, "section": "年报 > 2 区域经营数据", "assets": [
                {"asset": "9f/chart.jpg", "kind": "image", "caption": "图 1 区域营收", "page": 2},
                {"asset": "ae/missing.png", "kind": "table", "page": 3},
            ]}
            lines = _provenance(meta, ["/nonexistent", tmp])
        self.assertEqual(lines[0], "出处：第 2-3 页　年报 > 2 区域经营数据")
        self.assertEqual(lines[1], f"图片：第 2 页 「图 1 区域营收」 → {Path(tmp) / '9f' / 'chart.jpg'}")
        self.assertEqual(lines[2], "表格截图：第 3 页 （无图注） → ae/missing.png（资产库中未找到）")

    def test_ask_refuses_empty_index(self):
        from learn_rag.cli import _load_index
        from learn_rag.core.config import load_config
        from learn_rag.pipeline.rag import RagPipeline

        cfg = load_config("configs/default.yaml")
        pipe = RagPipeline.from_config(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            cfg["index"] = {"type": "flat", "path": tmp}   # 落盘目录下什么都没有
            with self.assertRaises(SystemExit):
                _load_index(pipe, cfg)


if __name__ == "__main__":
    unittest.main()
