"""
文档解析与结构切分的测试。

样例文件都在临时目录里现场生成，不提交二进制文件；
MinerU、Docling、VLM OCR 这类重型依赖一律用假对象或 mock，不依赖 GPU 和网络。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.core.registry import registry
from learn_rag.core.types import Document, Element
from learn_rag.ingest.structure import StructureChunker
from learn_rag.parsing.external import MinerUParser, content_list_to_elements, docling_to_elements
from learn_rag.parsing.markdown import markdown_to_elements, table_to_markdown
from learn_rag.parsing.source import FileSource
from learn_rag.parsing.web import HtmlParser


def _kinds(elements: list[Element]) -> list[str]:
    return [e.kind for e in elements]


class TestMarkdown(unittest.TestCase):
    def test_blocks(self):
        text = (
            "# 总则\n\n第一段，\n同一段的第二行。\n\n## 计费\n\n"
            "| 项目 | 价格 |\n| --- | --- |\n| A | 1 |\n\n"
            "```\nprint(1)\n```\n\n$$\nE=mc^2\n$$\n\n![架构图](a.png)\n\n"
            "<table><tr><td>合并</td></tr>\n</table>\n"
        )
        elements = markdown_to_elements(text, page=3)
        self.assertEqual(_kinds(elements), ["heading", "text", "heading", "table", "code", "formula", "image", "table"])
        self.assertEqual((elements[2].text, elements[2].level), ("计费", 2))
        self.assertEqual(elements[1].text, "第一段，\n同一段的第二行。")
        self.assertEqual(elements[5].text, "E=mc^2")
        self.assertEqual(elements[6].extra["src"], "a.png")
        self.assertTrue(all(e.page == 3 for e in elements))

    def test_table_to_markdown(self):
        md = table_to_markdown([["名称", "说明"], ["a|b", "多\n行"], ["c"], ["", None]])
        self.assertEqual(md.splitlines(), ["| 名称 | 说明 |", "| --- | --- |", "| a\\|b | 多 行 |", "| c |  |"])

    def test_document_from_elements_renders_markdown(self):
        doc = Document.from_elements("d", [Element("heading", "标题", level=2), Element("text", "正文")])
        self.assertEqual(doc.text, "## 标题\n\n正文")
        self.assertEqual(len(doc.elements), 2)


class TestHtml(unittest.TestCase):
    def test_structure_and_noise(self):
        html = """<html><head><style>p{}</style><script>var x=1;</script></head><body>
        <nav>首页 | 关于</nav>
        <h1>产品手册</h1>
        <p>第一行<br>第二行
           仍是第二行</p>
        <h2>价格</h2>
        <table><tr><th>型号</th><th>价格</th></tr><tr><td>A1</td><td>100</td></tr></table>
        <pre>line1
line2</pre>
        <footer>版权所有</footer></body></html>"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.html"
            path.write_text(html, encoding="utf-8")
            elements = HtmlParser().parse(path)
        self.assertEqual(_kinds(elements), ["heading", "text", "heading", "table", "code"])
        self.assertEqual(elements[1].text, "第一行\n第二行 仍是第二行")
        self.assertIn("| A1 | 100 |", elements[3].text)
        self.assertEqual(elements[4].text, "line1\nline2")
        self.assertNotIn("版权所有", " ".join(e.text for e in elements))


class TestOffice(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_docx_keeps_table_position(self):
        try:
            import docx
        except ImportError:
            self.skipTest("未安装 python-docx")
        from learn_rag.parsing.office import DocxParser

        d = docx.Document()
        d.add_heading("报销制度", level=1)
        d.add_paragraph("适用于全体员工。")
        table = d.add_table(rows=2, cols=2)
        table.cell(0, 0).text, table.cell(0, 1).text = "城市", "上限"
        table.cell(1, 0).text, table.cell(1, 1).text = "一线", "600"
        d.add_heading("附则", level=2)
        path = self.dir / "a.docx"
        d.save(path)

        elements = DocxParser().parse(path)
        self.assertEqual(_kinds(elements), ["heading", "text", "table", "heading"])
        self.assertEqual((elements[0].level, elements[3].level), (1, 2))
        self.assertIn("| 一线 | 600 |", elements[2].text)

    def test_pptx_slides_tables_notes(self):
        try:
            from pptx import Presentation
            from pptx.util import Inches
        except ImportError:
            self.skipTest("未安装 python-pptx")
        from learn_rag.parsing.office import PptxParser

        prs = Presentation()
        slide = prs.slides.add_slide(prs.slide_layouts[5])   # 只有标题的版式
        slide.shapes.title.text = "季度营收"
        table = slide.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
        table.cell(0, 0).text, table.cell(0, 1).text = "区域", "营收"
        table.cell(1, 0).text, table.cell(1, 1).text = "华东", "480"
        slide.notes_slide.notes_text_frame.text = "华东贡献最大"
        prs.slides.add_slide(prs.slide_layouts[6])            # 空白页：用页码作为标题
        path = self.dir / "a.pptx"
        prs.save(path)

        elements = PptxParser().parse(path)
        self.assertEqual(_kinds(elements), ["heading", "table", "text", "heading"])
        self.assertEqual(elements[0].text, "季度营收")
        self.assertIn("| 华东 | 480 |", elements[1].text)
        self.assertEqual(elements[2].text, "备注：华东贡献最大")
        self.assertEqual((elements[3].text, elements[3].page), ("第 2 页", 2))

    def test_xlsx_drops_empty_rows_and_columns(self):
        try:
            import openpyxl
        except ImportError:
            self.skipTest("未安装 openpyxl")
        from learn_rag.parsing.office import XlsxParser

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "订单"
        ws["A1"], ws["C1"] = "区域", "数量"
        ws["A2"], ws["C2"] = "华东", 120
        ws["A5"] = "合计"
        ws["C5"] = "=C2"
        path = self.dir / "a.xlsx"
        wb.save(path)

        elements = XlsxParser().parse(path)
        self.assertEqual(_kinds(elements), ["heading", "table"])
        self.assertEqual(elements[0].text, "订单")
        self.assertEqual(elements[1].text.splitlines()[0], "| 区域 | 数量 |")
        self.assertEqual(len(elements[1].text.splitlines()), 4)   # 表头 + 分隔线 + 两行数据


class TestPdf(unittest.TestCase):
    def setUp(self):
        try:
            import pymupdf  # noqa: F401
        except ImportError:
            self.skipTest("未安装 pymupdf")
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "a.pdf"

    def tearDown(self):
        self.tmp.cleanup()

    def _make_pdf(self, scanned_last_page: bool = False):
        import pymupdf

        pdf = pymupdf.open()
        for number in (1, 2, 3):
            page = pdf.new_page()
            page.insert_text((72, 30), "Company Confidential", fontsize=9)       # 页眉
            page.insert_text((290, 820), f"- {number} -", fontsize=9)           # 页脚页码
            if number == 1:
                page.insert_text((72, 100), "Travel Policy", fontsize=20)
                page.insert_text((72, 140), "Hotels in tier one cities are capped at 600 per night and", fontsize=11)
            elif number == 2:
                page.insert_text((72, 100), "other cities at 400 per night.", fontsize=11)
                page.insert_text((72, 140), "Meals", fontsize=15)
                page.insert_text((72, 170), "Meal allowance is 120 per day for every employee.", fontsize=11)
            elif not scanned_last_page:
                page.insert_text((72, 100), "Receipts must be submitted within thirty days.", fontsize=11)
        pdf.save(self.path)

    def test_headings_noise_and_cross_page_merge(self):
        from learn_rag.parsing.pdf import PdfParser

        self._make_pdf()
        elements = PdfParser().parse(self.path)
        texts = [e.text for e in elements]
        self.assertEqual(_kinds(elements), ["heading", "text", "heading", "text", "text"])
        self.assertEqual((elements[0].text, elements[0].level), ("Travel Policy", 1))
        self.assertEqual((elements[2].text, elements[2].level), ("Meals", 2))
        self.assertIn("600 per night and other cities at 400 per night.", texts[1])
        self.assertEqual((elements[1].page, elements[1].extra.get("page_end")), (1, 2))
        self.assertFalse(any("Confidential" in t or t.startswith("-") for t in texts))
        self.assertFalse(any("size" in e.extra or "margin" in e.extra for e in elements))

    def test_scanned_page_goes_to_ocr(self):
        from learn_rag.parsing.pdf import PdfParser

        class FakeOcr:
            def __init__(self):
                self.pages = []

            def recognize(self, image: bytes, *, page=None, mime="image/png"):
                self.pages.append(page)
                self.assertion = image.startswith(b"\x89PNG")
                return [Element("text", "OCR 识别出的内容。", page=page)]

        self._make_pdf(scanned_last_page=True)
        ocr = FakeOcr()
        elements = PdfParser(ocr=ocr, min_chars_per_page=40).parse(self.path)
        self.assertEqual(ocr.pages, [3])
        self.assertTrue(ocr.assertion)
        self.assertEqual((elements[-1].text, elements[-1].page), ("OCR 识别出的内容。", 3))

    def test_table_detected(self):
        import pymupdf

        from learn_rag.parsing.pdf import PdfParser

        pdf = pymupdf.open()
        page = pdf.new_page()
        page.insert_text((72, 80), "Price list for all products in the current catalogue.", fontsize=11)
        x = [72, 222, 372]
        y = [120, 145, 170]
        for yy in y:
            page.draw_line((x[0], yy), (x[-1], yy))
        for xx in x:
            page.draw_line((xx, y[0]), (xx, y[-1]))
        for row, (a, b) in enumerate([("Model", "Price"), ("A1", "100")]):
            page.insert_text((x[0] + 5, y[row] + 17), a, fontsize=11)
            page.insert_text((x[1] + 5, y[row] + 17), b, fontsize=11)
        pdf.save(self.path)

        elements = PdfParser().parse(self.path)
        self.assertEqual(_kinds(elements), ["text", "table"])
        self.assertIn("| A1 | 100 |", elements[1].text)


class TestVlmOcr(unittest.TestCase):
    def test_sends_image_and_parses_markdown(self):
        from learn_rag.parsing.ocr import VlmOcrParser

        sent = {}

        def fake_chat(messages, **_):
            sent["messages"] = messages
            return "```markdown\n# 发票\n\n金额：100 元\n```"

        parser = VlmOcrParser(model="ocr-model", base_url="http://localhost/v1", api_key="k")
        with patch.object(parser.llm, "chat", side_effect=fake_chat):
            elements = parser.recognize(b"\x89PNGfake", page=2)
        content = sent["messages"][0]["content"]
        self.assertTrue(content[0]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(_kinds(elements), ["heading", "text"])
        self.assertEqual((elements[1].text, elements[1].page, elements[1].extra["ocr"]), ("金额：100 元", 2, True))

    def test_requires_model(self):
        from learn_rag.parsing.ocr import VlmOcrParser

        with patch.dict("os.environ", {"OCR_MODEL": ""}):
            with self.assertRaises(ValueError):
                VlmOcrParser()


class TestExternalAdapters(unittest.TestCase):
    CONTENT_LIST = [
        {"type": "header", "text": "内部资料", "page_idx": 0},
        {"type": "text", "text": "第一章 总则", "text_level": 1, "page_idx": 0, "bbox": [1, 2, 3, 4]},
        {"type": "text", "text": "正文内容。", "page_idx": 0},
        {"type": "table", "table_caption": ["表1 价格"], "table_body": "<table><tr><td>A</td></tr></table>",
         "table_footnote": [], "img_path": "images/t.jpg", "page_idx": 1},
        {"type": "image", "image_caption": ["图1 架构"], "img_path": "images/i.jpg", "page_idx": 1},
        {"type": "image", "image_caption": [], "img_path": "images/j.jpg", "page_idx": 1},
        {"type": "equation", "text": "$$\nE=mc^2\n$$", "page_idx": 1},
        {"type": "page_number", "text": "2", "page_idx": 1},
    ]

    def test_mineru_content_list(self):
        elements = content_list_to_elements(self.CONTENT_LIST)
        self.assertEqual(_kinds(elements), ["heading", "text", "table", "image", "formula"])
        self.assertEqual((elements[0].level, elements[0].page, elements[0].bbox), (1, 1, [1, 2, 3, 4]))
        self.assertTrue(elements[2].text.startswith("表1 价格\n<table>"))
        self.assertEqual(elements[4].text, "E=mc^2")

    def test_mineru_runs_cli_and_reads_output(self):
        def fake_run(cmd, **_):
            out = Path(cmd[cmd.index("-o") + 1]) / "a" / "auto"
            out.mkdir(parents=True)
            (out / "a_content_list.json").write_text(json.dumps(self.CONTENT_LIST), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with patch("learn_rag.parsing.external.subprocess.run", side_effect=fake_run) as run:
            elements = MinerUParser(backend="pipeline").parse(Path("a.pdf"))
        self.assertIn("pipeline", run.call_args[0][0])
        self.assertEqual(len(elements), 5)

    def test_mineru_missing_command(self):
        with patch("learn_rag.parsing.external.subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaisesRegex(RuntimeError, "mineru"):
                MinerUParser().parse(Path("a.pdf"))

    def test_docling_items(self):
        def item(label, text="", page=1, **extra):
            box = SimpleNamespace(l=0, t=1, r=2, b=3)
            return SimpleNamespace(label=SimpleNamespace(value=label), text=text,
                                   prov=[SimpleNamespace(page_no=page, bbox=box)], **extra)

        table = item("table", export_to_markdown=lambda doc: "| a |\n| --- |", caption_text=lambda doc: "表1")
        items = [
            item("page_header", "页眉"), item("title", "手册"), item("section_header", "安装", level=1),
            item("text", "步骤一。"), table, item("caption", "表1"),
            item("picture", caption_text=lambda doc: "流程图"), item("list_item", "要点", page=2),
        ]
        doc = SimpleNamespace(iterate_items=lambda: [(i, 0) for i in items])
        elements = docling_to_elements(doc)
        self.assertEqual(_kinds(elements), ["heading", "heading", "text", "table", "image", "text"])
        self.assertEqual([e.level for e in elements[:2]], [1, 2])
        self.assertEqual(elements[3].text, "表1\n| a |\n| --- |")
        self.assertEqual((elements[5].page, elements[5].bbox), (2, [0, 1, 2, 3]))


@registry.register("parser", "_test_broken")
def _broken_parser(**_kwargs):
    class Broken:
        def parse(self, path):
            raise RuntimeError("boom")
    return Broken()


class TestFileSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "docs"
        (self.root / "sub").mkdir(parents=True)
        (self.root / "a.md").write_text("# 标题\n\n内容一", encoding="utf-8")
        (self.root / "sub" / "b.html").write_text("<h1>网页</h1><p>内容二</p>", encoding="utf-8")
        (self.root / "c.bin").write_bytes(b"\x00\x01")
        (self.root / "empty.txt").write_text("   ", encoding="utf-8")
        self.cache = Path(self.tmp.name) / "cache"

    def tearDown(self):
        self.tmp.cleanup()

    def test_routes_and_metadata(self):
        docs = list(FileSource(str(self.root), cache_dir=str(self.cache)).load())
        self.assertEqual([d.doc_id for d in docs], ["a.md", "sub/b.html"])
        self.assertEqual(docs[0].text, "# 标题\n\n内容一")
        self.assertEqual((docs[1].metadata["parser"], docs[1].metadata["file_type"]), ("html", "html"))
        self.assertEqual(len(docs[0].metadata["file_hash"]), 64)

    def test_cache_skips_second_parse(self):
        list(FileSource(str(self.root), cache_dir=str(self.cache)).load())
        with patch("learn_rag.parsing.markdown.markdown_to_elements", side_effect=AssertionError("不应再解析")):
            docs = list(FileSource(str(self.root), cache_dir=str(self.cache)).load())
        self.assertEqual(docs[0].elements[0], Element("heading", "标题", level=1))

    def test_fallback_to_next_parser(self):
        source = FileSource(str(self.root / "a.md"), parsers={".md": ["_test_broken", "markdown"]}, cache_dir=None)
        with self.assertLogs("learn_rag.parsing.source", level="WARNING"):
            docs = list(source.load())
        self.assertEqual((docs[0].doc_id, docs[0].metadata["parser"]), ("a.md", "markdown"))

    def test_bad_ocr_config_does_not_break_pdf(self):
        try:
            import pymupdf
        except ImportError:
            self.skipTest("未安装 pymupdf")
        pdf = pymupdf.open()
        pdf.new_page().insert_text((72, 100), "A normal text page with enough characters to parse.", fontsize=11)
        pdf.save(self.root / "d.pdf")
        with patch.dict("os.environ", {"OCR_MODEL": ""}):
            source = FileSource(str(self.root / "d.pdf"), ocr={"type": "vlm_ocr"}, cache_dir=None)
            with self.assertLogs("learn_rag.parsing.source", level="WARNING"):
                docs = list(source.load())
        self.assertIn("normal text page", docs[0].text)


class TestStructureChunker(unittest.TestCase):
    def _doc(self, elements):
        return Document.from_elements("manual.pdf", elements, {"title": "员工手册"})

    def test_sections_paths_and_pages(self):
        doc = self._doc([
            Element("heading", "报销", level=1, page=1),
            Element("text", "差旅标准如下。", page=1),
            Element("heading", "住宿", level=2, page=2),
            Element("text", "一线城市 600 元。", page=2, extra={"page_end": 3}),
            Element("heading", "餐饮", level=2, page=3),   # 只有标题没有内容，不产生块
            Element("heading", "补贴", level=2, page=3),
            Element("text", "每天 120 元。", page=4),
        ])
        chunks = StructureChunker(chunk_size=200).split(doc)
        self.assertEqual([c.text.splitlines()[0] for c in chunks],
                         ["[员工手册 > 报销]", "[员工手册 > 报销 > 住宿]", "[员工手册 > 报销 > 补贴]"])
        self.assertEqual((chunks[1].metadata["page_start"], chunks[1].metadata["page_end"]), (2, 3))
        self.assertEqual(chunks[1].metadata["section"], "报销 > 住宿")
        self.assertEqual(len({c.chunk_id for c in chunks}), 3)

    def test_packs_paragraphs_until_full(self):
        doc = self._doc([Element("heading", "章", level=1)] + [Element("text", "字" * 40) for _ in range(5)])
        chunks = StructureChunker(chunk_size=100, chunk_overlap=20).split(doc)
        self.assertEqual([c.text.count("字" * 40) for c in chunks], [2, 2, 1])

    def test_table_is_isolated_and_split_with_header(self):
        rows = [["型号", "价格"]] + [[f"M{i}", str(i)] for i in range(30)]
        doc = self._doc([
            Element("heading", "价格表", level=1),
            Element("text", "以下为报价。"),
            Element("table", table_to_markdown(rows)),
            Element("text", "价格含税。"),
        ])
        chunks = StructureChunker(chunk_size=150).split(doc)
        tables = [c for c in chunks if c.metadata["kinds"] == ["table"]]
        self.assertGreater(len(tables), 1)
        for c in tables:
            lines = c.text.splitlines()
            self.assertEqual(lines[1:3], ["| 型号 | 价格 |", "| --- | --- |"])
        all_rows = [l for c in tables for l in c.text.splitlines()[3:]]
        self.assertEqual(len(all_rows), 30)
        self.assertIn("以下为报价。", chunks[0].text)
        self.assertIn("价格含税。", chunks[-1].text)

    def test_long_paragraph_is_split(self):
        doc = self._doc([Element("text", "。".join(["这是一句话"] * 100))])
        chunks = StructureChunker(chunk_size=120, chunk_overlap=20).split(doc)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(c.text.startswith("[员工手册]\n") for c in chunks))

    def test_plain_document_falls_back(self):
        chunks = StructureChunker(chunk_size=50, chunk_overlap=10).split(Document("d", "纯文本。" * 30))
        self.assertGreater(len(chunks), 1)


class TestEndToEnd(unittest.TestCase):
    def test_parse_chunk_answer(self):
        from learn_rag.core.config import load_config
        from learn_rag.pipeline.rag import RagPipeline

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "policy.md").write_text(
                "# 报销制度\n\n## 住宿标准\n\n一线城市每晚上限 600 元。\n\n## 餐饮补贴\n\n每人每天 120 元。",
                encoding="utf-8",
            )
            (root / "faq.html").write_text("<h1>常见问题</h1><p>发票需在 30 天内提交。</p>", encoding="utf-8")
            cfg = load_config("configs/default.yaml", "configs/parsing.yaml")
            cfg["parsing"]["cache_dir"] = str(root / "cache")
            pipe = RagPipeline.from_config(cfg)
            stats = pipe.index(FileSource(str(root), **cfg["parsing"]).load())
            result = pipe.answer("餐饮补贴每天多少钱？")
        self.assertEqual(stats["documents"], 2)
        self.assertIn("120", result.answer.text)
        self.assertEqual(result.contexts[0].chunk.metadata["section"], "报销制度 > 餐饮补贴")


if __name__ == "__main__":
    unittest.main()
