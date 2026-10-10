"""
可视化建库后端的测试：上传 → 推荐 → 新建 → 后台建库 → 查看分块 → 删除文档 → 重建 → 删除知识库。

走真实的 HTTP 接口（FastAPI TestClient），embedding 用 hashing，不依赖网络。
fastapi 是可选依赖，没装时跳过。
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _has(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


_GUIDE = """# 部署指南

## 第一章 环境准备

部署前需要准备以下环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10 |
| Chroma | 1.5 |

## 第二章 安装步骤

""" + "\n\n".join(f"第{i}步：执行安装命令并检查输出，确认服务 svc-{i} 已经启动。" * 4 for i in range(6))

_LONG = "# 长段落\n\n" + "".join(f"第{i}句讲的是编号 X-{i} 的配置细节，务必逐项核对。" for i in range(40))

_CSV = "订单号,客户,金额\n" + "\n".join(f"XH-{1000 + i},客户{i},{i * 10}" for i in range(30))

_HASHING = {"type": "hashing", "model": "hashing"}


@unittest.skipUnless(_has("fastapi") and _has("httpx") and _has("multipart"), "未安装 fastapi / httpx / python-multipart")
class TestServer(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from learn_rag.server.app import create_app

        self.tmp = tempfile.mkdtemp()
        self.app = create_app(self.tmp)
        self.client = TestClient(self.app)
        self.library = self.app.state.library

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------------------------------------------------------------- 工具
    def _upload(self, files: dict[str, str], upload_id: str | None = None) -> dict:
        data = {"paths": list(files)}
        if upload_id:
            data["upload_id"] = upload_id
        response = self.client.post(
            "/api/uploads", data=data,
            files=[("files", (Path(p).name, io.BytesIO(text.encode("utf-8")))) for p, text in files.items()])
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _create(self, settings: dict, files: dict[str, str] | None = None) -> tuple[str, dict]:
        upload = self._upload(files or {"手册/部署指南.md": _GUIDE, "数据/订单.csv": _CSV})
        response = self.client.post("/api/kbs", json={"name": "测试库", "upload_id": upload["upload_id"],
                                                      "settings": settings})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        job = self.library.jobs.wait(body["job"]["id"])
        self.assertEqual(job.status, "done", job.error)
        return body["kb"]["id"], job.snapshot()

    # ---------------------------------------------------------------- 用例
    def test_env_greys_out_local_models(self):
        env = self.client.get("/api/env").json()
        local = next(e for e in env["encoders"] if e["type"] == "sentence_transformers")
        self.assertFalse(local["available"])
        self.assertTrue(next(e for e in env["encoders"] if e["type"] == "hashing")["available"])
        self.assertTrue(all("configured" in k and "value" not in k for k in env["keys"]))   # 密钥只报是否配置

    def test_upload_keeps_folders_and_rejects_escapes(self):
        upload = self._upload({"a/说明.md": "# 标题\n\n正文", "b/说明.md": "# 另一份\n\n正文", "工具.exe": "x"})
        paths = {f["path"]: f["supported"] for f in upload["files"]}
        self.assertEqual(paths, {"a/说明.md": True, "b/说明.md": True, "工具.exe": False})
        bad = self.client.post("/api/uploads", data={"paths": ["../../etc/passwd"]},
                               files=[("files", ("passwd", io.BytesIO(b"x")))])
        self.assertEqual(bad.status_code, 400)
        removed = self.client.delete(f"/api/uploads/{upload['upload_id']}/files", params={"path": "工具.exe"}).json()
        self.assertEqual(len(removed["files"]), 2)

    def test_recommend_explains_every_choice(self):
        upload = self._upload({"长文.md": _LONG, "订单.csv": _CSV, "订单2.csv": _CSV})
        result = self.client.post(f"/api/uploads/{upload['upload_id']}/recommend").json()
        chunker = result["settings"]["chunker"]
        self.assertEqual(chunker["type"], "structure")
        self.assertEqual(chunker["chunk_size"], 400)            # 三个文件里两个是表格
        self.assertEqual(chunker["overlap_sentences"], 1)       # 长段落比子块还长
        for field in ("chunker", "chunk_size", "parent_size", "overlap_sentences", "encoder", "index"):
            self.assertTrue(result["reasons"].get(field), field)
        self.assertIn("3 个文件", result["summary"][0])

    def test_build_view_and_remove_document(self):
        kb_id, job = self._create({"chunker": {"type": "structure", "chunk_size": 60, "parent_size": 240},
                                   "encoder": _HASHING, "index": {"type": "flat"}})
        self.assertEqual(job["counts"]["done"], 2)
        self.assertTrue(job["verify"]["ok"], job["verify"]["problems"])

        info = self.client.get(f"/api/kbs/{kb_id}").json()
        docs = {d["doc_id"]: d for d in info["documents"]}
        self.assertEqual(set(docs), {"手册/部署指南.md", "数据/订单.csv"})
        self.assertTrue(all(d["status"] == "done" and d["chunks"] > 0 for d in docs.values()))
        self.assertEqual(info["chunks"], sum(d["chunks"] for d in docs.values()))

        view = self.client.get(f"/api/kbs/{kb_id}/documents/手册/部署指南.md").json()
        self.assertEqual(view["stats"]["children"], docs["手册/部署指南.md"]["chunks"])
        self.assertFalse(any(c["body"].startswith("[") for c in view["children"]))   # 块首路径已去掉
        self.assertTrue(view["parents"])
        members = {cid for p in view["parents"] for cid in p["children"]}
        self.assertEqual(members, {c["id"] for c in view["children"] if c["parent_id"]})
        table = next(c for c in view["children"] if "table" in c["kinds"])
        self.assertIn("第一章", table["section"])

        self.assertEqual(self.client.delete(f"/api/kbs/{kb_id}/documents/数据/订单.csv").status_code, 200)
        self.assertEqual(self.client.get(f"/api/kbs/{kb_id}/documents/数据/订单.csv").status_code, 404)
        # 换一个进程重新打开（flat 从落盘文件读），删除依然生效
        from learn_rag.server.library import Library

        reopened = Library(self.tmp)
        self.assertEqual([d["doc_id"] for d in reopened.info(kb_id)["documents"]], ["手册/部署指南.md"])
        self.assertEqual(reopened._pipeline(kb_id).kb.documents(), ["手册/部署指南.md"])
        self.assertTrue(reopened._pipeline(kb_id).kb.verify()["ok"])

    def test_overlap_is_marked_between_split_paragraph_chunks(self):
        kb_id, _ = self._create({"chunker": {"type": "structure", "chunk_size": 80, "parent_size": 400,
                                             "overlap_sentences": 1},
                                 "encoder": _HASHING, "index": {"type": "flat"}}, files={"长文.md": _LONG})
        children = self.client.get(f"/api/kbs/{kb_id}/documents/长文.md").json()["children"]
        self.assertGreater(len(children), 2)
        self.assertEqual(children[0]["overlap"], 0)
        for previous, current in zip(children, children[1:]):
            if previous["parent_id"] != current["parent_id"]:   # 父块之间不重叠
                self.assertEqual(current["overlap"], 0)
                continue
            self.assertGreater(current["overlap"], 0)
            self.assertTrue(previous["body"].endswith(current["body"][:current["overlap"]]))

    def test_rebuild_with_new_chunk_size_and_resume(self):
        kb_id, _ = self._create({"chunker": {"type": "structure", "chunk_size": 60, "parent_size": 240},
                                 "encoder": _HASHING, "index": {"type": "flat"}})
        before = self.client.get(f"/api/kbs/{kb_id}").json()["chunks"]
        job = self.client.post(f"/api/kbs/{kb_id}/rebuild", json={"settings": {
            "chunker": {"type": "structure", "chunk_size": 150, "parent_size": 600},
            "encoder": _HASHING, "index": {"type": "flat"}}}).json()
        job = self.library.jobs.wait(job["id"])
        self.assertEqual(job.status, "done", job.error)
        info = self.client.get(f"/api/kbs/{kb_id}").json()
        self.assertLess(info["chunks"], before)
        self.assertEqual(info["settings"]["chunker"]["chunk_size"], 150)
        self.assertTrue(info["verify"]["ok"], info["verify"]["problems"])
        # 参数不变再重建一次：全部跳过，不重复向量化
        job = self.library.jobs.wait(self.client.post(f"/api/kbs/{kb_id}/rebuild", json={}).json()["id"])
        self.assertEqual(job.snapshot()["counts"]["skipped"], 2)

    def test_append_files_and_report_parse_failures(self):
        kb_id, _ = self._create({"encoder": _HASHING, "index": {"type": "flat"}})
        upload = self._upload({"空白.md": "   \n\n  ", "补充.md": "# 补充说明\n\n新增的一段内容。"})
        job = self.library.jobs.wait(self.client.post(f"/api/kbs/{kb_id}/files",
                                                       json={"upload_id": upload["upload_id"]}).json()["id"])
        statuses = {f.path: (f.status, f.message) for f in job.files}
        self.assertEqual(statuses["补充.md"][0], "done")
        self.assertEqual(statuses["空白.md"][0], "failed")
        self.assertIn("没有解析出任何内容", statuses["空白.md"][1])
        self.assertEqual(len(self.client.get(f"/api/kbs/{kb_id}").json()["documents"]), 4)

    def test_invalid_settings_are_rejected_with_reason(self):
        upload = self._upload({"a.md": "# 标题\n\n正文"})
        for settings, hint in [
            ({"chunker": {"type": "structure", "chunk_size": 300, "parent_size": 200}}, "parent_size"),
            ({"encoder": {"type": "sentence_transformers"}}, "本地模型"),
            ({"chunker": {"type": "magic"}}, "切分器"),
        ]:
            response = self.client.post("/api/kbs", json={"name": "x", "upload_id": upload["upload_id"],
                                                          "settings": settings})
            self.assertEqual(response.status_code, 400, settings)
            self.assertIn(hint, response.json()["detail"])

    @unittest.skipUnless(_has("chromadb"), "未安装 chromadb")
    def test_chroma_kb_survives_restart_and_can_be_deleted(self):
        kb_id, job = self._create({"encoder": _HASHING, "index": {"type": "chroma"}})
        self.assertTrue(job["verify"]["ok"], job["verify"]["problems"])
        from learn_rag.server.library import Library

        reopened = Library(self.tmp)
        self.assertEqual(len(reopened._pipeline(kb_id).kb.documents()), 2)
        self.assertEqual([k["id"] for k in self.client.get("/api/kbs").json()], [kb_id])
        self.assertEqual(self.client.delete(f"/api/kbs/{kb_id}").status_code, 200)
        self.assertEqual(self.client.get("/api/kbs").json(), [])
        self.assertEqual(self.client.get(f"/api/kbs/{kb_id}").status_code, 404)

    def test_job_events_stream_until_finished(self):
        kb_id, job = self._create({"encoder": _HASHING, "index": {"type": "flat"}})
        with self.client.stream("GET", f"/api/jobs/{job['id']}/events") as response:
            events = [line for line in response.iter_lines() if line.startswith("data: ")]
        self.assertEqual(len(events), 1)          # 已结束的任务推一次最终快照就关闭
        self.assertIn('"status": "done"', events[0])


class TestHelpers(unittest.TestCase):
    def test_safe_relative(self):
        from learn_rag.server.uploads import safe_relative

        self.assertEqual(safe_relative("docs\\a\\b.pdf"), "docs/a/b.pdf")
        self.assertEqual(safe_relative("/abs/./x.md"), "abs/x.md")
        for bad in ("../x", "a/../../x", "", "C:/x"):
            with self.assertRaises(ValueError):
                safe_relative(bad)

    def test_overlap_requires_real_repeat(self):
        from learn_rag.server.chunk_view import _overlap

        self.assertEqual(_overlap("甲句。乙句很长很长。", "乙句很长很长。丙句。"), len("乙句很长很长。"))
        self.assertEqual(_overlap("结尾。", "。开头"), 0)


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
    return path


@unittest.skipUnless(shutil.which("bash"), "没有 bash")
class TestMinerUService(unittest.TestCase):
    """用假的启动 / 停止脚本代替 scripts/start_mineru.sh、stop_mineru.sh。"""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.env = self.tmp / ".mineru.env"
        self.env.write_text("MINERU_MODEL_VLM_SERVER_URL=http://127.0.0.1:30000\n", encoding="utf-8")
        self.log = self.tmp / "calls.log"
        self.stop = _script(self.tmp / "stop.sh", f"echo stop >> {self.log}\necho 已停止 VLM 服务\n")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _service(self, start_body: str):
        from learn_rag.server.mineru_service import MinerUService

        start = _script(self.tmp / "start.sh", f"echo start >> {self.log}\n" + start_body)
        return MinerUService(env_file=self.env, start_script=start, stop_script=self.stop)

    def _calls(self) -> list[str]:
        return self.log.read_text().split() if self.log.exists() else []

    def test_starts_in_background_and_stops_on_exit(self):
        service = self._service("sleep 0.3\necho MinerU 已就绪\n")
        service.start()
        self.assertEqual(service.state, "starting")            # 不阻塞页面服务启动
        self.assertTrue(service.wait(10))
        self.assertEqual(service.status()["url"], "http://127.0.0.1:30000")
        service.stop()
        service.stop()                                          # 重复调用只关一次
        self.assertEqual(self._calls(), ["start", "stop"])
        self.assertEqual(service.state, "stopped")

    def test_failure_is_reported_with_script_output(self):
        service = self._service("echo 模型目录不存在 >&2\nexit 1\n")
        service.start()
        self.assertFalse(service.wait(10))
        self.assertEqual(service.state, "failed")
        self.assertIn("模型目录不存在", service.message)

    def test_stop_during_startup_kills_the_start_script(self):
        service = self._service("sleep 30\n")
        service.start()
        service.stop()
        self.assertFalse(service.wait(5))                        # 启动脚本连同子进程被结束，不会卡住
        self.assertTrue(service._done.is_set())
        self.assertEqual(service.state, "stopped")
        self.assertIn("stop", self._calls())

    def test_without_env_file_nothing_runs(self):
        self.env.unlink()
        service = self._service("echo 不该运行\n")
        service.start()
        service.stop()
        self.assertEqual(service.state, "disabled")
        self.assertEqual(self._calls(), [])

    @unittest.skipUnless(_has("fastapi") and _has("httpx") and _has("multipart"), "未安装 fastapi")
    def test_app_lifespan_starts_and_stops_mineru(self):
        from fastapi.testclient import TestClient

        from learn_rag.server.app import create_app

        service = self._service("echo MinerU 已就绪\n")
        data = tempfile.mkdtemp()
        try:
            with TestClient(create_app(data, mineru=service)) as client:
                self.assertTrue(service.wait(10))
                self.assertEqual(client.get("/api/env").json()["mineru"]["state"], "ready")
            self.assertEqual(self._calls(), ["start", "stop"])   # 服务退出时一并关闭
        finally:
            shutil.rmtree(data, ignore_errors=True)

    @unittest.skipUnless(_has("fastapi"), "未安装 fastapi")
    def test_build_waits_for_mineru_when_it_is_starting(self):
        from learn_rag.server.jobs import Job
        from learn_rag.server.library import Library

        service = self._service("sleep 0.5\necho MinerU 已就绪\n")
        service.start()
        data = tempfile.mkdtemp()
        try:
            library = Library(data, mineru=service)
            job = Job(id="j", kb_id="k", kind="build", files=[])
            library._wait_for_mineru(job, {"parsing": {"parsers": {".pdf": ["mineru", "pdf"]}}})
            self.assertEqual(service.state, "ready")              # 返回时 MinerU 已经就绪
            self.assertEqual(job.notice, "")
            self.assertGreaterEqual(job.version, 2)              # 等待期间页面能看到提示
        finally:
            service.stop()
            shutil.rmtree(data, ignore_errors=True)


class TestMinerUCommand(unittest.TestCase):
    def test_found_in_separate_python_env(self):
        from learn_rag.parsing.external import find_mineru_command

        with tempfile.TemporaryDirectory() as tmp:
            kit = Path(tmp) / "bin" / "mineru-kit"
            kit.parent.mkdir()
            kit.write_text("#!/bin/sh\n")
            kit.chmod(0o755)
            self.assertEqual(find_mineru_command(env={"PATH": "/nonexistent", "MINERU_PYTHON_ENV": tmp}), str(kit))
            self.assertIsNone(find_mineru_command(env={"PATH": "/nonexistent", "MINERU_PYTHON_ENV": tmp + "/x"}))


class TestFrontendBuild(unittest.TestCase):
    def test_skips_when_up_to_date_and_degrades_without_npm(self):
        import os
        import time
        from unittest.mock import patch

        from learn_rag.server.frontend import ensure_built

        with tempfile.TemporaryDirectory() as tmp:
            web = Path(tmp)
            (web / "src").mkdir()
            (web / "src" / "main.ts").write_text("x")
            with patch("learn_rag.server.frontend.shutil.which", return_value=None):
                self.assertIn("无法构建前端", ensure_built(web))         # 没有构建、也没有 npm
                (web / "dist").mkdir()
                (web / "dist" / "index.html").write_text("<html>")
                future = time.time() + 10
                os.utime(web / "dist" / "index.html", (future, future))
                self.assertEqual(ensure_built(web), "前端已是最新")
                os.utime(web / "src" / "main.ts", (future + 10, future + 10))
                self.assertIn("继续使用上次的构建", ensure_built(web))    # 源码更新了但没有 npm


def _make_pdf(path: Path) -> None:
    """两页的英文 PDF：页眉 + 标题 + 几段正文，第二页一段。"""
    import pymupdf

    doc = pymupdf.open()
    paragraphs = [
        "Retrieval augmented generation combines a retriever with a generator to answer questions.",
        "Hybrid retrieval fuses dense vectors and sparse keywords with reciprocal rank fusion.",
        "Parent child chunking indexes small chunks and returns their larger parents to the model.",
    ]
    for number, texts in enumerate([paragraphs[:2], paragraphs[2:]], start=1):
        page = doc.new_page()
        page.insert_text((72, 40), f"Internal handbook page {number}", fontsize=8)
        y = 90
        if number == 1:
            page.insert_text((72, y), "Overview", fontsize=16)
            y += 40
        for text in texts:
            page.insert_textbox(pymupdf.Rect(72, y, 520, y + 60), text, fontsize=11)
            y += 70
    doc.save(path)


@unittest.skipUnless(_has("fastapi") and _has("httpx") and _has("multipart") and _has("pymupdf"),
                     "未安装 fastapi / pymupdf")
class TestPagePreview(unittest.TestCase):
    """原文视图：页面图片 + 每个分块在页面上的位置 + 未进入分块的文字。"""

    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from learn_rag.server.app import create_app

        self.tmp = tempfile.mkdtemp()
        self.client = TestClient(create_app(self.tmp))
        self.library = self.client.app.state.library
        pdf = Path(self.tmp) / "handbook.pdf"
        _make_pdf(pdf)
        files = [("files", ("handbook.pdf", pdf.read_bytes())), ("files", ("notes.md", b"# Notes\n\nplain text"))]
        upload = self.client.post("/api/uploads", data={"paths": ["handbook.pdf", "notes.md"]}, files=files).json()
        settings = {"chunker": {"type": "structure", "chunk_size": 30, "parent_size": 120},
                    "encoder": {"type": "hashing", "model": "hashing"}, "index": {"type": "flat"},
                    "parsers": {".pdf": ["pdf"]}}
        body = self.client.post("/api/kbs", json={"name": "预览", "upload_id": upload["upload_id"],
                                                  "settings": settings}).json()
        self.assertEqual(self.library.jobs.wait(body["job"]["id"]).status, "done")
        self.kb = body["kb"]["id"]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_every_chunk_is_located_on_its_page(self):
        layout = self.client.get(f"/api/kbs/{self.kb}/layout", params={"doc": "handbook.pdf"}).json()
        self.assertTrue(layout["available"])
        self.assertEqual(len(layout["pages"]), 2)
        view = self.client.get(f"/api/kbs/{self.kb}/documents/handbook.pdf").json()
        for child in view["children"]:
            self.assertEqual(layout["methods"][child["id"]], "text", child["body"])
            pages = {r["page"] for r in layout["regions"][child["id"]]}
            self.assertEqual(pages, {child["page_start"]} if child["page_start"] == child["page_end"] else pages)
            for region in layout["regions"][child["id"]]:
                x0, y0, x1, y1 = region["bbox"]
                self.assertTrue(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1)
        # 页眉被解析器丢弃，它是唯一没进入分块的文字；标题也算覆盖
        self.assertTrue(layout["uncovered"])
        self.assertTrue(all(r["bbox"][1] < 0.1 for r in layout["uncovered"]), layout["uncovered"])
        self.assertGreater(layout["coverage"], 0.8)

    def test_page_images_and_unavailable_formats(self):
        response = self.client.get(f"/api/kbs/{self.kb}/page", params={"doc": "handbook.pdf", "n": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(self.client.get(f"/api/kbs/{self.kb}/page", params={"doc": "handbook.pdf", "n": 9}).status_code, 404)
        markdown = self.client.get(f"/api/kbs/{self.kb}/layout", params={"doc": "notes.md"}).json()
        self.assertFalse(markdown["available"])
        self.assertIn("只能按文本查看", markdown["reason"])


class TestPreviewMatching(unittest.TestCase):
    def test_small_differences_do_not_break_alignment(self):
        from learn_rag.server.preview import _match, _normalize

        stream = _normalize("页眉文字。本报告总结公司 2025 年经营情况，全年营业收入 4,860 万元，同比增长 18.6%。页脚")
        target = _normalize("本报告总结公司2025年经营情况。全年营业收人4860万元，同比增长18.6%")   # 一个错字、标点不同
        spans, ratio = _match(stream, target, 0)
        self.assertGreater(ratio, 0.8)
        self.assertEqual(len(spans), 1)
        self.assertTrue(stream[spans[0][0]:spans[0][1]].startswith("本报告"))
        self.assertTrue(stream[spans[0][0]:spans[0][1]].endswith("186"))     # 块尾不足一段的几个字也对上了

    def test_heading_prefers_its_own_line_over_page_header(self):
        from collections import Counter

        from learn_rag.server.preview import _find_heading

        stream = "星河科技产品手册v2" + "产品手册" + "正文"
        lines = [1] * 10 + [2] * 4 + [3] * 2           # 第 1 行是页眉，第 2 行是标题
        boxes = [(1, line, 0, 0, 0, 0) for line in lines]
        self.assertEqual(_find_heading(stream, boxes, Counter(lines), "产品手册"), [10, 14])


if __name__ == "__main__":
    unittest.main()
