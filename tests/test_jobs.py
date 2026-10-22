"""
第 5 阶段（任务队列与 worker）的测试：任务落库、同一知识库不会被两个 worker 同时写、中断的任务重新排队、
向量化失败自动重试、worker 停止时把任务放回队列，以及真实的 worker 进程接手别的进程留下的任务。
"""

from __future__ import annotations

import io
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.server.jobs import JobStore

_GUIDE = "# 部署指南\n\n## 环境\n\n" + "安装前确认服务器内存不少于十六GB，并开放八零八零端口。" * 4
_FAQ = "# 常见问题\n\n## 登录\n\n" + "忘记密码时在登录页点击找回密码，按短信验证码重置。" * 4
_SETTINGS = {"chunker": {"type": "structure", "chunk_size": 60, "parent_size": 240},
             "encoder": {"type": "hashing", "model": "hashing"}, "index": {"type": "flat"}}


def _dead_pid() -> int:
    """一个已经退出的进程号：模拟执行任务的 worker 被杀掉了。"""
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


class TestJobStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.store = JobStore(Path(self.tmp) / "jobs.db", keep=2)

    def test_one_running_job_per_kb(self):
        first = self.store.submit("kb1", "build", ["a.md"])
        second = self.store.submit("kb1", "append", ["b.md"])
        other = self.store.submit("kb2", "build", ["c.md"])
        self.assertEqual(self.store.claim("w1").id, first.id)
        self.assertEqual(self.store.claim("w2").id, other.id)             # 另一个知识库可以并行
        self.assertIsNone(self.store.claim("w3"))                          # kb1 正在执行，第二个任务等着
        self.assertTrue(self.store.busy("kb1"))
        job = self.store.get(first.id)
        self.assertEqual((job.status, job.attempts, job.worker), ("running", 1, "w1"))
        job.status = "done"
        job.touch()
        self.assertEqual(self.store.claim("w3").id, second.id)

    def test_progress_is_persisted(self):
        job = self.store.submit("kb1", "build", ["a.md"])
        claimed = self.store.claim("w1")
        claimed.files[0].status, claimed.notice = "embedding", "处理中"
        claimed.touch()
        reread = JobStore(Path(self.tmp) / "jobs.db").get(job.id)          # 另一个进程打开同一个队列
        self.assertEqual((reread.files[0].status, reread.notice), ("embedding", "处理中"))
        self.assertGreater(reread.version, job.version)

    def test_interrupted_jobs_are_requeued(self):
        job = self.store.submit("kb1", "build", ["a.md", "b.md"])
        claimed = self.store.claim(f"{socket.gethostname()}:{_dead_pid()}:1")
        claimed.files[0].status, claimed.files[1].status = "done", "embedding"
        claimed.touch()
        self.assertEqual(self.store.recover(stale_after=3600), [job.id])  # 进程不在了：不用等心跳超时
        job = self.store.get(job.id)
        self.assertEqual(job.status, "queued")
        self.assertEqual([f.status for f in job.files], ["done", "waiting"])  # 做完的不重做
        self.assertIn("中断", job.notice)

    def test_stale_heartbeat_and_exhausted_attempts(self):
        job = self.store.submit("kb1", "build", ["a.md"], max_attempts=1)
        self.store.claim("other-host:123:1")                               # 别的机器上的 worker，只能看心跳
        self.assertEqual(self.store.recover(stale_after=3600), [])
        self.assertEqual(self.store.recover(stale_after=0), [job.id])
        job = self.store.get(job.id)
        self.assertEqual(job.status, "failed")                             # 次数用完：不再重试
        self.assertIn("中断", job.error)

    def test_prune_keeps_recent_finished_jobs(self):
        ids = []
        for i in range(4):
            job = self.store.submit("kb1", "build", [f"{i}.md"])
            job.status = "done"
            job.touch()
            ids.append(job.id)
            time.sleep(0.01)
        self.store.prune("kb1")
        self.assertEqual([j.id for j in self.store.history("kb1")], ids[:1:-1])


class TestWorker(unittest.TestCase):
    """worker 和页面服务在同一进程里（不起线程，用 run_once 一步步推进）。"""

    def setUp(self) -> None:
        from learn_rag.server.library import Library
        from learn_rag.server.worker import JobWorker

        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.library = Library(self.tmp)
        self.worker = JobWorker(self.library, retry_delay=0)

    def _create(self) -> tuple[str, str]:
        upload_id = None
        for name, text in {"部署指南.md": _GUIDE, "常见问题.md": _FAQ}.items():
            upload_id, _ = self.library.staging.save(upload_id, name, io.BytesIO(text.encode("utf-8")))
        created = self.library.create("测试库", _SETTINGS, upload_id)
        return created["kb"]["id"], created["job"]["id"]

    def test_embedding_failure_is_retried(self):
        from learn_rag.embedding.encoders import HashingEncoder

        kb_id, job_id = self._create()
        original = HashingEncoder.encode
        calls = {"n": 0}

        def flaky(encoder, texts, is_query=False):
            calls["n"] += 1
            if calls["n"] == 1:                                            # 第一次调用失败：服务限流
                raise TimeoutError("embedding 服务限流")
            return original(encoder, texts, is_query=is_query)

        with patch.object(HashingEncoder, "encode", flaky):
            self.assertTrue(self.worker.run_once())
            job = self.library.jobs.get(job_id)
            self.assertEqual(job.status, "queued")                         # 有文件失败：排队重试
            self.assertIn("自动重试", job.notice)
            self.assertEqual(sorted(f.status for f in job.files), ["done", "waiting"])
            self.assertTrue(self.worker.run_once())
        job = self.library.jobs.get(job_id)
        self.assertEqual((job.status, job.attempts), ("done", 2))
        self.assertEqual([f.status for f in job.files], ["done", "done"])
        self.assertTrue(job.verify["ok"], job.verify["problems"])
        info = self.library.info(kb_id)
        self.assertTrue(all(d["status"] == "done" and d["seconds"]["index"] >= 0 for d in info["documents"]))

    def test_parse_failures_are_not_retried(self):
        kb_id, job_id = self._create()
        with patch("learn_rag.parsing.source.FileSource.load_file", return_value=None):
            self.worker.run_once()
        job = self.library.jobs.get(job_id)
        self.assertEqual((job.status, job.attempts), ("done", 1))          # 解析失败重试也一样，直接结束
        self.assertEqual([f.status for f in job.files], ["failed", "failed"])

    def test_stop_puts_the_job_back(self):
        kb_id, job_id = self._create()
        self.worker.stopping.set()                                          # 第一个文件开始前就要停
        self.worker.run_once()
        job = self.library.jobs.get(job_id)
        self.assertEqual((job.status, job.attempts), ("queued", 0))        # 主动停止不算一次执行
        self.worker.stopping.clear()
        self.worker.run_once()
        self.assertEqual(self.library.jobs.get(job_id).status, "done")

    def test_remove_document_runs_as_a_job(self):
        kb_id, job_id = self._create()
        self.worker.run_once()
        removal = self.library.remove_document(kb_id, "常见问题.md")
        self.assertEqual(removal.kind, "remove")
        self.worker.run_once()
        self.assertEqual(self.library.jobs.get(removal.id).status, "done")
        self.assertEqual([d["doc_id"] for d in self.library.info(kb_id)["documents"]], ["部署指南.md"])
        self.assertEqual(self.library.pipeline(kb_id).kb.documents(), ["部署指南.md"])

    def test_deleted_kb_fails_without_retry(self):
        kb_id, job_id = self._create()
        shutil.rmtree(self.library.kbs_dir / kb_id)
        self.worker.run_once()
        job = self.library.jobs.get(job_id)
        self.assertEqual((job.status, job.attempts), ("failed", 1))


class TestWorkerProcess(unittest.TestCase):
    """真实的 learn-rag worker 进程：接手被杀掉的 worker 留下的任务，页面服务进程看到它写入的结果。"""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_worker_process_takes_over(self):
        from learn_rag.server.library import Library
        from learn_rag.server.worker import WorkerProcess

        library = Library(self.tmp)                                          # 页面服务：只提交、只读
        upload_id, _ = library.staging.save(None, "部署指南.md", io.BytesIO(_GUIDE.encode("utf-8")))
        created = library.create("测试库", _SETTINGS, upload_id)
        kb_id, job_id = created["kb"]["id"], created["job"]["id"]
        library.jobs.claim(f"{socket.gethostname()}:{_dead_pid()}:1")      # 上一个 worker 领了任务就死了

        worker = WorkerProcess(self.tmp)
        worker.start()
        try:
            job = library.jobs.wait(job_id, timeout=60)
        finally:
            worker.stop()
        log = (Path(self.tmp) / "worker.log").read_text(encoding="utf-8", errors="replace")
        self.assertEqual(job.status, "done", log[-2000:])
        self.assertEqual(job.attempts, 2)                                    # 被中断的那次也算
        self.assertTrue(job.verify["ok"], job.verify["problems"])
        view = library.document(kb_id, "部署指南.md")                       # 另一个进程写的，这里重新打开后可见
        self.assertTrue(view["children"])
        self.assertEqual(worker.status()["state"], "stopped")


if __name__ == "__main__":
    unittest.main()
