"""
server.jobs —— 后台任务：建库、追加文件、重建。

解析一个 PDF 可能要几分钟，不能放在请求里同步做。这里用一个后台线程按提交顺序逐个执行任务，
页面通过任务快照（轮询或 SSE）看每个文件走到了哪一步。不引入 Celery / Redis：
单机单用户的学习工具，一个线程足够，任务一个接一个跑也避免了同一个知识库被两个任务同时写。

每个任务带一个 version，任何状态变化都加一，SSE 只在 version 变了时推送。
任务只保存在内存里：重启后历史任务消失，但每个文件的最终状态已经写进知识库的清单（kb.json）。
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# 文件状态：等待 → 解析中 → 向量化中 → 完成 / 未变化 / 失败
WAITING, PARSING, EMBEDDING, DONE, SKIPPED, FAILED = "waiting", "parsing", "embedding", "done", "skipped", "failed"


@dataclass
class FileProgress:
    path: str
    status: str = WAITING
    chunks: int = 0
    parents: int = 0
    message: str = ""


@dataclass
class Job:
    id: str
    kb_id: str
    kind: str                                   # build / append / rebuild
    files: list[FileProgress]
    status: str = "queued"                      # queued / running / done / failed
    error: str = ""
    notice: str = ""                            # 任务级提示，如"等待 MinerU 服务就绪"
    verify: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    version: int = 0

    def touch(self) -> None:
        self.version += 1

    @property
    def finished(self) -> bool:
        return self.status in ("done", "failed")

    def snapshot(self) -> dict[str, Any]:
        data = asdict(self)
        counts = {s: 0 for s in (WAITING, PARSING, EMBEDDING, DONE, SKIPPED, FAILED)}
        for f in self.files:
            counts[f.status] += 1
        data["counts"] = counts
        data["processed"] = counts[DONE] + counts[SKIPPED] + counts[FAILED]
        return data


class JobRunner:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[tuple[Job, Callable[[Job], None]]] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="learn-rag-jobs", daemon=True)
        self._thread.start()

    def submit(self, kb_id: str, kind: str, files: list[str], work: Callable[[Job], None]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kb_id=kb_id, kind=kind, files=[FileProgress(p) for p in files])
        self._jobs[job.id] = job
        self._queue.put((job, work))
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def latest(self, kb_id: str) -> Job | None:
        jobs = [j for j in self._jobs.values() if j.kb_id == kb_id]
        return max(jobs, key=lambda j: j.created_at) if jobs else None

    def busy(self, kb_id: str) -> bool:
        return any(j.kb_id == kb_id and not j.finished for j in self._jobs.values())

    def wait(self, job_id: str, timeout: float = 60) -> Job:
        """等任务结束（测试和脚本用）。"""
        deadline = time.time() + timeout
        job = self._jobs[job_id]
        while not job.finished and time.time() < deadline:
            time.sleep(0.05)
        return job

    def _loop(self) -> None:
        while True:
            job, work = self._queue.get()
            job.status = "running"
            job.touch()
            try:
                work(job)
                job.status = "done"
            except Exception as exc:   # 任务失败只影响这一个任务，线程继续处理后面的
                logger.exception("任务 %s（%s）失败", job.id, job.kind)
                job.status, job.error = "failed", str(exc)
            job.finished_at = time.time()
            job.touch()


class _Collector(logging.Handler):
    def __init__(self, thread: int) -> None:
        super().__init__(logging.WARNING)
        self.thread = thread
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.thread == self.thread:
            self.messages.append(record.getMessage())


@contextmanager
def capture_warnings() -> Iterator[list[str]]:
    """
    收集当前线程在这段代码里记下的警告和错误日志。
    解析器和知识库失败时只写日志、不抛异常（单个文件失败不能中断整批），页面要显示失败原因就从日志里取。
    """
    collector = _Collector(threading.get_ident())
    root = logging.getLogger("learn_rag")
    root.addHandler(collector)
    try:
        yield collector.messages
    finally:
        root.removeHandler(collector)
