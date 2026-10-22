"""
server.jobs —— 建库任务队列：任务存在 SQLite 里，worker 进程领取执行，页面服务只负责提交和查看。

为什么要把任务落库、把执行挪到单独的进程：
  - 页面服务重启、升级时，正在建的库不会丢：任务还在队列里，worker 接着做（已经写入的文件靠指纹跳过）；
  - 解析和向量化最容易出事（MinerU 卡死、本地模型吃光内存），出事的是 worker，页面照常能看；
  - worker 被杀掉或机器重启，任务停在"执行中"：worker 定期写心跳，心跳停了的任务会被重新排队。

不引入 Celery / Redis：SQLite 是标准库，一个文件就是一个队列，WAL 模式下页面服务读、worker 写互不阻塞。
同一个知识库同一时刻只有一个任务在执行（领取时检查），不同知识库可以由多个 worker 并行处理。

失败重试分两种：
  - 任务级：执行中抛异常、worker 中断，整个任务重新排队；
  - 文件级：向量化失败（embedding 服务限流、超时）和增强没完成的文件标记 retry，任务结束后隔一段时间只重做这些文件。
    解析失败不重试：解析器已经按配置逐个降级过，同一个文件再解析一遍结果还是一样。
重试次数用完后任务照常结束，失败原因留在文件上。
"""

from __future__ import annotations

import json
import logging
import os
import socket
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# 文件状态：等待 → 解析中 → 向量化中 → 完成 / 未变化 / 失败；删除文档时是"删除中"
WAITING, PARSING, EMBEDDING, REMOVING, DONE, SKIPPED, FAILED = (
    "waiting", "parsing", "embedding", "removing", "done", "skipped", "failed")
# 任务状态：排队中（含等待重试）→ 执行中 → 完成 / 失败
QUEUED, RUNNING, JOB_DONE, JOB_FAILED = "queued", "running", "done", "failed"
_IN_PROGRESS = (PARSING, EMBEDDING, REMOVING)


@dataclass
class FileProgress:
    path: str
    status: str = WAITING
    chunks: int = 0
    parents: int = 0
    message: str = ""
    retry: bool = False                         # 失败可以重试（向量化失败、增强没完成）
    seconds: dict[str, float] = field(default_factory=dict)    # 各阶段耗时：parse / index


@dataclass
class Job:
    id: str
    kb_id: str
    kind: str                                   # build / append / rebuild / remove
    files: list[FileProgress]
    status: str = QUEUED
    error: str = ""
    notice: str = ""                            # 任务级提示，如"等待 MinerU 服务就绪""30 秒后重试"
    verify: dict[str, Any] | None = None
    reset: bool = False                         # 执行前清空片段库（换了 embedding 模型或索引参数）
    attempts: int = 0                           # 已经执行过几次（含这一次）
    max_attempts: int = 3
    available_at: float = 0.0                   # 排队中的任务最早什么时候可以执行（重试退避）
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    heartbeat_at: float | None = None
    worker: str = ""
    version: int = 0

    def __post_init__(self) -> None:
        self._store: JobStore | None = None

    def touch(self) -> None:
        """状态变了：版本号加一并写回队列，页面据此推送。"""
        self.version += 1
        if self._store is not None:
            self._store.save(self)

    @property
    def finished(self) -> bool:
        return self.status in (JOB_DONE, JOB_FAILED)

    def snapshot(self) -> dict[str, Any]:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["files"] = [asdict(f) for f in self.files]
        counts = {s: 0 for s in (WAITING, PARSING, EMBEDDING, REMOVING, DONE, SKIPPED, FAILED)}
        for f in self.files:
            counts[f.status] += 1
        data["counts"] = counts
        data["processed"] = counts[DONE] + counts[SKIPPED] + counts[FAILED]
        return data


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kb_id TEXT NOT NULL,
    status TEXT NOT NULL,
    available_at REAL NOT NULL,
    created_at REAL NOT NULL,
    heartbeat_at REAL,
    worker TEXT NOT NULL DEFAULT '',
    body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs (status, available_at, created_at);
CREATE INDEX IF NOT EXISTS jobs_kb ON jobs (kb_id, created_at);
"""


class JobStore:
    """
    任务队列。领取、心跳、重新排队要用到的几列单独存，其余（文件进度、核对结果）整体存成 JSON：
    任务的结构还会变，不想每加一个字段就迁移一次表。
    """

    def __init__(self, path: str | Path, *, keep: int = 20) -> None:
        """keep：每个知识库保留最近多少个已结束的任务，更早的删掉。"""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self._submitted = threading.Event()     # 同一进程里的 worker 线程不用等下一次轮询
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)

    # ------------------------------------------------------------------ 页面服务用
    def submit(self, kb_id: str, kind: str, files: list[str], *, reset: bool = False, max_attempts: int = 3) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kb_id=kb_id, kind=kind, files=[FileProgress(p) for p in files],
                  reset=reset, max_attempts=max_attempts, available_at=time.time())
        with self._connect() as db:
            db.execute("INSERT INTO jobs (id, kb_id, status, available_at, created_at, body) VALUES (?, ?, ?, ?, ?, ?)",
                       (job.id, kb_id, job.status, job.available_at, job.created_at, _dump(job)))
        self._submitted.set()
        return self._attach(job)

    def get(self, job_id: str) -> Job | None:
        with self._connect() as db:
            row = db.execute("SELECT body FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._attach(_load(row[0])) if row else None

    def latest(self, kb_id: str) -> Job | None:
        with self._connect() as db:
            row = db.execute("SELECT body FROM jobs WHERE kb_id = ? ORDER BY created_at DESC LIMIT 1",
                             (kb_id,)).fetchone()
        return self._attach(_load(row[0])) if row else None

    def history(self, kb_id: str, limit: int = 20) -> list[Job]:
        with self._connect() as db:
            rows = db.execute("SELECT body FROM jobs WHERE kb_id = ? ORDER BY created_at DESC LIMIT ?",
                              (kb_id, limit)).fetchall()
        return [_load(r[0]) for r in rows]

    def busy(self, kb_id: str) -> bool:
        with self._connect() as db:
            row = db.execute("SELECT 1 FROM jobs WHERE kb_id = ? AND status IN (?, ?) LIMIT 1",
                             (kb_id, QUEUED, RUNNING)).fetchone()
        return row is not None

    def wait(self, job_id: str, timeout: float = 60) -> Job:
        """等任务结束（测试和脚本用）。"""
        deadline = time.time() + timeout
        job = self.get(job_id)
        while job is not None and not job.finished and time.time() < deadline:
            time.sleep(0.05)
            job = self.get(job_id)
        if job is None:
            raise LookupError(f"任务 {job_id} 不存在")
        return job

    # ------------------------------------------------------------------ worker 用
    def save(self, job: Job) -> None:
        with self._connect() as db:
            db.execute("UPDATE jobs SET status = ?, available_at = ?, heartbeat_at = ?, worker = ?, body = ? WHERE id = ?",
                       (job.status, job.available_at, job.heartbeat_at, job.worker, _dump(job), job.id))

    def claim(self, worker: str) -> Job | None:
        """
        领取最早一个可以执行的任务：到了重试时间，并且同一个知识库没有正在执行的任务。
        BEGIN IMMEDIATE 先拿写锁，两个 worker 同时领取也不会领到同一个任务。
        """
        now = time.time()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT body FROM jobs WHERE status = ? AND available_at <= ? "
                "AND kb_id NOT IN (SELECT kb_id FROM jobs WHERE status = ?) ORDER BY created_at LIMIT 1",
                (QUEUED, now, RUNNING)).fetchone()
            if row is None:
                return None
            job = _load(row[0])
            job.status, job.worker, job.attempts = RUNNING, worker, job.attempts + 1
            job.started_at = job.heartbeat_at = now
            job.notice, job.version = "", job.version + 1
            db.execute("UPDATE jobs SET status = ?, heartbeat_at = ?, worker = ?, body = ? WHERE id = ?",
                       (job.status, now, worker, _dump(job), job.id))
        return self._attach(job)

    def wait_for_work(self, timeout: float) -> None:
        """队列空时等一会儿：本进程提交了任务立即返回；其他进程提交的要等到超时（轮询）。"""
        self._submitted.wait(timeout)
        self._submitted.clear()

    def heartbeat(self, job_id: str) -> None:
        with self._connect() as db:
            db.execute("UPDATE jobs SET heartbeat_at = ? WHERE id = ? AND status = ?", (time.time(), job_id, RUNNING))

    def recover(self, stale_after: float) -> list[str]:
        """
        把中断了的任务重新排队，返回它们的 id：心跳超过 stale_after 秒没更新的，
        或者执行它的 worker 和自己在同一台机器上、进程已经不在了的（不用等心跳超时，重启后立刻接着做）。
        做到一半的文件回到"等待"；中断次数也算重试次数，用完了任务就失败，免得一个必崩的文件反复拖垮 worker。
        """
        now, host = time.time(), socket.gethostname()
        recovered = []
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT body FROM jobs WHERE status = ?", (RUNNING,)).fetchall()
            for (body,) in rows:
                job = _load(body)
                dead = _local_pid(job.worker, host)
                if not (dead is not None and not _alive(dead)) and (job.heartbeat_at or 0) > now - stale_after:
                    continue
                for f in job.files:
                    if f.status in _IN_PROGRESS:
                        f.status, f.message = WAITING, ""
                if job.attempts >= job.max_attempts:
                    job.status, job.finished_at = JOB_FAILED, now
                    job.error = f"执行了 {job.attempts} 次都被中断（worker 退出或卡死），已放弃；点“重建”可以续写"
                else:
                    job.status, job.available_at = QUEUED, now
                    job.notice = f"上次执行被中断（{job.worker}），已重新排队"
                job.version += 1
                db.execute("UPDATE jobs SET status = ?, available_at = ?, body = ? WHERE id = ?",
                           (job.status, job.available_at, _dump(job), job.id))
                recovered.append(job.id)
        for job_id in recovered:
            logger.warning("任务 %s 的执行被中断，已重新排队或标记失败", job_id)
        return recovered

    def prune(self, kb_id: str) -> None:
        """每个知识库只留最近 keep 个已结束的任务。"""
        with self._connect() as db:
            db.execute(
                "DELETE FROM jobs WHERE kb_id = ? AND status IN (?, ?) AND id NOT IN ("
                "SELECT id FROM jobs WHERE kb_id = ? ORDER BY created_at DESC LIMIT ?)",
                (kb_id, JOB_DONE, JOB_FAILED, kb_id, self.keep))

    def forget(self, kb_id: str) -> None:
        """知识库删掉后，它的任务记录也删掉。"""
        with self._connect() as db:
            db.execute("DELETE FROM jobs WHERE kb_id = ?", (kb_id,))

    # ------------------------------------------------------------------ 内部
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """每次操作单独连接：线程之间、进程之间都不共享连接对象，锁冲突时等 10 秒。"""
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        try:
            yield db
            if db.in_transaction:
                db.execute("COMMIT")
        except BaseException:
            if db.in_transaction:
                db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _attach(self, job: Job) -> Job:
        job._store = self
        return job


def worker_name() -> str:
    """worker 的名字：机器名 + 进程号，重新排队时据此判断执行它的进程还在不在。"""
    return f"{socket.gethostname()}:{os.getpid()}:{threading.get_ident()}"


def _local_pid(worker: str, host: str) -> int | None:
    parts = worker.split(":")
    if len(parts) >= 2 and parts[0] == host and parts[1].isdigit():
        return int(parts[1])
    return None


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:          # 进程在，只是不属于当前用户
        return True
    return True


def _dump(job: Job) -> str:
    return json.dumps(job.snapshot(), ensure_ascii=False)


def _load(body: str) -> Job:
    data = json.loads(body)
    data["files"] = [FileProgress(**{k: v for k, v in f.items() if k in _FILE_FIELDS}) for f in data["files"]]
    return Job(**{k: v for k, v in data.items() if k in _JOB_FIELDS})


_FILE_FIELDS = {f.name for f in fields(FileProgress)}
_JOB_FIELDS = {f.name for f in fields(Job)}


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
