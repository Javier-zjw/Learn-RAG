"""
server.worker —— 执行建库任务的 worker：从队列（jobs.py）领取任务，交给 Library.run_job 执行。

两种运行方式，逻辑完全相同：
  - 单独的进程：learn-rag worker（learn-rag serve 默认会启动一个并在退出时关掉）。
    写知识库的只有 worker，页面服务只读 —— 本地 Chroma 不支持多个进程同时写，
    另一个进程写入的向量也不会出现在本进程已打开的 HNSW 索引里；
  - 页面服务进程里的一个线程：测试和脚本里直接 Library(...) 时用，省得再起进程。

执行期间另一个线程定时写心跳；收到停止信号时，当前文件处理完就把任务放回队列，下次启动接着做。
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .jobs import DONE, FAILED, JOB_DONE, JOB_FAILED, QUEUED, SKIPPED, WAITING, Job, worker_name

if TYPE_CHECKING:
    from .library import Library

logger = logging.getLogger(__name__)


class Interrupted(Exception):
    """worker 要停了：当前任务放回队列，下次接着做。"""


class JobWorker:
    def __init__(self, library: "Library", *, poll: float = 0.5, heartbeat: float = 5.0,
                 stale_after: float = 60.0, retry_delay: float = 30.0) -> None:
        """
        poll：队列空时多久查一次。heartbeat：心跳间隔。stale_after：心跳停了多久算中断（要远大于心跳间隔）。
        retry_delay：第一次重试前等多久，之后每次翻倍 —— 限流和服务抖动需要时间恢复，立刻重试多半还是失败。
        """
        self.library = library
        self.poll, self.heartbeat, self.stale_after, self.retry_delay = poll, heartbeat, stale_after, retry_delay
        self.name = worker_name()
        self.stopping = threading.Event()

    def start(self) -> threading.Thread:
        """在后台线程里运行（页面服务进程内的 worker）。"""
        thread = threading.Thread(target=self.run, name="learn-rag-worker", daemon=True)
        thread.start()
        return thread

    def run(self) -> None:
        """一直运行到 stop()。"""
        self.name = worker_name()       # 在执行线程里取名字，线程号才对
        logger.info("worker %s 已启动，任务队列：%s", self.name, self.library.jobs.path)
        while not self.stopping.is_set():
            if not self.library.jobs.path.exists():      # 数据目录被删了（测试的临时目录）：没有活可干了
                break
            try:
                busy = self.run_once()
            except Exception as exc:  # noqa: BLE001 —— 队列偶尔读写失败（数据库忙），过一会儿再试，worker 不能退出
                logger.warning("worker 读写任务队列失败：%s", exc)
                busy = False
            if not busy and not self.stopping.is_set():
                self.library.jobs.wait_for_work(self.poll)
        logger.info("worker %s 已停止", self.name)

    def stop(self) -> None:
        self.stopping.set()

    def run_once(self) -> bool:
        """领取并执行一个任务，没有可执行的任务时返回 False。"""
        jobs = self.library.jobs
        jobs.recover(self.stale_after)
        job = jobs.claim(self.name)
        if job is None:
            return False
        logger.info("领取任务 %s（%s，知识库 %s，第 %d 次）", job.id, job.kind, job.kb_id, job.attempts)
        beat = threading.Event()
        threading.Thread(target=self._beat, args=(job.id, beat), daemon=True).start()
        error: Exception | None = None
        try:
            self.library.run_job(job, should_stop=self.stopping.is_set)
        except Interrupted:
            self._requeue(job, notice="worker 停止，任务已放回队列，下次启动接着做", count=False)
            return True
        except (ValueError, LookupError) as exc:     # 参数不合法、知识库被删了：原因说得清，不打堆栈
            logger.error("任务 %s（%s）失败：%s", job.id, job.kind, exc)
            error = exc
        except Exception as exc:  # noqa: BLE001 —— 任务失败只影响这一个任务
            logger.exception("任务 %s（%s）失败", job.id, job.kind)
            error = exc
        finally:
            beat.set()
        self._finish(job, error)
        return True

    def _finish(self, job: Job, error: Exception | None) -> None:
        """按结果决定：完成、失败，还是隔一段时间重试（只重做标了 retry 的文件）。"""
        retry = [f for f in job.files if f.retry]
        # 参数不合法、知识库被删了：重试也一样，直接失败
        permanent = isinstance(error, (ValueError, LookupError))
        if ((error is not None and not permanent) or (error is None and retry)) and job.attempts < job.max_attempts:
            delay = self.retry_delay * 2 ** (job.attempts - 1)
            what = f"出错：{error}" if error is not None else f"{len(retry)} 个文件没有完成"
            self._requeue(job, notice=f"{what}；{delay:.0f} 秒后自动重试（第 {job.attempts + 1}/{job.max_attempts} 次）",
                          delay=delay)
            return
        job.status = JOB_FAILED if error is not None else JOB_DONE
        job.error = str(error) if error is not None else ""
        job.notice = (f"重试 {job.max_attempts - 1} 次后仍有 {len(retry)} 个文件没有完成，原因见各文件"
                      if retry and error is None else "")
        job.finished_at = time.time()
        job.touch()
        logger.info("任务 %s %s，用时 %.1f 秒", job.id, "完成" if job.status == JOB_DONE else "失败",
                    job.finished_at - (job.started_at or job.finished_at))
        self.library.jobs.prune(job.kb_id)

    def _requeue(self, job: Job, *, notice: str, delay: float = 0.0, count: bool = True) -> None:
        """
        放回队列：没完成的文件回到"等待"，已完成的保持原样，下次只做剩下的。
        count=False（worker 主动停止）时不算一次执行。
        """
        for f in job.files:
            if f.status not in (DONE, SKIPPED, FAILED) or f.retry:
                f.status, f.retry = WAITING, False
            if f.status == WAITING:
                f.message = ""
        if not count:
            job.attempts -= 1
        job.status, job.notice, job.available_at = QUEUED, notice, time.time() + delay
        job.touch()

    def _beat(self, job_id: str, done: threading.Event) -> None:
        while not done.wait(self.heartbeat):
            try:
                self.library.jobs.heartbeat(job_id)
            except Exception as exc:  # noqa: BLE001 —— 心跳偶尔写失败（数据库忙）不影响任务
                logger.warning("任务 %s 写心跳失败：%s", job_id, exc)


class WorkerProcess:
    """
    learn-rag serve 启动的 worker 子进程：意外退出时自动拉起（间隔逐次加长，连续失败太多次就放弃并在页面上提示），
    页面服务退出时一并关闭。日志写到 <数据目录>/worker.log，不和页面服务的输出混在一起。
    """

    def __init__(self, data_dir: str | Path, *, max_restarts: int = 5, stop_timeout: float = 15) -> None:
        self.data_dir = Path(data_dir)
        self.log_path = self.data_dir / "worker.log"
        self.max_restarts, self.stop_timeout = max_restarts, stop_timeout
        self.state, self.message, self.restarts = "stopped", "", 0
        self._process: subprocess.Popen[bytes] | None = None
        self._stopping = threading.Event()
        self._lock = threading.Lock()

    def start(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._stopping.clear()
        self._spawn()
        threading.Thread(target=self._supervise, name="worker-supervisor", daemon=True).start()

    def _spawn(self) -> None:
        with self._lock:
            if self._stopping.is_set():
                return
            log = open(self.log_path, "ab")
            cmd = [sys.executable, "-m", "learn_rag.cli", "worker", "--data", str(self.data_dir)]
            self._process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close()
            self.state, self.message = "running", f"worker 进程 {self._process.pid} 运行中，日志：{self.log_path}"

    def _supervise(self) -> None:
        while not self._stopping.is_set():
            process = self._process
            if process is None or process.poll() is None:
                self._stopping.wait(1)
                continue
            if self._stopping.is_set():
                return
            self.restarts += 1
            if self.restarts > self.max_restarts:
                self.state = "failed"
                self.message = (f"worker 进程连续退出 {self.max_restarts} 次，已停止拉起，建库任务不会执行；"
                                f"原因见 {self.log_path}，修复后重启 learn-rag serve")
                logger.error(self.message)
                return
            delay = min(60, 2 ** self.restarts)
            self.state = "restarting"
            self.message = f"worker 进程意外退出（退出码 {process.returncode}），{delay} 秒后重新启动；日志：{self.log_path}"
            logger.warning(self.message)
            if self._stopping.wait(delay):
                return
            self._spawn()

    def stop(self) -> None:
        """先请 worker 自己停（当前文件做完、任务放回队列），超时再强制结束。可以重复调用。"""
        self._stopping.set()
        with self._lock:
            process, self._process = self._process, None
        if process is None or process.poll() is not None:
            self.state = "stopped"
            return
        process.terminate()
        try:
            process.wait(self.stop_timeout)
        except subprocess.TimeoutExpired:
            # 正在解析一个大文件：强制结束，任务停在"执行中"，下次 worker 启动时发现进程已不在，立刻重新排队
            os.killpg(process.pid, signal.SIGKILL) if hasattr(os, "killpg") else process.kill()
            process.wait()
        self.state, self.message = "stopped", "worker 已停止"

    def status(self) -> dict[str, Any]:
        return {"state": self.state, "message": self.message, "restarts": self.restarts}


def run_worker(data_dir: str | Path) -> None:
    """learn-rag worker 的主循环：收到 SIGTERM / Ctrl+C 时做完当前文件，把任务放回队列再退出。"""
    from .library import Library

    worker = JobWorker(Library(data_dir))

    def _stop(*_: Any) -> None:
        logger.info("收到停止信号，当前文件处理完后退出")
        worker.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    worker.run()
