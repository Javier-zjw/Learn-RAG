"""
server.mineru_service —— 页面服务启动时一并启动本机 MinerU 的 VLM 服务，退出时一并关闭。

启停直接复用 scripts/start_mineru.sh 和 scripts/stop_mineru.sh：它们已经处理好 .mineru.env、健康检查、
进程号文件和残留的文档库服务，这里不重写一遍，只负责"什么时候调用"和"把状态告诉页面"。

VLM 服务第一次启动要加载模型（CPU 上要几十秒），所以放在后台线程里启动，页面服务立刻可用；
建库任务要用 MinerU 时先等它就绪（Library 调用 wait），不会因为服务还没起来就白白退回 PyMuPDF。
退出时不管 VLM 服务是这次启动的还是之前手动启动的，都一并关闭。没有 .mineru.env 的机器不启动，也不报错。
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path
from typing import Any

from ..parsing.external import read_mineru_env

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 状态：没配置 / 启动中 / 已就绪 / 启动失败 / 已停止
DISABLED, STARTING, READY, FAILED, STOPPED = "disabled", "starting", "ready", "failed", "stopped"


class MinerUService:
    def __init__(
            self,
            env_file: Path = _PROJECT_ROOT / ".mineru.env",
            start_script: Path = _PROJECT_ROOT / "scripts" / "start_mineru.sh",
            stop_script: Path = _PROJECT_ROOT / "scripts" / "stop_mineru.sh",
            timeout: int = 180,
    ) -> None:
        self.env_file, self.start_script, self.stop_script = Path(env_file), Path(start_script), Path(stop_script)
        self.timeout = timeout
        self.state = DISABLED
        self.message = "未找到 .mineru.env，不启动 MinerU；PDF 用 PyMuPDF 解析"
        self._process: subprocess.Popen[str] | None = None
        self._done = threading.Event()
        self._done.set()
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """在后台启动 VLM 服务，立即返回。"""
        if not self.env_file.is_file():
            return
        if shutil.which("bash") is None:
            self.state, self.message = FAILED, "找不到 bash，无法运行 scripts/start_mineru.sh"
            return
        self.state, self.message = STARTING, "正在启动 MinerU VLM 服务（首次加载模型需要几十秒）"
        self._started = True
        self._done.clear()
        threading.Thread(target=self._run_start, name="mineru-start", daemon=True).start()
        logger.info(self.message)

    def _run_start(self) -> None:
        try:
            # 和 stop 用同一把锁：stop 先到时不再启动；启动后 stop 一定能看到这个进程
            with self._lock:
                if self.state != STARTING:
                    return
                # 单独的进程组：启动途中退出时连同脚本里还在等待的子进程一起结束
                self._process = subprocess.Popen(["bash", str(self.start_script)], stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT, text=True, start_new_session=True)
            output, _ = self._process.communicate(timeout=self.timeout)
            lines = [line for line in (output or "").splitlines() if line.strip()]
            if self.state != STARTING:          # 启动过程中已经被 stop 了
                return
            if self._process.returncode == 0:
                self.state, self.message = READY, f"MinerU VLM 服务已就绪：{self.url or ''}".rstrip("：")
                logger.info(self.message)
            else:
                self.state = FAILED
                self.message = "MinerU 启动失败，PDF 将退回 PyMuPDF 解析：" + "\n".join(lines[-8:])
                logger.error(self.message)
        except subprocess.TimeoutExpired:
            self._kill_start_script()
            self.state, self.message = FAILED, f"MinerU 启动脚本 {self.timeout} 秒内没有结束，已放弃等待"
            logger.error(self.message)
        except OSError as exc:
            self.state, self.message = FAILED, f"运行 MinerU 启动脚本失败：{exc}"
            logger.error(self.message)
        finally:
            self._done.set()

    def wait(self, timeout: float | None = None) -> bool:
        """等启动结束，返回是否就绪。没配置 MinerU 时立即返回 False。"""
        self._done.wait(timeout)
        return self.state == READY

    def stop(self) -> None:
        """关闭 VLM 服务和残留的文档库服务。可以重复调用，只有启动过才会真正执行。"""
        with self._lock:
            if not self._started:
                return
            self._started = False
            self._kill_start_script()            # 启动脚本可能还在等服务就绪
            self.state, self.message = STOPPED, "MinerU 已停止"
            try:
                result = subprocess.run(["bash", str(self.stop_script)], capture_output=True, text=True, timeout=60)
                output = (result.stdout + result.stderr).strip()
                logger.info("已关闭 MinerU：%s", output.replace("\n", "；"))
            except (OSError, subprocess.TimeoutExpired) as exc:
                logger.error("关闭 MinerU 失败，请手动运行 bash scripts/stop_mineru.sh：%s", exc)

    def _kill_start_script(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        try:
            os.killpg(self._process.pid, signal.SIGTERM)
        except (OSError, AttributeError):   # 进程已经结束，或者系统没有进程组
            self._process.terminate()

    @property
    def url(self) -> str:
        return read_mineru_env(str(self.env_file)).get("MINERU_MODEL_VLM_SERVER_URL", "")

    def status(self) -> dict[str, Any]:
        return {"state": self.state, "message": self.message, "url": self.url if self.state != DISABLED else ""}
