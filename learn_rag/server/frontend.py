"""
server.frontend —— 启动页面服务前确保前端已经构建好。

web/dist 不存在，或者 web/ 下的源码、依赖清单比它新时，自动 npm install（依赖有变化时）和 npm run build，
用户不用记前端命令，拉下新代码后直接 learn-rag serve 就是最新的页面。
没装 Node.js 时不中断：有旧的构建就继续用，没有就只提供接口并说明怎么装。
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[2] / "web"
_SOURCES = ("src", "index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json")


def ensure_built(web_dir: Path = WEB_DIR) -> str:
    """需要时构建前端，返回一句给用户看的状态。"""
    if not web_dir.is_dir():
        return "未找到 web/ 目录，只提供接口"
    built = web_dir / "dist" / "index.html"
    if built.is_file() and _newest(web_dir) <= built.stat().st_mtime:
        return "前端已是最新"
    npm = shutil.which("npm")
    if npm is None:
        if built.is_file():
            return "前端源码有更新，但没有安装 Node.js（npm），继续使用上次的构建"
        return "没有安装 Node.js（npm），无法构建前端，只提供接口。安装 Node.js 20+ 后重新运行即可"

    installed = web_dir / "node_modules" / ".package-lock.json"
    lock = web_dir / "package-lock.json"
    if not installed.is_file() or (lock.is_file() and lock.stat().st_mtime > installed.stat().st_mtime):
        logger.info("正在安装前端依赖（npm install，只在依赖变化时运行）……")
        error = _run([npm, "install", "--no-audit", "--no-fund"], web_dir)
        if error:
            return f"前端依赖安装失败：{error}"
    logger.info("正在构建前端（npm run build）……")
    error = _run([npm, "run", "build"], web_dir)
    if error:
        return f"前端构建失败{'，继续使用上次的构建' if built.is_file() else ''}：{error}"
    return "前端已重新构建"


def _newest(web_dir: Path) -> float:
    """源码和依赖清单里最新的修改时间。"""
    times = []
    for name in _SOURCES:
        path = web_dir / name
        if path.is_dir():
            times += [p.stat().st_mtime for p in path.rglob("*") if p.is_file()]
        elif path.is_file():
            times.append(path.stat().st_mtime)
    return max(times, default=0.0)


def _run(cmd: list[str], cwd: Path) -> str:
    """运行命令，成功返回空字符串，失败返回输出的最后几行。"""
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)
    if result.returncode == 0:
        return ""
    lines = [line for line in (result.stdout + result.stderr).splitlines() if line.strip()]
    return "\n".join(lines[-10:])
