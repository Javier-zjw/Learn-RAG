"""
server.uploads —— 上传暂存区。

文件先传到暂存区，扫描、推荐参数、用户确认之后才移进知识库。这样"一键推荐"能看到真实文件，
用户中途删掉某个文件或放弃新建时，知识库目录里也不会留下半成品。

每次上传会话一个目录（upload_id），浏览器传来的相对路径（选文件夹时带子目录）原样保留：
不同子目录里的同名文件不会互相覆盖，文档 id 也保留了目录信息。超过一天没用的暂存目录启动时清理。
"""

from __future__ import annotations

import re
import shutil
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from ..parsing.source import DEFAULT_PARSERS

_ID = re.compile(r"^[0-9a-f]{12}$")
_STALE_SECONDS = 24 * 3600


def safe_relative(path: str) -> str:
    """
    把浏览器传来的相对路径整理成安全的 POSIX 相对路径。
    含 .. 或绝对路径时报错：上传的路径决定写到磁盘哪里，不能让它跳出暂存目录。
    """
    parts = [p for p in PurePosixPath(path.replace("\\", "/")).parts if p not in ("", ".", "/")]
    if not parts or any(p == ".." for p in parts) or re.match(r"^[A-Za-z]:$", parts[0]):
        raise ValueError(f"不合法的文件路径：{path!r}")
    return "/".join(parts)


def describe(path: Path, rel: str) -> dict[str, Any]:
    suffix = path.suffix.lower()
    return {
        "path": rel,
        "size": path.stat().st_size,
        "type": suffix.lstrip("."),
        "supported": suffix in DEFAULT_PARSERS and not path.name.startswith(("~$", ".")),
    }


class Staging:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._clean_stale()

    def save(self, upload_id: str | None, rel_path: str, stream: BinaryIO) -> tuple[str, dict[str, Any]]:
        """保存一个文件，返回 (upload_id, 文件信息)。upload_id 为空时开一个新的上传会话。"""
        upload_id = upload_id or uuid.uuid4().hex[:12]
        rel = safe_relative(rel_path)
        target = self.directory(upload_id, create=True) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            shutil.copyfileobj(stream, fh)
        return upload_id, describe(target, rel)

    def files(self, upload_id: str) -> list[dict[str, Any]]:
        root = self.directory(upload_id)
        return [describe(p, p.relative_to(root).as_posix()) for p in sorted(root.rglob("*")) if p.is_file()]

    def paths(self, upload_id: str) -> list[Path]:
        """暂存区里能解析的文件。"""
        root = self.directory(upload_id)
        return [root / f["path"] for f in self.files(upload_id) if f["supported"]]

    def remove(self, upload_id: str, rel_path: str) -> None:
        target = self.directory(upload_id) / safe_relative(rel_path)
        target.unlink(missing_ok=True)

    def take(self, upload_id: str, dest: Path) -> list[str]:
        """把能解析的文件移进 dest（同名覆盖），删除暂存目录，返回移入的相对路径。"""
        root = self.directory(upload_id)
        moved = []
        for item in self.files(upload_id):
            if not item["supported"]:
                continue
            target = dest / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root / item["path"]), target)
            moved.append(item["path"])
        shutil.rmtree(root, ignore_errors=True)
        return moved

    def directory(self, upload_id: str, *, create: bool = False) -> Path:
        if not _ID.match(upload_id or ""):
            raise ValueError(f"不合法的上传编号：{upload_id!r}")
        path = self.root / upload_id
        if create:
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            raise LookupError(f"上传会话 {upload_id} 不存在或已过期，请重新选择文件")
        return path

    def _clean_stale(self) -> None:
        now = time.time()
        for path in self.root.iterdir():
            if path.is_dir() and now - path.stat().st_mtime > _STALE_SECONDS:
                shutil.rmtree(path, ignore_errors=True)
