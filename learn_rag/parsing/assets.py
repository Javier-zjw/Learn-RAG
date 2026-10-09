"""
parsing.assets —— 图片资产的统一落盘。

解析器只负责"读懂文档"，图片二进制统一交给这里保存：按内容 SHA-256 命名、
前两位分桶，同一张图（公司 logo、重复插图）不管出现在多少份文档里都只存一份。
检索层永远只索引图注 / 描述文本，原图通过 Element.extra.asset（资产库内的
相对路径）按需取回，两边各自演进、互不拖累。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tiff": "image/tiff",
}


def save_asset(data: bytes, assets_dir: str | Path, *, ext: str, mime: str = "") -> dict[str, str]:
    """保存一份二进制资产，返回 {"asset": 相对路径, "sha256": ..., "mime": ...}。

    同一内容天然去重（哈希即文件名）；临时文件带进程号，多进程同时写同一张图
    互不覆盖，最后用原子 rename 落位。
    """
    sha = hashlib.sha256(data).hexdigest()
    rel = f"{sha[:2]}/{sha}.{ext}"
    path = Path(assets_dir) / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return {"asset": rel, "sha256": sha, "mime": mime}


def mime_for(suffix: str) -> str:
    """按文件后缀猜 MIME 类型；未知类型给通用值，不猜空。"""
    return _MIME_BY_SUFFIX.get(suffix.lower(), "application/octet-stream")
