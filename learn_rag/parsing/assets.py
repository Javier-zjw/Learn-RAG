"""
parsing.assets —— 图片资产的统一落盘。

解析器只负责"读懂文档"，图片二进制统一交给这里保存：按内容 SHA-256 命名、
前两位分桶，同一张图（公司 logo、重复插图）不管出现在多少份文档里都只存一份。
检索层永远只索引图注 / 描述文本，原图通过 Element.extra.asset（资产库内的
相对路径）按需取回，两边各自演进、互不拖累。

资产库目录写成相对路径时一律按项目根解析：解析器写入和命令行读取用的是同一条规则，
从哪个目录启动程序都找得到同一张图。
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

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


def resolve_assets_dir(assets_dir: str | Path) -> Path:
    """资产库目录：绝对路径原样使用，相对路径按项目根解析。"""
    path = Path(assets_dir)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def locate_asset(asset: str, assets_dirs: list[str | Path]) -> Path | None:
    """在若干资产库目录里找到一张图的文件（不同解析器可以配置不同的目录），找不到返回 None。"""
    for assets_dir in assets_dirs:
        path = resolve_assets_dir(assets_dir) / asset
        if path.is_file():
            return path
    return None


def configured_assets_dirs(parsing: dict | None) -> list[str]:
    """解析配置里各解析器的资产库目录（MinerU、Docling 可以各配一个）。"""
    options = (parsing or {}).get("options") or {}
    return [o["assets_dir"] for o in options.values() if isinstance(o, dict) and o.get("assets_dir")]
