"""
server —— 可视化建库的后端（FastAPI）。

页面上的操作都由 library.Library 完成，app.py 只是把它暴露成 HTTP 接口：
  上传（uploads）→ 一键推荐（recommend）→ 新建知识库 → 后台建库（jobs）→ 查看分块（chunk_view）。
fastapi / uvicorn 是可选依赖：pip install -e ".[web]"，只在启动服务时才导入。
"""

from __future__ import annotations

from .library import Busy, Library, NotFound

__all__ = ["Busy", "Library", "NotFound"]
