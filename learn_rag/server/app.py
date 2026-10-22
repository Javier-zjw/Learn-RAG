"""
server.app —— HTTP 接口。只做参数解析和错误码翻译，所有逻辑都在 Library 里。

    GET    /api/env                              模型、依赖、密钥是否配置，表单默认值
    POST   /api/uploads                          上传文件到暂存区（可多次调用，追加到同一个 upload_id）
    GET    /api/uploads/{upload_id}              暂存区里的文件
    DELETE /api/uploads/{upload_id}/files?path=  移除一个暂存文件
    POST   /api/uploads/{upload_id}/recommend    一键推荐参数
    GET    /api/kbs                              知识库列表
    POST   /api/kbs                              新建知识库并开始建库
    GET    /api/kbs/{kb_id}                      知识库详情（参数、文档、最近的任务）
    PATCH  /api/kbs/{kb_id}                      改名（名称只用于显示，不影响数据）
    PUT    /api/kbs/{kb_id}/retrieval            改精排模型和召回参数（只影响查询，不用重建）
    DELETE /api/kbs/{kb_id}
    POST   /api/kbs/{kb_id}/files                追加暂存区里的文件
    POST   /api/kbs/{kb_id}/rebuild              按新参数重建
    GET    /api/kbs/{kb_id}/documents/{doc_id}   一篇文档的全部分块
    DELETE /api/kbs/{kb_id}/documents/{doc_id}   删除文档（提交给 worker 执行，返回任务）
    GET    /api/kbs/{kb_id}/layout?doc=           原文预览：每页尺寸、每个子块在页面上的位置、未进入分块的文字
    GET    /api/kbs/{kb_id}/page?doc=&n=          第 n 页的页面图片
    GET    /api/kbs/{kb_id}/assets/{asset}       图片资产
    GET    /api/kbs/{kb_id}/metrics?days=        运行监控：查询的 P50 / P95、空召回率、错误率，建库耗时和最近的任务
    GET    /api/jobs/{job_id}                    任务快照
    GET    /api/jobs/{job_id}/events             任务进度（SSE，状态变化时推送快照，任务结束后关闭）

构建好的前端（web/dist）存在时一并托管；传入 MinerUService 时随服务启停 MinerU，传入 WorkerProcess 时
随服务启停 worker 进程（不传时在本进程起一个 worker 线程）。用户只需要启动这一个服务。
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from . import environment
from .library import Busy, Library, NotFound
from .mineru_service import MinerUService
from .worker import WorkerProcess
from .recommend import profile, recommend

_WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"


class CreateKb(BaseModel):
    name: str
    upload_id: str
    settings: dict[str, Any] = {}


class Rename(BaseModel):
    name: str


class Retrieval(BaseModel):
    reranker: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None


class AddFiles(BaseModel):
    upload_id: str


class Rebuild(BaseModel):
    settings: dict[str, Any] | None = None


def create_app(data_dir: str | Path = "data/web", mineru: MinerUService | None = None,
               worker: WorkerProcess | None = None) -> FastAPI:
    """
    mineru 不为空时，服务启动时在后台启动 MinerU，服务退出时把它一并关闭。
    worker 不为空时建库任务交给这个 worker 进程（随服务启停）；为空时在本进程里起一个 worker 线程。
    """
    library = Library(data_dir, worker=worker is None)
    library.recover()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if mineru:
            mineru.start()
        if worker:
            worker.start()
        yield
        if worker:
            await run_in_threadpool(worker.stop)
        if mineru:
            await run_in_threadpool(mineru.stop)

    app = FastAPI(title="Learn-RAG", docs_url="/api/docs", openapi_url="/api/openapi.json", lifespan=lifespan)
    app.state.library = library

    @app.exception_handler(NotFound)
    async def _not_found(_: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(LookupError)
    async def _lookup(_: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def _invalid(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(Busy)
    async def _busy(_: Request, exc: Busy) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    # ---------------------------------------------------------------- 环境
    @app.get("/api/env")
    def env() -> dict[str, Any]:
        running = {"state": "running", "message": "worker 线程在页面服务进程内运行", "restarts": 0}
        return {**environment.report(), "mineru": mineru.status() if mineru else None,
                "worker": worker.status() if worker else running}

    # ---------------------------------------------------------------- 上传暂存区
    @app.post("/api/uploads")
    def upload(files: list[UploadFile] = File(...), paths: list[str] = Form(...),
               upload_id: str | None = Form(None)) -> dict[str, Any]:
        if len(files) != len(paths):
            raise HTTPException(400, "files 和 paths 的数量必须一致")
        saved = []
        for file, path in zip(files, paths):
            upload_id, entry = library.staging.save(upload_id, path, file.file)
            saved.append(entry)
        return {"upload_id": upload_id, "files": saved}

    @app.get("/api/uploads/{upload_id}")
    def staged(upload_id: str) -> dict[str, Any]:
        return {"upload_id": upload_id, "files": library.staging.files(upload_id)}

    @app.delete("/api/uploads/{upload_id}/files")
    def unstage(upload_id: str, path: str) -> dict[str, Any]:
        library.staging.remove(upload_id, path)
        return {"upload_id": upload_id, "files": library.staging.files(upload_id)}

    @app.post("/api/uploads/{upload_id}/recommend")
    def recommend_settings(upload_id: str) -> dict[str, Any]:
        files = library.staging.paths(upload_id)
        if not files:
            raise HTTPException(400, "还没有可以解析的文件")
        stats = profile(files)
        return {**recommend(stats), "profile": stats}

    # ---------------------------------------------------------------- 知识库
    @app.get("/api/kbs")
    def list_kbs() -> list[dict[str, Any]]:
        return library.list()

    @app.post("/api/kbs")
    def create_kb(body: CreateKb) -> dict[str, Any]:
        return library.create(body.name, body.settings, body.upload_id)

    @app.get("/api/kbs/{kb_id}")
    def kb_info(kb_id: str) -> dict[str, Any]:
        return library.info(kb_id)

    @app.patch("/api/kbs/{kb_id}")
    def rename_kb(kb_id: str, body: Rename) -> dict[str, Any]:
        return library.rename(kb_id, body.name)

    @app.put("/api/kbs/{kb_id}/retrieval")
    def set_retrieval(kb_id: str, body: Retrieval) -> dict[str, Any]:
        return library.set_retrieval(kb_id, body.model_dump(exclude_none=True))

    @app.delete("/api/kbs/{kb_id}")
    def delete_kb(kb_id: str) -> dict[str, Any]:
        library.delete(kb_id)
        return {"ok": True}

    @app.post("/api/kbs/{kb_id}/files")
    def add_files(kb_id: str, body: AddFiles) -> dict[str, Any]:
        return library.add_files(kb_id, body.upload_id).snapshot()

    @app.post("/api/kbs/{kb_id}/rebuild")
    def rebuild(kb_id: str, body: Rebuild) -> dict[str, Any]:
        return library.rebuild(kb_id, body.settings).snapshot()

    @app.get("/api/kbs/{kb_id}/documents/{doc_id:path}")
    def document(kb_id: str, doc_id: str) -> dict[str, Any]:
        return library.document(kb_id, doc_id)

    @app.delete("/api/kbs/{kb_id}/documents/{doc_id:path}")
    def remove_document(kb_id: str, doc_id: str) -> dict[str, Any]:
        return library.remove_document(kb_id, doc_id).snapshot()

    @app.get("/api/kbs/{kb_id}/metrics")
    def metrics(kb_id: str, days: float = 7) -> dict[str, Any]:
        return library.metrics(kb_id, days)

    @app.get("/api/kbs/{kb_id}/layout")
    def layout(kb_id: str, doc: str) -> dict[str, Any]:
        return library.layout(kb_id, doc)

    @app.get("/api/kbs/{kb_id}/page")
    def page_image(kb_id: str, doc: str, n: int) -> FileResponse:
        # 页面图片按文件内容缓存，内容不变 URL 就不变，浏览器可以长期缓存
        return FileResponse(library.page_image(kb_id, doc, n), media_type="image/png",
                            headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/kbs/{kb_id}/assets/{asset:path}")
    def asset(kb_id: str, asset: str) -> FileResponse:
        return FileResponse(library.asset(kb_id, asset))

    # ---------------------------------------------------------------- 任务
    def _job(job_id: str):
        job = library.jobs.get(job_id)
        if job is None:
            raise NotFound(f"任务 {job_id} 不存在（知识库删除后任务记录一并删除，每个知识库只保留最近的任务）")
        return job

    @app.get("/api/jobs/{job_id}")
    def job_snapshot(job_id: str) -> dict[str, Any]:
        return _job(job_id).snapshot()

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        _job(job_id)

        async def stream():
            # 任务在 worker 进程里推进，这里定时从队列读最新快照，版本号变了才推送
            version = -1
            while True:
                job = await run_in_threadpool(library.jobs.get, job_id)
                if job is None:
                    return
                if job.version != version:
                    version = job.version
                    yield f"data: {json.dumps(job.snapshot(), ensure_ascii=False)}\n\n"
                if job.finished:
                    return
                await asyncio.sleep(0.3)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ---------------------------------------------------------------- 前端
    if _WEB_DIST.is_dir():
        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str) -> FileResponse:
            """静态文件直接返回，其余路径交给前端路由（刷新 /kb/xxx 也能打开）。"""
            target = (_WEB_DIST / path).resolve()
            if path and target.is_file() and _WEB_DIST.resolve() in target.parents:
                return FileResponse(target)
            if path.startswith("api/"):
                raise HTTPException(404, f"没有这个接口：/{path}")
            return FileResponse(_WEB_DIST / "index.html")

    return app
