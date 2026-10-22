"""
server.library —— 知识库管理。页面上的每个操作（新建、追加文件、重建、删除文档、查看分块）都落到这里。

一个知识库就是一个目录，参数在新建时定下：
    <数据目录>/kbs/<id>/
        kb.json      清单：名称、参数、每个文件的状态和片段数、最近一次完整性检查
        files/       上传的原文件（保留浏览器传来的相对路径，也就是文档 id）
        index/       片段库：Chroma 在 index/chroma，flat 直接存在 index/

为什么每个知识库的参数固定：同一个库里混用 embedding 模型会被知识库拒绝，不同 chunk_size 的块混在一起
也没法比较效果。改参数就是"按新参数重建"：清空片段库，用解析缓存重新切分、向量化。

建库本身完全复用 KnowledgeBase：增量跳过、模型一致性检查、完整性核对都在那里，这里只负责
把页面上的参数翻译成配置、逐个文件推进并记录状态。知识库 id 是随机生成的，删掉再建同名知识库
也不会复用旧目录（Chroma 在同一进程里会缓存打开过的路径）。

写知识库的操作（建库、追加、重建、删除文档）都变成任务提交到队列（jobs.py），由 worker 执行（run_job）；
页面服务和 worker 可以是两个进程：
  - 清单 kb.json 的读改写用跨进程的文件锁保护（_KbLock）；
  - 片段库每次被写入，清单里的 revision 加一，读的一方发现 revision 变了就重新打开知识库 ——
    另一个进程写进 Chroma 的数据，本进程已打开的对象看不到（派生的 BM25、文档清单也是旧的）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:      # Windows
    fcntl = None

from ..core.config import load_config
from ..core.registry import registry
from ..parsing.assets import configured_assets_dirs, locate_asset
from ..parsing.source import FileSource
from ..core.types import RagResult
from ..pipeline.rag import RagPipeline
from ..pipeline.trace import QueryLog, percentile
from . import environment
from .chunk_view import document_view
from ..core.types import ENRICH_KEY
from ..parsing.external import mineru_ready, mineru_server_url
from .preview import Preview, PreviewUnavailable
from .jobs import (DONE, EMBEDDING, FAILED, PARSING, REMOVING, SKIPPED, WAITING, FileProgress, Job, JobStore,
                   capture_warnings)
from .uploads import Staging, safe_relative
from .worker import Interrupted, JobWorker

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 页面没有覆盖的配置（召回、精排、MinerU 选项、解析缓存目录）沿用这两份
_BASE_CONFIGS = [_PROJECT_ROOT / "configs" / "default.yaml", _PROJECT_ROOT / "configs" / "parsing.yaml"]
_KB_ID = re.compile(r"^[0-9a-f]{8}$")
_CHUNKER_FIELDS = {
    "structure": ("chunk_size", "parent_size", "overlap_sentences"),
    "recursive": ("chunk_size", "chunk_overlap"),
    "fixed": ("chunk_size", "chunk_overlap"),
}
_INDEX_FIELDS = ("ef_construction", "max_neighbors", "ef_search")
CHANNELS = ("vector", "bm25", "sparse")      # 可以参与融合的召回通道，和 retriever 注册名一致
_UNFINISHED = (WAITING, PARSING, EMBEDDING)


class NotFound(LookupError):
    """知识库、文档或资产不存在。"""


class Busy(RuntimeError):
    """知识库正在建库，暂时不能做这个操作。"""


class Library:
    def __init__(self, root: str | Path, *, worker: bool = False, mineru_wait: float = 180) -> None:
        """
        worker=True 时在本进程里起一个 worker 线程执行任务（测试、脚本、不想多开进程时用）；
        默认不起：任务由 learn-rag worker 进程执行（learn-rag serve 会启动它）。
        命令行评测、调参打开 Library 只是为了读，千万不能顺手执行队列里的任务。
        mineru_wait：任务要用 MinerU 而服务还没就绪时最多等多少秒。
        """
        self.root = Path(root)
        self.kbs_dir = self.root / "kbs"
        self.kbs_dir.mkdir(parents=True, exist_ok=True)
        self.staging = Staging(self.root / "uploads")
        self.preview = Preview(self.root / "preview")      # 按文件内容缓存，多个知识库共用
        self.jobs = JobStore(self.root / "jobs.db")
        self.mineru_wait = mineru_wait
        self._digests: dict[tuple[str, int, int], str] = {}
        self._pipes: dict[str, tuple[str, RagPipeline]] = {}   # kb_id -> (参数和 revision, 管线)
        self._locks: dict[str, _KbLock] = {}
        self._guard = threading.Lock()
        self.worker: JobWorker | None = None
        if worker:
            self.worker = JobWorker(self)
            self.worker.start()

    # ------------------------------------------------------------------ 知识库
    def pipeline(self, kb_id: str) -> RagPipeline:
        """知识库的完整管线（召回、精排、生成），评测和以后的提问页用它；和建库用的是同一个对象。"""
        with self._lock(kb_id):
            return self._pipeline(kb_id)

    def list(self) -> list[dict[str, Any]]:
        summaries = [self._summary(m) for m in self._manifests()]
        return sorted(summaries, key=lambda s: s["updated_at"], reverse=True)

    def create(self, name: str, settings: dict[str, Any], upload_id: str) -> dict[str, Any]:
        settings = self.validate(settings)
        kb_id = uuid.uuid4().hex[:8]
        directory = self.kbs_dir / kb_id
        (directory / "files").mkdir(parents=True)
        files = self.staging.take(upload_id, directory / "files")
        if not files:
            shutil.rmtree(directory, ignore_errors=True)
            raise ValueError("没有可以解析的文件：请上传 PDF、Word、PPT、Excel、Markdown、网页等支持的格式")
        now = time.time()
        manifest = {
            "id": kb_id, "name": _clean_name(name), "settings": settings,
            "created_at": now, "updated_at": now, "verify": None,
            "files": {path: _file_entry(directory / "files" / path) for path in files},
        }
        self._write_manifest(manifest)
        job = self.jobs.submit(kb_id, "build", files)
        return {"kb": self.info(kb_id), "job": job.snapshot()}

    def info(self, kb_id: str) -> dict[str, Any]:
        manifest = self._manifest(kb_id)
        job = self.jobs.latest(kb_id)
        live = {f.path: f for f in job.files} if job and not job.finished else {}
        documents = []
        for path, entry in sorted(manifest["files"].items()):
            row = {"doc_id": path, "type": Path(path).suffix.lower().lstrip("."), **entry}
            if path in live:   # 正在处理的文件用任务里的实时状态
                row["status"], row["message"] = live[path].status, live[path].message
            documents.append(row)
        return {**self._summary(manifest), "documents": documents, "verify": manifest.get("verify"),
                "job": job.snapshot() if job else None}

    def rename(self, kb_id: str, name: str) -> dict[str, Any]:
        """
        改名。名称只用于显示：目录、Chroma 集合、文档 id 都不含名称，改名不影响任何数据，建库过程中也可以改。
        """
        with self._lock(kb_id):
            manifest = self._manifest(kb_id)
            manifest["name"] = _clean_name(name)
            self._write_manifest(manifest)
        return self._summary(manifest)

    def delete(self, kb_id: str) -> None:
        manifest = self._manifest(kb_id)
        if self.jobs.busy(kb_id):
            raise Busy("这个知识库正在建库，完成后再删除")
        with self._lock(kb_id):
            self._pipes.pop(kb_id, None)
            if manifest["settings"]["index"]["type"] == "chroma":
                # 先清空 Chroma 集合再删目录：Chroma 在进程里还开着这些文件
                registry.build("index", self._config(kb_id, manifest["settings"], reset=True)["index"])
            shutil.rmtree(self.kbs_dir / kb_id, ignore_errors=True)
        self.jobs.forget(kb_id)

    def add_files(self, kb_id: str, upload_id: str) -> Job:
        """追加文件，沿用知识库的参数；同名文件覆盖旧版本（知识库按指纹识别出变化，替换旧片段）。"""
        manifest = self._manifest(kb_id)
        files = self.staging.take(upload_id, self.kbs_dir / kb_id / "files")
        if not files:
            raise ValueError("没有可以解析的文件")
        with self._lock(kb_id):
            manifest = self._manifest(kb_id)
            for path in files:
                manifest["files"][path] = _file_entry(self.kbs_dir / kb_id / "files" / path)
            self._write_manifest(manifest)
        return self.jobs.submit(kb_id, "append", files)

    def rebuild(self, kb_id: str, settings: dict[str, Any] | None = None) -> Job:
        """
        按新参数（不传则按原参数）重新处理全部文件。解析结果有缓存，不会重新解析。

        换了 embedding 模型或索引参数时清空片段库重建：两个模型的向量不能混在一起，HNSW 参数只在建集合时生效。
        只改切分参数时不清空：文档指纹变了，知识库会逐篇替换旧片段；参数没变时未变化的文档直接跳过，
        相当于续写上次没完成的建库，不会重复调用 embedding。
        """
        if self.jobs.busy(kb_id):
            raise Busy("这个知识库正在建库，完成后再重建")
        with self._lock(kb_id):
            manifest = self._manifest(kb_id)
            old = manifest["settings"]
            new = self.validate(settings) if settings is not None else old
            reset = (new["encoder"], new["index"]) != (old["encoder"], old["index"])
            manifest["settings"] = new
            for entry in manifest["files"].values():
                entry.update(status=WAITING, message="")
            self._write_manifest(manifest)
        return self.jobs.submit(kb_id, "rebuild", sorted(manifest["files"]), reset=reset)

    # ------------------------------------------------------------------ 文档
    def remove_document(self, kb_id: str, doc_id: str) -> Job:
        """删除文档也是写片段库，和建库一样交给 worker 执行，返回任务。"""
        manifest = self._manifest(kb_id)
        if doc_id not in manifest["files"]:
            raise NotFound(f"知识库里没有文档 {doc_id}")
        if self.jobs.busy(kb_id):
            raise Busy("这个知识库正在建库，完成后再删除文档")
        return self.jobs.submit(kb_id, "remove", [doc_id])

    def document(self, kb_id: str, doc_id: str) -> dict[str, Any]:
        manifest = self._manifest(kb_id)
        with self._lock(kb_id):
            chunks = self._pipeline(kb_id).kb.chunks_of(doc_id)
        if not chunks:
            raise NotFound(f"知识库里没有文档 {doc_id} 的片段（可能解析失败或还在处理）")
        chunker = manifest["settings"]["chunker"]
        view = document_view(chunks, overlap=chunker.get("overlap_sentences", 0) > 0)
        return {**view, "kb_id": kb_id, "kb_name": manifest["name"], "settings": manifest["settings"],
                "file": manifest["files"].get(doc_id, {})}

    def layout(self, kb_id: str, doc_id: str) -> dict[str, Any]:
        """
        原文预览：每页尺寸、每个子块在页面上的位置、没有进入任何分块的文字（见 preview.py）。
        不能预览时（Markdown、CSV 等没有版面，或缺少 LibreOffice）返回 available=False 和原因，页面退回文本视图。
        """
        file = self._source_file(kb_id, doc_id)
        with self._lock(kb_id):
            chunks = self._pipeline(kb_id).kb.chunks_of(doc_id)
        if not chunks:
            raise NotFound(f"知识库里没有文档 {doc_id} 的片段")
        children = document_view(chunks)["children"]
        targets = [{"id": c["id"], "text": c["body"], "regions": c["metadata"].get("regions") or []} for c in children]
        headings = list(dict.fromkeys(h for c in children for h in str(c["section"]).split(" > ") if h))
        try:
            digest = self._digest(file)
            # version 跟着文件内容变，页面图片的 URL 带上它，同名文件被替换后浏览器不会用旧图
            return {"available": True, "version": digest[:12], **self.preview.layout(file, digest, targets, headings)}
        except PreviewUnavailable as exc:
            return {"available": False, "reason": str(exc)}

    def page_image(self, kb_id: str, doc_id: str, page: int) -> Path:
        file = self._source_file(kb_id, doc_id)
        try:
            return self.preview.page_image(file, self._digest(file), page)
        except PreviewUnavailable as exc:
            raise NotFound(str(exc)) from None

    def _source_file(self, kb_id: str, doc_id: str) -> Path:
        self._manifest(kb_id)
        file = self.kbs_dir / kb_id / "files" / safe_relative(doc_id)
        if not file.is_file():
            raise NotFound(f"知识库里没有文件 {doc_id}")
        return file

    def _digest(self, file: Path) -> str:
        """文件内容哈希，按路径、修改时间和大小缓存：翻页时每张图都要用到它，不能每次都把整个文件读一遍。"""
        stat = file.stat()
        key = (str(file), stat.st_mtime_ns, stat.st_size)
        if key not in self._digests:
            self._digests[key] = hashlib.sha256(file.read_bytes()).hexdigest()
        return self._digests[key]

    # ------------------------------------------------------------------ 问答与监控
    def ask(self, kb_id: str, question: str, *, history: list[dict[str, str]] | None = None,
            source: str = "chat") -> RagResult:
        """
        线上问答的统一入口（命令行 chat，以后的提问页）：答完记一条 trace，出错也记，再把异常抛给调用方。
        评测、调参直接用 pipeline()，不记 trace —— 它们一次跑几百题，会把线上统计冲掉。
        注意：worker 在另一个进程写过的 Chroma，本进程已打开的 HNSW 索引看不到新向量，
        所以问答要在新进程里做（命令行），或者等以后页面问答时在这里处理。
        """
        log = self._query_log(kb_id)
        started = time.perf_counter()
        try:
            result = self.pipeline(kb_id).answer(question, history=history)
        except Exception as exc:
            log.record(question, error=str(exc), source=source, seconds=time.perf_counter() - started)
            raise
        log.record(question, result, source=source)
        return result

    def metrics(self, kb_id: str, days: float = 7) -> dict[str, Any]:
        """
        运行监控：最近 days 天的查询统计（P50 / P95、空召回率、错误率），和建库统计
        （每个文件解析、入库的耗时分布，最近任务的排队时间、执行时间、重试次数、失败文件数）。
        """
        manifest = self._manifest(kb_id)
        stages: dict[str, list[float]] = {}
        for entry in manifest["files"].values():
            for stage, value in (entry.get("seconds") or {}).items():
                stages.setdefault(stage, []).append(value)
        jobs = []
        for job in self.jobs.history(kb_id, limit=10):
            started, finished = job.started_at, job.finished_at
            jobs.append({
                "id": job.id, "kind": job.kind, "status": job.status, "attempts": job.attempts,
                "files": len(job.files), "failed": sum(1 for f in job.files if f.status == FAILED),
                "created_at": job.created_at, "queued_seconds": round(started - job.created_at, 2) if started else None,
                "run_seconds": round(finished - started, 2) if started and finished else None,
            })
        return {
            "queries": self._query_log(kb_id).summary(days=days),
            "build": {
                "files": len(manifest["files"]),
                "stages": {stage: {"p50": percentile(v, 50), "p95": percentile(v, 95), "max": max(v), "total": sum(v)}
                           for stage, v in sorted(stages.items())},
                "jobs": jobs,
            },
        }

    def _query_log(self, kb_id: str) -> QueryLog:
        self._manifest(kb_id)          # 知识库不存在时报 NotFound
        return QueryLog(self.kbs_dir / kb_id / "traces")

    def asset(self, kb_id: str, asset: str) -> Path:
        manifest = self._manifest(kb_id)
        parsing = self._config(kb_id, manifest["settings"])["parsing"]
        path = locate_asset(safe_relative(asset), configured_assets_dirs(parsing))
        if path is None:
            raise NotFound(f"资产库里没有 {asset}")
        return path

    # ------------------------------------------------------------------ 参数
    def validate(self, settings: dict[str, Any] | None) -> dict[str, Any]:
        """补齐缺省值并检查参数，不合法时抛 ValueError（消息直接显示在页面上）。"""
        base = environment.default_settings()
        settings = settings or {}
        chunker = {**base["chunker"], **(settings.get("chunker") or {})}
        kind = chunker.get("type")
        if kind not in _CHUNKER_FIELDS:
            raise ValueError(f"不支持的切分器：{kind}")
        if kind != "structure" and "chunk_overlap" not in chunker:
            chunker["chunk_overlap"] = 0
        try:
            chunker = {"type": kind, **{k: int(chunker[k]) for k in _CHUNKER_FIELDS[kind]}}
            registry.build("chunker", chunker)   # 切分器自己会检查参数之间的约束
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"切分参数不合法：{exc}") from None
        if chunker["chunk_size"] < 20:
            raise ValueError("子块大小至少 20")

        # 不和默认值合并：换了模型类型却没给模型名时，要用这个类型自己的默认模型，而不是沿用默认类型的模型名
        encoder = settings.get("encoder") or base["encoder"]
        option = next((e for e in environment.encoders() if e["type"] == encoder.get("type")), None)
        if option is None or not option["available"]:
            raise ValueError(f"embedding 模型不可用：{option['note'] if option else encoder.get('type')}")
        encoder = {"type": option["type"], "model": str(encoder.get("model") or option["model"])}

        index = {**base["index"], **(settings.get("index") or {})}
        if index.get("type") == "chroma":
            if not environment.has_chroma():
                raise ValueError("未安装 chromadb，只能选 flat 索引")
            if index.get("space") not in ("cosine", "l2", "ip"):
                raise ValueError(f"不支持的距离度量：{index.get('space')}")
            index = {"type": "chroma", "space": index["space"], **{k: int(index[k]) for k in _INDEX_FIELDS}}
        elif index.get("type") == "flat":
            index = {"type": "flat"}
        else:
            raise ValueError(f"不支持的索引类型：{index.get('type')}")

        choices = environment.parser_choices()
        parsers = {**base["parsers"], **(settings.get("parsers") or {})}
        for suffix, chain in parsers.items():
            allowed = [c["value"] for c in choices.get(suffix, []) if c["available"]]
            if list(chain) not in allowed:
                raise ValueError(f"{suffix} 的解析器 {chain} 不可用")
        return {"chunker": chunker, "encoder": encoder, "index": index,
                "parsers": {k: list(v) for k, v in parsers.items()}, "ocr": bool(settings.get("ocr", False)),
                "enrich": _validate_enrich(settings.get("enrich")),
                **self._validate_retrieval(settings, base, encoder)}

    def _validate_retrieval(self, settings: dict[str, Any], base: dict[str, Any],
                            encoder: dict[str, Any]) -> dict[str, Any]:
        """精排模型、召回候选数、交给大模型的片段数、各路召回的权重、RRF 常数、查询改写。"""
        ranker = settings.get("reranker") or base["reranker"]
        option = next((r for r in environment.rerankers() if r["type"] == ranker.get("type")), None)
        if option is None or not option["available"]:
            raise ValueError(f"精排模型不可用：{option['note'] if option else ranker.get('type')}")
        retrieval = {**base["retrieval"], **(settings.get("retrieval") or {})}
        try:
            top_k, candidate_k, rrf_k = int(retrieval["top_k"]), int(retrieval["candidate_k"]), int(retrieval["rrf_k"])
            weights = {name: float((retrieval.get("weights") or {}).get(name, 0.0)) for name in CHANNELS}
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"检索参数不合法：{retrieval}") from None
        if not 1 <= top_k <= 20:
            raise ValueError("交给大模型的片段数要在 1～20 之间")
        if not top_k <= candidate_k <= 200:
            raise ValueError("召回候选数要不少于交给大模型的片段数，且不超过 200")
        if not 1 <= rrf_k <= 1000:
            raise ValueError("RRF 常数 k 要在 1～1000 之间")
        if any(w < 0 or w > 10 for w in weights.values()) or not any(weights.values()):
            raise ValueError("各路召回的权重要在 0～10 之间，且至少有一路大于 0")
        if weights["sparse"] and not _supports_sparse(encoder):
            raise ValueError("稀疏召回需要 embedding 选“BGE-M3（稠密 + 稀疏）”：稀疏向量在建库时和稠密向量一起生成")
        transform = str(retrieval.get("query_transform") or "none")
        choice = next((t for t in environment.query_transforms() if t["type"] == transform), None)
        if choice is None or not choice["available"]:
            raise ValueError(f"查询改写不可用：{choice['note'] if choice else transform}")
        return {"reranker": {"type": option["type"], "model": str(ranker.get("model") or option["model"])},
                "retrieval": {"candidate_k": candidate_k, "top_k": top_k, "weights": weights, "rrf_k": rrf_k,
                              "query_transform": transform}}

    def set_retrieval(self, kb_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        """
        改精排模型和召回参数。它们只影响查询，不影响存进去的向量和分块，所以直接生效，不用重建。
        """
        with self._lock(kb_id):
            manifest = self._manifest(kb_id)
            current = _with_defaults(manifest["settings"])
            manifest["settings"] = {**current, **self._validate_retrieval(settings, environment.default_settings(),
                                                                          current["encoder"])}
            self._write_manifest(manifest)
        return self._summary(manifest)

    def _config(self, kb_id: str, settings: dict[str, Any], *, reset: bool = False) -> dict[str, Any]:
        """页面参数 → 完整配置。页面没有的部分沿用 configs/default.yaml 和 configs/parsing.yaml。"""
        cfg = load_config(*_BASE_CONFIGS)
        index = dict(settings["index"])
        if index["type"] == "chroma":
            index.update(path=str(self.kbs_dir / kb_id / "index" / "chroma"), collection="kb", reset=reset)
        parsing = dict(cfg.get("parsing") or {})
        cache_dir = Path(parsing.get("cache_dir") or ".cache/parsed")
        parsing["cache_dir"] = str(cache_dir if cache_dir.is_absolute() else _PROJECT_ROOT / cache_dir)
        parsing["parsers"] = {**(parsing.get("parsers") or {}), **settings["parsers"]}
        parsing.pop("ocr", None)
        if settings.get("ocr"):
            parsing["ocr"] = {"type": "vlm_ocr"}
        settings = _with_defaults(settings)
        retrieval = settings["retrieval"]
        cfg.update(chunker=dict(settings["chunker"]), encoder=_encoder_config(settings["encoder"]),
                   index=index, parsing=parsing, reranker=_reranker_config(settings["reranker"]),
                   retriever=_retriever_config(retrieval),
                   pipeline={**(cfg.get("pipeline") or {}), "candidate_k": retrieval["candidate_k"],
                             "top_k": retrieval["top_k"]})
        if environment.has_llm():
            # 有大模型时：回答由大模型生成，多轮对话的追问先补全再检索，查询改写也可以开。
            # 地址和密钥由 LLM 客户端从环境变量读取，不写进 kb.json
            cfg["llm"] = load_config(_PROJECT_ROOT / "configs" / "models_env.yaml")["llm"]
            cfg["generator"] = {"type": "stuff"}
            cfg["pipeline"]["condense"] = True
        if _enrich_on(settings["enrich"]):
            cfg["enrich"] = _enrich_config(settings["enrich"], parsing)
        return cfg

    # ------------------------------------------------------------------ 建库任务（worker 里执行）
    def run_job(self, job: Job, *, should_stop: Callable[[], bool] = lambda: False) -> None:
        """
        执行一个任务：逐个处理还在"等待"的文件（重试时只剩上次没完成的），最后落盘并核对完整性。
        should_stop() 为真时在两个文件之间停下（抛 Interrupted），任务放回队列，下次接着做。
        """
        if job.kind == "remove":
            self._remove(job)
            return
        kb_id = job.kb_id
        with self._lock(kb_id):
            pipe = self._pipeline(kb_id, reset=job.reset)
            cfg = self._config(kb_id, self._manifest(kb_id)["settings"])
        if job.reset:              # 已经清空过了：中断后重试不能再清空一遍，否则做完的又要重做
            job.reset = False
            job.touch()
        self._wait_for_mineru(job, cfg, should_stop)
        files_dir = self.kbs_dir / kb_id / "files"
        source = FileSource(str(files_dir), **cfg["parsing"])
        for progress in job.files:
            if progress.status != WAITING:
                continue
            if should_stop():
                raise Interrupted()
            info = self._process(job, progress, pipe, source, files_dir)
            self._record(kb_id, progress, info)
        with self._lock(kb_id):
            pipe.kb.save(str(self._persist_dir(kb_id)))
            chunks = pipe.kb.stats()["chunks"]
            job.verify = pipe.kb.verify(sample=min(10, chunks))
            manifest = self._manifest(kb_id)
            manifest["verify"] = job.verify
            manifest["updated_at"] = time.time()
            self._data_changed(manifest)
        job.touch()

    def _remove(self, job: Job) -> None:
        kb_id, progress = job.kb_id, job.files[0]
        progress.status = REMOVING
        job.touch()
        with self._lock(kb_id):
            pipe = self._pipeline(kb_id)
            pipe.kb.remove([progress.path])
            pipe.kb.save(str(self._persist_dir(kb_id)))
            (self.kbs_dir / kb_id / "files" / safe_relative(progress.path)).unlink(missing_ok=True)
            manifest = self._manifest(kb_id)
            manifest["files"].pop(progress.path, None)
            manifest["updated_at"] = time.time()
            self._data_changed(manifest)
        progress.status = DONE
        job.touch()

    def _wait_for_mineru(self, job: Job, cfg: dict[str, Any], should_stop: Callable[[], bool]) -> None:
        """
        要用 MinerU 解析而它的服务还没就绪（learn-rag serve 刚启动、模型还在加载）时先等一会儿，
        否则这批 PDF 会全部退回 PyMuPDF。worker 可能在另一个进程，所以用 HTTP 健康检查判断，而不是看内存里的状态。
        """
        url = mineru_server_url()
        uses_mineru = any("mineru" in chain for chain in cfg["parsing"]["parsers"].values())
        if not (url and uses_mineru) or mineru_ready(url, cache=0):
            return
        job.notice = "等待 MinerU 服务就绪（首次启动要加载模型）……"
        job.touch()
        deadline = time.time() + self.mineru_wait
        while time.time() < deadline and not mineru_ready(url, cache=0):
            if should_stop():
                raise Interrupted()
            time.sleep(2)
        ready = mineru_ready(url, cache=0)
        job.notice = "" if ready else f"MinerU 服务 {self.mineru_wait:.0f} 秒内没有就绪，PDF 退回 PyMuPDF 解析"
        job.touch()

    def _process(self, job: Job, progress: FileProgress, pipe: RagPipeline, source: FileSource,
                 files_dir: Path) -> dict[str, Any]:
        """
        处理一个文件，失败只记在这个文件上，不中断整批。返回要记进清单的解析信息。
        向量化失败、增强没完成的文件标记 retry，worker 稍后重试；解析失败不重试（解析器已经逐个降级过）。
        """
        progress.status, progress.retry, progress.seconds = PARSING, False, {}
        job.touch()
        started = time.perf_counter()
        with capture_warnings() as notes:
            try:
                document = source.load_file(files_dir / progress.path)
            except Exception as exc:   # load_file 自己不抛异常，这里防的是意料之外的错误
                notes.append(str(exc))
                document = None
        progress.seconds["parse"] = round(time.perf_counter() - started, 3)
        if document is None:
            progress.status, progress.message = FAILED, "\n".join(notes) or "没有解析出任何内容"
            job.touch()
            return {"seconds": progress.seconds}
        info = {"parser": document.metadata.get("parser", ""),
                "pages": max((e.extra.get("page_end") or e.page or 0 for e in document.elements), default=0) or None}

        progress.status = EMBEDDING
        job.touch()
        started = time.perf_counter()
        with capture_warnings() as notes, self._lock(job.kb_id):
            stats = pipe.kb.add([document])
            chunks = pipe.kb.chunks_of(document.doc_id)
        progress.seconds["index"] = round(time.perf_counter() - started, 3)
        info["seconds"] = progress.seconds
        parent_ids = {c.metadata.get("parent_id") for c in chunks} - {None, ""}
        unfinished = sum(1 for c in chunks if (c.metadata.get(ENRICH_KEY) or {}).get("failed"))
        if stats["failed"]:
            progress.status, progress.retry = FAILED, True
            progress.message = "\n".join(notes) or "向量化失败"
        else:
            progress.status = SKIPPED if stats["skipped"] else DONE
            progress.message = "内容和参数都没变，跳过向量化" if stats["skipped"] else ""
            progress.parents = sum(1 for c in chunks if c.chunk_id in parent_ids)
            progress.chunks = len(chunks) - progress.parents
            if unfinished:
                progress.retry = True
                progress.message = f"{unfinished} 个片段的增强没有完成（调用大模型失败），稍后自动重试"
        job.touch()
        return info

    def _record(self, kb_id: str, progress: FileProgress, info: dict[str, Any]) -> None:
        with self._lock(kb_id):
            manifest = self._manifest(kb_id)
            entry = manifest["files"].setdefault(progress.path, {})
            entry.update(status=progress.status, message=progress.message, updated_at=time.time(), **info)
            if progress.status in (DONE, SKIPPED):
                entry.update(chunks=progress.chunks, parents=progress.parents)
            self._data_changed(manifest)

    # ------------------------------------------------------------------ 内部
    def _pipeline(self, kb_id: str, *, reset: bool = False) -> RagPipeline:
        """打开（并缓存）知识库的管线；参数变了（重建）时重新打开。reset=True 时清空片段库。"""
        manifest = self._manifest(kb_id)
        key = _cache_key(manifest)
        cached = self._pipes.get(kb_id)
        settings = manifest["settings"]
        if reset or cached is None or cached[0] != key:
            pipe = RagPipeline.from_config(self._config(kb_id, settings, reset=reset))
            persist = self._persist_dir(kb_id)
            if not reset and (persist / "meta.json").exists():
                pipe.kb.load(str(persist))
            self._pipes[kb_id] = (key, pipe)
        return self._pipes[kb_id][1]

    def _persist_dir(self, kb_id: str) -> Path:
        return self.kbs_dir / kb_id / "index"

    def _lock(self, kb_id: str) -> "_KbLock":
        with self._guard:
            if kb_id not in self._locks:
                self._locks[kb_id] = _KbLock(self.kbs_dir / kb_id / ".lock")
            return self._locks[kb_id]

    def _data_changed(self, manifest: dict[str, Any]) -> None:
        """
        片段库刚被本进程写过：revision 加一写回清单，别的进程据此重新打开知识库；
        本进程缓存的管线就是刚写入的那个，数据是新的，只更新缓存键，不用重新打开。
        """
        manifest["revision"] = manifest.get("revision", 0) + 1
        self._write_manifest(manifest)
        cached = self._pipes.get(manifest["id"])
        if cached is not None:
            self._pipes[manifest["id"]] = (_cache_key(manifest), cached[1])

    def _summary(self, manifest: dict[str, Any]) -> dict[str, Any]:
        files = manifest["files"].values()
        return {
            "id": manifest["id"], "name": manifest["name"], "settings": _with_defaults(manifest["settings"]),
            "created_at": manifest["created_at"], "updated_at": manifest["updated_at"],
            "files": len(manifest["files"]),
            "chunks": sum(f.get("chunks", 0) for f in files),
            "parents": sum(f.get("parents", 0) for f in files),
            "failed": sum(1 for f in files if f.get("status") == FAILED),
            "busy": self.jobs.busy(manifest["id"]),
        }

    def _manifest(self, kb_id: str) -> dict[str, Any]:
        path = self.kbs_dir / kb_id / "kb.json"
        if not _KB_ID.match(kb_id or "") or not path.is_file():
            raise NotFound(f"知识库 {kb_id} 不存在")
        return json.loads(path.read_text(encoding="utf-8"))

    def _manifests(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.kbs_dir.glob("*/kb.json")):
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                logger.warning("知识库清单 %s 读取失败，已忽略：%s", path, exc)
        return out

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        """先写临时文件再替换：写到一半被杀也不会留下坏掉的清单。"""
        path = self.kbs_dir / manifest["id"] / "kb.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def recover(self) -> None:
        """
        页面服务启动时调用：清单里停在"处理中"、队列里却没有对应任务的文件标成失败。
        任务都在队列里，中断的会由 worker 接着做；这里只清理升级前（任务只存在内存里时）遗留的状态。
        """
        for manifest in self._manifests():
            if self.jobs.busy(manifest["id"]):
                continue
            stale = [e for e in manifest["files"].values() if e.get("status") in _UNFINISHED]
            for entry in stale:
                entry.update(status=FAILED, message="建库被中断（服务重启），点击“重建”即可续写，已写入的不会重复向量化")
            if stale:
                self._write_manifest(manifest)


def _cache_key(manifest: dict[str, Any]) -> str:
    """打开的管线什么时候要重新打开：参数变了（重建），或者片段库被别的进程写过（revision 变了）。"""
    return json.dumps([manifest["settings"], manifest.get("revision", 0)], sort_keys=True)


class _KbLock:
    """
    一个知识库的读改写锁：同一进程的线程之间用 RLock，进程之间（页面服务和 worker）用文件锁。
    可重入：同一线程嵌套加锁时只在最外层拿文件锁 —— 同一进程对同一文件再 flock 一次会把自己锁死。
    没有 fcntl 的系统（Windows）只有线程锁，此时页面服务要用进程内的 worker（--embedded-worker）。
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._thread_lock = threading.RLock()
        self._depth = 0
        self._file: Any = None

    def __enter__(self) -> "_KbLock":
        self._thread_lock.acquire()
        if self._depth == 0 and fcntl is not None and self.path.parent.is_dir():
            self._file = open(self.path, "a")
            fcntl.flock(self._file, fcntl.LOCK_EX)
        self._depth += 1
        return self

    def __exit__(self, *_: Any) -> None:
        self._depth -= 1
        if self._depth == 0 and self._file is not None:
            fcntl.flock(self._file, fcntl.LOCK_UN)
            self._file.close()
            self._file = None
        self._thread_lock.release()


def _clean_name(name: str) -> str:
    """去掉首尾空白和换行，最长 40 个字；空名称用"未命名知识库"。"""
    name = " ".join((name or "").split())[:40]
    return name or "未命名知识库"


def _file_entry(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "uploaded_at": time.time(), "status": WAITING, "message": "",
            "chunks": 0, "parents": 0}


# 建库增强的结果缓存，各知识库共用（见 _enrich_config）
_ENRICH_CACHE = _PROJECT_ROOT / ".cache" / "enrich"


def _with_defaults(settings: dict[str, Any]) -> dict[str, Any]:
    """
    精排和召回参数是后来分几次加的，之前建的知识库清单里缺其中几项：按当时的实际行为补上
    （词重叠精排、召回 20 个候选、交给大模型 5 个、向量和 BM25 等权、RRF 常数 60、不改写），
    老知识库的检索结果不会因为升级而改变。
    """
    retrieval = {"candidate_k": 20, "top_k": 5, **environment.DEFAULT_FUSION, **(settings.get("retrieval") or {})}
    return {"reranker": {"type": "lexical", "model": ""}, "enrich": dict(environment.ENRICH_OFF), **settings,
            "retrieval": retrieval}


def _validate_enrich(enrich: dict[str, Any] | None) -> dict[str, Any]:
    """建库增强的开关和假设问题个数；开了任何一项都要配置大模型。"""
    merged = {**environment.ENRICH_OFF, **(enrich or {})}
    try:
        out = {"context": bool(merged["context"]), "describe": bool(merged["describe"]),
               "questions": int(merged["questions"]), "summaries": bool(merged["summaries"])}
    except (TypeError, ValueError):
        raise ValueError(f"建库增强参数不合法：{enrich}") from None
    if not 0 <= out["questions"] <= 5:
        raise ValueError("每个子块的假设问题要在 0～5 个之间")
    if _enrich_on(out) and not environment.has_llm():
        raise ValueError("建库增强需要大模型：在 .env 配置 LLM_BASE_URL、LLM_API_KEY")
    return out


def _enrich_on(enrich: dict[str, Any]) -> bool:
    return any(enrich.values())


def _enrich_config(enrich: dict[str, Any], parsing: dict[str, Any]) -> dict[str, Any]:
    """
    增强结果缓存在项目的 .cache/enrich，各知识库共用：同样的片段、同样的模型，换个知识库也不用重新调用。
    视觉模型的地址和密钥只在运行时从环境变量读取，不写进 kb.json。
    """
    cfg: dict[str, Any] = {**enrich, "cache_dir": str(_ENRICH_CACHE),
                           "assets_dirs": configured_assets_dirs(parsing)}
    if enrich["describe"] and environment.has_vlm():
        cfg["vlm"] = {"type": "openai_compat", "model": os.getenv("OCR_MODEL"),
                      "base_url": os.getenv("OCR_BASE_URL") or None, "api_key": os.getenv("OCR_API_KEY") or None}
    return cfg


def _supports_sparse(encoder: dict[str, Any]) -> bool:
    """只有 FlagEmbedding 版的 BGE-M3 在向量化时同时生成稀疏向量。"""
    return encoder.get("type") == "bge_m3"


def _retriever_config(retrieval: dict[str, Any]) -> dict[str, Any]:
    """各路召回（权重为 0 的不参与）→ RRF 融合 → 可选的查询改写套在最外层（每个改写后的问题都走一遍多路召回）。"""
    channels = [name for name in CHANNELS if retrieval["weights"].get(name, 0) > 0]
    hybrid = {"type": "hybrid", "rrf_k": retrieval["rrf_k"], "channels": [{"type": c} for c in channels],
              "weights": [retrieval["weights"][c] for c in channels]}
    if retrieval["query_transform"] == "none":
        return hybrid
    return {"type": "transformed", "transformer": {"type": retrieval["query_transform"]}, "inner": hybrid}


def _reranker_config(reranker: dict[str, Any]) -> dict[str, Any]:
    if reranker["type"] == "cross_encoder":
        return {"type": "cross_encoder", "model_name": reranker["model"]}
    if reranker["type"] == "api":
        # 地址和密钥由精排器自己从环境变量（RERANK_*）读取，不写进清单
        return {"type": "api", "model": reranker["model"]}
    return {"type": reranker["type"]}


def _encoder_config(encoder: dict[str, Any]) -> dict[str, Any]:
    if encoder["type"] == "hashing":
        return {"type": "hashing", "dimension": 512}
    if encoder["type"] in ("sentence_transformers", "bge_m3"):
        return {"type": encoder["type"], "model_name": encoder["model"]}
    # 地址和密钥由 encoder 自己从环境变量读取，不经过页面，也不写进清单
    return {"type": "openai_compat", "model": encoder["model"], "dimension": int(os.getenv("EMBEDDING_DIM") or 1536)}
