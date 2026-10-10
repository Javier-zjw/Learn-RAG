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
from pathlib import Path
from typing import Any

from ..core.config import load_config
from ..core.registry import registry
from ..parsing.assets import configured_assets_dirs, locate_asset
from ..parsing.source import FileSource
from ..pipeline.rag import RagPipeline
from . import environment
from .chunk_view import document_view
from .mineru_service import STARTING, MinerUService
from .preview import Preview, PreviewUnavailable
from .jobs import DONE, EMBEDDING, FAILED, PARSING, SKIPPED, WAITING, FileProgress, Job, JobRunner, capture_warnings
from .uploads import Staging, safe_relative

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
_UNFINISHED = (WAITING, PARSING, EMBEDDING)


class NotFound(LookupError):
    """知识库、文档或资产不存在。"""


class Busy(RuntimeError):
    """知识库正在建库，暂时不能做这个操作。"""


class Library:
    def __init__(self, root: str | Path, runner: JobRunner | None = None, mineru: MinerUService | None = None) -> None:
        self.root = Path(root)
        self.mineru = mineru
        self.kbs_dir = self.root / "kbs"
        self.kbs_dir.mkdir(parents=True, exist_ok=True)
        self.staging = Staging(self.root / "uploads")
        self.preview = Preview(self.root / "preview")      # 按文件内容缓存，多个知识库共用
        self._digests: dict[tuple[str, int, int], str] = {}
        self.jobs = runner or JobRunner()
        self._pipes: dict[str, tuple[str, RagPipeline]] = {}   # kb_id -> (参数, 管线)
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()
        self._mark_interrupted()

    # ------------------------------------------------------------------ 知识库
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
            "id": kb_id, "name": name.strip() or "未命名知识库", "settings": settings,
            "created_at": now, "updated_at": now, "verify": None,
            "files": {path: _file_entry(directory / "files" / path) for path in files},
        }
        self._write_manifest(manifest)
        job = self._submit(kb_id, "build", files, reset=False)
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
        return self._submit(kb_id, "append", files, reset=False)

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
        return self._submit(kb_id, "rebuild", sorted(manifest["files"]), reset=reset)

    # ------------------------------------------------------------------ 文档
    def remove_document(self, kb_id: str, doc_id: str) -> None:
        manifest = self._manifest(kb_id)
        if doc_id not in manifest["files"]:
            raise NotFound(f"知识库里没有文档 {doc_id}")
        if self.jobs.busy(kb_id):
            raise Busy("这个知识库正在建库，完成后再删除文档")
        with self._lock(kb_id):
            pipe = self._pipeline(kb_id)
            pipe.kb.remove([doc_id])
            pipe.kb.save(str(self._persist_dir(kb_id)))
            (self.kbs_dir / kb_id / "files" / safe_relative(doc_id)).unlink(missing_ok=True)
            manifest = self._manifest(kb_id)
            manifest["files"].pop(doc_id, None)
            self._write_manifest(manifest)

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

        encoder = {**base["encoder"], **(settings.get("encoder") or {})}
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
                "parsers": {k: list(v) for k, v in parsers.items()}, "ocr": bool(settings.get("ocr", False))}

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
        cfg.update(chunker=dict(settings["chunker"]), encoder=_encoder_config(settings["encoder"]),
                   index=index, parsing=parsing)
        return cfg

    # ------------------------------------------------------------------ 建库任务
    def _submit(self, kb_id: str, kind: str, files: list[str], *, reset: bool) -> Job:
        return self.jobs.submit(kb_id, kind, files, lambda job: self._run(job, reset=reset))

    def _run(self, job: Job, *, reset: bool) -> None:
        """在后台线程里执行：逐个文件解析、切分、向量化，最后落盘并核对完整性。"""
        kb_id = job.kb_id
        with self._lock(kb_id):
            pipe = self._pipeline(kb_id, reset=reset)
            cfg = self._config(kb_id, self._manifest(kb_id)["settings"])
        self._wait_for_mineru(job, cfg)
        files_dir = self.kbs_dir / kb_id / "files"
        source = FileSource(str(files_dir), **cfg["parsing"])
        for progress in job.files:
            info = self._process(job, progress, pipe, source, files_dir)
            self._record(kb_id, progress, info)
        with self._lock(kb_id):
            pipe.kb.save(str(self._persist_dir(kb_id)))
            chunks = pipe.kb.stats()["chunks"]
            job.verify = pipe.kb.verify(sample=min(10, chunks))
            manifest = self._manifest(kb_id)
            manifest["verify"] = job.verify
            manifest["updated_at"] = time.time()
            self._write_manifest(manifest)
        job.touch()

    def _wait_for_mineru(self, job: Job, cfg: dict[str, Any]) -> None:
        """要用 MinerU 解析而它还在启动时先等它就绪，否则这批 PDF 会全部退回 PyMuPDF。"""
        uses_mineru = any("mineru" in chain for chain in cfg["parsing"]["parsers"].values())
        if not (self.mineru and uses_mineru and self.mineru.state == STARTING):
            return
        job.notice = "等待 MinerU 服务就绪（首次启动要加载模型）……"
        job.touch()
        ready = self.mineru.wait(self.mineru.timeout)
        job.notice = "" if ready else self.mineru.message
        job.touch()

    def _process(self, job: Job, progress: FileProgress, pipe: RagPipeline, source: FileSource,
                 files_dir: Path) -> dict[str, Any]:
        """处理一个文件，失败只记在这个文件上，不中断整批。返回要记进清单的解析信息。"""
        progress.status = PARSING
        job.touch()
        with capture_warnings() as notes:
            try:
                document = source.load_file(files_dir / progress.path)
            except Exception as exc:   # load_file 自己不抛异常，这里防的是意料之外的错误
                notes.append(str(exc))
                document = None
        if document is None:
            progress.status, progress.message = FAILED, "\n".join(notes) or "没有解析出任何内容"
            job.touch()
            return {}
        info = {"parser": document.metadata.get("parser", ""),
                "pages": max((e.extra.get("page_end") or e.page or 0 for e in document.elements), default=0) or None}

        progress.status = EMBEDDING
        job.touch()
        with capture_warnings() as notes, self._lock(job.kb_id):
            stats = pipe.kb.add([document])
            chunks = pipe.kb.chunks_of(document.doc_id)
        parent_ids = {c.metadata.get("parent_id") for c in chunks} - {None, ""}
        if stats["failed"]:
            progress.status, progress.message = FAILED, "\n".join(notes) or "向量化失败"
        else:
            progress.status = SKIPPED if stats["skipped"] else DONE
            progress.message = "内容和参数都没变，跳过向量化" if stats["skipped"] else ""
            progress.parents = sum(1 for c in chunks if c.chunk_id in parent_ids)
            progress.chunks = len(chunks) - progress.parents
        job.touch()
        return info

    def _record(self, kb_id: str, progress: FileProgress, info: dict[str, Any]) -> None:
        with self._lock(kb_id):
            manifest = self._manifest(kb_id)
            entry = manifest["files"].setdefault(progress.path, {})
            entry.update(status=progress.status, message=progress.message, updated_at=time.time(), **info)
            if progress.status in (DONE, SKIPPED):
                entry.update(chunks=progress.chunks, parents=progress.parents)
            self._write_manifest(manifest)

    # ------------------------------------------------------------------ 内部
    def _pipeline(self, kb_id: str, *, reset: bool = False) -> RagPipeline:
        """打开（并缓存）知识库的管线；参数变了（重建）时重新打开。reset=True 时清空片段库。"""
        settings = self._manifest(kb_id)["settings"]
        key = json.dumps(settings, sort_keys=True)
        cached = self._pipes.get(kb_id)
        if reset or cached is None or cached[0] != key:
            pipe = RagPipeline.from_config(self._config(kb_id, settings, reset=reset))
            persist = self._persist_dir(kb_id)
            if not reset and (persist / "meta.json").exists():
                pipe.kb.load(str(persist))
            self._pipes[kb_id] = (key, pipe)
        return self._pipes[kb_id][1]

    def _persist_dir(self, kb_id: str) -> Path:
        return self.kbs_dir / kb_id / "index"

    def _lock(self, kb_id: str) -> threading.RLock:
        with self._guard:
            return self._locks.setdefault(kb_id, threading.RLock())

    def _summary(self, manifest: dict[str, Any]) -> dict[str, Any]:
        files = manifest["files"].values()
        return {
            "id": manifest["id"], "name": manifest["name"], "settings": manifest["settings"],
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

    def _mark_interrupted(self) -> None:
        """服务重启时还没处理完的文件标成失败：任务只在内存里，重启后不会自动继续。"""
        for manifest in self._manifests():
            stale = [e for e in manifest["files"].values() if e.get("status") in _UNFINISHED]
            for entry in stale:
                entry.update(status=FAILED, message="建库被中断（服务重启），点击“重建”即可续写，已写入的不会重复向量化")
            if stale:
                self._write_manifest(manifest)


def _file_entry(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "uploaded_at": time.time(), "status": WAITING, "message": "",
            "chunks": 0, "parents": 0}


def _encoder_config(encoder: dict[str, Any]) -> dict[str, Any]:
    if encoder["type"] == "hashing":
        return {"type": "hashing", "dimension": 512}
    # 地址和密钥由 encoder 自己从环境变量读取，不经过页面，也不写进清单
    return {"type": "openai_compat", "model": encoder["model"], "dimension": int(os.getenv("EMBEDDING_DIM") or 1536)}
