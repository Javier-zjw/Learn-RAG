"""
ingest.enrich —— 建库时用大模型增强片段，补上"正文里没有那几个词"造成的漏召回。

切分后的片段离开了上下文：一块只写着"保留期为 30 天"，没说是谁的保留期；一张表只有数字，没说统计的是什么；
一张图在索引里只剩一行图注。用户的问法又常和原文措辞不同。增强给每块补上帮助"找到它"的文本：

  context      这一块在全文里的背景（Anthropic 的 Contextual Retrieval）：补全省略的主语、产品名、时间
  description  表格的概括；图片和图表交给视觉模型描述
  questions    这一块能回答的几个问题（假设问题），各自生成一个向量
  summary      章节（父块）摘要，生成一个向量，对应"这一节讲什么"一类的概括性提问

context 和 description 拼在正文前面进向量、BM25 和稀疏索引（Chunk.index_text），questions 和 summary 是
单独的向量记录（Chunk.index_views）。交给大模型和页面展示的始终是原样的正文。

成本：每个子块一次大模型调用（context / description / questions 合在一次里），每张图一次视觉模型调用，
开了摘要时每个父块再一次。所以：
  - 结果按"任务 + 模型 + 输入"缓存在磁盘上，重建、改检索参数、换 embedding 模型都不会重复调用；
  - 多线程并发调用；
  - 单次调用失败只记在片段的 enrich.failed 里，不中断建库，下次建库只重试失败的那几次。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..core.interfaces import LLM, ChunkEnricher
from ..core.registry import registry
from ..core.types import ENRICH_KEY, Chunk, Document
from ..parsing.assets import locate_asset, mime_for

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 提示词或输出格式改了就加一：缓存键和文档指纹都带着它，旧的增强结果自动失效
VERSION = 1

CHUNK_PROMPT = """下面是《{title}》中的一个片段，以及它所在章节的内容（帮你理解上下文）。按要求输出一个 JSON 对象，不要输出其他内容。

文档结构：
{outline}

所在章节：
{section}

片段：
{chunk}

要求（JSON 的字段）：
{tasks}"""

_TASKS = {
    "context": '"context"：一句话（60 字以内）说明这个片段在全文中的位置和讨论的对象，补全片段里省略的主语、产品名、时间等，'
               '让它脱离全文也能看懂。不要复述片段内容。',
    "description": '"description"：两三句话概括片段里的表格：统计的是什么、有哪些列、关键的数值或结论。',
    "questions": '"questions"：{n} 个用户可能会问、并且能用这个片段回答的问题（字符串列表），措辞彼此不同，'
                 '要写出具体对象，不要用"本文""该表"这类指代。',
}

SUMMARY_PROMPT = """用两三句话概括《{title}》中下面这一节的内容，写清主题和关键结论、数字，让用户用概括性的问题也能找到它。只输出摘要。

{section}
{text}"""

IMAGE_PROMPT = ("这是文档《{title}》中的一张图{caption}。用两三句话描述图中的内容：是什么类型的图、展示了什么、"
                "关键的数据或结论。只输出描述。")


@registry.register("enricher", "llm")
class LLMEnricher(ChunkEnricher):
    def __init__(
            self,
            llm: LLM,
            vlm: LLM | None = None,
            context: bool = True,
            describe: bool = True,
            questions: int = 0,
            summaries: bool = False,
            workers: int = 4,
            cache_dir: str = ".cache/enrich",
            assets_dirs: Sequence[str] = (),
            max_chars: int = 3000,
    ) -> None:
        """
        vlm：描述图片用的视觉模型，没有时不描述图片（表格仍由 llm 概括）。
        questions：每个子块生成几个假设问题，0 表示不生成。summaries：是否给每个父块写摘要。
        assets_dirs：解析器保存图片的资产库目录（相对路径按项目根解析）。
        max_chars：交给大模型的章节内容最多多少字，控制每次调用的长度。
        """
        self.llm, self.vlm = llm, vlm
        self.context, self.describe, self.questions, self.summaries = context, describe, questions, summaries
        self.workers = max(1, workers)
        cache = Path(cache_dir)
        self.cache_dir = cache if cache.is_absolute() else _PROJECT_ROOT / cache
        self.assets_dirs = list(assets_dirs)
        self.max_chars = max_chars

    def signature(self) -> str:
        vlm = _model_name(self.vlm) if self.vlm is not None and self.describe else "none"
        return (f"enrich-v{VERSION}|llm={_model_name(self.llm)}|context={self.context}|describe={self.describe}"
                f"|questions={self.questions}|summaries={self.summaries}|vlm={vlm}")

    def enrich(self, document: Document, chunks: list[Chunk]) -> None:
        parent_ids = {c.metadata["parent_id"] for c in chunks if c.metadata.get("parent_id")}
        children = [c for c in chunks if c.chunk_id not in parent_ids]
        by_id = {c.chunk_id: c for c in chunks}
        title = str(document.metadata.get("title") or document.doc_id)
        outline = _outline(chunks)

        # 先列出全部调用，再并发执行；每个任务的结果写回对应片段的哪个字段由 apply 决定
        jobs: list[tuple[str, list[Chunk], Callable[[], Any], Callable[[Chunk, Any], None]]] = []
        for chunk in children:
            fields = self._chunk_fields(chunk)
            if fields:
                section = by_id.get(chunk.metadata.get("parent_id", ""))
                prompt = self._chunk_prompt(title, outline, section.text if section else _siblings(chunk, children),
                                            chunk, fields)
                jobs.append(("chunk", [chunk], self._cached("chunk", prompt, lambda p=prompt, f=fields: self._ask_json(p, f)),
                             _apply_fields))
        if self.describe and self.vlm is not None:
            for asset, owners, caption in _images(children):
                jobs.append(("image", owners, self._cached("image", asset + caption, lambda a=asset, c=caption: self._see(title, a, c)),
                             _apply_image(caption)))
        if self.summaries:
            for parent in (by_id[pid] for pid in parent_ids if pid in by_id):
                prompt = SUMMARY_PROMPT.format(title=title, section=parent.metadata.get("section", ""),
                                               text=parent.text[: self.max_chars])
                jobs.append(("summary", [parent], self._cached("summary", prompt, lambda p=prompt: self._ask_text(p)),
                             _apply_summary))
        if not jobs:
            return

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [(name, owners, apply, pool.submit(run)) for name, owners, run, apply in jobs]
            failed = 0
            for name, owners, apply, future in futures:
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 —— 增强失败不中断建库，记下来下次重试
                    failed += 1
                    logger.warning("增强失败（%s，%s）：%s", name, owners[0].chunk_id, exc)
                    for owner in owners:
                        _enrich_of(owner).setdefault("failed", []).append(name)
                    continue
                for owner in owners:
                    apply(owner, result)
        if failed:
            logger.warning("文档 %s：%d / %d 次增强调用失败，下次建库会重试", document.doc_id, failed, len(jobs))

    # ------------------------------------------------------------------ 单次调用
    def _chunk_fields(self, chunk: Chunk) -> list[str]:
        fields = ["context"] if self.context else []
        if self.describe and "table" in (chunk.metadata.get("kinds") or []):
            fields.append("description")
        if self.questions > 0:
            fields.append("questions")
        return fields

    def _chunk_prompt(self, title: str, outline: str, section: str, chunk: Chunk, fields: list[str]) -> str:
        tasks = "\n".join(f"- {_TASKS[f].format(n=self.questions)}" for f in fields)
        return CHUNK_PROMPT.format(title=title, outline=outline, section=_clip(section, self.max_chars),
                                   chunk=chunk.text, tasks=tasks)

    def _ask_json(self, prompt: str, fields: list[str]) -> dict[str, Any]:
        """要求的字段缺了或格式不对算失败（抛异常），下次重试，而不是悄悄存一个空增强。"""
        reply = _parse_json_object(self.llm.ask(prompt))
        if reply is None:
            raise ValueError("模型输出不是 JSON 对象")
        out: dict[str, Any] = {}
        for field in fields:
            value = reply.get(field)
            if field == "questions":
                value = [str(q).strip() for q in value if str(q).strip()][: self.questions] if isinstance(value, list) else None
            elif isinstance(value, str):
                value = value.strip()
            if not value:
                raise ValueError(f"模型输出缺少字段 {field}")
            out[field] = value
        return out

    def _ask_text(self, prompt: str) -> str:
        text = self.llm.ask(prompt).strip()
        if not text:
            raise ValueError("模型没有输出")
        return text

    def _see(self, title: str, asset: str, caption: str) -> str:
        path = locate_asset(asset, self.assets_dirs)
        if path is None:
            raise FileNotFoundError(f"资产库里找不到图片 {asset}（资产库目录：{self.assets_dirs}）")
        url = f"data:{mime_for(path.suffix)};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
        prompt = IMAGE_PROMPT.format(title=title, caption=f"（图注：{caption}）" if caption else "")
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                 {"type": "text", "text": prompt}]}]
        text = self.vlm.chat(messages).strip()
        if not text:
            raise ValueError("视觉模型没有输出描述")
        return text

    def _cached(self, task: str, key_text: str, run: Callable[[], Any]) -> Callable[[], Any]:
        """
        按"任务 + 模型 + 输入"缓存结果：同样的片段、同样的模型，第二次建库直接读盘。
        输入包含章节内容和文档结构，所以文档改了（哪怕只改了别的章节标题）相关片段会重新增强。
        """
        model = _model_name(self.vlm if task == "image" else self.llm)
        key = hashlib.sha256(json.dumps([VERSION, task, model, key_text], ensure_ascii=False).encode("utf-8")).hexdigest()
        path = self.cache_dir / key[:2] / f"{key}.json"

        def call() -> Any:
            if path.is_file():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))["result"]
                except (ValueError, KeyError):
                    logger.warning("增强缓存 %s 损坏，重新调用", path)
            result = run()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"result": result}, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)
            return result

        return call


def _enrich_of(chunk: Chunk) -> dict[str, Any]:
    return chunk.metadata.setdefault(ENRICH_KEY, {})


def _apply_fields(chunk: Chunk, result: dict[str, Any]) -> None:
    enrich = _enrich_of(chunk)
    for field, value in result.items():
        if field == "description" and enrich.get("description"):     # 图片描述可能先写进来了
            value = f"{value}\n{enrich['description']}"
        enrich[field] = value


def _apply_summary(chunk: Chunk, text: str) -> None:
    _enrich_of(chunk)["summary"] = text


def _apply_image(caption: str) -> Callable[[Chunk, str], None]:
    def apply(chunk: Chunk, text: str) -> None:
        enrich = _enrich_of(chunk)
        line = f"图（{caption}）：{text}" if caption else f"图：{text}"
        enrich["description"] = f"{enrich['description']}\n{line}" if enrich.get("description") else line
    return apply


def _images(children: list[Chunk]) -> list[tuple[str, list[Chunk], str]]:
    """块里的图片和图表（表格截图除外，表格已经是文字）：(资产路径, 包含它的子块, 图注)，同一张图只描述一次。"""
    found: dict[str, tuple[list[Chunk], str]] = {}
    for chunk in children:
        for entry in chunk.metadata.get("assets") or []:
            if entry.get("kind") == "table" or not entry.get("asset"):
                continue
            owners, _ = found.setdefault(entry["asset"], ([], str(entry.get("caption") or "")))
            owners.append(chunk)
    return [(asset, owners, caption) for asset, (owners, caption) in found.items()]


def _outline(chunks: list[Chunk], limit: int = 40) -> str:
    sections = list(dict.fromkeys(str(c.metadata.get("section") or "") for c in chunks))
    lines = [s for s in sections if s][:limit]
    return "\n".join(lines) or "（无章节结构）"


def _siblings(chunk: Chunk, children: list[Chunk]) -> str:
    """没有父块时，用同一小节的子块拼出章节内容。"""
    section = chunk.metadata.get("section")
    return "\n".join(c.text for c in children if c.metadata.get("section") == section)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "……"


def _model_name(llm: LLM | None) -> str:
    return str(getattr(llm, "model", None) or type(llm).__name__)


def _parse_json_object(reply: str) -> dict[str, Any] | None:
    """模型常在 JSON 外面包一层 ``` 或加几句说明，取第一个 { 到最后一个 } 之间的部分。"""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(reply[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None
