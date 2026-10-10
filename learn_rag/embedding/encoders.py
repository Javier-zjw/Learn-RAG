"""
embedding.encoders —— 文本向量化的若干实现。

三个实现覆盖三类场景：
  hashing        : 零依赖、确定性、离线可跑 —— 保证你 clone 下来立刻能跑通全流程
  openai_compat  : 任何 OpenAI 兼容的 embedding 服务（含本地部署的 Youtu-Embedding / bge / TEI）
  sentence_tf    : 本地 sentence-transformers 模型

它们对上层完全等价（同一个 TextEncoder 接口），
所以"先用 hashing 跑通链路和评测，再换成真模型看指标提升"是本项目推荐的学习路径。

CachingEncoder 是"装饰器模式"：它自己也是 TextEncoder，
包住任意 encoder 加上缓存。这就是通用接口带来的复利 ——
写一次缓存，所有实现都能用。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from ..core.interfaces import TextEncoder
from ..core.registry import registry
from ..core.text import char_ngrams, index_tokens

logger = logging.getLogger(__name__)


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 1e-12 else vec


@registry.register("encoder", "hashing")
class HashingEncoder(TextEncoder):
    """特征哈希（hashing trick）向量：把 token / 字符 n-gram 散列到固定维度。

    它不是语义模型，本质是"稀疏词袋的稠密投影"，
    但它有三个教学上极其宝贵的性质：
      1. 零依赖、零下载、毫秒级，任何机器都能跑；
      2. 完全确定性，评测结果可复现；
      3. 效果明显弱于真模型 —— 于是你换成真 embedding 后，
         能在指标上**看到**语义检索到底带来了多少提升。
    """

    def __init__(self, dimension: int = 512, use_ngrams: bool = True, ngram: int = 3) -> None:
        self.dimension = dimension
        self.use_ngrams, self.ngram = use_ngrams, ngram

    def signature(self) -> str:
        return f"hashing:dim={self.dimension},ngram={self.ngram if self.use_ngrams else 0}"

    def _features(self, text: str) -> list[str]:
        feats = index_tokens(text)
        if self.use_ngrams:
            feats += char_ngrams(text, self.ngram)
        return feats

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimension
            for feat in self._features(text):
                h = int(hashlib.md5(feat.encode("utf-8")).hexdigest(), 16)
                idx = h % self.dimension
                sign = 1.0 if (h >> 8) & 1 else -1.0  # 有符号哈希，减小碰撞偏置
                vec[idx] += sign
            out.append(_l2_normalize(vec))
        return out


@registry.register("encoder", "openai_compat")
class OpenAICompatEncoder(TextEncoder):
    """任何 OpenAI 兼容 /v1/embeddings 服务。

    刻意用标准库 urllib 而不是 openai SDK：
    少一个依赖，也让你看清楚"请求究竟长什么样"。

    【批量上限自适应】
    各家服务商对"单次请求最多几条文本"限制不同（有的 10 条，有的 25 条，有的 2048 条），
    超了就回 400。早期版本只会抛一句 "Bad Request"，而且只有语料变大时才暴露 ——
    12 篇时一个请求 12 条能过，60 篇时一个请求 32 条就挂。

    现在的处理：遇到 400/413 且批量大于 1 时，**自动把批量减半重试**，
    直到成功或减到 1 条。成功后记住这个批量，后续请求直接用它。
    这是把"服务商有批量上限"这个异常情况在 encoder 内部定义掉，
    上层（知识库、检索器、评测）不需要知道任何一家服务商的限制。

    【错误信息】
    服务商在 400 的响应体里会写明原因（"batch size exceeds 10"、"input is empty"……），
    urllib 默认把它吞掉。这里把响应体读出来拼进异常，排查时一眼就能看到。

    batch_size 默认读环境变量 EMBEDDING_BATCH_SIZE，没设则为 32。
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        api_key: str | None = None,
        dimension: int = 1536,
        batch_size: int | None = None,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "")
        self.dimension = dimension
        self.batch_size = int(batch_size or os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        self.timeout, self.max_retries = timeout, max_retries

    def signature(self) -> str:
        return f"openai_compat:{self.model}"

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        vectors: list[list[float]] = []
        i = 0
        while i < len(texts):
            batch = list(texts[i : i + self.batch_size])
            try:
                vectors.extend(self._request(batch))
                i += len(batch)
            except _BatchTooLarge as exc:
                if self.batch_size <= 1:
                    raise RuntimeError(f"Embedding 请求失败（单条也被拒绝，不是批量问题）：{exc}") from None
                old = self.batch_size
                self.batch_size = max(1, self.batch_size // 2)
                logger.warning("服务端拒绝 %d 条/批，自动降为 %d 条/批重试（可在 .env 设 EMBEDDING_BATCH_SIZE=%d 跳过探测）",
                               old, self.batch_size, self.batch_size)
        if vectors:
            self.dimension = len(vectors[0])
        return vectors

    def _request(self, batch: list[str]) -> list[list[float]]:
        # 空字符串是另一个常见的 400 来源，发出去之前就挡掉
        safe = [t if t.strip() else " " for t in batch]
        payload = json.dumps({"model": self.model, "input": safe}).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                f"{self.base_url}/embeddings",
                data=payload,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read())
                items = sorted(data["data"], key=lambda d: d.get("index", 0))  # 按 index 对齐，防乱序
                if len(items) != len(batch):
                    raise RuntimeError(f"返回 {len(items)} 条向量，但发送了 {len(batch)} 条文本")
                return [_l2_normalize(item["embedding"]) for item in items]
            except urllib.error.HTTPError as exc:
                detail = _http_error_detail(exc)
                if exc.code in (400, 413) and len(batch) > 1:
                    raise _BatchTooLarge(f"HTTP {exc.code}：{detail}") from None
                if exc.code in (401, 403, 404):  # 配置错误，重试没用，立刻报出来
                    raise RuntimeError(
                        f"Embedding HTTP {exc.code}：{detail}\n"
                        f"  base_url={self.base_url}  model={self.model}  批量={len(batch)}"
                    ) from None
                last = RuntimeError(f"HTTP {exc.code}：{detail}")
            except (urllib.error.URLError, TimeoutError) as exc:
                last = exc
            time.sleep(2 ** attempt)     # 429 / 5xx / 网络抖动：退避重试
        raise RuntimeError(f"Embedding 请求失败（已重试 {self.max_retries} 次）：{last}")


class _BatchTooLarge(Exception):
    """内部信号：服务端拒绝了这一批，交给 encode() 减半重试。"""


def _http_error_detail(exc: "urllib.error.HTTPError") -> str:
    """把 HTTPError 的响应体读出来。服务商的真实报错原因就在这里。"""
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        body = ""
    try:
        parsed = json.loads(body)
        msg = parsed.get("error", parsed)
        if isinstance(msg, dict):
            msg = msg.get("message") or msg
        body = str(msg)
    except Exception:
        pass
    return (body or exc.reason or "无响应体")[:500]


@registry.register("encoder", "sentence_transformers")
class SentenceTransformerEncoder(TextEncoder):
    """本地 sentence-transformers 模型（bge / gte / Youtu-Embedding 等）。

    query_prefix/doc_prefix 用于非对称检索模型：
    很多中文模型要求查询侧加 "为这个句子生成表示以用于检索相关文章：" 之类的指令前缀，
    忘了加会明显掉点 —— 把它做成参数，是为了把"模型的坑"关进配置里。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        query_prefix: str = "",
        doc_prefix: str = "",
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # 延迟导入：不装也能用别的实现

        self._model = SentenceTransformer(model_name, device=device)
        self.query_prefix, self.doc_prefix, self.batch_size = query_prefix, doc_prefix, batch_size
        self.dimension = int(self._model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def signature(self) -> str:
        # 文档前缀会改变存进索引的向量，查询前缀决定查询向量和文档向量是否可比，两者都算模型身份的一部分
        return f"sentence_transformers:{self.model_name}|query={self.query_prefix}|doc={self.doc_prefix}"

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        prefix = self.query_prefix if is_query else self.doc_prefix
        payload = [prefix + t for t in texts] if prefix else list(texts)
        arr = self._model.encode(payload, batch_size=self.batch_size, normalize_embeddings=True)
        return [list(map(float, row)) for row in arr]


@registry.register("encoder", "cached")
class CachingEncoder(TextEncoder):
    """给任意 encoder 套一层缓存（内存 + 可选磁盘）。

    它自身实现 TextEncoder，所以对上层"透明"。
    评测经常要反复跑同一批语料，这一层能省掉大量重复的模型调用和 API 费用。
    """

    def __init__(self, inner: TextEncoder, cache_path: str | None = None) -> None:
        self.inner = inner
        self.dimension = inner.dimension
        self._lock = threading.Lock()
        self._mem: dict[str, list[float]] = {}
        self._path = Path(cache_path) if cache_path else None
        if self._path and self._path.exists():
            self._mem = json.loads(self._path.read_text(encoding="utf-8"))

    def signature(self) -> str:
        return self.inner.signature()

    def _key(self, text: str, is_query: bool) -> str:
        # 键里带上模型身份：换了模型、缓存文件没换时，不会把旧模型的向量当成新模型的返回
        return hashlib.md5(f"{self.signature()}|{int(is_query)}|{text}".encode("utf-8")).hexdigest()

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        keys = [self._key(t, is_query) for t in texts]
        missing = [(i, t) for i, (t, k) in enumerate(zip(texts, keys)) if k not in self._mem]
        if missing:
            fresh = self.inner.encode([t for _, t in missing], is_query=is_query)
            with self._lock:
                for (i, _), vec in zip(missing, fresh):
                    self._mem[keys[i]] = vec
        self.dimension = self.inner.dimension
        return [self._mem[k] for k in keys]

    def flush(self) -> None:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._mem), encoding="utf-8")