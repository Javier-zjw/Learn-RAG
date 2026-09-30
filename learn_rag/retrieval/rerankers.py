"""
retrieval.rerankers —— 精排层。

召回（recall）要"快而全"，精排（precision）要"准"。
两阶段拆开是检索系统的经典结构：先用便宜的方法从海量里捞 50 条，
再用昂贵的方法把这 50 条排准，取前 5 条进 prompt。

为什么精排值得单独一层：进入 prompt 的上下文越少越准（噪声会稀释注意力），
但召回又不能太少（否则漏掉答案）。精排就是解决这个矛盾的。
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Sequence

from ..core.interfaces import LLM, Reranker
from ..core.registry import registry
from ..core.text import index_tokens
from ..core.types import ScoredChunk

logger = logging.getLogger(__name__)


@registry.register("reranker", "identity")
class IdentityReranker(Reranker):
    """
    不做任何事，只截断。基线用 —— 有它才能量化出重排到底涨了多少分
    """

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return list(candidates)[:top_k]


@registry.register("reranker", "lexical")
class LexicalReranker(Reranker):
    """
    零依赖的词重叠精排：查询覆盖率 + 片段长度惩罚。

    效果远不如 CrossEncoder，但它在离线/无网络时保证链路完整，
    并且能让你观察到"精排这一步在指标上确实有区别"。
    """

    def __init__(self, alpha: float = 0.7, length_penalty: float = 0.15) -> None:
        self.alpha = alpha
        self.length_penalty = length_penalty

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        q_tokens = set(index_tokens(query))
        if not q_tokens:
            return list(candidates)[:top_k]
        rescored: list[ScoredChunk] = []
        for cand in candidates:
            c_tokens = set(index_tokens(cand.text))
            coverage = len(q_tokens & c_tokens) / len(q_tokens)
            penalty = self.length_penalty * min(len(cand.text) / 1000, 1.0)
            score = self.alpha * coverage - penalty + (1 - self.alpha) * _sigmoid(cand.score)

            rescored.append(
                ScoredChunk(chunk=cand.chunk, score=score, source="rerank:lexical", debug={"coverage": coverage})
            )
        return sorted(rescored, key=lambda h: -h.score)[:top_k]


@registry.register("reranker", "cross_encoder")
class CrossEncoderReranker(Reranker):
    """
    CrossEncoder 精排（bge-reranker / jina-reranker 等）。

    与双塔向量的关键差别：query 和 chunk 一起进模型做交互注意力，
    因此判别力强得多，但复杂度是 O(候选数) 次前向，只能用在小候选集上。
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", batch_size: int = 16,
                 device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size)
        ranked = sorted(zip(candidates, scores), key=lambda kv: -float(kv[1]))[:top_k]
        return [ScoredChunk(chunk=c.chunk, score=float(s), source="rerank:cross") for c, s in ranked]


@registry.register("reranker", "api")
class ApiReranker(Reranker):
    """
    远程 rerank 服务（Jina / Cohere / SiliconFlow / 火山 等）。

    rerank API 的调用协议已经事实标准化了：

        POST {base_url}/rerank
        {"model": ..., "query": ..., "documents": ["文本1", "文本2"], "top_n": 5}
        →  {"results": [{"index": 0, "relevance_score": 0.93}, ...]}

    与 embedding 一样用标准库 urllib，不引入任何 SDK。

    【配置示例】
      环境变量：
        export RERANK_BASE_URL="https://api.siliconflow.cn/v1"
        export RERANK_API_KEY="sk-xxx"
        export RERANK_MODEL="BAAI/bge-reranker-v2-m3"
      yaml：
        reranker:
          type: api
          model: BAAI/bge-reranker-v2-m3
          base_url: https://api.siliconflow.cn/v1
          api_key: sk-xxx

    【常见服务商】
      SiliconFlow  BAAI/bge-reranker-v2-m3      https://api.siliconflow.cn/v1
      Jina         jina-reranker-v2-base-multilingual   https://api.jina.ai/v1
      Cohere       rerank-multilingual-v3.0     https://api.cohere.com/v1
      通义(DashScope) gte-rerank                 https://dashscope.aliyuncs.com/api/v1

    【接口路径】
      endpoint 默认是标准的 /rerank；base_url 含 dashscope 时默认改用通义原生路径
      /services/rerank/text-rerank/text-rerank。其他非标准服务可以在 yaml 里显式指定 endpoint。
      响应同时兼容 {"results": [...]}、{"data": [...]} 与通义的 {"output": {"results": [...]}}。
    """

    def __init__(
            self,
            model: str = "BAAI/bge-reranker-v2-m3",
            base_url: str | None = None,
            api_key: str | None = None,
            max_chars: int = 1000,
            timeout: int = 60,
            max_retries: int = 2,
            endpoint: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("RERANK_BASE_URL", "https://api.siliconflow.cn/v1")).rstrip("/")
        if endpoint is None:
            endpoint = "/services/rerank/text-rerank/text-rerank" if "dashscope" in self.base_url else "/rerank"
        self.endpoint = "/" + endpoint.lstrip("/")
        self.api_key = api_key or os.getenv("RERANK_API_KEY", "")
        self.max_chars = max_chars
        self.timeout = timeout
        self.max_retries = max_retries

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        import time
        import urllib.request

        documents = [c.text[: self.max_chars] for c in candidates]
        top_n = min(top_k, len(candidates))
        if self.endpoint.startswith("/services/"):
            # 通义原生协议：参数分别包在 input / parameters 里
            body = {"model": self.model, "input": {"query": query, "documents": documents},
                    "parameters": {"top_n": top_n, "return_documents": False}}
        else:
            body = {"model": self.model, "query": query, "documents": documents, "top_n": top_n}
        payload = json.dumps(body).encode("utf-8")

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}{self.endpoint}",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    items = _result_items(json.loads(resp.read()))

                ranked = [
                    ScoredChunk(
                        chunk=candidates[int(item["index"])].chunk,
                        score=float(item.get("relevance_score", item.get("score", 0.0))),
                        source="rerank:api",
                    )
                    for item in items
                    if 0 <= int(item.get("index", -1)) < len(candidates)
                ]
                return ranked[: top_k] if ranked else list(candidates)[: top_k]
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(2 ** attempt)

        logger.warning("rerank API 调用失败，退回召回顺序：%s", last_error)
        return list(candidates)[: top_k]


def _result_items(body: dict) -> list[dict]:
    """从不同服务商的响应里取出结果列表：标准协议在顶层，通义原生协议包在 output 里。"""
    container = body.get("output") if isinstance(body.get("output"), dict) else body
    return container.get("results") or container.get("data") or []


@registry.register("reranker", "llm")
class LLMReranker(Reranker):
    """
    让 LLM 直接给每个片段打 0-10 分（listwise 打分）。

    优点是无需额外模型、可解释；缺点是慢且贵。
    解析失败时回退到原顺序 —— 任何"聪明"的组件都必须有降级路径。
    """

    PROMPT = (
        "给定问题和若干候选片段，为每个片段与问题的相关性打 0-10 分。\n"
        '只输出 JSON 数组，形如 [{{"id": 0, "score": 8}}]，不要其他内容。\n\n'
        "问题：{q}\n\n候选：\n{c}"
    )

    def __init__(self, llm: LLM, max_chars: int = 300) -> None:
        self.llm = llm
        self.max_chars = max_chars

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not candidates:
            return []

        listing = "\n".join(f"[{i}] {c.text[: self.max_chars]}" for i, c in enumerate(candidates))
        try:
            raw = self.llm.ask(self.PROMPT.format(q=query, c=listing))
            payload = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
            scores = {int(item["id"]): float(item["score"]) for item in payload}

        except Exception:
            return list(candidates)[:top_k]

        ranked = sorted(enumerate(candidates), key=lambda kv: -scores.get(kv[0], 0.0))[:top_k]

        return [
            ScoredChunk(chunk=c.chunk, score=scores.get(i, 0.0), source="rerank:llm") for i, c in ranked
        ]


def _sigmoid(x: float) -> float:
    import math

    return 1 / (1 + math.exp(-max(min(x, 30), -30)))
