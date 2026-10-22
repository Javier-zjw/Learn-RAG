"""
eval.synth —— 用大模型从自己的文档里生成评测集。

没有自己的评测集，检索的每个改动都只能凭感觉判断好坏；公开数据集的分布和企业文档差别很大。
这里从知识库的片段里抽样，让大模型针对每段原文出几个问题，并逐字摘录回答所需的原文（证据）。

生成的题目质量决定评测的可信度，所以每道题都要过几道检查，不合格的直接丢弃并计数：
  - 证据必须真的出现在原文里（大模型常常"改写"原文，改写过的证据没法用来判断是否召回）；
  - 问题不能依赖"本文""上文""根据资料"这类指代 —— 真实用户提问时并不知道是哪篇文档，
    这类问题放到整个知识库里检索是不公平的；
  - 同一个问题只留一次。
生成后仍建议人工抽查一遍（jsonl 文件可以直接编辑、删除）。

抽样对象是"交给大模型的上下文"那一级：有父块时用父块（信息完整，出的题更像真实问题），
没有父块的子块直接用。按文档轮流抽取，每篇文档都有题，不会被长文档占满。
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..core.interfaces import LLM
from ..core.text import count_tokens, coverage, match_key
from ..core.types import Chunk, EvalSample
from ..ingest.structure import chunk_body

logger = logging.getLogger(__name__)

TYPES = ("fact", "number", "table", "reasoning", "code")

_SYSTEM = "你是为企业知识库问答系统构建评测集的标注专家，只输出 JSON。"

_PROMPT = """下面是知识库中的一段资料（来自《{title}》，章节：{section}）。
请针对这段资料出 {count} 道评测题，用来检验检索系统能不能在整个知识库里找到这段资料。

要求：
1. 问题必须能且只能根据这段资料回答，答案明确、唯一；
2. 问题要像真实用户在不知道是哪篇文档时会问的那样，带上足够的限定信息（如产品名、年份、编号、部门），
   不要出现"本文""上文""文中""这段资料""根据资料"之类的说法；
3. 问题使用和资料相同的语言；尽量覆盖不同题型：fact（事实）、number（数值）、table（表格里的数据）、
   reasoning（需要综合两句以上）、code（命令、配置、代码）。资料里有表格时至少出一道 table 题；
4. answer 简短，只写答案本身；
5. evidence 从资料中**逐字摘录**一段连续原文（1～3 句），包含回答所需的全部信息，不要改写、不要省略中间的字。

只输出 JSON 数组，不要任何解释：
[{{"question": "...", "answer": "...", "evidence": "...", "type": "fact"}}]

资料：
{text}"""

# 依赖文档上下文的指代：真实用户不会这么问，放到全库检索时这类问题无法公平评估
_DEICTIC = re.compile(r"本文|上文|下文|文中|该文|这段|此段|资料中|根据(以上|上述)?(资料|材料|文档|内容)|"
                      r"the (passage|text|document|above)|according to the", re.I)


class QuestionSynthesizer:
    def __init__(self, llm: LLM, *, per_passage: int = 2, min_tokens: int = 60, max_chars: int = 3000,
                 workers: int = 4) -> None:
        """
        per_passage 每段资料出几道题；min_tokens 太短的片段信息量不够，不出题；
        max_chars 资料过长时截断，避免超出模型上下文；workers 并发调用大模型的数量。
        """
        self.llm = llm
        self.per_passage = per_passage
        self.min_tokens = min_tokens
        self.max_chars = max_chars
        self.workers = max(1, workers)
        self.stats: Counter[str] = Counter()

    def generate(self, chunks: Iterable[Chunk], n: int, *, seed: int = 0) -> list[EvalSample]:
        """从片段里抽样出约 n 道题。stats 记录生成和各类丢弃的数量。"""
        self.stats = Counter()
        passages = self._sample(list(chunks), -(-n // self.per_passage), seed)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            batches = list(pool.map(self._ask, passages))
        samples: list[EvalSample] = []
        seen: set[str] = set()
        for passage, items in zip(passages, batches):
            for item in items:
                sample = self._check(passage, item, seen)
                if sample is not None:
                    samples.append(sample)
        logger.info("生成评测题 %d 道（抽样片段 %d 段）；%s", len(samples), len(passages),
                    "，".join(f"{k} {v}" for k, v in self.stats.items()))
        return samples[:n]

    def _sample(self, chunks: list[Chunk], count: int, seed: int) -> list[Chunk]:
        """交给大模型的那一级片段（父块，或没有父块的子块），按文档轮流抽取。"""
        parents = {c.metadata.get("parent_id") for c in chunks} - {None, ""}
        units = [c for c in chunks
                 if (c.chunk_id in parents or not c.metadata.get("parent_id"))
                 and count_tokens(chunk_body(c)) >= self.min_tokens]
        by_doc: dict[str, list[Chunk]] = {}
        for chunk in units:
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
        rng = random.Random(seed)
        for group in by_doc.values():
            rng.shuffle(group)
        picked: list[Chunk] = []
        while len(picked) < count and any(by_doc.values()):
            for doc_id in sorted(by_doc):
                if by_doc[doc_id] and len(picked) < count:
                    picked.append(by_doc[doc_id].pop())
        return picked

    def _ask(self, passage: Chunk) -> list[dict[str, Any]]:
        meta = passage.metadata
        prompt = _PROMPT.format(title=meta.get("title", passage.doc_id), section=meta.get("section") or "—",
                                count=self.per_passage, text=chunk_body(passage)[: self.max_chars])
        try:
            reply = self.llm.ask(prompt, system=_SYSTEM, temperature=0.3)
        except Exception as exc:          # 一段出错只少几道题，不中断整批
            logger.warning("为 %s 出题失败：%s", passage.chunk_id, exc)
            self.stats["调用失败"] += 1
            return []
        items = _parse_json_list(reply)
        if items is None:
            self.stats["输出不是 JSON"] += 1
            return []
        return items

    def _check(self, passage: Chunk, item: dict[str, Any], seen: set[str]) -> EvalSample | None:
        question = str(item.get("question") or "").strip()
        answer = str(item.get("answer") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        kind = str(item.get("type") or "fact").strip().lower()
        if not question or not answer or not evidence:
            self.stats["缺少字段"] += 1
            return None
        if _DEICTIC.search(question):
            self.stats["问题依赖上下文指代"] += 1
            return None
        if coverage(evidence, chunk_body(passage)) < 0.9:
            self.stats["证据不是原文"] += 1
            return None
        key = match_key(question)
        if key in seen:
            self.stats["重复问题"] += 1
            return None
        seen.add(key)
        self.stats["通过"] += 1
        return EvalSample(
            sample_id=f"{passage.doc_id}#q{len(seen)}",
            question=question,
            answers=[answer],
            gold_doc_ids=[passage.doc_id],
            evidence=[evidence],
            metadata={"type": kind if kind in TYPES else "fact", "source_chunk": passage.chunk_id,
                      "section": passage.metadata.get("section", "")},
        )


def to_jsonl_row(sample: EvalSample) -> dict[str, Any]:
    """评测集的一行，和 JsonlQADataset 读取的格式一致。"""
    return {"id": sample.sample_id, "question": sample.question, "answers": sample.answers,
            "gold_doc_ids": sample.gold_doc_ids, "evidence": sample.evidence, **sample.metadata}


def _parse_json_list(reply: str) -> list[dict[str, Any]] | None:
    """模型常在 JSON 外面包一层 ``` 或加几句说明，取第一个 [ 到最后一个 ] 之间的部分。"""
    start, end = reply.find("["), reply.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(reply[start:end + 1])
    except ValueError:
        return None
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else None
