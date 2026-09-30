"""
generation.generators —— 生成层。

生成层要解决三件容易被忽略的事：
  1. **上下文装配**：按预算截断（token/字符），保证不超模型窗口，
     且优先保留排名靠前的证据。
  2. **引用可追溯**：给每段证据编号，要求模型输出 [1][2]，
     再把编号映射回 chunk_id。没有引用的 RAG 无法被审计，也无法算忠实度。
  3. **拒答**：证据不足时必须允许模型说"资料中没有提到"，
     这是抑制幻觉最有效的一招（也是 faithfulness 指标的主要来源）。
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..core.interfaces import LLM, Generator
from ..core.registry import registry
from ..core.text import index_tokens, normalize
from ..core.types import Answer, ScoredChunk

SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。请只依据【参考资料】回答问题。\n"
    "要求：\n"
    "1. 答案必须来自参考资料，不得编造；资料中没有的信息请回答“资料中未提及”。\n"
    "2. 在使用了某段资料的句子末尾用 [编号] 标注出处，可标多个，如 [1][3]。\n"
    "3. 回答简洁直接，先给结论再给必要的解释。"
)

USER_TEMPLATE = "【参考资料】\n{context}\n\n【问题】\n{question}\n\n【回答】"


def pack_contexts(contexts: Sequence[ScoredChunk], max_chars: int) -> tuple[str, list[ScoredChunk]]:
    """
    按预算装配证据文本，返回 (拼接文本, 实际用到的片段)。

    单独抽成函数是为了让"装配策略"可被单测和替换
    （将来换成按 token 计费、按文档聚合等，只改这里）。
    """
    used: list[ScoredChunk] = []
    blocks: list[str] = []
    budget = max_chars
    for i, ctx in enumerate(contexts, 1):
        title = ctx.chunk.metadata.get("title", ctx.doc_id)
        block = f"[{i}] （来源：{title}）\n{ctx.text}"
        if len(block) > budget and used:
            break
        blocks.append(block[:budget])
        used.append(ctx)
        budget -= len(block)
        if budget <= 0:
            break

    return "\n\n".join(blocks), used


@registry.register("generator", "stuff")
class StuffGenerator(Generator):
    """
    最常用的 "stuff" 策略：把所有证据一次性塞进 prompt。

    适用于证据总量能放进上下文窗口的场景（绝大多数知识库问答）。
    证据超长时的 map-reduce / refine 策略可以再实现一个 Generator，
    上层依旧不用改。
    """

    def __init__(
            self,
            llm: LLM,
            max_context_chars: int = 4000,
            system_prompt: str = SYSTEM_PROMPT,
            user_template: str = USER_TEMPLATE,
    ) -> None:
        self.llm = llm
        self.max_context_chars = max_context_chars
        self.system_prompt = system_prompt
        self.user_template = user_template

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        if not contexts:
            return Answer(text="资料中未提及。", citations=[], raw="")
        context_text, used = pack_contexts(contexts, self.max_context_chars)
        prompt = self.user_template.format(context=context_text, question=question)
        raw = self.llm.ask(prompt, system=self.system_prompt)
        return Answer(text=raw.strip(), citations=self._citations(raw, used), raw=raw)

    @staticmethod
    def _citations(text: str, used: Sequence[ScoredChunk]) -> list[str]:
        """
        把答案里的 [1][2] 还原成 chunk_id，做到证据可追溯
        """

        ids: list[str] = []
        for token in re.findall(r"\[(\d+)\]", text):
            idx = int(token) - 1
            if 0 <= idx < len(used) and used[idx].chunk.chunk_id not in ids:
                ids.append(used[idx].chunk.chunk_id)

        return ids


@registry.register("generator", "extractive")
class ExtractiveGenerator(Generator):
    """
    离线抽取式生成器：不调用任何模型，从证据里挑最相关的句子作答。

    它存在的意义有两个：
      1. 让你在**没有任何 API Key** 的情况下就能跑通端到端评测，
         先把"指标怎么算、流程怎么串"学会；
      2. 作为生成层的下限基线 —— 换成真 LLM 后指标涨了多少，一目了然。
    """

    _SENT = re.compile(r"[^。！？!?\n]+[。！？!?]?")

    def __init__(self, max_sentences: int = 2) -> None:
        self.max_sentences = max_sentences

    def generate(self, question: str, contexts: Sequence[ScoredChunk]) -> Answer:
        q_tokens = set(index_tokens(question))
        scored: list[tuple[float, str, str]] = []
        for ctx in contexts:
            for sent in self._SENT.findall(ctx.text):
                sent = sent.strip()
                if len(normalize(sent)) < 2:
                    continue
                s_tokens = set(index_tokens(sent))
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens) / (len(q_tokens) or 1)

                scored.append((overlap - 0.0005 * len(sent), sent, ctx.chunk.chunk_id))

        if not scored:
            return Answer(text="资料中未提及")
        scored.sort(key=lambda item: -item[0])
        picked = scored[: self.max_sentences]
        return Answer(
            text=" ".join(s for _, s, _ in picked),
            citations=list(dict.fromkeys(cid for _, _, cid in picked))
        )
