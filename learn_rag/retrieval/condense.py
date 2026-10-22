"""
retrieval.condense —— 多轮对话的问题补全：把依赖上文的追问改写成能单独检索的完整问题。

用户问完"数据库多久备份一次"接着问"那保留多久呢"，第二句直接拿去检索什么也找不到：
"那""呢"没有检索价值，真正的主语"数据库备份"在上一轮里。召回只看这一句，所以要先补全再检索；
生成也用补全后的问题，回答和检索到的证据说的是同一件事。

补全只是锦上添花：大模型调用失败、输出不像一个问题时，退回原问题，不让一次补全失败拖垮问答。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence

from ..core.interfaces import LLM

logger = logging.getLogger(__name__)

PROMPT = """下面是用户和知识库助手的对话，以及用户的最新问题。
请把最新问题改写成一个不依赖对话上下文、可以单独拿去检索的完整问题：补全被省略的主语、把"它""这个""那"换成具体所指。
如果最新问题本身已经完整，原样输出。只输出改写后的问题，不要解释。

对话：
{history}

最新问题：{question}"""

# 模型常在答案前加的前缀
_PREFIX = re.compile(r"^(改写后的问题|改写后|完整问题|问题)\s*[:：]\s*")


class QuestionCondenser:
    def __init__(self, llm: LLM, max_turns: int = 3, max_chars: int = 300) -> None:
        """max_turns：只看最近几轮（一问一答算一轮）；max_chars：每条历史消息最多取多少字，控制提示词长度。"""
        self.llm = llm
        self.max_turns = max_turns
        self.max_chars = max_chars

    def condense(self, question: str, history: Sequence[dict[str, str]]) -> str:
        """history 是 OpenAI 格式的消息列表（role 为 user / assistant）。没有历史时不调用大模型。"""
        recent = [m for m in history if m.get("role") in ("user", "assistant") and m.get("content")]
        recent = recent[-2 * self.max_turns:]
        if not recent:
            return question
        lines = [f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'].strip()[: self.max_chars]}" for m in recent]
        try:
            text = self.llm.ask(PROMPT.format(history="\n".join(lines), question=question))
        except Exception as exc:  # noqa: BLE001 —— 补全失败不影响问答，用原问题检索
            logger.warning("问题补全失败，用原问题检索：%s", exc)
            return question
        return _clean(text, question)


def _clean(text: str, question: str) -> str:
    """取第一行非空输出，去掉前缀和引号；输出为空或长得不像一个问题时用原问题。"""
    line = next((s.strip() for s in text.splitlines() if s.strip()), "")
    line = _PREFIX.sub("", line).strip().strip("\"'“”「」")
    if not line or len(line) > 4 * len(question) + 100:
        logger.warning("问题补全的输出不像一个问题，用原问题检索：%r", text[:200])
        return question
    return line
