"""
多轮对话问题补全的测试：追问结合上文补全后再检索，补全失败时退回原问题。大模型一律用假的实现。
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.core.interfaces import LLM
from learn_rag.retrieval.condense import QuestionCondenser

_HISTORY = [
    {"role": "user", "content": "数据库多久备份一次"},
    {"role": "assistant", "content": "数据库每晚两点全量备份。"},
]


class _ScriptedLLM(LLM):
    """补全的提示词返回固定改写，其余（生成回答）返回固定答案；记下每次收到的提示词。"""

    def __init__(self, rewrite: str = "改写后的问题：“数据库备份保留多久？”") -> None:
        self.rewrite = rewrite
        self.prompts: list[str] = []

    def chat(self, messages, **options):
        prompt = messages[-1]["content"]
        self.prompts.append(prompt)
        return self.rewrite if "最新问题" in prompt else "保留三十天。"


class TestQuestionCondenser(unittest.TestCase):
    def test_no_history_means_no_llm_call(self):
        llm = _ScriptedLLM()
        self.assertEqual(QuestionCondenser(llm).condense("那保留多久呢", []), "那保留多久呢")
        self.assertEqual(llm.prompts, [])

    def test_rewrite_is_cleaned_and_history_trimmed(self):
        llm = _ScriptedLLM()
        history = [{"role": "user", "content": f"很早以前的问题{i}"} for i in range(5)] + _HISTORY
        question = QuestionCondenser(llm, max_turns=1).condense("那保留多久呢", history)
        self.assertEqual(question, "数据库备份保留多久？")             # 去掉前缀和引号
        self.assertIn("数据库每晚两点全量备份", llm.prompts[0])
        self.assertNotIn("很早以前的问题", llm.prompts[0])               # 只看最近一轮

    def test_falls_back_to_original_question(self):
        class _Down(LLM):
            def chat(self, messages, **options):
                raise TimeoutError("服务超时")

        self.assertEqual(QuestionCondenser(_Down()).condense("那保留多久呢", _HISTORY), "那保留多久呢")
        rambling = _ScriptedLLM(rewrite="好的。" * 200)
        self.assertEqual(QuestionCondenser(rambling).condense("那保留多久呢", _HISTORY), "那保留多久呢")
        empty = _ScriptedLLM(rewrite="\n\n")
        self.assertEqual(QuestionCondenser(empty).condense("那保留多久呢", _HISTORY), "那保留多久呢")


class TestChatWithKnowledgeBase(unittest.TestCase):
    """页面建好的知识库 + .env 里配置了大模型：追问补全后检索，回答由大模型生成。"""

    def setUp(self) -> None:
        from learn_rag.server.library import Library

        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = str(Path(self.tmp) / "web")
        library = Library(self.data)
        docs = {
            "运维.md": "# 运维手册\n\n## 备份\n\n" + "数据库每晚两点全量备份，备份文件保留三十天。" * 3,
            "考勤.md": "# 考勤制度\n\n## 加班\n\n" + "节假日加班按三倍工资结算，工作日加班可以调休。" * 3,
        }
        upload_id = None
        for name, text in docs.items():
            upload_id, _ = library.staging.save(upload_id, name, io.BytesIO(text.encode("utf-8")))
        created = library.create("制度库", {"encoder": {"type": "hashing", "model": "hashing"},
                                          "index": {"type": "flat"}, "retrieval": {"top_k": 1}}, upload_id)
        self.assertEqual(library.jobs.wait(created["job"]["id"]).status, "done")
        self.kb = created["kb"]["id"]
        self.llm = _ScriptedLLM()
        for p in (patch.dict(os.environ, {"LLM_API_KEY": "test", "LLM_BASE_URL": "http://llm.local/v1"}),
                  patch("learn_rag.generation.llms.OpenAICompatLLM.chat", side_effect=self.llm.chat)):
            p.start()
            self.addCleanup(p.stop)

    def test_follow_up_is_condensed_before_retrieval(self):
        from learn_rag.server.library import Library

        pipe = Library(self.data).pipeline(self.kb)
        result = pipe.answer("那保留多久呢", history=_HISTORY)
        self.assertEqual(result.question, "那保留多久呢")
        self.assertEqual(result.extras["standalone_question"], "数据库备份保留多久？")
        self.assertEqual(result.contexts[0].doc_id, "运维.md")
        self.assertEqual(result.answer.text, "保留三十天。")
        self.assertIn("condense", result.timings)

        calls = len(self.llm.prompts)
        first = pipe.retrieve_only("节假日加班怎么算")                    # 第一轮没有历史：不调用大模型补全
        self.assertEqual(len(self.llm.prompts), calls)
        self.assertNotIn("standalone_question", first.extras)

    def test_cli_chat_keeps_history(self):
        from learn_rag import cli

        out = io.StringIO()
        argv = ["learn-rag", "--log-level", "ERROR", "chat", "--kb", self.kb, "--data", self.data]
        with patch.object(sys, "argv", argv), redirect_stdout(out), \
                patch("builtins.input", side_effect=["数据库多久备份一次", "那保留多久呢", "exit"]):
            cli.main()
        printed = out.getvalue()
        self.assertEqual(printed.count("补全为"), 1)                     # 只有第二句是追问
        self.assertIn("补全为：数据库备份保留多久？", printed)
        self.assertIn("数据库多久备份一次", self.llm.prompts[-2])           # 第二轮的补全看到了第一轮


if __name__ == "__main__":
    unittest.main()
