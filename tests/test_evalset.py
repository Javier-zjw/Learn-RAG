"""
第 1 阶段（评测集）的测试：证据比对、证据级指标、用大模型出题的检查规则，以及"建库 → 出题 → 评测"整条流程。
大模型一律用假的实现，不依赖网络和密钥。
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.core.interfaces import LLM
from learn_rag.core.text import coverage, match_key
from learn_rag.core.types import Answer, Chunk, EvalSample, RagResult, ScoredChunk
from learn_rag.eval.metrics import EvidenceAtK
from learn_rag.eval.synth import QuestionSynthesizer, to_jsonl_row


def _result(*texts: str) -> RagResult:
    contexts = [ScoredChunk(Chunk(f"d#{i}", "d", text), 1.0 / (i + 1)) for i, text in enumerate(texts)]
    return RagResult(question="q", answer=Answer(""), contexts=contexts)


class TestCoverage(unittest.TestCase):
    def test_ignores_layout_noise(self):
        self.assertEqual(match_key("| 版本 | 年费（元）|\n| 标准版 | 29,990 |"), "版本年费元标准版29990")
        self.assertEqual(coverage("全年营业收入 4,860 万元。", "摘要：全年营业\n收入4860万元，同比增长"), 1.0)
        self.assertEqual(match_key("访问 http://<服务器IP>:8080"), "访问http服务器ip8080")    # 不是 HTML 标签的尖括号保留

    def test_tolerates_a_typo_but_not_different_text(self):
        evidence = "第二阶段把召回的五十个候选交给交叉编码器重新打分，只保留前五个"
        self.assertGreater(coverage(evidence, evidence.replace("交叉", "交差")), 0.8)
        self.assertLess(coverage(evidence, "完全不相关的一段文字，讲的是报销流程和审批节点"), 0.2)


class TestEvidenceMetric(unittest.TestCase):
    def test_rank_and_recall(self):
        sample = EvalSample("1", "q", ["a"], evidence=["报销单需在三十天内提交", "超过一万元需要总监审批"])
        scores = EvidenceAtK(k=3).compute(sample, _result("差旅标准见附件", "员工报销单需在三十天内提交审批", "其他"))
        self.assertEqual(scores["evidence_hit@3"], 1.0)
        self.assertEqual(scores["evidence_recall@3"], 0.5)
        self.assertEqual(scores["evidence_mrr@3"], 0.5)

    def test_evidence_split_across_two_contexts_counts_when_both_retrieved(self):
        sample = EvalSample("1", "q", evidence=["报销单需在三十天内提交，超过一万元需要总监审批"])
        scores = EvidenceAtK(k=2).compute(sample, _result("报销单需在三十天内提交，", "超过一万元需要总监审批。"))
        self.assertEqual(scores["evidence_mrr@2"], 0.5)          # 两块都召回后才凑齐，算在第 2 位

    def test_not_in_top_k(self):
        sample = EvalSample("1", "q", evidence=["报销单需在三十天内提交"])
        scores = EvidenceAtK(k=1).compute(sample, _result("其他内容", "报销单需在三十天内提交"))
        self.assertEqual((scores["evidence_hit@1"], scores["evidence_mrr@1"]), (0.0, 0.0))

    def test_runner_skips_samples_without_evidence(self):
        from learn_rag.eval.runner import Evaluator

        class _Pipe:
            def retrieve_only(self, question):
                return _result("报销单需在三十天内提交")

        class _Data:
            name = "t"

            def corpus(self):
                return []

            def samples(self):
                return [EvalSample("a", "q", evidence=["报销单需在三十天内提交"]), EvalSample("b", "q")]

        from learn_rag.eval.metrics import ExactMatch

        report = Evaluator(_Pipe(), [EvidenceAtK(1), ExactMatch()], retrieval_only=True).run(_Data())
        self.assertEqual([bool(c.scores) for c in report.cases], [True, False])   # 没标证据的题不记 0 分
        self.assertEqual(report.summary["evidence_hit@1"], 1.0)
        self.assertNotIn("exact_match", report.summary)                          # 只评检索时不评答案


class _FakeLLM(LLM):
    """按提示词里的资料出题：第一句原文作证据。另外故意出几道不合格的题，检验过滤规则。"""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, messages, **options):
        self.calls += 1
        prompt = messages[-1]["content"]
        title = re.search(r"来自《(.+?)》", prompt).group(1)
        text = prompt.rsplit("资料：\n", 1)[1]
        first = re.split(r"(?<=。)", text.strip())[0]
        items = [
            {"question": f"{title}中{first[:12]}的具体规定是什么？", "answer": first[-8:], "evidence": first, "type": "fact"},
            {"question": "本文提到的流程是什么？", "answer": "x", "evidence": first, "type": "fact"},          # 指代
            {"question": f"{title}的核心观点？", "answer": "y", "evidence": "一句原文里根本没有的话，模型自己编的", "type": "fact"},
            {"question": f"{title}中{first[:12]}的具体规定是什么？", "answer": "z", "evidence": first, "type": "fact"},  # 重复
        ]
        return "好的，题目如下：\n```json\n" + json.dumps(items, ensure_ascii=False) + "\n```"


def _chunks() -> list[Chunk]:
    """两篇文档：a 有一个父块和两个子块，b 只有一个没有父块的子块。"""
    long_a = "差旅报销需在出差结束后三十天内提交，逾期需部门负责人说明原因。" * 3
    parent = Chunk("a#0", "a", "[制度 > 报销]\n" + long_a + "住宿标准按城市分档执行。" * 3,
                   metadata={"title": "制度", "section": "报销"})
    children = [Chunk(f"a#{i}", "a", "[制度 > 报销]\n" + long_a, metadata={"title": "制度", "section": "报销",
                                                                         "parent_id": "a#0"}) for i in (1, 2)]
    other = Chunk("b#0", "b", "[手册 > 安装]\n" + "安装前确认服务器内存不少于十六GB并开放八零八零端口。" * 4,
                  metadata={"title": "手册", "section": "安装"})
    return [parent, *children, other]


class TestSynthesizer(unittest.TestCase):
    def test_keeps_only_well_formed_questions(self):
        llm = _FakeLLM()
        synth = QuestionSynthesizer(llm, per_passage=2, min_tokens=20, workers=1)
        samples = synth.generate(_chunks(), n=10)
        self.assertEqual(llm.calls, 2)                                  # a 的父块 + b 的子块，a 的子块不单独出题
        self.assertEqual({s.metadata["source_chunk"] for s in samples}, {"a#0", "b#0"})
        self.assertEqual(synth.stats["通过"], 2)
        for reason in ("问题依赖上下文指代", "证据不是原文", "重复问题"):
            self.assertEqual(synth.stats[reason], 2, reason)
        sample = samples[0]
        self.assertEqual(coverage(sample.evidence[0], _chunks()[0].text), 1.0)
        row = to_jsonl_row(sample)
        self.assertEqual(set(row) >= {"id", "question", "answers", "gold_doc_ids", "evidence", "type"}, True)

    def test_bad_model_output_is_counted_not_raised(self):
        class _Broken(LLM):
            def chat(self, messages, **options):
                return "抱歉，我无法完成"

        synth = QuestionSynthesizer(_Broken(), min_tokens=20, workers=1)
        self.assertEqual(synth.generate(_chunks(), n=4), [])
        self.assertEqual(synth.stats["输出不是 JSON"], 2)


def _has(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


class TestEvalsetEndToEnd(unittest.TestCase):
    """页面建好的知识库 → learn-rag evalset 出题 → learn-rag eval --kb 评测。"""

    def setUp(self) -> None:
        from learn_rag.server.library import Library

        self.tmp = tempfile.mkdtemp()
        self.data = str(Path(self.tmp) / "web")
        library = Library(self.data, worker=True)
        docs = {
            "报销制度.md": "# 报销制度\n\n## 时限\n\n" + "差旅报销需在出差结束后三十天内提交，逾期需部门负责人书面说明原因。" * 3
                         + "\n\n## 标准\n\n" + "一线城市住宿标准为每晚六百元，其他城市为每晚四百元。" * 3,
            "安装手册.md": "# 安装手册\n\n## 环境\n\n" + "安装前确认服务器内存不少于十六GB，并开放八零八零端口。" * 3,
        }
        upload_id = None
        for name, text in docs.items():
            upload_id, _ = library.staging.save(upload_id, name, io.BytesIO(text.encode("utf-8")))
        created = library.create("制度库", {"chunker": {"type": "structure", "chunk_size": 40, "parent_size": 160},
                                          "encoder": {"type": "hashing", "model": "hashing"},
                                          "index": {"type": "flat"}}, upload_id)
        self.assertEqual(library.jobs.wait(created["job"]["id"]).status, "done")
        self.kb = created["kb"]["id"]

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cli(self, *argv: str) -> str:
        from learn_rag import cli

        out = io.StringIO()
        with patch.object(sys, "argv", ["learn-rag", "--log-level", "ERROR", *argv]), redirect_stdout(out):
            cli.main()
        return out.getvalue()

    def test_generate_then_evaluate(self):
        qa = Path(self.tmp) / "qa.jsonl"
        fake = _FakeLLM()
        with patch.dict(os.environ, {"LLM_API_KEY": "test", "LLM_BASE_URL": "http://llm.local/v1"}), \
                patch("learn_rag.generation.llms.OpenAICompatLLM.chat", side_effect=fake.chat):
            printed = self._cli("evalset", "--kb", self.kb, "--data", self.data, "-n", "6", "--workers", "1",
                                "--out", str(qa))
        rows = [json.loads(line) for line in qa.read_text(encoding="utf-8").splitlines()]
        self.assertGreaterEqual(len(rows), 2)
        self.assertIn("道题", printed)
        self.assertTrue(all(r["evidence"] and r["gold_doc_ids"] for r in rows))

        report = Path(self.tmp) / "report.json"
        printed = self._cli("eval", "--kb", self.kb, "--data", self.data, "--qa", str(qa), "--retrieval-only",
                            "--out", str(report))
        summary = json.loads(report.read_text(encoding="utf-8"))["summary"]
        self.assertIn("evidence_hit@5", summary)
        self.assertGreater(summary["evidence_hit@5"], 0.5)            # 问题里带着原文的前几个字，hashing 也能找回
        self.assertIn("制度库", printed)

    def test_unknown_kb_lists_existing_ones(self):
        with self.assertRaises(SystemExit) as ctx:
            self._cli("eval", "--kb", "ffffffff", "--data", self.data, "--qa", "x.jsonl")
        self.assertIn(self.kb, str(ctx.exception))

    def test_evalset_requires_an_llm(self):
        with patch.dict(os.environ, {"LLM_API_KEY": "", "LLM_BASE_URL": ""}):
            with self.assertRaises(SystemExit) as ctx:
                self._cli("evalset", "--kb", self.kb, "--data", self.data)
        self.assertIn("LLM_API_KEY", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
