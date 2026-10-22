"""
第 5 阶段（监控）的测试：查询 trace 的记录与统计（P50 / P95、疑似空召回、错误率、过期清理），
知识库的问答入口记 trace、评测入口不记，以及 learn-rag stats 的输出。
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.core.types import Answer, Chunk, RagResult, ScoredChunk
from learn_rag.pipeline.trace import QueryLog, percentile


def _result(question: str, *texts: str, total: float = 0.1) -> RagResult:
    contexts = [ScoredChunk(Chunk(f"d#{i}", "d", t), 1.0 / (i + 1)) for i, t in enumerate(texts)]
    return RagResult(question=question, answer=Answer("答"), contexts=contexts,
                     timings={"retrieve": total / 2, "rerank": total / 4, "total": total})


class TestQueryLog(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.log = QueryLog(self.tmp)

    def test_percentile_is_nearest_rank(self):
        values = [float(i) for i in range(1, 101)]
        self.assertEqual((percentile(values, 50), percentile(values, 95)), (50.0, 95.0))
        self.assertEqual(percentile([0.3], 95), 0.3)
        self.assertEqual(percentile([], 95), 0.0)

    def test_empty_recall_detection(self):
        hit = self.log.record("数据库备份保留多久", _result("q", "数据库每晚备份，备份文件保留三十天。"))
        miss = self.log.record("数据库备份保留多久", _result("q", "节假日加班按三倍工资结算。"))
        nothing = self.log.record("数据库备份保留多久", _result("q"))
        self.assertFalse(hit["empty"])
        self.assertTrue(miss["empty"])                                    # 召回了东西，但和问题毫不相干
        self.assertEqual(miss["overlap"], 0.0)
        self.assertTrue(nothing["empty"])
        self.assertEqual(hit["top"][0], {"chunk": "d#0", "score": 1.0})

    def test_summary(self):
        for i in range(20):
            self.log.record(f"数据库备份问题{i}", _result("q", "数据库备份", total=(i + 1) / 10))
        self.log.record("加班怎么算", _result("q", "数据库备份"))
        self.log.record("出错的问题", error="LLM 调用失败", seconds=2.0)
        s = self.log.summary(days=1)
        self.assertEqual((s["queries"], s["errors"]), (22, 1))
        self.assertAlmostEqual(s["error_rate"], 1 / 22, places=3)
        self.assertAlmostEqual(s["empty_rate"], 1 / 21, places=3)
        self.assertEqual(s["latency"]["total"]["p95"], 1.9)               # 21 个值里第 20 个
        self.assertEqual(s["latency"]["total"]["max"], 2.0)
        self.assertEqual(s["slowest"][0]["question"], "数据库备份问题19")
        self.assertEqual(s["recent_empty"][0]["question"], "加班怎么算")
        self.assertEqual(s["recent_errors"][0]["error"], "LLM 调用失败")

    def test_window_and_retention(self):
        old = Path(self.tmp) / "queries-20000101.jsonl"
        old.write_text(json.dumps({"ts": 946684800, "question": "很久以前", "timings": {}}) + "\n", encoding="utf-8")
        (Path(self.tmp) / f"queries-{time.strftime('%Y%m%d')}.jsonl").unlink(missing_ok=True)
        self.log.record("今天的问题", _result("q", "今天"))                  # 当天第一次写入时清理过期文件
        self.assertFalse(old.exists())
        self.assertEqual(self.log.summary(days=1)["queries"], 1)

    def test_corrupt_lines_are_skipped(self):
        self.log.record("问题", _result("q", "问题"))
        path = next(Path(self.tmp).glob("queries-*.jsonl"))
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"ts": 1, "quest')                                   # 进程被杀时留下的半行
        self.assertEqual(self.log.summary(days=1)["queries"], 1)


class TestKnowledgeBaseMetrics(unittest.TestCase):
    def setUp(self) -> None:
        from learn_rag.server.library import Library

        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.library = Library(self.tmp, worker=True)
        text = "# 运维手册\n\n## 备份\n\n" + "数据库每晚两点全量备份，备份文件保留三十天。" * 3
        upload_id, _ = self.library.staging.save(None, "运维.md", io.BytesIO(text.encode("utf-8")))
        created = self.library.create("运维", {"encoder": {"type": "hashing", "model": "hashing"},
                                               "index": {"type": "flat"}}, upload_id)
        self.assertEqual(self.library.jobs.wait(created["job"]["id"]).status, "done")
        self.kb = created["kb"]["id"]

    def test_ask_records_traces_but_pipeline_does_not(self):
        self.library.ask(self.kb, "数据库备份保留多久")
        self.library.pipeline(self.kb).retrieve_only("评测用的问题")         # 评测直接用管线，不记 trace
        with patch("learn_rag.pipeline.rag.RagPipeline.answer", side_effect=RuntimeError("服务不可用")):
            with self.assertRaises(RuntimeError):
                self.library.ask(self.kb, "会出错的问题")
        m = self.library.metrics(self.kb)
        self.assertEqual((m["queries"]["queries"], m["queries"]["errors"]), (2, 1))
        self.assertEqual(m["queries"]["empty_rate"], 0.0)
        self.assertIn("retrieve", m["queries"]["latency"])
        self.assertEqual(set(m["build"]["stages"]), {"parse", "index"})
        job = m["build"]["jobs"][0]
        self.assertEqual((job["kind"], job["status"], job["failed"]), ("build", "done", 0))
        self.assertGreaterEqual(job["run_seconds"], 0)

    def test_stats_command(self):
        from learn_rag import cli

        self.library.ask(self.kb, "加班工资怎么算")
        out = io.StringIO()
        argv = ["learn-rag", "--log-level", "ERROR", "stats", "--kb", self.kb, "--data", self.tmp]
        with patch.object(sys, "argv", argv), redirect_stdout(out):
            cli.main()
        printed = out.getvalue()
        self.assertIn("查询 1 次", printed)
        self.assertIn("P95", printed)
        self.assertIn("加班工资怎么算", printed)                             # 和文档不相干：列为疑似空召回
        self.assertIn("切分与向量化", printed)


if __name__ == "__main__":
    unittest.main()
