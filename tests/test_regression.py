"""
第 5 阶段（质量回归）的测试：和基线比较时只看质量指标、掉得超过容差算回归，learn-rag eval --baseline 有回归时退出码为 1。
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from learn_rag.eval.regression import compare, load_baseline, regressions, save_baseline


class TestCompare(unittest.TestCase):
    def test_statuses(self):
        rows = {r.metric: r for r in compare(
            {"recall@5": 0.80, "mrr@5": 0.70, "ndcg@5": 0.95, "latency_total": 9.0, "faithfulness": 0.9},
            {"recall@5": 0.85, "mrr@5": 0.71, "ndcg@5": 0.90, "latency_total": 0.1, "evidence_hit@5": 0.8},
            tolerance=0.02)}
        self.assertEqual(rows["recall@5"].status, "regressed")
        self.assertEqual(rows["mrr@5"].status, "ok")                       # 掉了 0.01，在容差内
        self.assertEqual(rows["ndcg@5"].status, "improved")
        self.assertNotIn("latency_total", rows)                           # 延迟不参与比较
        self.assertEqual(rows["faithfulness"].status, "missing")
        self.assertEqual(rows["evidence_hit@5"].status, "missing")
        self.assertEqual([r.metric for r in regressions(list(rows.values()))], ["recall@5"])

    def test_baseline_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.json"
            save_baseline(path, {"recall@5": 0.912345, "latency_total": 1.0}, note="x")
            self.assertEqual(load_baseline(path), {"recall@5": 0.9123})
            full = Path(tmp) / "report.json"                              # --out 写的完整报告也能当基线
            full.write_text(json.dumps({"summary": {"mrr@5": 0.5}, "cases": []}), encoding="utf-8")
            self.assertEqual(load_baseline(full), {"mrr@5": 0.5})


class TestEvalBaseline(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        docs = [{"id": "d1", "title": "备份", "text": "数据库每晚两点全量备份，备份文件保留三十天。"},
                {"id": "d2", "title": "加班", "text": "节假日加班按三倍工资结算，工作日加班可以调休。"}]
        qa = [{"id": "1", "question": "备份文件保留多久", "answers": ["三十天"], "gold_doc_ids": ["d1"]},
              {"id": "2", "question": "节假日加班工资怎么算", "answers": ["三倍工资"], "gold_doc_ids": ["d2"]}]
        self.corpus, self.qa = Path(self.tmp) / "corpus.jsonl", Path(self.tmp) / "qa.jsonl"
        self.corpus.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
        self.qa.write_text("\n".join(json.dumps(q, ensure_ascii=False) for q in qa), encoding="utf-8")
        self.baseline = Path(self.tmp) / "baseline.json"

    def _eval(self, *extra: str) -> str:
        from learn_rag import cli

        out = io.StringIO()
        argv = ["learn-rag", "--log-level", "ERROR", "eval", "--dataset", "jsonl", "--qa", str(self.qa),
                "--corpus", str(self.corpus), "--retrieval-only", *extra]
        with patch.object(sys, "argv", argv), redirect_stdout(out):
            cli.main()
        return out.getvalue()

    def test_save_then_compare(self):
        self._eval("--save-baseline", str(self.baseline))
        self.assertIn("recall@5", load_baseline(self.baseline))
        self.assertIn("没有回归", self._eval("--baseline", str(self.baseline)))

        data = json.loads(self.baseline.read_text(encoding="utf-8"))
        data["summary"]["mrr@5"] = 1.5                                    # 基线高得不可能：一定回归
        self.baseline.write_text(json.dumps(data), encoding="utf-8")
        with self.assertRaises(SystemExit) as ctx:
            self._eval("--baseline", str(self.baseline))
        self.assertIn("mrr@5", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
