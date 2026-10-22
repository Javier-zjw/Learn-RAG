"""
learn-rag tune 的测试：调参时模拟出来的排序必须和线上管线一模一样，否则调出来的参数没有意义。
知识库用 hashing + flat，精排用词重叠（不能缓存分数）和一个逐个打分的假精排器（可以缓存）各测一遍。
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

from learn_rag.core.interfaces import QueryTransformer, Reranker
from learn_rag.core.registry import registry
from learn_rag.core.text import index_tokens
from learn_rag.core.types import EvalSample, ScoredChunk
from learn_rag.eval.tuning import FusionTuner, default_grid
from learn_rag.pipeline.rag import RagPipeline
from learn_rag.retrieval.retrievers import HybridRetriever

_DOCS = {
    "报销制度.md": "# 报销制度\n\n## 时限\n\n" + "差旅报销需在出差结束后三十天内提交，逾期需部门负责人书面说明原因。" * 3
                 + "\n\n## 标准\n\n" + "一线城市住宿标准为每晚六百元，其他城市为每晚四百元。" * 3,
    "安装手册.md": "# 安装手册\n\n## 环境\n\n" + "安装前确认服务器内存不少于十六GB，并开放八零八零端口。" * 3
                 + "\n\n## 备份\n\n" + "数据库每晚两点全量备份，保留三十天，备份失败自动告警。" * 3,
    "考勤制度.md": "# 考勤制度\n\n## 请假\n\n" + "病假三天以内由直属主管审批，超过三天需提交医院证明。" * 3
                 + "\n\n## 加班\n\n" + "工作日加班可以调休，节假日加班按三倍工资结算。" * 3,
}

_QA = [
    ("报销要在出差后多少天内提交", "差旅报销需在出差结束后三十天内提交"),
    ("一线城市住宿每晚多少钱", "一线城市住宿标准为每晚六百元"),
    ("服务器内存要求", "安装前确认服务器内存不少于十六GB"),
    ("数据库备份保留多久", "数据库每晚两点全量备份，保留三十天"),
    ("病假超过三天要什么材料", "超过三天需提交医院证明"),
    ("节假日加班怎么算", "节假日加班按三倍工资结算"),
]


class _OverlapReranker(Reranker):
    """逐个打分：分数只看问题和片段共有多少个词，同分很多，正好检验同分时的顺序也和线上一致。"""

    pointwise = True

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, candidates, top_k):
        self.calls += 1
        q = set(index_tokens(query))
        scored = [ScoredChunk(c.chunk, float(len(q & set(index_tokens(c.text)))), source="rerank:overlap")
                  for c in candidates]
        return sorted(scored, key=lambda h: -h.score)[:top_k]


class _Suffix(QueryTransformer):
    def transform(self, query):
        return [query, query + "的规定"]


class TestDefaultGrid(unittest.TestCase):
    def test_weights_deduplicated_by_ratio(self):
        grid = default_grid(["vector", "bm25"], top_k=5)
        weight_sets = {tuple(c["weights"].values()) for c in grid}
        self.assertNotIn((0.0, 0.0), weight_sets)
        self.assertIn((1.0, 0.0), weight_sets)                        # 同一比例保留最大权重为 1 的写法
        self.assertNotIn((2.0, 0.0), weight_sets)                    # 和 1:0 是同一种排序
        self.assertEqual(sum(1 for w in weight_sets if w[0] == w[1]), 1)   # 0.5:0.5、1:1、2:2 只留一个
        self.assertEqual(len(weight_sets), 7)                          # 1:0 0:1 1:1 0.5:1 1:0.5 0.5:2 2:0.5
        self.assertTrue(all(c["candidate_k"] >= 5 for c in grid))
        self.assertFalse(any(c["candidate_k"] < 50 for c in default_grid(["vector"], top_k=30)))


class TestFusionTuner(unittest.TestCase):
    def setUp(self) -> None:
        from learn_rag.server.library import Library

        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.data = str(Path(self.tmp) / "web")
        self.library = Library(self.data, worker=True)
        upload_id = None
        for name, text in _DOCS.items():
            upload_id, _ = self.library.staging.save(upload_id, name, io.BytesIO(text.encode("utf-8")))
        created = self.library.create("制度库", {"chunker": {"type": "structure", "chunk_size": 30, "parent_size": 120},
                                               "encoder": {"type": "hashing", "model": "hashing"},
                                               "reranker": {"type": "lexical"},
                                               "retrieval": {"candidate_k": 20, "top_k": 3},
                                               "index": {"type": "flat"}}, upload_id)
        self.assertEqual(self.library.jobs.wait(created["job"]["id"]).status, "done")
        self.kb = created["kb"]["id"]
        self.pipe = self.library.pipeline(self.kb)
        self.channels = {n: registry.build("retriever", {"type": n}, kb=self.pipe.kb) for n in ("vector", "bm25")}
        self.samples = [EvalSample(sample_id=str(i), question=q, evidence=[e]) for i, (q, e) in enumerate(_QA)]

    def _assert_same_as_pipeline(self, reranker) -> None:
        self.pipe.reranker = reranker
        tuner = FusionTuner(self.pipe, self.channels)
        grid = default_grid(list(self.channels), self.pipe.top_k)
        fetch = max(c["candidate_k"] for c in grid) * 2
        for sample in self.samples:
            lists, scores = tuner._prepare(sample, fetch)
            for config in grid[::5]:
                names = [n for n, w in config["weights"].items() if w > 0]
                real = RagPipeline(kb=self.pipe.kb, retriever=HybridRetriever(
                    [self.channels[n] for n in names], [config["weights"][n] for n in names], rrf_k=config["rrf_k"]),
                    generator=self.pipe.generator, reranker=reranker, top_k=self.pipe.top_k,
                    candidate_k=config["candidate_k"])
                expected = [h.chunk.chunk_id for h in real.retrieve_only(sample.question).contexts]
                simulated = [h.chunk.chunk_id for h in tuner._simulate(sample.question, lists, scores, config)]
                self.assertEqual(simulated, expected, (sample.question, config))

    def test_simulation_matches_pipeline_with_lexical_reranker(self):
        self._assert_same_as_pipeline(self.pipe.reranker)

    def test_simulation_matches_pipeline_with_cached_pointwise_scores(self):
        reranker = _OverlapReranker()
        self._assert_same_as_pipeline(reranker)

    def test_pointwise_reranker_scores_each_question_once(self):
        reranker = _OverlapReranker()
        self.pipe.reranker = reranker
        rows = FusionTuner(self.pipe, self.channels).run(self.samples)
        self.assertEqual(reranker.calls, len(self.samples))           # 几十组参数，每题只精排一次
        keys = [r.key(3) for r in rows]
        self.assertEqual(keys, sorted(keys, reverse=True))
        self.assertGreater(rows[0].scores["evidence_hit@3"], 0.5)

    def test_compare_transforms_uses_the_real_pipeline(self):
        tuner = FusionTuner(self.pipe, self.channels)
        config = {"weights": {"vector": 1.0, "bm25": 1.0}, "rrf_k": 60, "candidate_k": 20}
        row = tuner.run(self.samples, [config])[0]
        report = tuner.compare_transforms(self.samples, config, {"none": None, "suffix": _Suffix()})
        self.assertEqual(set(report), {"none", "suffix"})
        self.assertEqual(report["none"], row.scores)                  # 真实检索和模拟的分数一致

    def test_samples_without_evidence_are_rejected(self):
        with self.assertRaises(ValueError):
            FusionTuner(self.pipe, self.channels).run([EvalSample(sample_id="x", question="随便问问")])

    def test_cli_ranks_and_applies(self):
        from learn_rag import cli

        qa = Path(self.tmp) / "qa.jsonl"
        qa.write_text("\n".join(json.dumps({"id": str(i), "question": q, "evidence": [e]}, ensure_ascii=False)
                                for i, (q, e) in enumerate(_QA)), encoding="utf-8")
        out, report = io.StringIO(), Path(self.tmp) / "tune.json"
        argv = ["learn-rag", "--log-level", "ERROR", "tune", "--kb", self.kb, "--data", self.data, "--qa", str(qa),
                "--apply", "--out", str(report)]
        with patch.object(sys, "argv", argv), redirect_stdout(out):
            cli.main()
        printed = out.getvalue()
        self.assertIn("← 当前", printed)
        rows = json.loads(report.read_text(encoding="utf-8"))
        best = rows[0]["config"]

        from learn_rag.server.library import Library

        retrieval = Library(self.data).info(self.kb)["settings"]["retrieval"]
        self.assertEqual(retrieval["rrf_k"], best["rrf_k"])
        self.assertEqual(retrieval["candidate_k"], best["candidate_k"])
        self.assertEqual({n: retrieval["weights"][n] for n in ("vector", "bm25")}, best["weights"])
        self.assertEqual(retrieval["top_k"], 3)                        # 没参与调参的设置保持不变
        self.assertEqual(retrieval["query_transform"], "none")


if __name__ == "__main__":
    unittest.main()
