"""
eval.datasets —— 公开数据集适配层。

统一接口 EvalDataset：corpus() 给语料、samples() 给问题。
不管底层是 HotpotQA 的嵌套 JSON、CMRC 的 SQuAD 结构，还是你自己的 jsonl，
评测器看到的都一样 —— **适配的复杂度被关在这一层**。

已内置：
  jsonl        通用格式（推荐你自己的业务集用这个）
  hotpotqa     多跳英文问答，自带干扰文档 + 支撑句标注（检索指标的黄金测试集）
  squad_style  SQuAD 结构，覆盖中文 CMRC2018 / DuReader-robust / SQuAD v1.1
  beir_style   BEIR 三件套 corpus.jsonl / queries.jsonl / qrels.tsv（纯检索评测标准格式）

新增数据集 = 新增一个类 + 注册一行，评测器与指标一行不改。
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..core.interfaces import EvalDataset
from ..core.registry import registry
from ..core.types import Document, EvalSample


@registry.register("dataset", "jsonl")
class JsonlQADataset(EvalDataset):
    """通用 jsonl 数据集。

    qa.jsonl     : {"id", "question", "answers": [...], "gold_doc_ids": [...]}
    corpus.jsonl : {"id", "title", "text"}   （可选；不给就用已建好的知识库）

    字段名全部可配，因此绝大多数开源 QA 数据集导出成 jsonl 后都能直接吃。
    """

    def __init__(
        self,
        qa_path: str,
        corpus_path: str | None = None,
        *,
        name: str = "jsonl",
        question_field: str = "question",
        answer_field: str = "answers",
        gold_field: str = "gold_doc_ids",
        id_field: str = "id",
        limit: int | None = None,
    ) -> None:
        self.qa_path, self.corpus_path = Path(qa_path), Path(corpus_path) if corpus_path else None
        self.name = name
        self.question_field, self.answer_field = question_field, answer_field
        self.gold_field, self.id_field, self.limit = gold_field, id_field, limit

    def corpus(self) -> Iterable[Document]:
        if not self.corpus_path:
            return []
        docs = []
        for line in self.corpus_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(
                Document(
                    doc_id=str(row.get("id") or row.get("doc_id")),
                    text=row.get("text", ""),
                    metadata={"title": row.get("title", "")},
                )
            )
        return docs

    def samples(self) -> Iterable[EvalSample]:
        rows = [json.loads(l) for l in self.qa_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for i, row in enumerate(rows[: self.limit] if self.limit else rows):
            answers = row.get(self.answer_field, [])
            if isinstance(answers, str):
                answers = [answers]
            yield EvalSample(
                sample_id=str(row.get(self.id_field, i)),
                question=row[self.question_field],
                answers=[str(a) for a in answers],
                gold_doc_ids=[str(g) for g in row.get(self.gold_field, [])],
                # 透传全部额外字段：expected_tools（路由标注）、gold_weights（偏好权重）
                # 等 agentic 指标都从这里取值。**数据集不该为指标做特判**。
                metadata={k: v for k, v in row.items() if k not in {self.question_field, self.answer_field}},
            )


@registry.register("dataset", "hotpotqa")
class HotpotQADataset(EvalDataset):
    """HotpotQA（distractor 设置）。

    下载：https://hotpotqa.github.io  →  hotpot_dev_distractor_v1.json
    为什么它特别适合练 RAG 评测：
      - 每题给 10 篇段落，其中 2 篇是金标准、8 篇是**干扰项** → 能真实考验检索；
      - 自带 supporting_facts（支撑句）标注 → Recall/NDCG 等指标有金标准可算；
      - 需要跨两篇文档推理 → 单纯提高 top_k 也救不了，能逼出检索策略的差距。
    文档 id 直接用段落标题（HotpotQA 里标题唯一）。
    """

    name = "hotpotqa"

    def __init__(self, path: str, limit: int | None = None) -> None:
        self.path, self.limit = Path(path), limit
        self._raw: list[dict[str, Any]] | None = None

    def _load(self) -> list[dict[str, Any]]:
        if self._raw is None:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._raw = data[: self.limit] if self.limit else data
        return self._raw

    def corpus(self) -> Iterable[Document]:
        seen: set[str] = set()
        for item in self._load():
            for title, sentences in item.get("context", []):
                if title in seen:
                    continue
                seen.add(title)
                yield Document(doc_id=title, text="".join(sentences), metadata={"title": title})

    def samples(self) -> Iterable[EvalSample]:
        for i, item in enumerate(self._load()):
            gold = sorted({title for title, _ in item.get("supporting_facts", [])})
            yield EvalSample(
                sample_id=str(item.get("_id", i)),
                question=item["question"],
                answers=[item["answer"]],
                gold_doc_ids=gold,
                metadata={"type": item.get("type"), "level": item.get("level")},
            )


@registry.register("dataset", "squad_style")
class SquadStyleDataset(EvalDataset):
    """SQuAD 结构的数据集：中文 CMRC2018、DuReader-robust、英文 SQuAD v1.1 通用。

    下载参考：
      CMRC2018        https://github.com/ymcui/cmrc2018 （中文抽取式阅读理解）
      DuReader-robust https://github.com/baidu/DuReader
      SQuAD v1.1      https://rajpurkar.github.io/SQuAD-explorer/

    结构：data -> paragraphs -> {context, qas:[{id, question, answers:[{text}]}]}
    每个 paragraph 作为一篇文档，其自身即为该问题的金标准文档。
    注意：这类数据集段落较少、干扰弱，检索指标会偏高，适合入门与快速回归。
    """

    name = "squad_style"

    def __init__(self, path: str, limit: int | None = None) -> None:
        self.path, self.limit = Path(path), limit
        self._paragraphs: list[tuple[str, str, list[dict[str, Any]]]] | None = None

    def _load(self) -> list[tuple[str, str, str, list[dict[str, Any]]]]:
        """返回 (doc_id, title, context, qas) 列表。

        doc_id 优先用数据集自带的段落 id（CMRC 有 "DEV_0_QUERY_0" 这类字段），
        没有才退回"文章序号#段落序号"。
        早期版本用"标题#段落序号"，同名文章会撞 id，导致检索到**另一篇**
        同名文章也被算作命中 —— 召回虚高，而且完全不报错。
        """
        if self._paragraphs is None:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            out: list[tuple[str, str, str, list[dict[str, Any]]]] = []
            seen: set[str] = set()
            for a_idx, article in enumerate(raw.get("data", [])):
                title = article.get("title", f"article{a_idx}")
                art_id = str(article.get("id") or f"a{a_idx}")
                for p_idx, para in enumerate(article.get("paragraphs", [])):
                    doc_id = str(para.get("id") or f"{art_id}#{p_idx}")
                    if doc_id in seen:          # 兜底：数据集自带 id 也重复时强制加后缀
                        doc_id = f"{doc_id}#{a_idx}_{p_idx}"
                    seen.add(doc_id)
                    out.append((doc_id, title, para.get("context", ""), para.get("qas", [])))
            self._paragraphs = out
        return self._paragraphs

    def corpus(self) -> Iterable[Document]:
        for doc_id, title, context, _ in self._load():
            yield Document(doc_id=doc_id, text=context, metadata={"title": title})

    def samples(self) -> Iterable[EvalSample]:
        count = 0
        for doc_id, _, _, qas in self._load():
            for qa in qas:
                if self.limit and count >= self.limit:
                    return
                answers = [a["text"] for a in qa.get("answers", []) if a.get("text")]
                yield EvalSample(
                    sample_id=str(qa.get("id", count)),
                    question=qa["question"],
                    answers=answers or [""],
                    gold_doc_ids=[doc_id],
                )
                count += 1


@registry.register("dataset", "beir_style")
class BeirStyleDataset(EvalDataset):
    """BEIR 标准格式（纯检索评测）。

    目录结构：corpus.jsonl（_id/title/text）、queries.jsonl（_id/text）、
              qrels/test.tsv（query-id \t corpus-id \t score）
    数据集：https://github.com/beir-cellar/beir （MS MARCO、NQ、FiQA、SciFact…）
    中文可用 C-MTEB 检索子集，格式一致。

    这类数据集**没有标准答案**，只有相关性标注，
    因此只跑检索类指标；本框架会自动跳过需要答案的指标（见 runner）。
    """

    name = "beir"

    def __init__(self, root: str, split: str = "test", limit: int | None = None) -> None:
        self.root, self.split, self.limit = Path(root), split, limit

    def corpus(self) -> Iterable[Document]:
        for line in (self.root / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            yield Document(
                doc_id=str(row["_id"]),
                text=(row.get("title", "") + "\n" + row.get("text", "")).strip(),
                metadata={"title": row.get("title", "")},
            )

    def samples(self) -> Iterable[EvalSample]:
        queries = {}
        for line in (self.root / "queries.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                queries[str(row["_id"])] = row["text"]
        qrels: dict[str, list[str]] = {}
        with (self.root / "qrels" / f"{self.split}.tsv").open(encoding="utf-8") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader, None)
            for row in reader:
                if len(row) < 3:
                    continue
                qid, did, score = row[0], row[1], row[2]
                if float(score) > 0:
                    qrels.setdefault(qid, []).append(did)
        for i, (qid, docs) in enumerate(qrels.items()):
            if self.limit and i >= self.limit:
                return
            if qid in queries:
                yield EvalSample(sample_id=qid, question=queries[qid], answers=[], gold_doc_ids=docs)
