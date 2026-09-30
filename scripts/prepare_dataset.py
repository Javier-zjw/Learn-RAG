"""
把公开数据集转换成本项目的通用 jsonl 格式（corpus.jsonl + qa.jsonl）。

为什么要转换：本项目已经内置了 hotpotqa / squad_style / beir_style 三个适配器，
可以**直接评测**；但转成统一 jsonl 后，语料和问题都能被 grep、被抽样、被人肉检查 ——
评测出问题时，第一步永远是去看数据本身。

下载地址（都可以直接 wget）：
  HotpotQA (dev, distractor, 英文多跳，含干扰文档与支撑句标注)
    http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json
  CMRC2018 (中文抽取式阅读理解，SQuAD 结构)
    https://github.com/ymcui/cmrc2018  → squad-style-data/cmrc2018_dev.json
  DuReader-robust (中文，SQuAD 结构)
    https://github.com/baidu/DuReader
  SQuAD v1.1 (英文)
    https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
  BEIR (纯检索基准，MS MARCO / NQ / SciFact / FiQA…)
    https://github.com/beir-cellar/beir

用法：
    python scripts/prepare_dataset.py --kind hotpotqa \
        --src data/hotpot_dev_distractor_v1.json --out data/hotpot --limit 500
    python scripts/prepare_dataset.py --kind squad_style --src data/cmrc2018_dev.json --out data/cmrc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minirag.eval.datasets import BeirStyleDataset, HotpotQADataset, SquadStyleDataset

BUILDERS = {
    "hotpotqa": HotpotQADataset,
    "squad_style": SquadStyleDataset,
    "beir_style": BeirStyleDataset,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=sorted(BUILDERS))
    ap.add_argument("--src", required=True, help="原始文件路径（beir 传目录）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    dataset = BUILDERS[args.kind](args.src, limit=args.limit)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n_doc = 0
    with (out / "corpus.jsonl").open("w", encoding="utf-8") as fh:
        for doc in dataset.corpus():
            fh.write(
                json.dumps(
                    {"id": doc.doc_id, "title": doc.metadata.get("title", ""), "text": doc.text},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_doc += 1

    n_qa = 0
    with (out / "qa.jsonl").open("w", encoding="utf-8") as fh:
        for s in dataset.samples():
            fh.write(
                json.dumps(
                    {
                        "id": s.sample_id,
                        "question": s.question,
                        "answers": s.answers,
                        "gold_doc_ids": s.gold_doc_ids,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n_qa += 1

    print(f"完成：{out}/corpus.jsonl（{n_doc} 篇） {out}/qa.jsonl（{n_qa} 条）")


if __name__ == "__main__":
    main()
