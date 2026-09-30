"""
检查 data/ 下的数据集是否能被本项目正确读取。

**在跑评测之前先跑这个。** 数据集的目录层级、文件名、字段名千奇百怪，
解压出来多一层目录是家常便饭。花 10 秒确认结构，比对着一堆 0 分的指标
猜半小时强得多。

用法：
    python scripts/inspect_dataset.py                    # 扫描整个 data/
    python scripts/inspect_dataset.py --path data/scifact
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    """安全地显示相对路径：传进来的可能是相对路径，也可能在项目外。"""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _head(path: Path, n: int = 1) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def _count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def check_beir(root: Path) -> bool:
    """BEIR 三件套：corpus.jsonl / queries.jsonl / qrels/*.tsv"""
    corpus, queries = root / "corpus.jsonl", root / "queries.jsonl"
    qrels_dir = root / "qrels"
    if not (corpus.exists() and queries.exists() and qrels_dir.is_dir()):
        return False

    print(f"\n✅ BEIR 格式：{rel(root)}")
    print(f"   corpus.jsonl   {_count_lines(corpus):>8,} 篇")
    print(f"   queries.jsonl  {_count_lines(queries):>8,} 条")
    splits = sorted(p.stem for p in qrels_dir.glob("*.tsv"))
    print(f"   qrels/         {splits}")
    for split in splits:
        n = sum(1 for _ in (qrels_dir / f"{split}.tsv").open(encoding="utf-8")) - 1  # 减去表头
        print(f"     {split}.tsv  {n:>8,} 条标注")

    sample = _head(corpus)[0]
    print(f"   corpus 字段    {list(sample.keys())}")
    if "_id" not in sample or "text" not in sample:
        print("   ⚠ 缺少 _id 或 text 字段，BeirStyleDataset 读不了")
        return True
    print(f"   示例文档       _id={sample['_id']!r} title={str(sample.get('title',''))[:30]!r}")
    print(f"   → 用法：--dataset beir_style --qa {rel(root)}")
    print("   → 注意：BEIR **没有标准答案**，只能跑检索类指标（生成类会被自动跳过）")
    return True


def check_squad(path: Path) -> bool:
    """SQuAD 结构：data -> paragraphs -> qas"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(raw, dict) or "data" not in raw:
        return False

    articles = raw["data"]
    paragraphs = sum(len(a.get("paragraphs", [])) for a in articles)
    qas = sum(len(p.get("qas", [])) for a in articles for p in a.get("paragraphs", []))
    print(f"\n✅ SQuAD 结构：{rel(path)}")
    print(f"   文章 {len(articles):,} 篇 / 段落 {paragraphs:,} 个 / 问题 {qas:,} 条")
    first_qa = next((q for a in articles for p in a.get("paragraphs", []) for q in p.get("qas", [])), None)
    if first_qa:
        print(f"   示例问题       {str(first_qa.get('question',''))[:40]!r}")
        print(f"   示例答案       {[a.get('text') for a in first_qa.get('answers', [])][:2]}")
    print(f"   → 用法：--dataset squad_style --qa {rel(path)}")
    return True


def check_hotpot(path: Path) -> bool:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(raw, list) or not raw or "supporting_facts" not in raw[0]:
        return False
    print(f"\n✅ HotpotQA：{rel(path)}")
    print(f"   样本 {len(raw):,} 条，每题 {len(raw[0].get('context', []))} 段（含干扰项）")
    print(f"   → 用法：--dataset hotpotqa --qa {rel(path)} --limit 300")
    return True


def check_jsonl_qa(path: Path) -> bool:
    """本项目的通用 jsonl 格式"""
    try:
        rows = _head(path, 2)
    except Exception:
        return False
    if not rows or "question" not in rows[0]:
        return False
    print(f"\n✅ 通用 QA jsonl：{rel(path)}  {_count_lines(path):,} 条")
    print(f"   字段           {list(rows[0].keys())}")
    has_gold = "gold_doc_ids" in rows[0] and rows[0]["gold_doc_ids"]
    print(f"   证据标注       {'有 → 检索指标可算' if has_gold else '无 → 只能靠 context_recall 评检索'}")
    return True


def scan(base: Path) -> None:
    found = 0
    # 目录型（BEIR）
    for d in sorted([base, *[p for p in base.rglob("*") if p.is_dir()]]):
        if check_beir(d):
            found += 1
    # 文件型
    for f in sorted(base.rglob("*.json")):
        if check_hotpot(f) or check_squad(f):
            found += 1
    for f in sorted(base.rglob("*.jsonl")):
        if f.name in {"corpus.jsonl", "queries.jsonl"}:
            continue  # 已在 BEIR 检查里覆盖
        if check_jsonl_qa(f):
            found += 1

    if found == 0:
        print(f"\n❌ 在 {base} 下没找到可识别的数据集。")
        print("   常见原因：解压后多了一层目录，比如 data/scifact/scifact/corpus.jsonl")
        print("   处理办法：mv data/scifact/scifact/* data/scifact/ && rmdir data/scifact/scifact")
    else:
        print(f"\n共识别 {found} 个数据集。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data", help="要扫描的目录")
    args = ap.parse_args()
    base = Path(args.path)
    if not base.exists():
        raise SystemExit(f"目录不存在：{base}")
    print(f"扫描 {base} ...")
    scan(base)


if __name__ == "__main__":
    main()
