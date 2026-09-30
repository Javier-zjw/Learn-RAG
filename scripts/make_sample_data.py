"""生成内置示例数据集（离线可跑通全流程）。

语料主题就是 RAG 本身 —— 你一边评测，一边把这些概念读一遍。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DOCS = [
    ("doc_rag_overview", "RAG 概述",
     "检索增强生成（Retrieval-Augmented Generation，简称 RAG）是一种把外部知识库检索与大语言模型生成结合起来的技术。"
     "它的典型流程分为离线与在线两部分：离线阶段完成文档解析、切分、向量化与索引构建；"
     "在线阶段完成查询改写、召回、精排、上下文装配与答案生成。RAG 的主要价值是缓解模型知识过时和幻觉问题，"
     "并让答案具备可追溯的出处。"),
    ("doc_chunking", "文档切分",
     "切分（Chunking）是把长文档拆成检索单元的过程。常见策略包括定长滑窗切分、递归切分和结构感知切分。"
     "定长切分实现简单但容易在句子中间截断；递归切分优先按段落、句号等语义边界切分，超长时才降级为硬切；"
     "结构感知切分会利用标题层级，把标题路径拼接回正文，缓解片段脱离上下文后语义丢失的问题。"
     "切分粒度过小会丢失上下文，过大则会稀释语义并引入噪声，通常需要通过评测来确定合适的 chunk_size 和重叠长度。"),
    ("doc_embedding", "文本向量化",
     "向量化（Embedding）把文本映射成稠密向量，使语义相近的文本在向量空间中距离更近。"
     "检索时通常先对向量做 L2 归一化，这样内积就等价于余弦相似度。"
     "非对称检索模型要求查询侧和文档侧使用不同的指令前缀，忘记添加前缀会导致检索效果明显下降。"),
    ("doc_bm25", "BM25 稀疏检索",
     "BM25 是基于词频统计的经典检索算法，其打分公式包含三个关键部分：逆文档频率 idf 抑制高频无信息词；"
     "参数 k1 让词频收益饱和，避免一个词出现二十次就获得四倍分数；参数 b 惩罚长文档，因为长文档天然更容易包含查询词。"
     "BM25 擅长专有名词、型号和编号这类精确匹配场景，与稠密向量检索形成互补。"),
    ("doc_hybrid", "混合检索与 RRF",
     "混合检索把稠密向量召回与稀疏关键词召回的结果融合在一起。常用的融合方法是倒数排名融合 RRF，"
     "其公式为 score 等于各通道权重除以常数 k 加名次之和。RRF 只依赖名次而不依赖原始分数，"
     "因此天然免疫不同通道分数量纲不可比的问题，是工程上最省心且很难被打败的融合方式。常用的 k 取 60。"),
    ("doc_rerank", "重排精排",
     "重排（Rerank）是在召回结果内部重新打分排序的精排阶段。CrossEncoder 把查询和候选片段拼接后一起送入模型做交互注意力，"
     "判别力远强于双塔向量模型，但计算复杂度与候选数量成正比，只能用于小候选集。"
     "两阶段结构的意义在于：召回负责快而全，精排负责准，从而兼顾覆盖率与上下文纯净度。"),
    ("doc_query_rewrite", "查询改写与 HyDE",
     "查询改写把用户问题转换成更适合检索的形式。多查询改写让模型生成若干语义等价但用词不同的查询，分别召回后合并。"
     "HyDE 的做法是先让模型针对问题编写一段假设性答案，再用这段假设答案去检索，"
     "原理是假想答案与真实文档在表述风格上更接近，因此比原始问句更容易命中正确文档。"),
    ("doc_generation", "生成与拒答",
     "生成阶段需要解决三件事：按预算装配上下文、要求模型标注引用编号以保证可追溯、以及允许模型在证据不足时拒答。"
     "明确允许模型回答“资料中未提及”，是抑制幻觉最有效的手段之一。"
     "把所有证据一次性放入提示词的策略通常称为 stuff 策略。"),
    ("doc_metric_retrieval", "检索指标",
     "检索指标衡量证据找得准不准。Recall@k 表示前 k 个结果中命中的金标准文档数除以金标准文档总数，"
     "回答“该找的证据找回来了多少”。Precision@k 表示前 k 个结果中相关结果的比例。"
     "MRR 是第一个相关结果名次的倒数，衡量正确证据排得有多靠前。"
     "NDCG 在累积增益上按位置做对数折损并用理想排序归一化，是兼顾相关性与位置的综合排序指标。"),
    ("doc_metric_generation", "生成指标",
     "生成指标衡量答得对不对。EM 精确匹配要求归一化后完全一致，最为严格。"
     "Token 级 F1 计算预测与标准答案的词袋重叠，允许部分正确得分。"
     "ROUGE-L 基于最长公共子序列，与 F1 的区别在于它考虑词序。"
     "忠实度衡量答案是否完全由检索到的上下文支持，是幻觉的反向指标。"),
    ("doc_llm_judge", "LLM 裁判",
     "LLM-as-a-Judge 用更强的模型对答案打分，通常包含正确性、忠实性和相关性三个维度。"
     "使用时应把裁判模型的温度设为零，并要求模型先给出理由再输出分数，思维链能显著提升裁判的一致性。"
     "裁判模型应当强于被测系统，否则打分不可信。"),
    ("doc_eval_practice", "评测方法论",
     "RAG 评测必须分层进行，否则只能知道效果差却不知道差在哪。先看检索指标，检索不行就调切分、向量模型和融合策略；"
     "检索好而生成差，应调整提示词、更换生成模型或调整 top_k；检索与生成都好但忠实度差，说明模型在自由发挥，"
     "需要收紧提示词并强制要求标注引用。评测还应包含延迟等工程指标，用于权衡效果与成本。"),
]

QA = [
    ("q1", "RRF 融合公式里常用的常数 k 取多少？", ["60"], ["doc_hybrid"]),
    ("q2", "为什么向量要做 L2 归一化？", ["内积就等价于余弦相似度"], ["doc_embedding"]),
    ("q3", "BM25 里参数 b 的作用是什么？", ["惩罚长文档"], ["doc_bm25"]),
    ("q4", "MRR 指标衡量的是什么？", ["正确证据排得有多靠前"], ["doc_metric_retrieval"]),
    ("q5", "抑制幻觉最有效的手段之一是什么？", ["允许模型回答“资料中未提及”", "拒答"], ["doc_generation"]),
    ("q6", "HyDE 的基本原理是什么？", ["先让模型编写一段假设性答案，再用这段假设答案去检索"], ["doc_query_rewrite"]),
    ("q7", "CrossEncoder 为什么只能用于小候选集？", ["计算复杂度与候选数量成正比"], ["doc_rerank"]),
    ("q8", "ROUGE-L 和 Token F1 的区别是什么？", ["它考虑词序"], ["doc_metric_generation"]),
    ("q9", "结构感知切分是怎么缓解语义丢失的？", ["利用标题层级，把标题路径拼接回正文"], ["doc_chunking"]),
    ("q10", "LLM 裁判通常从哪三个维度打分？", ["正确性、忠实性和相关性"], ["doc_llm_judge"]),
    ("q11", "检索指标好但生成差应该怎么办？", ["调整提示词、更换生成模型或调整 top_k"], ["doc_eval_practice"]),
    ("q12", "RAG 的主要价值是什么？", ["缓解模型知识过时和幻觉问题"], ["doc_rag_overview"]),
]


def build_distractors(docs: list[tuple[str, str, str]], per_doc: int = 4) -> list[tuple[str, str, str]]:
    """为每篇文档生成若干"干扰文档"。

    为什么必须有干扰项:12 篇互不相似的文档,随便什么检索器都能拿满分,
    消融实验因此完全没有区分度(实测四个维度的 recall 全是 1.0000)。
    HotpotQA 每题配 8 篇干扰段落,正是同一个道理。

    构造方法:拿走含答案的那句,再掺入其他主题的句子。
    于是干扰文档与正确文档**共享大部分词汇却不含答案** ——
    这才能真正区分开 BM25、向量、混合三种召回。
    """
    import random

    random.seed(7)
    sentences: dict[str, list[str]] = {
        doc_id: [s for s in text.split("。") if s] for doc_id, _, text in docs
    }
    out: list[tuple[str, str, str]] = []
    for doc_id, title, text in docs:
        own = sentences[doc_id]
        others = [d for d in docs if d[0] != doc_id]
        for i in range(per_doc):
            keep = own[: max(1, len(own) // 2)]          # 保留前半段（共享词汇）
            borrowed = random.choice(sentences[random.choice(others)[0]])
            body = "。".join(keep[1:] + [borrowed]) + "。"  # 丢掉首句（通常含答案）
            out.append((f"{doc_id}_distract{i}", f"{title}（相关讨论{i+1}）", body))
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data"
    root.mkdir(parents=True, exist_ok=True)
    # 干净版：12 篇，互不相似。用于跑通链路、wiki 编译演示。
    with (root / "sample_corpus.jsonl").open("w", encoding="utf-8") as fh:
        for doc_id, title, text in DOCS:
            fh.write(json.dumps({"id": doc_id, "title": title, "text": text}, ensure_ascii=False) + "\n")

    # 困难版：12 + 48 篇干扰文档，写到**另一个文件**，不覆盖干净版。
    # 两者分开的原因：干扰项是为了给检索消融制造区分度，
    # 但它对 wiki 编译是病态输入（60 篇近似文档会互相刷矛盾标注）。
    # 不同实验用不同语料，别混为一谈。
    corpus = list(DOCS)
    if "--with-distractors" in sys.argv:
        corpus = list(DOCS) + build_distractors(DOCS)
        with (root / "sample_corpus_hard.jsonl").open("w", encoding="utf-8") as fh:
            for doc_id, title, text in corpus:
                fh.write(json.dumps({"id": doc_id, "title": title, "text": text}, ensure_ascii=False) + "\n")
        print(f"已生成困难版语料 {root/'sample_corpus_hard.jsonl'}（{len(corpus)} 篇，含干扰项）")
    with (root / "sample_qa.jsonl").open("w", encoding="utf-8") as fh:
        for sid, q, answers, gold in QA:
            fh.write(
                json.dumps(
                    {"id": sid, "question": q, "answers": answers, "gold_doc_ids": gold}, ensure_ascii=False
                )
                + "\n"
            )
    print(f"已生成 {root/'sample_corpus.jsonl'}（{len(DOCS)} 篇）与 {root/'sample_qa.jsonl'}")


if __name__ == "__main__":
    main()
