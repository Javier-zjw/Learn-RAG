"""
eval.metrics —— RAG 评测指标。

RAG 的评测必须**分层**做，否则你只知道"效果差"，不知道差在哪：

  ① 检索指标（Retrieval）：证据找没找对 —— 决定了系统能力的上限。
  ② 生成指标（Generation）：给了证据，答得对不对 —— 衡量模型的利用能力。
  ③ 忠实/相关（Faithfulness / Relevance）：答案有没有超出证据乱说 —— 衡量幻觉。
  ④ 工程指标（Latency）：延迟与成本。

排障口诀：**先看①，①不行调检索；①好②差，调 prompt / 换模型 / 调 top_k；
①②都好但③差，说明模型在自由发挥，要收紧 prompt 并强制引用。**

所有指标共享同一个 Metric 接口，因此 runner 里没有一行 if 判断指标类型。
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

from ..core.interfaces import LLM, Metric
from ..core.registry import registry
from ..core.types import EvalSample, RagResult
from ..core.text import normalize, tokenize


# ① 检索指标：需要 gold_doc_ids 标注

class _RetrievalMetric(Metric):
    """
    检索指标基类：统一处理 top-k 截断与"命中列表"计算。

    把 relevance 序列（[1,0,0,1,...]）算好交给子类，
    子类只写自己那一行公式 —— 又是"基类吸收共性"的深类做法。
    """

    needs_gold_docs = True

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def _relevance(self, sample: EvalSample, result: RagResult) -> list[int]:
        gold = set(sample.gold_doc_ids)
        return [1 if doc_id in gold else 0 for doc_id in result.retrieved_doc_ids[: self.k]]

@registry.register("metric", "recall")
class RecallAtK(_RetrievalMetric):
    """
    Recall@k = 前 k 个结果里命中的金标准文档数 / 金标准文档总数。

    含义：**该找的证据找回来了多少**。RAG 最重要的单一指标 ——
    证据没召回，后面再强的模型也只能瞎猜。多跳问答（HotpotQA）里
    一个问题有多篇金标准文档，Recall 会明显低于 HitRate。
    """
    name = "recall"

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        gold = set(sample.gold_doc_ids)
        hit = len(gold & set(result.retrieved_doc_ids[: self.k]))
        return {f"recall@{self.k}": hit / len(gold) if gold else 0.0}

@registry.register("metric", "precision")
class PrecisionAtK(_RetrievalMetric):
    """
    Precision@k = 前 k 个结果中相关的比例。

    含义：**塞进 prompt 的东西有多干净**。它和 Recall 是一对矛盾：
    k 调大 Recall 涨、Precision 跌，噪声变多还会拖累生成质量。
    """

    name = "precision"
    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        rel = self._relevance(sample, result)
        return {f"precision@{self.k}": sum(rel) / len(rel) if rel else 0.0}

@registry.register("metric", "hit_rate")
class HitRateAtK(_RetrievalMetric):
    """
    HitRate@k（命中率 / Success@k）：前 k 个里**至少有一个**相关就记 1。

    含义：**有多少比例的问题，系统至少给对了一条线索**。
    对单跳问答，它约等于"检索的及格线"。
    """

    name = "hit_rate"

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        return {f"hit_rate@{self.k}": 1.0 if any(self._relevance(sample, result)) else 0.0}

@registry.register("metric", "mrr")
class MRRAtK(_RetrievalMetric):
    """
    MRR@k（平均倒数排名）= 1 / 第一个相关结果的名次。

    含义：**正确证据排得有多靠前**。第 1 位命中得 1.0，第 2 位 0.5，第 5 位 0.2。
    重排（rerank）好不好使，看它最灵敏 —— Recall 不变而 MRR 涨，就是排序变好了。
    """

    name = "mrr"

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        for i, rel in enumerate(self._relevance(sample, result), 1):
            if rel:
                return {f"mrr@{self.k}": 1.0 / i}
        return {f"mrr@{self.k}": 0.0}

@registry.register("metric", "map")
class MAPAtK(_RetrievalMetric):
    """
    MAP@k（平均精度均值）：在每个命中位置上算 Precision 再取平均。

    含义：比 MRR 更全面的排序质量指标 —— MRR 只看第一个命中，
    MAP 关心**所有**相关文档是否都排在前面。多证据场景（多跳、多文档）更有区分度。
    """

    name = "map"

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        rel = self._relevance(sample, result)
        hits, precisions = 0, []
        for i, r in enumerate(rel, 1):
            if r:
                hits += 1
                precisions.append(hits / i)

        denom = min(len(sample.gold_doc_ids), self.k) or 1

        return {f"map@{self.k}": sum(precisions) / denom}

@registry.register("metric", "ndcg")
class NDCGAtK(_RetrievalMetric):
    """
    NDCG@k（归一化折损累积增益）。

    DCG = Σ rel_i / log2(i+1)：名次越靠后，同样的相关性贡献越小（"折损"）。
    再除以理想排序的 IDCG 做归一化，得到 0~1 的分数。

    含义：**兼顾"相关性强弱"和"位置先后"的综合排序质量**。
    支持分级相关性（很相关=2，一般=1，不相关=0），
    是搜索/推荐里最主流的排序指标；本实现用二值相关性。
    """

    name = "ndcg"

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        rel = self._relevance(sample, result)
        dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rel, 1))
        ideal = [1] * min(len(sample.gold_doc_ids), self.k)
        idcg = sum(r / math.log2(i + 1) for i, r in enumerate(ideal, 1))
        return {f"ndcg@{self.k}": dcg / idcg if idcg else 0.0}


# ② 上下文质量指标：不需要证据标注，只需要标准答案
@registry.register("metric", "context_recall")
class ContextRecall(Metric):
    """
    上下文召回（无标注版）：标准答案的词有多少能在检索到的上下文里找到。

    含义：**检索到的资料是否包含了回答所需的信息**。
    价值极高 —— 现实业务数据集通常没有"金标准文档"标注，
    但一定有标准答案；这个指标让你在零标注成本下也能评检索。
    同时还输出 answer_in_context（答案是否整体出现在上下文里，0/1）。
    """

    name = "context_recall"
    needs_gold_answer = True

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        context = normalize(" ".join(result.context_texts))
        best_cover, contained = 0.0, 0.0

        for ans in sample.answers:
            tokens = tokenize(ans)
            if not tokens:
                continue

            cover = sum(1 for t in tokens if t in context) / len(tokens)
            best_cover = max(best_cover, cover)
            if normalize(ans) and normalize(ans) in context:
                contained = 1.0

        return {"context_recall": best_cover, "answer_in_context": contained}

# ③ 生成指标
@registry.register("metric", "exact_match")
class ExactMatch(Metric):
    """
    EM（精确匹配）：归一化后预测与任一标准答案完全相同才得 1 分。

    含义：**最严格的正确性**。适合短答案抽取型任务（HotpotQA、CMRC、NQ）。
    对开放式长答案几乎恒为 0，此时应看 F1 或 LLM Judge。
    """

    name = "exact_match"
    needs_gold_answer = True

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        pred = normalize(result.answer.text)
        return {"exact_match": 1.0 if any(pred == normalize(a) for a in sample.answers) else 0.0}

@registry.register("metric", "answer_contains")
class AnswerContains(Metric):
    """
    包含匹配：标准答案是否作为子串出现在预测里（取多答案最优）。

    含义：**宽松版正确性**。生成式模型常在正确答案外附带解释，
    EM 会误判为错，这个指标能更贴近人的判断（但也更容易"蒙对"）。
    """

    name = "answer_contains"
    needs_gold_answer = True

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        pred = normalize(result.answer.text)
        hit = any(normalize(a) and normalize(a) in pred for a in sample.answers)
        return {"answer_contains": 1.0 if hit else 0.0}

@registry.register("metric", "token_f1")
class TokenF1(Metric):
    """
    Token 级 F1（SQuAD 口径）：预测与标准答案的词袋重叠。

    Precision = 命中词 / 预测词数，Recall = 命中词 / 标准答案词数，F1 是调和平均。
    含义：**部分正确也能拿分**。答对了但多说了几句 -> Precision 掉；
    只答对一半 -> Recall 掉。中文按字计算（见 core.text），无需分词器。
    """

    name = "token_f1"
    needs_gold_answer = True

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        pred = tokenize(result.answer.text)
        best = {"token_f1": 0.0, "token_precision": 0.0, "token_recall": 0.0}
        for ans in sample.answers:
            gold = tokenize(ans)
            score = _f1(pred, gold)
            if score["token_f1"] > best["token_f1"]:
                best = score
        return best

@registry.register("metric", "rouge_l")
class RougeL(Metric):
    """
    ROUGE-L：基于最长公共子序列（LCS）的 F1。

    含义：**考虑词序的重叠度**。与 Token-F1 的区别在于"顺序"：
    "北京是中国的首都" vs "首都中国是北京" —— Token-F1 满分，ROUGE-L 会扣分。
    摘要式/长答案评测常用。
    """

    name = "rouge_l"
    needs_gold_answer = True

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        pred = tokenize(result.answer.text)
        best = 0.0
        for ans in sample.answers:
            gold = tokenize(ans)
            if not pred or not gold:
                continue
            lcs = _lcs_length(pred, gold)
            if lcs:
                p, r = lcs / len(pred), lcs / len(gold)
                best = max(best, 2 * p * r / (p + r))

        return {"rouge_l": best}

@registry.register("metric", "faithfulness")
class LexicalFaithfulness(Metric):
    """
    忠实度（词汇代理版）：预测里的实词有多少能在上下文中找到依据。

    含义：**答案有多少是"有据可查"的**，即幻觉的反向指标。
    这是启发式近似（真正的忠实度评判需要 LLM，见 LLMJudge），
    但胜在零成本、可批量跑，非常适合做回归监控。
    同时输出 citation_rate：答案是否带了引用编号 —— 可审计性的量化。
    """

    name = "faithfulness"

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        context = set(tokenize(" ".join(result.context_texts)))
        pred = [t for t in tokenize(result.answer.text) if len(t) > 0]
        if not pred:
            return {"faithfulness": 0.0, "citation_rate": 0.0}

        supported = sum(1 for t in pred if t in context) / len(pred)

        return {
            "faithfulness": supported,
            "citation_rate": 1.0 if result.answer.citations else 0.0
        }

@registry.register("metric", "latency")
class Latency(Metric):
    """
    工程指标：各阶段耗时（秒）。

    含义：RAG 的**成本与体验**。上线前必须看它 ——
    加了 LLM 重排、多路改写之后，指标涨了 2 分但延迟涨了 5 秒，是否值得？
    这类权衡只有把指标摆在一起才谈得清楚。
    """

    name = "latency"

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        return {f"latency_{k}": v for k, v in result.timings.items()}

# ④ LLM Judge：用大模型当裁判
@registry.register("metric", "llm_judge")
class LLMJudge(Metric):
    """
    LLM-as-a-Judge：让强模型对答案打分（0~1）。

    三个维度（业界通用，RAGAS/TruLens 也是这套）：
      - correctness（正确性）：与标准答案是否语义一致。
        解决 EM/F1 无法处理"说法不同但意思对"的问题。
      - faithfulness（忠实性）：答案是否**完全**由给定上下文支持，有无编造。
      - relevance（相关性）：答案是否切题、有没有答非所问或答了一堆废话。

    注意事项：裁判模型要比被测系统强；temperature 设 0；
    prompt 里要求先给理由再给分（思维链能显著提升裁判一致性）。
    """

    name = "llm_judge"
    needs_gold_answer = False

    PROMPT = (
        "你是严格的评测裁判。请根据【问题】【参考资料】【标准答案】评价【待评答案】。\n"
        "按三个维度打分，每项 0~1 的小数：\n"
        "- correctness：与标准答案是否语义一致（无标准答案时，判断事实是否成立）\n"
        "- faithfulness：是否完全由参考资料支持，有无编造\n"
        "- relevance：是否切题、简洁\n"
        '先简短说明理由，最后单独一行输出 JSON：{{"correctness":x,"faithfulness":y,"relevance":z}}\n\n'
        "【问题】{question}\n\n【参考资料】\n{context}\n\n【标准答案】{gold}\n\n【待评答案】{pred}"
    )

    def __init__(self, llm: LLM, max_context_char: int = 3000) -> None:
        self.llm = llm
        self.max_context_char = max_context_char

    def compute(self, sample: EvalSample, result: RagResult) -> dict[str, float]:
        context = "\n---\n".join(result.context_texts)[: self.max_context_char]
        prompt = self.PROMPT.format(
            question=sample.question,
            context=context or "（无）",
            gold=" / ".join(sample.answers) or "（无标准答案）",
            pred=result.answer.text or "（空）"
        )

        try:
            raw = self.llm.ask(prompt)
            payload = json.loads(re.findall(r"\{[^{}]*\}", raw)[-1])
            return {
                "judge_correctness": float(payload.get("correctness", 0.0)),
                "judge_faithfulness": float(payload.get("faithfulness", 0.0)),
                "judge_relevance": float(payload.get("relevance", 0.0))
            }
        except Exception:
            return {}

# 工具函数
def _f1(pred: Sequence[str], gold: Sequence[str]) -> dict[str, float]:
    if not pred or not gold:
        return {"token_f1": 0.0, "token_precision": 0.0, "token_recall": 0.0}

    from collections import Counter

    common = Counter(pred) & Counter(gold)
    same = sum(common.values())
    if same == 0:
        return {"token_f1": 0.0, "token_precision": 0.0, "token_recall": 0.0}
    p, r = same / len(pred), same / len(gold)

    return {"token_f1": 2 * p * r / (p + r), "token_precision": p, "token_recall": r}

def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            prev, dp[j] = dp[j], prev + 1 if x == y else max(dp[j], dp[j - 1])

    return dp[len(b)]

def default_metrics(k: int = 5, llm: LLM | None = None) -> list[Metric]:
    """
    一套开箱即用的指标组合：检索 + 上下文 + 生成 + 忠实 + 延迟
    """

    metrics: list[Metric] = [
        RecallAtK(k),
        PrecisionAtK(k),
        HitRateAtK(k),
        MRRAtK(k),
        MAPAtK(k),
        NDCGAtK(k),
        ContextRecall(),
        ExactMatch(),
        AnswerContains(),
        TokenF1(),
        RougeL(),
        LexicalFaithfulness(),
        Latency()
    ]
    if llm is not None:
        metrics.append(LLMJudge(llm))
    return metrics