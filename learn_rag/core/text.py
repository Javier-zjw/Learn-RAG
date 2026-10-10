"""
core.text —— 中英混合的文本归一化与切词。

一个典型的"深模块"：接口只有三四个函数，但内部把标点归一、大小写、冠词、中文按字切、英文按词切、n-gram 这些琐碎细节一次性解决。
BM25、Hashing 向量、EM/F1 指标全部复用它，保证"索引时怎么切词，评测时就怎么切词"，避免口径不一致。
"""

from __future__ import annotations

import math
import re
import string
import unicodedata

# 中日韩统一表意文字
_CJK = r"\u4e00-\u9fff\u3400-\u4dbf"
_TOKEN_RE = re.compile(rf"[{_CJK}]|[a-zA-Z]+|[0-9]+")
_PUNCT = set(string.punctuation) | set("，。！？；：、“”‘’（）《》【】…—　")
_EN_STOP = {"a", "an", "the", "of", "and", "or", "is", "are", "was", "were", "to", "in", "on", "for"}


def normalize(text: str) -> str:
    """
    答案/文本归一化：全角转半角、小写、去标点、压缩空白
    EM/F1 等指标对归一化非常敏感，统一放在这里，杜绝各处各写一套
    """

    text = unicodedata.normalize("NFKC", text or "").lower()
    text = "".join(" " if ch in _PUNCT else ch for ch in text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str, *, drop_stopwords: bool = False) -> list[str]:
    """
    中文按字、英文/数字按词切分
    中文按字切是检索场景的稳健选择：不依赖任何分词词典和第三方库,配合下面的 bigram 也能拿到接近词级的效果
    """

    tokens = _TOKEN_RE.findall(normalize(text))
    if drop_stopwords:
        tokens = [t for t in tokens if t not in _EN_STOP]
    return tokens


def index_tokens(text: str) -> list[str]:
    """
    检索用切词：在基础 token 上追加中文 bigram，提升中文短语区分度
    例：'机器学习' -> ['机','器','学','习','机器','器学','学习']
    """
    tokens = tokenize(text, drop_stopwords=True)
    grams = [
        tokens[i] + tokens[i + 1]
        for i in range(len(tokens) - 1)
        if _is_cjk(tokens[i]) and _is_cjk(tokens[i + 1])
    ]
    return tokens + grams


# 编号：字母数字段，段之间可以用 - _ / . 连接，如 XH-2025-0386、SKU_12/A、v2.3.1、GPT4o、2025-10-08
_CODE_RE = re.compile(r"[a-z0-9]+(?:[-_/.][a-z0-9]+)*")


def code_tokens(text: str) -> list[str]:
    """
    整体保留的编号，供关键词检索使用：合同号、订单号、型号、版本号、日期。

    tokenize 会把 XH-2025-0386 拆成 xh、2025、0386 三个词：订单号里反复出现的 xh、日期里反复出现的 2025
    会压过真正有区分度的 0386，搜编号时反而排不到目标文档。把编号整体也作为一个词，它在语料里通常只出现
    在少数片段中，idf 很高，精确匹配时能稳稳排到前面。
    只保留含数字、并且含字母或连接符的片段：纯数字（2025）和纯字母单词已经由 tokenize 产出，不必重复。
    """
    text = unicodedata.normalize("NFKC", text or "").lower()
    return [
        m for m in _CODE_RE.findall(text)
        if any(ch.isdigit() for ch in m) and (any(ch.isalpha() for ch in m) or any(ch in "-_/." for ch in m))
    ]


def _is_cjk(token: str) -> bool:
    return len(token) == 1 and re.match(rf"[{_CJK}]", token) is not None


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """字符 n-gram，供 Hashing 向量与模糊匹配使用"""
    s = normalize(text).replace(" ", "")
    if len(s) < n:
        return [s] if s else []
    return [s[i: i + n] for i in range(len(s) - n + 1)]


# 估算 token 用的切分：中文单字、英文单词、数字串、连续符号各算一段
_COUNT_RE = re.compile(rf"([{_CJK}])|([a-zA-Z]+)|([0-9]+)|([^\s{_CJK}a-zA-Z0-9]+)")


def count_tokens(text: str) -> int:
    """
    估算文本的 token 数，切分器按它控制块大小。

    按字符数控制大小在中英混排时会失真：300 个字符的中文约 300 token，英文却只有 70 左右。
    这里按常见分词器的经验值估算：中文每字 1 个，英文每词约 1.3 个，数字每 3 位 1 个，
    一串连续符号（标点、Markdown 表格的 "| --- |"）1 个。误差在一两成以内，
    用来定块大小足够，不必为此引入具体模型的分词器。
    结果向上取整：切分器把各段的估算值相加来装箱，取整只会偏大，装出来的块就不会超出预算。
    """
    total = 0.0
    for cjk, word, digits, _symbols in _COUNT_RE.findall(text or ""):
        if cjk:
            total += 1
        elif word:
            total += 1.3
        elif digits:
            total += (len(digits) + 2) // 3
        else:
            total += 1
    return math.ceil(total)
