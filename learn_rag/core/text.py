"""
core.text —— 中英混合的文本归一化与切词。

一个典型的"深模块"：接口只有三四个函数，但内部把标点归一、大小写、冠词、中文按字切、英文按词切、n-gram 这些琐碎细节一次性解决。
BM25、Hashing 向量、EM/F1 指标全部复用它，保证"索引时怎么切词，评测时就怎么切词"，避免口径不一致。
"""

from __future__ import annotations

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


def _is_cjk(token: str) -> bool:
    return len(token) == 1 and re.match(rf"[{_CJK}]", token) is not None


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """字符 n-gram，供 Hashing 向量与模糊匹配使用"""
    s = normalize(text).replace(" ", "")
    if len(s) < n:
        return [s] if s else []
    return [s[i: i + n] for i in range(len(s) - n + 1)]
