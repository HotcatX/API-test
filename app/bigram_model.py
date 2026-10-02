"""一个仅使用 Python 标准库的二元词组（bigram）文本生成器。"""

import random
import re
from collections import defaultdict
from collections.abc import Iterable


def _tokenize(text: str) -> list[str]:
    """忽略标点和大小写；保留 don't 这样的单词中的撇号。"""
    normalized = text.lower().replace("’", "'")
    return re.findall(r"[^\W\d_]+(?:'[^\W\d_]+)*", normalized)


class BigramModel:
    """记录相邻的两个词，再根据当前词随机选择下一个词。"""

    def __init__(self, corpus: Iterable[str]) -> None:
        if isinstance(corpus, str):
            raise ValueError("Corpus must be an iterable of text entries, not one string.")

        self._transitions: dict[str, list[str]] = defaultdict(list)
        self._vocabulary: set[str] = set()
        for entry in corpus:
            if not isinstance(entry, str):
                raise ValueError("Each corpus entry must be a string.")
            words = _tokenize(entry)
            self._vocabulary.update(words)
            # 在每条文本内配对，避免把上一条末尾连接到下一条开头。
            for current, following in zip(words, words[1:]):
                # 保留重复项：出现得越多，被随机选中的概率就越大。
                self._transitions[current].append(following)

        if not self._vocabulary:
            raise ValueError("Corpus must contain at least one word.")

    @property
    def vocabulary(self) -> list[str]:
        """返回按字母顺序排列的可用起始词。"""
        return sorted(self._vocabulary)

    def generate_text(self, start_word: str, length: int) -> str:
        """生成最多 length 个词（包含起始词）；没有后续词时提前结束。"""
        if isinstance(length, bool) or not isinstance(length, int) or length < 1:
            raise ValueError("Length must be a positive integer.")
        if not isinstance(start_word, str):
            raise ValueError("Start word must be a string containing one word.")
        words = _tokenize(start_word)
        if len(words) != 1:
            raise ValueError("Start word must contain exactly one word.")
        if words[0] not in self._vocabulary:
            raise ValueError(f"Unknown start word: {words[0]}")

        while len(words) < length:
            successors = self._transitions.get(words[-1])
            if not successors:
                break
            words.append(random.choice(successors))
        return " ".join(words)
