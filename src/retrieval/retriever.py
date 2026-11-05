"""Tiny dependency-free BM25 retriever over the policy excerpts.

Retrieval is a SEPARATE component from the language model: it decides which policy
text is shown to the SLM. Quality of retrieval is measured on its own (hit rate)
so that retrieval errors are not confused with model errors.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "am", "do", "does", "i", "my", "me", "we", "our", "you", "your",
    "of", "to", "in", "on", "for", "and", "or", "can", "how", "what", "when", "who", "which", "will",
    "be", "it", "if", "at", "by", "with", "this", "that", "there", "get", "have", "has", "much", "many",
}


def tokenize(text: str) -> list[str]:
    toks = _TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in toks if t not in _STOPWORDS]


def _stem(tok: str) -> str:
    """Very light suffix stripping so that 'days'/'day' and 'leaves'/'leave' match."""
    for suf in ("ing", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


@dataclass
class Hit:
    policy_id: str
    title: str
    text: str
    score: float


class BM25Retriever:
    def __init__(self, policies: list[dict], k1: float = 1.5, b: float = 0.75):
        self.policies = policies
        self.k1, self.b = k1, b
        # Index title + text so that questions mentioning the topic name also match.
        self._docs = [tokenize(p["title"] + " " + p["text"]) for p in policies]
        self._doc_len = [len(d) for d in self._docs]
        self._avgdl = sum(self._doc_len) / max(len(self._docs), 1)
        df: Counter = Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self._tf = [Counter(d) for d in self._docs]

    def score(self, query: str) -> list[float]:
        q = tokenize(query)
        scores = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avgdl)
                s += self._idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            scores.append(s)
        return scores

    def search(self, query: str, top_k: int = 1, exclude_ids: set[str] | None = None) -> list[Hit]:
        scores = self.score(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            p = self.policies[i]
            if exclude_ids and p["id"] in exclude_ids:
                continue
            hits.append(Hit(p["id"], p["title"], p["text"], scores[i]))
            if len(hits) == top_k:
                break
        return hits
