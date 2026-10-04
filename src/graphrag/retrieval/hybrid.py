"""Baseline hybrid retriever, matching the MSc thesis retriever
(github.com/RohithkumarReddipogula/AI-Powered-Rag-System):

- BM25: rank_bm25 Okapi, k1 1.5, b 0.75, lowercase + punctuation stripped, whitespace split
- dense: E5 (intfloat/e5-base-v2) with "query: " / "passage: " prefixes, exact cosine in Neo4j
- fusion: min-max normalise each list (a missing score counts as 0), alpha * dense + (1 - alpha) * bm25;
  RRF (k = 60) as the alternative
- then the bge-reranker-base cross-encoder over the fused top-n

This module must never import graph code (enforced by tests/test_hybrid.py).
"""

import re
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from rank_bm25 import BM25Okapi

from graphrag.retrieval.base import Passage, passage_text

Ranked = list[tuple[str, float]]          # (chunk_id, score), best first


def tokenize(text: str) -> list[str]:
    """Thesis tokenizer: lowercase, punctuation to spaces, whitespace split."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if t]


class BM25Index:
    def __init__(self, chunks: list[dict], with_title: bool, k1: float = 1.5, b: float = 0.75):
        self.ids = [c["id"] for c in chunks]
        self._bm25 = BM25Okapi([tokenize(passage_text(c["title"], c["text"], with_title)) for c in chunks], k1=k1, b=b)

    def search(self, question: str, k: int) -> Ranked:
        scores = self._bm25.get_scores(tokenize(question))
        # Stable sort on (-score, id) so ties are deterministic.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self.ids[i]))[:k]
        return [(self.ids[i], float(scores[i])) for i in order]


def min_max(ranked: Ranked) -> dict[str, float]:
    """Thesis normalisation. All-equal scores map to 0 (no information to rank on)."""
    if not ranked:
        return {}
    values = [s for _, s in ranked]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {cid: 0.0 for cid, _ in ranked}
    return {cid: (s - lo) / (hi - lo) for cid, s in ranked}


def fuse_weighted(bm25: Ranked, dense: Ranked, alpha: float) -> Ranked:
    b, d = min_max(bm25), min_max(dense)
    fused = {cid: alpha * d.get(cid, 0.0) + (1 - alpha) * b.get(cid, 0.0) for cid in b.keys() | d.keys()}
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


def fuse_rrf(bm25: Ranked, dense: Ranked, k: int = 60) -> Ranked:
    fused: dict[str, float] = {}
    for ranked in (bm25, dense):
        for rank, (cid, _) in enumerate(ranked, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: (-x[1], x[0]))


@dataclass(frozen=True)
class HybridConfig:
    with_title: bool = True
    fusion: Literal["weighted", "rrf"] = "weighted"
    alpha: float = 0.70
    candidates: int = 50          # per retriever, before fusion
    rerank_top: int = 30          # fused candidates passed to the cross-encoder
    rerank: bool = True


class HybridRetriever:
    """Implements graphrag.retrieval.base.Retriever."""

    def __init__(
        self,
        chunks: list[dict],
        dense_search: Callable[[str, int, bool], Ranked],     # (question, k, with_title) -> ranked
        rerank_scores: Callable[[str, list[str]], list[float]] | None,
        config: HybridConfig = HybridConfig(),
    ):
        self.config = config
        self.chunks = {c["id"]: c for c in chunks}
        self._bm25 = BM25Index(chunks, config.with_title)
        self._dense_search = dense_search
        self._rerank_scores = rerank_scores

    def candidates(self, question: str) -> tuple[Ranked, Ranked]:
        c = self.config
        return self._bm25.search(question, c.candidates), self._dense_search(question, c.candidates, c.with_title)

    def fuse(self, bm25: Ranked, dense: Ranked) -> Ranked:
        c = self.config
        return fuse_weighted(bm25, dense, c.alpha) if c.fusion == "weighted" else fuse_rrf(bm25, dense)

    def rerank(self, question: str, fused: Ranked) -> Ranked:
        top = [cid for cid, _ in fused[: self.config.rerank_top]]
        texts = [passage_text(self.chunks[cid]["title"], self.chunks[cid]["text"], True) for cid in top]
        scores = self._rerank_scores(question, texts)
        return sorted(zip(top, map(float, scores)), key=lambda x: (-x[1], x[0]))

    def retrieve(self, question: str, k: int) -> list[Passage]:
        bm25, dense = self.candidates(question)
        ranked = self.fuse(bm25, dense)
        if self.config.rerank and self._rerank_scores is not None:
            ranked = self.rerank(question, ranked)
        return [
            Passage(chunk_id=cid, title=self.chunks[cid]["title"], text=self.chunks[cid]["text"], score=score)
            for cid, score in ranked[:k]
        ]


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import CrossEncoder

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model = CrossEncoder(model_name, device=device)

    def __call__(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return np.asarray(self._model.predict([(question, t) for t in texts], show_progress_bar=False)).tolist()
