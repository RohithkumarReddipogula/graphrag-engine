"""Candidate pairs for M3 (docs/PLAN.md, M3, step 2).

A pair of mentions is a candidate when the types are compatible and at least one of:
  (a) identical normalised names (each mention paired with its EXACT_K most similar same-name mentions);
  (b) rapidfuzz ratio of the normalised names >= FUZZY_MIN;
  (c) among the K nearest neighbours by E5 embedding of "name: description", cosine >= t_low, and the
      names share something (token_set_ratio >= NAME_GATE). The gate keeps two unrelated people with
      similar descriptions ("American actor") out; aliases without a shared word are missed on purpose.
Pairs of two page entities are never candidates: they never merge.
"""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rapidfuzz import fuzz, process

from graphrag.resolution.mentions import Mention

FUZZY_MIN = 85
NAME_GATE = 60
K = 10
EXACT_K = 3
STRICT_TYPES = {"PERSON", "PLACE", "ORG"}


def types_compatible(a: str, b: str) -> bool:
    if a == b or "OTHER" in (a, b):
        return True
    return {a, b} == {"FILM", "WORK"}


@dataclass
class Pair:
    a: str              # mention id (a < b)
    b: str
    exact: bool
    name_ratio: float
    cos: float
    sources: tuple[str, ...]   # which of exact / fuzzy / embedding produced it

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "exact": self.exact, "name_ratio": round(self.name_ratio, 1),
                "cos": round(self.cos, 4), "sources": list(self.sources)}


def mention_text(m: Mention) -> str:
    return f"{m.name}: {m.description}" if m.description else m.name


def candidate_pairs(mentions: list[Mention], emb: np.ndarray, t_low: float) -> list[Pair]:
    """emb: L2-normalised embeddings, row i for mentions[i]."""
    idx = {m.id: i for i, m in enumerate(mentions)}
    found: dict[tuple[str, str], set[str]] = defaultdict(set)

    def add(i: int, j: int, source: str) -> None:
        mi, mj = mentions[i], mentions[j]
        if i == j or (mi.is_page and mj.is_page) or not types_compatible(mi.type, mj.type):
            return
        key = (mi.id, mj.id) if mi.id < mj.id else (mj.id, mi.id)
        found[key].add(source)

    # (a) identical normalised names. Each mention is paired with its EXACT_K most similar same-name
    # mentions, not with all of them: clustering is transitive, so a group still links up, and frequent
    # names ("United States") do not produce n * n pairs.
    by_norm: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(mentions):
        by_norm[m.norm].append(i)
    for group in by_norm.values():
        if len(group) < 2:
            continue
        g = np.asarray(group)
        sims = emb[g] @ emb[g].T
        np.fill_diagonal(sims, -np.inf)
        for r, i in enumerate(group):
            for c in np.argsort(-sims[r])[: min(EXACT_K, len(group) - 1)]:
                add(i, int(g[c]), "exact")

    # (b) fuzzy names, computed on distinct names
    names = sorted(by_norm)
    scores = process.cdist(names, names, scorer=fuzz.ratio, score_cutoff=FUZZY_MIN, workers=-1, dtype=np.uint8)
    ii, jj = np.nonzero(np.triu(scores, 1))
    for x, y in zip(ii, jj):
        for i in by_norm[names[x]]:
            for j in by_norm[names[y]]:
                add(i, j, "fuzzy")

    # (c) embedding neighbours with a name gate
    for start in range(0, len(mentions), 1024):
        sims = emb[start:start + 1024] @ emb.T
        for r, row in enumerate(sims):
            i = start + r
            top = np.argpartition(-row, K)[: K + 1] if len(row) > K + 1 else np.arange(len(row))
            for j in top:
                if j != i and row[j] >= t_low and fuzz.token_set_ratio(mentions[i].norm, mentions[j].norm) >= NAME_GATE:
                    add(i, int(j), "embedding")

    pairs = []
    for (a, b), sources in found.items():
        ia, ib = idx[a], idx[b]
        pairs.append(Pair(
            a=a, b=b, exact=mentions[ia].norm == mentions[ib].norm,
            name_ratio=fuzz.ratio(mentions[ia].norm, mentions[ib].norm),
            cos=float(emb[ia] @ emb[ib]), sources=tuple(sorted(sources)),
        ))
    return sorted(pairs, key=lambda p: (p.a, p.b))
