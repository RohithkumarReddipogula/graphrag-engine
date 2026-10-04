import ast
from pathlib import Path

import numpy as np
import pytest

from graphrag.config import ROOT
from graphrag.embed import E5Encoder
from graphrag.eval.retrieval import aggregate, per_question
from graphrag.retrieval.hybrid import BM25Index, HybridConfig, HybridRetriever, fuse_rrf, fuse_weighted, min_max, tokenize

BM25 = [("a", 12.0), ("b", 6.0), ("c", 3.0)]
DENSE = [("c", 0.9), ("d", 0.8), ("a", 0.7)]


def test_hybrid_module_never_imports_graph_code():
    tree = ast.parse((ROOT / "src/graphrag/retrieval/hybrid.py").read_text())
    imported = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module] + [
        a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
    ]
    assert not [m for m in imported if "graph." in m.replace("graphrag.", "") or "extraction" in m or "resolution" in m]


def test_alpha_one_is_dense_ranking_and_alpha_zero_is_bm25_ranking():
    assert [c for c, _ in fuse_weighted(BM25, DENSE, 1.0)][:3] == ["c", "d", "a"]
    assert [c for c, _ in fuse_weighted(BM25, DENSE, 0.0)][:3] == ["a", "b", "c"]


def test_min_max_and_missing_scores_count_as_zero():
    assert min_max(BM25) == {"a": 1.0, "b": 1 / 3, "c": 0.0}
    assert min_max([("x", 2.0), ("y", 2.0)]) == {"x": 0.0, "y": 0.0}
    fused = dict(fuse_weighted(BM25, DENSE, 0.7))
    assert fused["b"] == pytest.approx(0.3 * (1 / 3))     # b has no dense score


def test_rrf_rewards_items_in_both_lists():
    ranked = [c for c, _ in fuse_rrf(BM25, DENSE)]
    assert set(ranked[:2]) == {"a", "c"}


def test_thesis_tokenizer():
    assert tokenize("Who directed 'Heat' (1995)?") == ["who", "directed", "heat", "1995"]


def test_e5_prefixes():
    seen = []

    class Fake:
        def encode(self, sentences, **kwargs):
            seen.extend(sentences)
            return np.zeros((len(sentences), 2))

    enc = E5Encoder("fake", model=Fake())
    enc.encode_queries(["q1"])
    enc.encode_passages(["p1"])
    assert seen == ["query: q1", "passage: p1"]


def test_retriever_end_to_end_with_fakes():
    chunks = [{"id": "x", "title": "X", "text": "alpha beta"}, {"id": "y", "title": "Y", "text": "gamma delta"}]
    dense = lambda q, k, with_title: [("y", 0.9), ("x", 0.1)]
    rerank = lambda q, texts: [1.0 if t.startswith("X") else 0.0 for t in texts]
    r = HybridRetriever(chunks, dense, rerank, HybridConfig(alpha=1.0))
    assert [p.chunk_id for p in r.retrieve("alpha", 2)] == ["x", "y"]          # reranker decides
    r2 = HybridRetriever(chunks, dense, rerank, HybridConfig(alpha=1.0, rerank=False))
    assert [p.chunk_id for p in r2.retrieve("alpha", 2)] == ["y", "x"]         # dense decides


def test_recall_metrics():
    m = per_question(["a", "b", "c"], ["a", "z"], ks=(2,))
    assert m == {"recall@2": 0.5, "all_gold@2": 0.0}
    agg = aggregate([{"type": "comparison", **m}, {"type": "single_hop", **per_question(["z"], ["z"], ks=(2,))}])
    assert agg["multi_hop"]["n"] == 1 and agg["all"]["recall@2"] == 0.75
