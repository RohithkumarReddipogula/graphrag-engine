"""Helpers shared by the M5 test run, with the same logic as in scripts/run_m4_eval.py (which is kept
unchanged so the committed M4 results stay reproducible)."""

from graphrag.generation import PackedContext, pack_context
from graphrag.resolution.mentions import norm_name
from graphrag.retrieval.base import Passage


def bridge_chunks(q: dict, corpus: dict) -> set[str]:
    """Gold paragraphs of bridge entities: objects of one gold triple that are the subject of another."""
    subjects = {norm_name(s) for s, _, _ in q["evidences"]}
    bridges = {norm_name(o) for _, _, o in q["evidences"]} & subjects
    return {c for c in q["gold_chunk_ids"] if norm_name(corpus[c]["title"]) in bridges}


def combine(graph_part: PackedContext, passages: list[Passage], budget: int) -> PackedContext:
    """Graph context first, then the hybrid passages that are not already included, in the remaining budget."""
    rest = [p for p in passages if p.chunk_id not in set(graph_part.chunk_ids)]
    chunk_part = pack_context(rest, budget - graph_part.tokens)
    text = "\n\n".join(t for t in (graph_part.text, chunk_part.text) if t)
    return PackedContext(text=text, chunk_ids=graph_part.chunk_ids + chunk_part.chunk_ids,
                         tokens=graph_part.tokens + chunk_part.tokens)
