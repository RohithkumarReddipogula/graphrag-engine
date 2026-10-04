"""Retrieval metrics against gold chunk ids. No LLM involved."""

from collections import defaultdict
from statistics import mean

KS = (2, 5, 10)


def per_question(ranked_ids: list[str], gold: list[str], ks=KS) -> dict[str, float]:
    gold_set = set(gold)
    out = {}
    for k in ks:
        hit = len(gold_set & set(ranked_ids[:k]))
        out[f"recall@{k}"] = hit / len(gold_set)          # share of gold paragraphs found
        out[f"all_gold@{k}"] = float(hit == len(gold_set))  # every gold paragraph found (what multi-hop needs)
    return out


def aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    """rows: {"type": ..., metric: value, ...}. Groups: each type, all multi-hop, single-hop, all."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
        groups["multi_hop" if r["type"] != "single_hop" else "single_hop_all"].append(r)
        groups["all"].append(r)
    metrics = [m for m in rows[0] if m != "type"]
    return {g: {"n": len(rs), **{m: round(mean(r[m] for r in rs), 4) for m in metrics}} for g, rs in sorted(groups.items())}
