"""M3 pilot: tune t_low on dev gold bridge links (tune split only), judge about 100 tune-split borderline
pairs with the LLM, measure the cost per pair and project the full LLM run for several t_high values.

Pairs come only from the tune split: a pair is tune if every non-page mention in it comes from a tune
paragraph. Test questions are never read. Writes results/m3/pilot.json and
results/m3/pilot_decisions.jsonl.
"""

import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from graphrag.config import get_settings
from graphrag.llm.client import make_llm
from graphrag.llm.spend import write_balance
from graphrag.resolution.candidates import EXACT_K, FUZZY_MIN, NAME_GATE, candidate_pairs
from graphrag.resolution.embed_mentions import embed_mentions
from graphrag.resolution.judge import PAIRS_PER_CALL, judge_batch
from graphrag.resolution.labels import bridge_links
from graphrag.resolution.mentions import build_mentions

SEED = 42
WORKERS = 8
COS_BINS = [(0.0, 0.85), (0.85, 0.88), (0.88, 0.90), (0.90, 0.92), (0.92, 1.01)]
PER_BIN = 10
PER_KIND = {"fuzzy name": 20, "embedding + name gate": 20, "exact name, 2+ pages": 10}
T_HIGH_OPTIONS = [0.85, 0.88, 0.90, 0.92, 1.01]


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    s = get_settings()
    corpus = {c["id"]: c for c in read_jsonl(s.data_dir / "corpus.jsonl")}
    mentions = build_mentions(read_jsonl(s.results_dir / "m2" / "extractions.jsonl"), corpus)
    by = {m.id: m for m in mentions}
    idx = {m.id: i for i, m in enumerate(mentions)}
    emb = embed_mentions(mentions, s.embedding_model, s.cache_dir)

    # t_low from dev gold only (tune split): 5th percentile of bridge-link similarity, rounded down.
    dev = read_jsonl(s.data_dir / "questions_dev.jsonl") + read_jsonl(s.data_dir / "single_hop_dev.jsonl")
    test_ids = {q["id"] for f in ("questions_test.jsonl", "single_hop_test.jsonl") for q in read_jsonl(s.data_dir / f)}
    assert not test_ids & {q["id"] for q in dev}, "test questions must never be used in M3"
    ev_aliases = json.loads((s.data_dir / "evidence_aliases.json").read_text(encoding="utf-8"))
    links = bridge_links(dev, corpus, mentions, ev_aliases)
    tune_cos = [float(emb[idx[l.mention_id]] @ emb[idx[l.page_id]]) for l in links
                if l.split == "tune" and l.mention_id and l.page_id]
    t_low = math.floor(float(np.percentile(tune_cos, 5)) * 100) / 100

    pairs = candidate_pairs(mentions, emb, t_low=t_low)
    pages_per_norm = Counter(m.norm for m in mentions if m.is_page)

    def kind(p):
        multi = max(pages_per_norm[by[p.a].norm], pages_per_norm[by[p.b].norm]) > 1
        if p.exact:
            return "exact name, 2+ pages" if multi else "exact name, single page"
        return "fuzzy name" if "fuzzy" in p.sources else "embedding + name gate"

    def split(p):
        side = [by[x] for x in (p.a, p.b) if not by[x].is_page]
        return "tune" if all(m.split == "tune" for m in side) else "report"

    tune = [p for p in pairs if split(p) == "tune"]
    rng = random.Random(f"{SEED}:m3-pilot")
    sample = []
    for lo, hi in COS_BINS:
        pool = [p for p in tune if kind(p) == "exact name, single page" and lo <= p.cos < hi]
        sample += [(f"exact single-page cos {lo:.2f}-{min(hi, 1):.2f}", p) for p in rng.sample(pool, min(PER_BIN, len(pool)))]
    for k, n in PER_KIND.items():
        pool = [p for p in tune if kind(p) == k]
        sample += [(k, p) for p in rng.sample(pool, min(n, len(pool)))]
    rng.shuffle(sample)

    llm = make_llm(s.extractor_model, s)
    items = [(f"p{i:03d}", stratum, p) for i, (stratum, p) in enumerate(sample)]
    batches = [items[i:i + PAIRS_PER_CALL] for i in range(0, len(items), PAIRS_PER_CALL)]

    def run(batch):
        return batch, judge_batch(llm, [(pid, by[p.a], by[p.b]) for pid, _, p in batch], corpus)

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(run, batches))

    rows, calls, failed = [], [], 0
    for batch, (decisions, call_results, ok) in results:
        calls += call_results
        failed += not ok
        for pid, stratum, p in batch:
            d = decisions[pid]
            rows.append({"pair_id": pid, "stratum": stratum, **p.to_dict(),
                         "a_name": by[p.a].name, "b_name": by[p.b].name, "verdict": d.verdict, "reason": d.reason})

    cost = sum(c.cost_usd for c in calls)
    per_pair = cost / len(rows)
    single = [p for p in pairs if kind(p) == "exact name, single page"]
    always_llm = len(pairs) - len(single)
    balance = write_balance(s)
    verdicts = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        c = Counter(r["verdict"] for r in rows if r["stratum"] == stratum)
        verdicts[stratum] = {v: c.get(v, 0) for v in ("same", "different", "unsure")}

    report = {
        "thresholds": {
            "t_low": t_low,
            "t_low_rule": "5th percentile of E5 cosine over tune-split dev bridge links, rounded down to 0.01",
            "tune_bridge_links": len(tune_cos),
            "tune_bridge_link_cos_percentiles_5_25_50": [round(float(x), 4) for x in np.percentile(tune_cos, [5, 25, 50])],
            "t_high": "not set: dev gold triples give no same-name negatives, so see t_high_options",
            "fuzzy_min": FUZZY_MIN, "name_gate": NAME_GATE, "exact_k": EXACT_K,
        },
        "candidates": {"total": len(pairs), "by_kind": dict(Counter(kind(p) for p in pairs)),
                       "by_split": dict(Counter(split(p) for p in pairs))},
        "pilot": {"pairs": len(rows), "calls": len(calls), "batches_failed_twice": failed,
                  "live_calls": sum(not c.cached for c in calls),
                  "input_tokens": sum(c.input_tokens for c in calls), "output_tokens": sum(c.output_tokens for c in calls),
                  "cost_usd": round(cost, 6), "cost_per_pair_usd": round(per_pair, 8),
                  "verdicts_by_stratum": verdicts},
        "t_high_options": [
            {"t_high": th, "auto_merged": sum(p.cos >= th for p in single),
             "llm_pairs": always_llm + sum(p.cos < th for p in single),
             "projected_usd": round((always_llm + sum(p.cos < th for p in single)) * per_pair, 3)}
            for th in T_HIGH_OPTIONS
        ],
        "remaining_credit_usd": balance["remaining_usd"],
    }
    out = s.results_dir / "m3"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "pilot_decisions.jsonl").open("w", encoding="utf-8") as f:
        for r in sorted(rows, key=lambda r: r["pair_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "pilot.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
