"""M3 pilot for judge prompt v2 (docs/NEXT.md, section 3, step 4): tests both directions before the full
re-run.

- Same direction: the tune-split bridge links from dev gold triples only (mention -> its correct page).
  Gate: at least 22 of 24 judged "same".
- Different direction: run 1's known false merges (results/m3/run1_flawed/): United States vs United
  Kingdom, Cannes vs Berlin vs Moscow film festivals, Mexico vs Mexico City. For each category, the pairs
  run 1 wrongly judged "same"; if run 1 merged a category only through chains, its most similar cross
  pairs. Gate: every pair judged "different".

Writes results/m3/pilot_v2.json and results/m3/pilot_v2_decisions.jsonl. Test questions are never read.
"""

import json
from collections import Counter
from itertools import combinations

from graphrag.config import get_settings
from graphrag.llm.client import make_llm
from graphrag.llm.spend import write_balance
from graphrag.resolution.embed_mentions import embed_mentions
from graphrag.resolution.judge import JUDGE_VERSION, PAIRS_PER_CALL, judge_batch
from graphrag.resolution.labels import bridge_links
from graphrag.resolution.mentions import build_mentions

SAME_GATE = 22
PER_CATEGORY = 5
# Negative categories: groups of names that run 1 merged but are different entities.
NEGATIVES = {
    "United States vs United Kingdom": [{"united states", "us", "u s", "usa", "america"},
                                        {"united kingdom", "uk", "u k", "britain", "great britain"}],
    "Cannes vs Berlin vs Moscow film festivals": [{"cannes film festival", "festival de cannes"},
                                                  {"berlin international film festival", "berlinale"},
                                                  {"moscow international film festival"}],
    "Mexico vs Mexico City": [{"mexico"}, {"mexico city"}],
}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    s = get_settings()
    out = s.results_dir / "m3"
    corpus = {c["id"]: c for c in read_jsonl(s.data_dir / "corpus.jsonl")}
    mentions = build_mentions(read_jsonl(s.results_dir / "m2" / "extractions.jsonl"), corpus)
    by = {m.id: m for m in mentions}
    idx = {m.id: i for i, m in enumerate(mentions)}
    emb = embed_mentions(mentions, s.embedding_model, s.cache_dir)
    cos = lambda a, b: float(emb[idx[a]] @ emb[idx[b]])

    dev = read_jsonl(s.data_dir / "questions_dev.jsonl") + read_jsonl(s.data_dir / "single_hop_dev.jsonl")
    test_ids = {q["id"] for f in ("questions_test.jsonl", "single_hop_test.jsonl") for q in read_jsonl(s.data_dir / f)}
    assert not test_ids & {q["id"] for q in dev}, "test questions must never be used in M3"
    links = bridge_links(dev, corpus, mentions, json.loads((s.data_dir / "evidence_aliases.json").read_text()))
    positives = [(l.mention_id, l.page_id) for l in links if l.split == "tune" and l.mention_id and l.page_id]

    run1_same = [(r["a"], r["b"]) for r in read_jsonl(out / "run1_flawed" / "decisions.jsonl") if r["verdict"] == "same"]
    negatives: list[tuple[str, str, str, str]] = []   # (category, a, b, how chosen)
    for category, groups in NEGATIVES.items():
        members = [[m.id for m in mentions if m.norm in g] for g in groups]
        chosen = []
        for gi, gj in combinations(range(len(groups)), 2):
            si, sj = set(members[gi]), set(members[gj])
            direct = [(a, b) for a, b in run1_same if (a in si and b in sj) or (a in sj and b in si)]
            if direct:
                chosen += [(a, b, "run 1 judged same") for a, b in direct]
            else:
                cross = sorted(((a, b) for a in si for b in sj), key=lambda p: -cos(*p))[:2]
                chosen += [(a, b, "most similar cross pair (run 1 merged it through a chain)") for a, b in cross]
        chosen = sorted(chosen, key=lambda c: -cos(c[0], c[1]))[:PER_CATEGORY]
        negatives += [(category, a, b, how) for a, b, how in chosen]

    items = [(f"pos{i:02d}", "same", "tune bridge link", a, b) for i, (a, b) in enumerate(positives)]
    items += [(f"neg{i:02d}", "different", cat, a, b) for i, (cat, a, b, _) in enumerate(negatives)]
    how = {f"neg{i:02d}": h for i, (_, _, _, h) in enumerate(negatives)}

    llm = make_llm(s.extractor_model, s)
    decisions, calls = {}, []
    for k in range(0, len(items), PAIRS_PER_CALL):
        batch = items[k:k + PAIRS_PER_CALL]
        dec, call_results, _ = judge_batch(llm, [(pid, by[a], by[b]) for pid, _, _, a, b in batch], corpus)
        decisions.update(dec)
        calls += call_results

    rows = []
    for pid, expected, category, a, b in items:
        d = decisions[pid]
        rows.append({"pair_id": pid, "expected": expected, "category": category, "how_chosen": how.get(pid, "dev gold"),
                     "a": a, "a_name": by[a].name, "a_role": "subject" if by[a].is_page else "mentioned",
                     "b": b, "b_name": by[b].name, "b_role": "subject" if by[b].is_page else "mentioned",
                     "cos": round(cos(a, b), 4), "verdict": d.verdict, "reason": d.reason})

    pos_same = sum(r["verdict"] == "same" for r in rows if r["expected"] == "same")
    neg_rows = [r for r in rows if r["expected"] == "different"]
    neg_ok = sum(r["verdict"] == "different" for r in neg_rows)
    balance = write_balance(s)
    report = {
        "judge_version": JUDGE_VERSION,
        "same_direction": {"pairs": len(positives), "judged_same": pos_same, "gate": SAME_GATE,
                           "pass": pos_same >= SAME_GATE,
                           "verdicts": dict(Counter(r["verdict"] for r in rows if r["expected"] == "same"))},
        "different_direction": {"pairs": len(neg_rows), "judged_different": neg_ok, "gate": "all",
                                "pass": neg_ok == len(neg_rows),
                                "by_category": {c: dict(Counter(r["verdict"] for r in neg_rows if r["category"] == c))
                                                for c in NEGATIVES}},
        "cost": {"calls": len(calls), "live_calls": sum(not c.cached for c in calls),
                 "cost_usd": round(sum(c.cost_usd for c in calls if not c.cached), 6)},
        "remaining_credit_usd": balance["remaining_usd"],
    }
    with (out / "pilot_v2_decisions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "pilot_v2.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
