"""M3 full run: the LLM judges every candidate pair (auto-merge off, approved 2026-10-08) with judge
prompt v2, then union-find clustering with the never-merge rules (two pages; cannot-link), cluster statistics, bridge link recall before and after
M3, and the two hand-check sheets.

Writes results/m3/decisions.jsonl, results/m3/clusters.jsonl, results/m3/resolution_run.json,
results/m3/merge_handcheck.md (+ merge_check_key.json) and results/m3/duplicate_handcheck.md
(+ duplicate_check_key.json). Sheets are never overwritten. Test questions are never read.
"""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from graphrag.config import get_settings
from graphrag.extraction.extract import QuotaExhausted
from graphrag.llm.client import make_llm
from graphrag.llm.spend import write_balance
from graphrag.resolution import handcheck
from graphrag.resolution.candidates import candidate_pairs
from graphrag.resolution.cluster import cluster
from graphrag.resolution.embed_mentions import embed_mentions
from graphrag.resolution.judge import JUDGE_VERSION, PAIRS_PER_CALL, judge_batch
from graphrag.resolution.labels import bridge_links
from graphrag.resolution.mentions import build_mentions


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    s = get_settings()
    out = s.results_dir / "m3"
    corpus = {c["id"]: c for c in read_jsonl(s.data_dir / "corpus.jsonl")}
    mentions = build_mentions(read_jsonl(s.results_dir / "m2" / "extractions.jsonl"), corpus)
    by = {m.id: m for m in mentions}
    emb = embed_mentions(mentions, s.embedding_model, s.cache_dir)
    t_low = json.loads((out / "pilot.json").read_text())["thresholds"]["t_low"]
    pairs = candidate_pairs(mentions, emb, t_low=t_low)

    # 1. LLM judges every pair.
    llm = make_llm(s.extractor_model, s)
    items = [(f"{p.a}||{p.b}", p) for p in pairs]
    batches = [items[i:i + PAIRS_PER_CALL] for i in range(0, len(items), PAIRS_PER_CALL)]
    t0 = time.time()
    decisions, calls, failed = {}, [], 0

    def run(batch):
        return batch, judge_batch(llm, [(pid, by[p.a], by[p.b]) for pid, p in batch], corpus)

    try:
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(run, b) for b in batches]
            for done, fut in enumerate(as_completed(futures), 1):
                batch, (dec, call_results, ok) = fut.result()
                decisions.update(dec)
                calls += call_results
                failed += not ok
                if done % 50 == 0 or done == len(batches):
                    print(f"[{time.time() - t0:6.0f}s] {done}/{len(batches)} batches  live calls "
                          f"{sum(not c.cached for c in calls)}  failed {failed}  "
                          f"cost {sum(c.cost_usd for c in calls if not c.cached):.4f} USD", flush=True)
    except QuotaExhausted as exc:
        print("STOPPED:", str(exc)[:300], "(finished batches are cached; re-run to resume)", flush=True)
        raise SystemExit(1)

    rows = []
    for pid, p in items:
        d = decisions[pid]
        rows.append({**p.to_dict(), "verdict": d.verdict, "reason": d.reason})
    write_jsonl(out / "decisions.jsonl", rows)

    # 2. Clusters.
    same = [(r["a"], r["b"], r["cos"]) for r in rows if r["verdict"] == "same"]
    different = [(r["a"], r["b"]) for r in rows if r["verdict"] == "different"]
    cl = cluster(mentions, same, different)
    write_jsonl(out / "clusters.jsonl", [{"mention": m.id, "cluster": cl.cluster_of[m.id], "name": m.name,
                                          "type": m.type, "is_page": m.is_page} for m in mentions])
    sizes = Counter({cid: len(ids) for cid, ids in cl.members.items()})
    page_clusters = {cid for cid, ids in cl.members.items() if any(by[i].is_page for i in ids)}
    linked = sum(1 for m in mentions if not m.is_page and cl.cluster_of[m.id] in page_clusters)

    def top_names(ids):
        return [n for n, _ in Counter(by[i].name for i in ids).most_common(3)]

    # 3. Bridge link recall (dev gold only), before (exact name match) and after M3.
    dev = read_jsonl(s.data_dir / "questions_dev.jsonl") + read_jsonl(s.data_dir / "single_hop_dev.jsonl")
    links = bridge_links(dev, corpus, mentions, json.loads((s.data_dir / "evidence_aliases.json").read_text()))
    pages_per_norm = Counter(m.norm for m in mentions if m.is_page)
    bridge = {}
    for split in ("report", "tune"):
        ls = [l for l in links if l.split == split]
        before = sum(1 for l in ls if l.mention_id and by[l.mention_id].norm == by[l.page_id].norm
                     and pages_per_norm[by[l.page_id].norm] == 1)
        after = sum(1 for l in ls if l.mention_id and cl.cluster_of[l.mention_id] == cl.cluster_of[l.page_id])
        bridge[split] = {"links": len(ls), "no_mention_extracted": sum(l.mention_id is None for l in ls),
                         "before_exact_name": before, "after_m3": after,
                         "recall_before": round(before / len(ls), 4), "recall_after": round(after / len(ls), 4)}

    # 4. Hand-check sheets (report split; never overwritten).
    direct = {(r["a"], r["b"]) for r in rows if r["verdict"] == "same"}
    if (out / "merge_handcheck.md").exists():
        print("kept existing merge_handcheck.md (it may hold hand-check marks)")
    else:
        picked = handcheck.merge_sample(cl, by, direct)
        (out / "merge_handcheck.md").write_text(handcheck.merge_sheet(picked, by, corpus), encoding="utf-8")
        (out / "merge_check_key.json").write_text(json.dumps(picked, indent=1, ensure_ascii=False) + "\n")
    if (out / "duplicate_handcheck.md").exists():
        print("kept existing duplicate_handcheck.md (it may hold hand-check marks)")
    else:
        sample = handcheck.duplicate_sample(mentions, emb)
        (out / "duplicate_handcheck.md").write_text(handcheck.duplicate_sheet(sample, by, corpus), encoding="utf-8")
        (out / "duplicate_check_key.json").write_text(json.dumps(sample, indent=1, ensure_ascii=False) + "\n")

    balance = write_balance(s)
    run = {
        "thresholds": {"t_low": t_low, "t_high": "off (every candidate pair judged by the LLM)"},
        "mentions": len(mentions), "page_entities": sum(m.is_page for m in mentions),
        "candidate_pairs": len(pairs),
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "llm": {"calls": len(calls), "live_calls": sum(not c.cached for c in calls), "batches_failed_twice": failed,
                "input_tokens": sum(c.input_tokens for c in calls), "output_tokens": sum(c.output_tokens for c in calls),
                "cost_usd_live": round(sum(c.cost_usd for c in calls if not c.cached), 6)},
        "judge_version": JUDGE_VERSION,
        "merges_applied": len(cl.applied),
        "merges_rejected": dict(Counter(reason for _, _, reason in cl.rejected)),
        "clusters": {"total": len(cl.members), "with_2_or_more_mentions": sum(v >= 2 for v in sizes.values()),
                     "singletons": sum(v == 1 for v in sizes.values()), "with_a_page": len(page_clusters),
                     "largest": [{"cluster": cid, "mentions": n, "names": top_names(cl.members[cid])}
                                 for cid, n in sizes.most_common(10)]},
        "non_page_mentions_linked_to_a_page": linked,
        "non_page_mentions": sum(not m.is_page for m in mentions),
        "bridge_link_recall": bridge,
        "remaining_credit_usd": balance["remaining_usd"],
        "seconds": round(time.time() - t0),
    }
    (out / "resolution_run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: run[k] for k in ("verdicts", "llm", "merges_applied", "merges_rejected",
                                          "clusters", "non_page_mentions_linked_to_a_page", "bridge_link_recall")},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
