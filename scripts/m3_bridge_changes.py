"""List the report-split bridge links that M3 lost or gained compared with exact name matching.
Writes results/m3/bridge_link_changes.json. Dev gold triples only; no LLM calls."""

import json
from collections import Counter

from graphrag.config import get_settings
from graphrag.resolution.labels import bridge_links
from graphrag.resolution.mentions import build_mentions


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    s = get_settings()
    corpus = {c["id"]: c for c in read_jsonl(s.data_dir / "corpus.jsonl")}
    mentions = build_mentions(read_jsonl(s.results_dir / "m2" / "extractions.jsonl"), corpus)
    by = {m.id: m for m in mentions}
    cluster_of = {r["mention"]: r["cluster"] for r in read_jsonl(s.results_dir / "m3" / "clusters.jsonl")}
    decisions = {(r["a"], r["b"]): r for r in read_jsonl(s.results_dir / "m3" / "decisions.jsonl")}
    pages_per_norm = Counter(m.norm for m in mentions if m.is_page)
    dev = read_jsonl(s.data_dir / "questions_dev.jsonl") + read_jsonl(s.data_dir / "single_hop_dev.jsonl")
    links = bridge_links(dev, corpus, mentions, json.loads((s.data_dir / "evidence_aliases.json").read_text()))

    changes = []
    for l in links:
        if l.split != "report" or not l.mention_id:
            continue
        m, p = by[l.mention_id], by[l.page_id]
        before = m.norm == p.norm and pages_per_norm[p.norm] == 1
        after = cluster_of[m.id] == cluster_of[p.id]
        if before != after:
            d = decisions.get(tuple(sorted((m.id, p.id))))
            changes.append({
                "change": "lost" if before else "gained", "relation": l.relation,
                "mention": m.name, "mention_paragraph": m.chunk_id, "page": p.chunk_id,
                "direct_verdict": d["verdict"] if d else None, "reason": d["reason"] if d else None,
                "linked_through_chain": bool(after and not (d and d["verdict"] == "same")),
            })
    out = {"split": "report", "lost": sum(c["change"] == "lost" for c in changes),
           "gained": sum(c["change"] == "gained" for c in changes), "changes": changes}
    (s.results_dir / "m3" / "bridge_link_changes.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("lost", "gained")}))


if __name__ == "__main__":
    main()
