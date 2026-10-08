"""Hand-check sheets for M3 (docs/PLAN.md, M3, quality bar). Report split only.

Merge sheet: 50 pairs of mentions sampled from inside the final clusters, so indirect merges (through
union-find chains) are checked too. Strata and quotas: direct same-name, direct different-name,
indirect, and mention-to-page links. Within a stratum a cluster is chosen uniformly first, then a pair
inside it, so one large cluster ("United States") cannot dominate. The sheet does not show how a pair
was merged; the stratum is kept in a separate key file.

Duplicate sheet: 60 seeded-random report-split mentions, each with up to 5 candidates chosen without the
resolver: other mentions whose names share something (token_set_ratio >= 60), most similar by E5 first.
"""

import random
from itertools import combinations

import numpy as np
from rapidfuzz import fuzz, process

from graphrag.resolution.cluster import Clustering
from graphrag.resolution.mentions import Mention, pair_split

SEED = 42
MERGE_QUOTAS = {"direct, same name": 13, "direct, different name": 12, "indirect (through a chain)": 13,
                "mention linked to page": 12}
PAIRS_PER_CLUSTER_CAP = 300
DUP_N, DUP_CANDIDATES, NAME_GATE = 60, 5, 60
SNIPPET_WORDS = 80


def _snippet(m: Mention, corpus: dict) -> str:
    c = corpus[m.chunk_id]
    words = c["text"].split()
    return f"[{c['title']}] " + " ".join(words[:SNIPPET_WORDS]) + (" ..." if len(words) > SNIPPET_WORDS else "")


def _stratum(a: Mention, b: Mention, direct: set[tuple[str, str]]) -> str:
    if a.is_page or b.is_page:
        return "mention linked to page"
    key = (a.id, b.id) if a.id < b.id else (b.id, a.id)
    if key in direct:
        return "direct, same name" if a.norm == b.norm else "direct, different name"
    return "indirect (through a chain)"


def merge_sample(cl: Clustering, by: dict[str, Mention], direct: set[tuple[str, str]]) -> list[dict]:
    rng = random.Random(f"{SEED}:m3-merge-check")
    pools: dict[str, dict[str, list[tuple[str, str]]]] = {s: {} for s in MERGE_QUOTAS}
    for cid, ids in sorted(cl.members.items()):
        if len(ids) < 2:
            continue
        pairs = list(combinations(ids, 2))
        if len(pairs) > PAIRS_PER_CLUSTER_CAP:
            pairs = rng.sample(pairs, PAIRS_PER_CLUSTER_CAP)
        for x, y in pairs:
            a, b = by[x], by[y]
            if pair_split(a, b) != "report":
                continue
            pools[_stratum(a, b, direct)].setdefault(cid, []).append((x, y))
    picked = []
    for stratum, quota in MERGE_QUOTAS.items():
        clusters = sorted(pools[stratum])
        rng.shuffle(clusters)
        for cid in clusters[:quota]:
            x, y = rng.choice(sorted(pools[stratum][cid]))
            picked.append({"stratum": stratum, "cluster": cid, "a": x, "b": y})
    rng.shuffle(picked)
    return picked


def merge_sheet(picked: list[dict], by: dict[str, Mention], corpus: dict) -> str:
    lines = [
        "# M3 merge hand check (report split)",
        "",
        f"{len(picked)} pairs of mentions that the resolver put into the same cluster. For each: are A and B",
        'the same real-world entity? Mark "ok" or write the problem. A verdict starting with "ok" counts as',
        "correct. Bar: at least 48 of 50.",
        "",
    ]
    for i, p in enumerate(picked, 1):
        a, b = by[p["a"]], by[p["b"]]
        lines += [
            f"## {i}. {a.name} / {b.name}",
            "",
            f"- A: {a.name} ({a.type}), {a.description}",
            f"  - Paragraph: {_snippet(a, corpus)}",
            f"- B: {b.name} ({b.type}), {b.description}",
            f"  - Paragraph: {_snippet(b, corpus)}",
            "- Verdict: ",
            "",
        ]
    return "\n".join(lines)


def duplicate_sample(mentions: list[Mention], emb: np.ndarray) -> list[dict]:
    rng = random.Random(f"{SEED}:m3-duplicate-check")
    report = sorted((i for i, m in enumerate(mentions) if m.split == "report"), key=lambda i: mentions[i].id)
    chosen = rng.sample(report, DUP_N)
    norms = [m.norm for m in mentions]
    out = []
    for i in chosen:
        hits = process.extract(norms[i], norms, scorer=fuzz.token_set_ratio, score_cutoff=NAME_GATE, limit=None)
        cand = [j for _, _, j in hits if j != i]
        cand.sort(key=lambda j: (-float(emb[i] @ emb[j]), mentions[j].id))
        out.append({"mention": mentions[i].id, "candidates": [mentions[j].id for j in cand[:DUP_CANDIDATES]]})
    return out


def duplicate_sheet(sample: list[dict], by: dict[str, Mention], corpus: dict) -> str:
    lines = [
        "# M3 duplicate hand check (report split)",
        "",
        f"{len(sample)} random mentions, each with up to {DUP_CANDIDATES} similar mentions from other places in the",
        "corpus. For each, list the candidate numbers that are the same real-world entity as the mention,",
        'for example "Same entity as: 1, 3", or "Same entity as: none".',
        "",
    ]
    for i, s in enumerate(sample, 1):
        m = by[s["mention"]]
        lines += [f"## {i}. {m.name} ({m.type})", "", f"- Mention: {m.name}, {m.description}",
                  f"  - Paragraph: {_snippet(m, corpus)}"]
        if not s["candidates"]:
            lines.append("- Candidates: none found")
        for k, cid in enumerate(s["candidates"], 1):
            c = by[cid]
            lines += [f"- Candidate {k}: {c.name} ({c.type}), {c.description}", f"  - Paragraph: {_snippet(c, corpus)}"]
        lines += ["- Same entity as: ", ""]
    return "\n".join(lines)
