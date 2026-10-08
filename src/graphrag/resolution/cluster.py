"""Union-find clustering of mentions with the never-merge rule (docs/PLAN.md, M3, step 4).

Only pairs the LLM judged "same" are merged ("unsure" and "different" never merge). Edges are applied in
a fixed order (higher E5 cosine first, then pair ids), and a merge that would put two different page
entities into one cluster is rejected and logged, so a chain A=B, B=C can never join two pages through C.
Cluster ids are stable: the page's chunk_id when the cluster has a page entity, else "m:" + the
smallest mention id in it.
"""

from dataclasses import dataclass, field

from graphrag.resolution.mentions import Mention


@dataclass
class Clustering:
    cluster_of: dict[str, str]                      # mention id -> cluster id
    members: dict[str, list[str]]                   # cluster id -> mention ids (sorted)
    applied: list[tuple[str, str]] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)


def cluster(mentions: list[Mention], same_edges: list[tuple[str, str, float]]) -> Clustering:
    """same_edges: (mention a, mention b, cosine) for pairs judged "same"."""
    parent = {m.id: m.id for m in mentions}
    page = {m.id: (m.chunk_id if m.is_page else None) for m in mentions}   # page chunk held by each root

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    applied, rejected = [], []
    for a, b, _ in sorted(same_edges, key=lambda e: (-e[2], e[0], e[1])):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if page[ra] and page[rb] and page[ra] != page[rb]:
            rejected.append((a, b))
            continue
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        parent[drop] = keep
        page[keep] = page[keep] or page[drop]
        applied.append((a, b))

    groups: dict[str, list[str]] = {}
    for m in mentions:
        groups.setdefault(find(m.id), []).append(m.id)
    cluster_of, members = {}, {}
    for root, ids in groups.items():
        cid = page[root] if page[root] else "m:" + min(ids)
        members[cid] = sorted(ids)
        for i in ids:
            cluster_of[i] = cid
    return Clustering(cluster_of=cluster_of, members=members, applied=applied, rejected=rejected)
