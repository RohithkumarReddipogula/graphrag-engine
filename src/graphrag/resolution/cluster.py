"""Union-find clustering of mentions with never-merge rules (docs/PLAN.md, M3, step 4; docs/NEXT.md).

Only pairs the LLM judged "same" are merged ("unsure" and "different" never merge). Merges are applied
in a fixed, deterministic order: most confident first, measured as the pair's E5 cosine (highest first,
ties broken by pair ids). A merge is refused when it would
- put two different page entities into one cluster ("two_pages"), or
- join two clusters that contain a pair judged "different" (cannot-link, "cannot_link").
Both checks run on whole clusters, so a chain A=B, B=C can never join what a direct verdict kept apart.
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
    rejected: list[tuple[str, str, str]] = field(default_factory=list)   # (a, b, reason)


def merge_order(same_edges: list[tuple[str, str, float]]) -> list[tuple[str, str, float]]:
    return sorted(same_edges, key=lambda e: (-e[2], e[0], e[1]))


def cluster(
    mentions: list[Mention],
    same_edges: list[tuple[str, str, float]],
    different_pairs: list[tuple[str, str]] = (),
) -> Clustering:
    """same_edges: (mention a, mention b, cosine) judged "same"; different_pairs: (a, b) judged "different"."""
    parent = {m.id: m.id for m in mentions}
    page = {m.id: (m.chunk_id if m.is_page else None) for m in mentions}   # page chunk held by each root
    group = {m.id: {m.id} for m in mentions}                               # members of each root
    forbidden: dict[str, set[str]] = {m.id: set() for m in mentions}       # mentions a root may not join
    for a, b in different_pairs:
        forbidden[a].add(b)
        forbidden[b].add(a)

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    applied, rejected = [], []
    for a, b, _ in merge_order(same_edges):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if page[ra] and page[rb] and page[ra] != page[rb]:
            rejected.append((a, b, "two_pages"))
            continue
        small, large = (ra, rb) if len(group[ra]) <= len(group[rb]) else (rb, ra)
        if group[small] & forbidden[large] or group[large] & forbidden[small]:
            rejected.append((a, b, "cannot_link"))
            continue
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        parent[drop] = keep
        page[keep] = page[keep] or page[drop]
        group[keep] |= group.pop(drop)
        forbidden[keep] |= forbidden.pop(drop)
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
