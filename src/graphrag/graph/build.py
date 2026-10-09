"""Build the M4 graph from the M2 extractions and the M3 clusters (docs/PLAN.md, M4.1). Deterministic.

- Entity: one per M3 cluster. name = most frequent mention name (ties: alphabetical), aliases = all
  mention names, type = most frequent coarse type, is_page / page_chunk_id from the cluster's page
  mention. Date facts are properties with their source chunks.
- REL: M2 relations after M2 post-processing. Each endpoint is resolved to the mention in the same
  paragraph (identical normalised name, else rapidfuzz ratio at least 90), then to its M3 cluster. One
  edge per (subject, type, other_label, object); n_sources counts the paragraphs that state it.
  Relations whose endpoint is not an extracted entity are dropped and counted; so are self loops.
- EXACT_TITLE: from the cluster of a non-page mention to the page entity whose normalised title equals
  the mention's normalised name, when exactly one page has that name and the two are different
  clusters. A separate, labelled edge type; never merged into the M3 clusters.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from graphrag.extraction.schema import DATE_RELATIONS
from graphrag.resolution.mentions import Mention, norm_name

ENDPOINT_MIN_RATIO = 90
DATE_PROPERTY = {r: r.replace(" ", "_") for r in DATE_RELATIONS}   # "date of birth" -> "date_of_birth"


@dataclass
class Entity:
    id: str
    name: str
    aliases: list[str]
    type: str
    is_page: bool
    page_chunk_id: str | None
    mention_ids: list[str]
    dates: dict[str, list[tuple[str, str]]] = field(default_factory=dict)   # property -> [(value, chunk_id)]


@dataclass(frozen=True)
class Rel:
    source: str
    type: str
    other_label: str | None
    target: str
    source_chunk_ids: tuple[str, ...]

    @property
    def n_sources(self) -> int:
        return len(self.source_chunk_ids)


@dataclass(frozen=True)
class ExactTitle:
    source: str          # cluster of the mention
    target: str          # page entity
    mention_ids: tuple[str, ...]


@dataclass
class GraphData:
    entities: dict[str, Entity]
    rels: list[Rel]
    exact_titles: list[ExactTitle]
    stats: dict


def _resolve(name: str, chunk_mentions: list[Mention]) -> Mention | None:
    n = norm_name(name)
    best, best_score = None, 0.0
    for m in chunk_mentions:
        score = 100.0 if m.norm == n else fuzz.ratio(m.norm, n)
        if score > best_score or (score == best_score and best is not None and m.id < best.id):
            best, best_score = m, score
    return best if best_score >= ENDPOINT_MIN_RATIO else None


def build_graph(mentions: list[Mention], cluster_of: dict[str, str], extraction_rows: list[dict]) -> GraphData:
    by_cluster: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions:
        by_cluster[cluster_of[m.id]].append(m)

    entities: dict[str, Entity] = {}
    for cid in sorted(by_cluster):
        ms = sorted(by_cluster[cid], key=lambda m: m.id)
        names = Counter(m.name for m in ms)
        types = Counter(m.type for m in ms)
        page = next((m for m in ms if m.is_page), None)
        entities[cid] = Entity(
            id=cid,
            name=min(names, key=lambda n: (-names[n], n)),
            aliases=sorted(names),
            type=min(types, key=lambda t: (-types[t], t)),
            is_page=page is not None,
            page_chunk_id=page.chunk_id if page else None,
            mention_ids=[m.id for m in ms],
        )

    by_chunk: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions:
        by_chunk[m.chunk_id].append(m)

    rel_sources: dict[tuple, set[str]] = defaultdict(set)
    dropped = Counter()
    for row in sorted(extraction_rows, key=lambda r: r["chunk_id"]):
        cid = row["chunk_id"]
        for t in row["relations"]:
            subj = _resolve(t["subject"], by_chunk[cid])
            if subj is None:
                dropped["subject is not an extracted entity"] += 1
                continue
            s = cluster_of[subj.id]
            if t["relation"] in DATE_RELATIONS:
                prop = DATE_PROPERTY[t["relation"]]
                values = entities[s].dates.setdefault(prop, [])
                if (t["object"], cid) not in values:
                    values.append((t["object"], cid))
                continue
            obj = _resolve(t["object"], by_chunk[cid])
            if obj is None:
                dropped["object is not an extracted entity"] += 1
                continue
            o = cluster_of[obj.id]
            if s == o:
                dropped["self loop after resolution"] += 1
                continue
            label = t.get("other_label") if t["relation"] == "OTHER" else None
            rel_sources[(s, t["relation"], label, o)].add(cid)

    rels = [Rel(source=s, type=r, other_label=lab, target=o, source_chunk_ids=tuple(sorted(src)))
            for (s, r, lab, o), src in sorted(rel_sources.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "", kv[0][3]))]
    for e in entities.values():
        for values in e.dates.values():
            values.sort()

    pages_by_norm: dict[str, list[Mention]] = defaultdict(list)
    for m in mentions:
        if m.is_page:
            pages_by_norm[m.norm].append(m)
    exact: dict[tuple[str, str], list[str]] = defaultdict(list)
    for m in sorted(mentions, key=lambda m: m.id):
        if m.is_page or len(pages_by_norm.get(m.norm, [])) != 1:
            continue
        src, tgt = cluster_of[m.id], cluster_of[pages_by_norm[m.norm][0].id]
        if src != tgt:
            exact[(src, tgt)].append(m.id)
    exact_titles = [ExactTitle(source=s, target=t, mention_ids=tuple(ids)) for (s, t), ids in sorted(exact.items())]

    degree = Counter()
    for r in rels:
        degree[r.source] += 1
        degree[r.target] += 1
    stats = {
        "entities": len(entities),
        "page_entities": sum(e.is_page for e in entities.values()),
        "rel_edges": len(rels),
        "rel_edges_by_type": dict(Counter(r.type for r in rels).most_common()),
        "exact_title_edges": len(exact_titles),
        "date_values": sum(len(v) for e in entities.values() for v in e.dates.values()),
        "dropped_relations": dict(dropped),
        "max_degree": max(degree.values()) if degree else 0,
    }
    return GraphData(entities=entities, rels=rels, exact_titles=exact_titles, stats=stats)


def degrees(rels: list[Rel]) -> Counter:
    """Degree over REL edges only (EXACT_TITLE links are identity links, not facts)."""
    d = Counter()
    for r in rels:
        d[r.source] += 1
        d[r.target] += 1
    return d
