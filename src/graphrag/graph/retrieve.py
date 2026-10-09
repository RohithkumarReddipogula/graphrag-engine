"""Graph retrieval (docs/PLAN.md, M4.2, approved 2026-10-09).

- Seeds (at most MAX_SEEDS): entity names and aliases of at least 3 characters found verbatim in the
  normalised question at word boundaries, longest match first (a match inside an already accepted span
  is skipped), page entities before other entities; then, if there are fewer than MAX_SEEDS, E5 nearest
  entities to the question with cosine at least SEED_MIN_COS.
- Expansion: up to 2 hops over REL edges in either direction. EXACT_TITLE edges are identity links: an
  entity and its exact-title partners share their neighbours, and following such a link is not a hop.
  At most MAX_NEIGHBOURS neighbours per node (most n_sources first, then name, then id).
- Hubs: an entity with REL degree above HUB_DEGREE can end a path but is never passed through. Every
  intermediate node multiplies the path score by 1 / log2(2 + degree).
- Path score: E5 cosine between the question ("query: ") and the path written out as text ("passage: "),
  times the hub factors. The TOP_PATHS best paths are kept; ties broken by the path text.
- Graph context: the kept paths as cited facts, then the page paragraphs of the entities on those paths,
  in path-score order.
"""

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from graphrag.generation import PackedContext, n_tokens
from graphrag.graph.build import DATE_PROPERTY, GraphData, degrees
from graphrag.resolution.mentions import norm_name

MAX_SEEDS = 5
SEED_MIN_COS = 0.85
MIN_ALIAS_CHARS = 3
MAX_NEIGHBOURS = 20
HUB_DEGREE = 25
TOP_PATHS = 10
MAX_NGRAM = 12
DATE_LABEL = {prop: prop.replace("_", " ") for prop in DATE_PROPERTY.values()}


@dataclass(frozen=True)
class Step:
    source: str          # entity id the edge starts at (as stored)
    rel: str             # relation label shown in text
    target: str
    chunk_ids: tuple[str, ...]


@dataclass
class Path:
    nodes: list[str]     # entity ids along the path, seed first
    steps: list[Step]
    text: str
    sources: list[str]   # chunk ids that state the facts on the path
    score: float = 0.0


class GraphIndex:
    def __init__(self, g: GraphData, entity_emb: dict[str, np.ndarray], use_exact_title: bool = True):
        self.g = g
        self.use_exact_title = use_exact_title
        self.degree = degrees(g.rels)
        self.ids = sorted(entity_emb)
        self.emb = np.stack([entity_emb[i] for i in self.ids]) if self.ids else np.zeros((0, 1))

        self.partners: dict[str, set[str]] = defaultdict(set)
        if use_exact_title:
            for x in g.exact_titles:
                self.partners[x.source].add(x.target)
                self.partners[x.target].add(x.source)

        raw: dict[str, list[tuple[Step, str, int]]] = defaultdict(list)
        for r in g.rels:
            label = f"{r.type}: {r.other_label}" if r.type == "OTHER" and r.other_label else r.type
            step = Step(source=r.source, rel=label, target=r.target, chunk_ids=r.source_chunk_ids)
            raw[r.source].append((step, r.target, r.n_sources))
            raw[r.target].append((step, r.source, r.n_sources))
        self.adj: dict[str, list[tuple[Step, str]]] = {}
        for node in set(raw) | set(self.partners):
            group = sorted({node} | self.partners.get(node, set()))
            options = [o for member in group for o in raw.get(member, [])]
            options.sort(key=lambda o: (-o[2], self.g.entities[o[1]].name, o[1], o[0].rel))
            seen, kept = set(), []
            for step, nb, _ in options:
                if nb in group or (step, nb) in seen:
                    continue
                seen.add((step, nb))
                kept.append((step, nb))
                if len(kept) == MAX_NEIGHBOURS:
                    break
            self.adj[node] = kept

        self.alias_index: dict[str, list[str]] = defaultdict(list)
        for e in g.entities.values():
            for a in e.aliases:
                n = norm_name(a)
                if len(n) >= MIN_ALIAS_CHARS and e.id not in self.alias_index[n]:
                    self.alias_index[n].append(e.id)

    # ---------- seeds ----------
    def seeds(self, question: str, q_emb: np.ndarray) -> list[tuple[str, str]]:
        """[(entity id, how found)]."""
        toks = norm_name(question).split()
        matches = []
        for i in range(len(toks)):
            for j in range(min(len(toks), i + MAX_NGRAM), i, -1):
                phrase = " ".join(toks[i:j])
                if phrase in self.alias_index:
                    matches.append((i, j, phrase))
        matches.sort(key=lambda m: (-(m[1] - m[0]), m[0]))
        taken, accepted = set(), []
        for i, j, phrase in matches:
            if any(k in taken for k in range(i, j)):
                continue
            taken.update(range(i, j))
            accepted.append((j - i, phrase))
        cands = {eid: length for length, phrase in accepted for eid in self.alias_index[phrase]}
        ordered = sorted(cands, key=lambda e: (not self.g.entities[e].is_page, -cands[e], e))
        out = [(e, "verbatim") for e in ordered[:MAX_SEEDS]]
        if len(out) < MAX_SEEDS and len(self.ids):
            sims = self.emb @ q_emb
            for k in np.argsort(-sims, kind="stable"):
                if sims[k] < SEED_MIN_COS or len(out) >= MAX_SEEDS:
                    break
                e = self.ids[k]
                if all(e != x for x, _ in out):
                    out.append((e, "embedding"))
        return out

    # ---------- paths ----------
    def _dates(self, eid: str) -> list[tuple[str, str]]:
        e = self.g.entities[eid]
        out = []
        for member in sorted({eid} | self.partners.get(eid, set())):
            for prop, values in self.g.entities[member].dates.items():
                out += [(f"{e.name}: {DATE_LABEL[prop]} {v}", c) for v, c in values]
        return sorted(set(out))

    def _make(self, nodes: list[str], steps: list[Step]) -> Path:
        name = lambda x: self.g.entities[x].name
        parts = [f"{name(s.source)} -{s.rel}-> {name(s.target)}" for s in steps]
        sources = sorted({c for s in steps for c in s.chunk_ids})
        for n in nodes:
            for text, chunk in self._dates(n):
                parts.append(text)
                sources = sorted(set(sources) | {chunk})
        if not parts:
            parts.append(name(nodes[0]))
        return Path(nodes=nodes, steps=steps, text="; ".join(parts), sources=sources)

    def paths(self, seed_ids: list[str]) -> list[Path]:
        out, seen = [], set()

        def add(p: Path):
            if p.text not in seen:
                seen.add(p.text)
                out.append(p)

        for s in seed_ids:
            add(self._make([s], []))
            for step1, v in self.adj.get(s, []):
                add(self._make([s, v], [step1]))
                if self.degree.get(v, 0) > HUB_DEGREE:
                    continue                      # hubs can end a path but are never passed through
                for step2, w in self.adj.get(v, []):
                    if w == s or w in self.partners.get(s, set()):
                        continue
                    add(self._make([s, v, w], [step1, step2]))
        return out

    def hub_factor(self, p: Path) -> float:
        f = 1.0
        for v in p.nodes[1:-1]:
            f /= math.log2(2 + self.degree.get(v, 0))
        return f

    def page_chunk(self, eid: str) -> str | None:
        for member in [eid] + sorted(self.partners.get(eid, set())):
            pc = self.g.entities[member].page_chunk_id
            if pc:
                return pc
        return None


def score_paths(paths: list[Path], q_emb: np.ndarray, path_emb: np.ndarray, index: GraphIndex) -> list[Path]:
    for p, e in zip(paths, path_emb):
        p.score = float(e @ q_emb) * index.hub_factor(p)
    return sorted(paths, key=lambda p: (-p.score, p.text))[:TOP_PATHS]


def graph_context(top: list[Path], index: GraphIndex, corpus: dict[str, dict], budget_tokens: int,
                  skip_chunks: set[str] = frozenset()) -> PackedContext:
    """Facts first, then page paragraphs of the entities on the kept paths, in path-score order. A block
    that does not fit is skipped and the next one is tried (as in M1's pack_context)."""
    parts, ids, used = [], [], 0
    for p in top:
        line = f"Fact: {p.text} [source: {', '.join(p.sources)}]"
        cost = n_tokens(line) + 1
        if used + cost <= budget_tokens:
            parts.append(line)
            used += cost
    for p in top:
        for v in p.nodes:
            pc = index.page_chunk(v)
            if not pc or pc in ids or pc in skip_chunks:
                continue
            block = f"[{corpus[pc]['title']}] {corpus[pc]['text']}"
            cost = n_tokens(block) + 2
            if used + cost <= budget_tokens:
                parts.append(block)
                ids.append(pc)
                used += cost
    return PackedContext(text="\n\n".join(parts), chunk_ids=ids, tokens=used)
