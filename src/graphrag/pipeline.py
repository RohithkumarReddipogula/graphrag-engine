"""The frozen GraphRAG system (`graph_plus_chunks_g0.5`, docs/PLAN.md M5.2) as a reusable pipeline for the
read-only API (M6.2). Same retrieval, context packing, prompt and generator as the M5 test run; the
committed scripts are not changed. tests/test_pipeline_parity.py checks, without LLM calls, that the
contexts it builds equal the committed M4 dev contexts of graph_plus_chunks_g0.5.
"""

import json
import logging
from dataclasses import dataclass

import numpy as np

from graphrag.config import Settings
from graphrag.embed import E5Encoder
from graphrag.generation import PackedContext, build_prompt, parse_answer
from graphrag.graph.build import GraphData
from graphrag.graph.retrieve import GraphIndex, Path, graph_context, score_paths
from graphrag.graph.store import fetch_graph
from graphrag.llm.client import make_llm
from graphrag.resolution.embed_mentions import embed_mentions
from graphrag.resolution.mentions import build_mentions
from graphrag.retrieval.hybrid import CrossEncoderReranker, HybridConfig, HybridRetriever
from graphrag.store.corpus_store import fetch_chunks, vector_search
from graphrag.store.neo4j_client import connect
from graphrag.systems import combine

logging.getLogger("neo4j").setLevel(logging.ERROR)
BUDGET = 1500
G = 0.5
SYSTEM_NAME = "graph_plus_chunks_g0.5"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@dataclass
class Retrieved:
    context: PackedContext
    top_paths: list[Path]


class GraphRAGPipeline:
    def __init__(self, settings: Settings):
        self.s = settings
        self.corpus = {c["id"]: c for c in read_jsonl(settings.data_dir / "corpus.jsonl")}
        mentions = build_mentions(read_jsonl(settings.results_dir / "m2" / "extractions.jsonl"), self.corpus)
        m_emb = embed_mentions(mentions, settings.embedding_model, settings.cache_dir)
        m_idx = {m.id: i for i, m in enumerate(mentions)}
        by_mention = {m.id: m for m in mentions}
        self.encoder = E5Encoder(settings.embedding_model)
        self.driver = connect(settings)
        self.graph: GraphData = fetch_graph(self.driver)

        def rep(e):
            ids = sorted(e.mention_ids)
            pages = [i for i in ids if by_mention[i].is_page]
            return m_emb[m_idx[(pages or ids)[0]]]

        self.index = GraphIndex(self.graph, {eid: rep(e) for eid, e in self.graph.entities.items()}, use_exact_title=True)
        chosen = json.loads((settings.results_dir / "m1" / "retrieval_dev.json").read_text())["chosen"]
        dense = lambda q, k, wt: vector_search(self.driver, self.encoder.encode_queries([q])[0].tolist(), k, wt)
        self.hybrid = HybridRetriever(fetch_chunks(self.driver), dense, CrossEncoderReranker(settings.reranker_model),
                                      HybridConfig(with_title=chosen["with_title"], fusion=chosen["fusion"], alpha=chosen["alpha"]))
        self.frozen = json.loads((settings.results_dir / "m5" / "frozen_settings.json").read_text())
        self._llm = None

    def retrieve(self, question: str) -> Retrieved:
        """Graph context with share G of the budget, then hybrid chunks; no LLM call."""
        q_emb = self.encoder.encode_queries([question])[0]
        seeds = self.index.seeds(question, q_emb)
        paths = self.index.paths([e for e, _ in seeds])
        top = score_paths(paths, q_emb, self.encoder.encode_passages([p.text for p in paths]), self.index) if paths else []
        gpart = graph_context(top, self.index, self.corpus, int(G * BUDGET))
        return Retrieved(context=combine(gpart, self.hybrid.retrieve(question, self.hybrid.config.rerank_top), BUDGET),
                         top_paths=top)

    def ask(self, question: str) -> dict:
        r = self.retrieve(question)
        if self._llm is None:
            self._llm = make_llm(self.s.generator_model, self.s)
        system_prompt, prompt = build_prompt(question, r.context)
        res = self._llm.complete(prompt, system=system_prompt)
        answer = parse_answer(res.text)
        return {
            "answer": answer,
            "unknown": answer.strip().lower() == "unknown",
            "facts": [{"path": p.text, "source_chunk_ids": p.sources} for p in r.top_paths
                      if f"Fact: {p.text}" in r.context.text],
            "paragraphs": [{"chunk_id": c, "title": self.corpus[c]["title"]} for c in r.context.chunk_ids],
            "config": {"system": SYSTEM_NAME, "m5_frozen_commit": self.frozen["commit"],
                       "m5_settings_hash": self.frozen["settings_hash"], "generator": self.s.generator_model,
                       "endpoint": self.s.generator_provider, "cached": res.cached},
        }

    def entity(self, entity_id: str, max_neighbours: int = 50) -> dict | None:
        e = self.graph.entities.get(entity_id)
        if e is None:
            return None
        rels = [r for r in self.graph.rels if entity_id in (r.source, r.target)]
        rels.sort(key=lambda r: (-r.n_sources, r.type, r.source, r.target))
        name = lambda x: self.graph.entities[x].name
        return {
            "id": e.id, "name": e.name, "aliases": e.aliases, "type": e.type, "is_page": e.is_page,
            "dates": {k: [{"value": v, "source_chunk_id": c} for v, c in vals] for k, vals in e.dates.items()},
            "relations": [{"source": r.source, "source_name": name(r.source), "relation": r.type,
                           "other_label": r.other_label, "target": r.target, "target_name": name(r.target),
                           "source_chunk_ids": list(r.source_chunk_ids)} for r in rels[:max_neighbours]],
            "relations_total": len(rels),
            "exact_title_links": [{"entity": x.target if x.source == entity_id else x.source,
                                   "name": name(x.target if x.source == entity_id else x.source)}
                                  for x in self.graph.exact_titles if entity_id in (x.source, x.target)],
            "page_paragraph": ({"chunk_id": e.page_chunk_id, "title": self.corpus[e.page_chunk_id]["title"],
                                "text": self.corpus[e.page_chunk_id]["text"]} if e.page_chunk_id else None),
        }

    def health(self, expected_counts: dict) -> dict:
        counts = dict(self.driver.execute_query(
            "MATCH (e:Entity) WITH count(e) AS entities OPTIONAL MATCH ()-[r:REL]->() WITH entities, count(r) AS rels "
            "OPTIONAL MATCH ()-[x:EXACT_TITLE]->() WITH entities, rels, count(x) AS exact "
            "OPTIONAL MATCH ()-[m:MENTIONED_IN]->() RETURN entities, rels, exact, count(m) AS mentioned_in").records[0])
        return {"neo4j": "reachable", "graph_counts": counts, "graph_matches_m4": counts == expected_counts,
                "models_loaded": True}

    def close(self) -> None:
        self.driver.close()
