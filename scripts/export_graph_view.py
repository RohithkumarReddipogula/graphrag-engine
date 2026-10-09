"""M6.3: export a small static subgraph for the portfolio website (docs/PLAN.md, approved 2026-10-09).

20 dev questions (5 per multi-hop type, seeded); test questions are never exported. For each: the
question, the gold answer, the graph_plus_chunks_g0.5 dev answer taken from the committed M4 run (no new
LLM calls), and the graph paths that were used as facts in its context, recomputed deterministically with
the same pipeline as the API. Writes docs/graph_view/subgraph.json: nodes, links and questions in the
node/link shape that force-graph libraries load directly. Names and relation labels only, no paragraph
text. Needs Neo4j running.
"""

import json
import random

from graphrag.config import ROOT, get_settings
from graphrag.pipeline import GraphRAGPipeline

SEED = 42
PER_TYPE = 5
TYPES = ["comparison", "inference", "compositional", "bridge_comparison"]
OUT = ROOT / "docs" / "graph_view" / "subgraph.json"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    s = get_settings()
    questions = read_jsonl(s.data_dir / "questions_dev.jsonl")
    answers = {r["id"]: r for r in read_jsonl(s.results_dir / "m4" / "generation_dev_graph_plus_chunks_g0.5.jsonl")}
    rng = random.Random(f"{SEED}:graph-view")
    picked = []
    for t in TYPES:
        pool = sorted((q for q in questions if q["type"] == t), key=lambda q: q["id"])
        picked += rng.sample(pool, PER_TYPE)

    pipe = GraphRAGPipeline(s)
    nodes, links, out_questions = {}, {}, []
    try:
        for q in picked:
            r = pipe.retrieve(q["question"])
            used = [p for p in r.top_paths if f"Fact: {p.text}" in r.context.text]
            node_ids, link_ids = [], []

            def add_node(n):
                e = pipe.graph.entities[n]
                nodes.setdefault(n, {"id": n, "label": e.name, "type": e.type, "is_page": e.is_page})
                if n not in node_ids:
                    node_ids.append(n)

            def add_link(lid, source, target, label, chunk_ids):
                links.setdefault(lid, {"id": lid, "source": source, "target": target, "label": label,
                                       "source_chunk_ids": chunk_ids})
                if lid not in link_ids:
                    link_ids.append(lid)

            for p in used:
                for n in p.nodes:
                    add_node(n)
                for st in p.steps:
                    add_link(f"{st.source}|{st.rel}|{st.target}", st.source, st.target, st.rel, list(st.chunk_ids))
                    # A step can sit on an exact-title partner of a path node (identity link, not a hop):
                    # export the partner and the exact-title link itself, so the crossing is visible.
                    for end in (st.source, st.target):
                        if end in p.nodes:
                            continue
                        add_node(end)
                        for n in p.nodes:
                            if end in pipe.index.partners.get(n, set()):
                                a, b = sorted((n, end))
                                add_link(f"{a}|same entity (exact title)|{b}", a, b, "same entity (exact title)", [])
            a = answers[q["id"]]
            out_questions.append({"id": q["id"], "type": q["type"], "question": q["question"], "gold": q["answer"],
                                  "answer": a["prediction"], "exact_match": a["em"] == 1.0,
                                  "facts": [p.text for p in used], "path_node_ids": node_ids, "path_link_ids": link_ids})
    finally:
        pipe.close()

    data = {
        "description": "Graph paths that the GraphRAG system (graph_plus_chunks_g0.5) used as facts for 20 dev "
                       "questions of 2WikiMultihopQA; answers from the committed dev run (results/m4).",
        "source": "github.com/RohithkumarReddipogula/graphrag-engine",
        "licence": "Code MIT; entity names and relation labels derived from Wikipedia and Wikidata via 2WikiMultihopQA",
        "nodes": sorted(nodes.values(), key=lambda n: n["id"]),
        "links": sorted(links.values(), key=lambda l: l["id"]),
        "questions": out_questions,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {len(nodes)} nodes, {len(links)} links, {len(out_questions)} questions, "
          f"{OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
