"""M4.1: build the graph from the M2 extractions and the M3 clusters and load it into Neo4j.
Writes results/m4/graph_stats.json. No LLM calls."""

import json
import logging

from graphrag.config import get_settings
from graphrag.graph.build import build_graph, degrees
from graphrag.graph.store import fetch_graph, load_graph
from graphrag.resolution.mentions import build_mentions
from graphrag.store.neo4j_client import connect

logging.getLogger("neo4j").setLevel(logging.ERROR)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    s = get_settings()
    corpus = {c["id"]: c for c in read_jsonl(s.data_dir / "corpus.jsonl")}
    rows = read_jsonl(s.results_dir / "m2" / "extractions.jsonl")
    mentions = build_mentions(rows, corpus)
    cluster_of = {r["mention"]: r["cluster"] for r in read_jsonl(s.results_dir / "m3" / "clusters.jsonl")}
    g = build_graph(mentions, cluster_of, rows)

    with connect(s) as driver:
        counts = load_graph(driver, g)
        back = fetch_graph(driver)
    assert len(back.entities) == len(g.entities) and len(back.rels) == len(g.rels) \
        and len(back.exact_titles) == len(g.exact_titles), "graph read back from Neo4j differs from the built graph"

    deg = degrees(g.rels)
    top = sorted(deg.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    stats = {**g.stats, "neo4j_counts": counts,
             "degree_above_25": sum(v > 25 for v in deg.values()),
             "top_degree": [{"entity": e, "name": g.entities[e].name, "degree": d} for e, d in top]}
    out = s.results_dir / "m4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
