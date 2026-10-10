"""M7.4 pilot: build the library's graph for 50 seeded corpus paragraphs and project the full build cost.

Needs the container graphrag-neo4j-m7 (docker compose up -d). The M7 database must be empty: the pilot
graph is deleted at the end unless --keep is given. Writes results/m7/pilot.json. Reads no question file.
"""

import argparse
import asyncio
import json
import random
import time

from graphrag.config import get_settings
from graphrag.embed import E5Encoder
from graphrag.llm.client import make_llm
from graphrag.m7 import (COST_CAP_USD, LIBRARY_VERSION, CachedLibraryLLM, E5LibraryEmbedder, build_paragraphs,
                         connect_m7, graph_counts, resolve_entities)

N_PILOT = 50
SEED = "42:m7-pilot"
MAX_FAILURE_SHARE = 0.10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the pilot graph in the M7 database")
    args = ap.parse_args()
    s = get_settings()
    corpus = [json.loads(line) for line in (s.data_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    sample = random.Random(SEED).sample(sorted(corpus, key=lambda c: c["id"]), N_PILOT)

    driver = connect_m7(s)
    try:
        existing = driver.execute_query("MATCH (n) RETURN count(n)").records[0][0]
        if existing:
            raise SystemExit(f"the M7 database is not empty ({existing} nodes); refusing to run the pilot into it")
        llm = CachedLibraryLLM(make_llm(s.generator_model, s))
        embedder = E5LibraryEmbedder(E5Encoder(s.embedding_model), "passage")
        t0 = time.time()
        runs = asyncio.run(build_paragraphs(sample, llm, embedder, driver))
        before = graph_counts(driver)
        resolution = resolve_entities(driver)
        after = graph_counts(driver)
        seconds = round(time.time() - t0, 1)
        if not args.keep:
            driver.execute_query("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()

    failed = [r for r in runs if not r["ok"]]
    spend = llm.summary()
    projected = spend["cost_usd"] / N_PILOT * len(corpus)
    out = {
        "library": f"neo4j-graphrag {LIBRARY_VERSION}", "model": s.generator_model, "endpoint": s.generator_provider,
        "paragraphs": N_PILOT, "seed": SEED, "corpus_paragraphs": len(corpus), "seconds": seconds,
        "failed": len(failed), "failures": failed, "spend": spend,
        "projected_full_build_usd": round(projected, 4), "cost_cap_usd": COST_CAP_USD,
        "graph_before_resolution": before, "resolution": resolution, "graph_after_resolution": after,
        "go": projected <= COST_CAP_USD and len(failed) / N_PILOT <= MAX_FAILURE_SHARE,
    }
    path = s.results_dir / "m7" / "pilot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("paragraphs", "failed", "spend", "projected_full_build_usd", "seconds", "go")}, indent=1))
    print("entities", after["entities"], "relations", after["relations"], "by type", after["relations_by_type"])


if __name__ == "__main__":
    main()
