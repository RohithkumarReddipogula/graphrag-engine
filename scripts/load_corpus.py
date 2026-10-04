"""M1: create the Neo4j schema and load data/corpus.jsonl with E5 embeddings (idempotent)."""

import json
import time

from graphrag.config import get_settings
from graphrag.embed import E5Encoder
from graphrag.store.corpus_store import ensure_schema, load_corpus
from graphrag.store.neo4j_client import connect

if __name__ == "__main__":
    settings = get_settings()
    chunks = [json.loads(line) for line in (settings.data_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    with connect(settings) as driver:
        ensure_schema(driver)
        t0 = time.time()
        stats = load_corpus(driver, chunks, E5Encoder(settings.embedding_model))
        print(f"{stats} in {time.time() - t0:.0f}s")
