"""Neo4j schema and idempotent corpus loader for M1.

(:Document {id, title})-[:HAS_CHUNK]->(:Chunk {id, title, text, text_hash, embedding_title, embedding_text})

Two embeddings per chunk, one of "title. text" and one of the text alone, each with its own vector index,
so whether to index the title can be decided on dev. One document per paragraph for now.
"""

import hashlib
from typing import Iterable

from neo4j import Driver

from graphrag.embed import E5Encoder
from graphrag.retrieval.base import passage_text

DIM = 768
VECTOR_INDEXES = {True: "chunk_embedding_title", False: "chunk_embedding_text"}
EMBEDDING_PROPERTY = {True: "embedding_title", False: "embedding_text"}


def text_hash(title: str, text: str, model_name: str) -> str:
    return hashlib.sha1(f"{model_name}\n{title}\n{text}".encode("utf-8")).hexdigest()


def ensure_schema(driver: Driver) -> None:
    driver.execute_query("CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE")
    driver.execute_query("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")
    for with_title, name in VECTOR_INDEXES.items():
        prop = EMBEDDING_PROPERTY[with_title]
        driver.execute_query(
            f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (c:Chunk) ON c.{prop} "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {DIM}, `vector.similarity_function`: 'cosine'}}}}"
        )
    driver.execute_query("CALL db.awaitIndexes(300)")


def load_corpus(driver: Driver, chunks: Iterable[dict], encoder: E5Encoder, batch_size: int = 128) -> dict[str, int]:
    """Upsert chunks; embed only new or changed ones; delete chunks that are no longer in the corpus.
    Running it twice in a row changes nothing."""
    chunks = list(chunks)
    wanted = {c["id"]: text_hash(c["title"], c["text"], encoder.model_name) for c in chunks}
    stored = {
        r["id"]: r["h"]
        for r in driver.execute_query("MATCH (c:Chunk) RETURN c.id AS id, c.text_hash AS h").records
    }

    stale = [cid for cid in stored if cid not in wanted]
    if stale:
        driver.execute_query(
            "MATCH (c:Chunk) WHERE c.id IN $ids "
            "OPTIONAL MATCH (d:Document)-[:HAS_CHUNK]->(c) DETACH DELETE c, d",
            ids=stale,
        )

    todo = [c for c in chunks if stored.get(c["id"]) != wanted[c["id"]]]
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        emb_title = encoder.encode_passages([passage_text(c["title"], c["text"], True) for c in batch])
        emb_text = encoder.encode_passages([passage_text(c["title"], c["text"], False) for c in batch])
        rows = [
            {
                "id": c["id"], "title": c["title"], "text": c["text"], "h": wanted[c["id"]],
                "et": emb_title[j].tolist(), "ex": emb_text[j].tolist(),
            }
            for j, c in enumerate(batch)
        ]
        driver.execute_query(
            "UNWIND $rows AS r "
            "MERGE (d:Document {id: r.id}) SET d.title = r.title "
            "MERGE (c:Chunk {id: r.id}) "
            "SET c.title = r.title, c.text = r.text, c.text_hash = r.h, "
            "    c.embedding_title = r.et, c.embedding_text = r.ex "
            "MERGE (d)-[:HAS_CHUNK]->(c)",
            rows=rows,
        )
    return {"wanted": len(wanted), "embedded": len(todo), "deleted": len(stale)}


def fetch_chunks(driver: Driver) -> list[dict]:
    """All chunks, ordered by id (BM25 is built from this at startup)."""
    records = driver.execute_query("MATCH (c:Chunk) RETURN c.id AS id, c.title AS title, c.text AS text ORDER BY c.id").records
    return [dict(r) for r in records]


def vector_search(driver: Driver, query_vec: list[float], k: int, with_title: bool) -> list[tuple[str, float]]:
    """Exact cosine top-k, computed in Neo4j over every chunk.

    The HNSW index is approximate: on 20 dev questions its top-50 missed up to 3 exact hits in the tail,
    and fusion uses the whole top-50. At about 2k chunks an exact scan is fast, so the baseline uses it.
    Ties are broken by chunk id so results are deterministic."""
    prop = EMBEDDING_PROPERTY[with_title]
    records = driver.execute_query(
        f"MATCH (c:Chunk) WITH c, vector.similarity.cosine(c.{prop}, $vec) AS score "
        "RETURN c.id AS id, score ORDER BY score DESC, id LIMIT $k",
        vec=query_vec, k=k,
    ).records
    return [(r["id"], float(r["score"])) for r in records]
