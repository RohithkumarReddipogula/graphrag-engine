"""The API pipeline must build exactly the contexts of the committed M4 dev run of graph_plus_chunks_g0.5.
No LLM calls. Skipped when Neo4j is not running (it needs the graph and the models)."""

import json

import pytest

from graphrag.config import get_settings

N_QUESTIONS = 5


def _neo4j_up():
    try:
        from graphrag.store.neo4j_client import connect

        with connect(get_settings()):
            return True
    except Exception:  # noqa: BLE001 - any failure means skip
        return False


@pytest.mark.skipif(not _neo4j_up(), reason="Neo4j is not running")
def test_api_contexts_equal_the_committed_m4_dev_contexts():
    from graphrag.pipeline import GraphRAGPipeline

    s = get_settings()
    rows = [json.loads(line) for line in (s.results_dir / "m4" / "generation_dev_graph_plus_chunks_g0.5.jsonl").read_text().splitlines()]
    questions = {json.loads(line)["id"]: json.loads(line) for line in (s.data_dir / "questions_dev.jsonl").read_text().splitlines()}
    picked = [r for r in rows if r["id"] in questions][:N_QUESTIONS]
    pipe = GraphRAGPipeline(s)
    try:
        for r in picked:
            assert pipe.retrieve(questions[r["id"]]["question"]).context.chunk_ids == r["context_chunk_ids"], r["id"]
    finally:
        pipe.close()
