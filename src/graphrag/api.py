"""Read-only API (docs/PLAN.md, M6.2). Runs locally from the venv:

    .venv/bin/uvicorn graphrag.api:app --host 127.0.0.1 --port 8000

GET /health, POST /ask, GET /entity/{id}. No ingestion and no writes. Each new /ask question makes one
cached call to the pinned generator (OpenRouter, prepaid credit); the key is read from .env and never
returned or logged.
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from graphrag.config import get_settings

MAX_QUESTION_CHARS = 500


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_QUESTION_CHARS)


def create_app(pipeline=None) -> FastAPI:
    """pipeline=None loads the real GraphRAGPipeline at startup (Neo4j + models); tests pass a stand-in."""
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if pipeline is None:
            from graphrag.pipeline import GraphRAGPipeline

            state["pipeline"] = GraphRAGPipeline(get_settings())
        else:
            state["pipeline"] = pipeline
        yield
        if pipeline is None:
            state["pipeline"].close()

    app = FastAPI(title="GraphRAG Engine (read-only)", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        s = get_settings()
        expected = json.loads((s.results_dir / "m4" / "graph_stats.json").read_text())["neo4j_counts"]
        return state["pipeline"].health(expected)

    @app.post("/ask")
    def ask(req: AskRequest):
        return state["pipeline"].ask(req.question.strip())

    @app.get("/entity/{entity_id:path}")
    def entity(entity_id: str):
        found = state["pipeline"].entity(entity_id)
        if found is None:
            raise HTTPException(status_code=404, detail="entity not found")
        return found

    return app


app = create_app()
