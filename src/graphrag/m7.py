"""M7: the `neo4j-graphrag` library as a third system (docs/PLAN.md M7). Dev split only.

Adapters that let the library's KG builder and retriever run on this project's LLM client (OpenRouter,
disk cache, spend ledger) and E5 encoder, the schema handed to the library, and the per-paragraph build.
The library is an optional dependency (extra `m7`); nothing else in the package imports this module.
"""

import asyncio
import threading
from dataclasses import asdict, dataclass

from neo4j import Driver, GraphDatabase
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.llm.types import LLMResponse

from graphrag.config import Settings
from graphrag.embed import E5Encoder
from graphrag.extraction.schema import DATE_RELATIONS, ENTITY_TYPES, RELATIONS
from graphrag.llm.client import CachedLLM

LIBRARY_VERSION = "1.22.0"
M7_NEO4J_URI = "bolt://localhost:7688"      # container graphrag-neo4j-m7 (docker-compose.yml)
COST_CAP_USD = 1.0


class CostCapReached(RuntimeError):
    pass


@dataclass(frozen=True)
class CallRecord:
    input_tokens: int
    output_tokens: int
    cost_usd: float     # cost of the original live call (a cache hit repeats it, it is not paid again)
    cached: bool


class CachedLibraryLLM(LLMInterface):
    """The library's LLM interface on top of CachedLLM. Every call is cached and, when live, written to the
    spend ledger by CachedLLM. The build cost is the sum over all calls, cached or not, so a re-run reports
    the same cost; a live call is refused once that sum has reached the cap."""

    supports_structured_output = False      # the library then uses its prompt-and-parse path

    def __init__(self, llm: CachedLLM, cost_cap_usd: float = COST_CAP_USD):
        super().__init__(model_name=llm.model)
        self._llm = llm
        self._cap = cost_cap_usd
        self._lock = threading.Lock()
        self.calls: list[CallRecord] = []

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    def invoke(self, input, message_history=None, system_instruction=None) -> LLMResponse:
        if message_history:
            raise NotImplementedError("message history is not used by the KG builder")
        with self._lock:
            if self.cost_usd >= self._cap:
                raise CostCapReached(f"M7 build cost reached the cap of {self._cap} USD")
        r = self._llm.complete(input, system=system_instruction)
        with self._lock:
            self.calls.append(CallRecord(r.input_tokens, r.output_tokens, r.cost_usd, r.cached))
        return LLMResponse(content=r.text)

    async def ainvoke(self, input, message_history=None, system_instruction=None) -> LLMResponse:
        return await asyncio.to_thread(self.invoke, input, message_history, system_instruction)

    def summary(self) -> dict:
        return {"calls": len(self.calls), "live_calls": sum(not c.cached for c in self.calls),
                "input_tokens": sum(c.input_tokens for c in self.calls),
                "output_tokens": sum(c.output_tokens for c in self.calls),
                "cost_usd": round(self.cost_usd, 6),
                "paid_now_usd": round(sum(c.cost_usd for c in self.calls if not c.cached), 6)}


class E5LibraryEmbedder(Embedder):
    """E5 needs a different prefix for passages and for queries; the library has one method for both, so the
    builder gets a passage instance and the retriever a query instance."""

    def __init__(self, encoder: E5Encoder, mode: str):
        assert mode in ("passage", "query")
        self._encoder, self._mode = encoder, mode
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            enc = self._encoder.encode_passages if self._mode == "passage" else self._encoder.encode_queries
            return enc([text])[0].tolist()


def rel_label(relation: str) -> str:
    return relation.upper().replace(" ", "_")


def library_schema() -> dict:
    """The same information our own extractor gets (M2): the 6 entity types and the 34 relations. As in our
    graph, the 4 date relations are node properties and the other 30 are relationships."""
    date_props = [{"name": r.replace(" ", "_"), "type": "STRING", "description": RELATIONS[r]} for r in sorted(DATE_RELATIONS)]
    props = [{"name": "name", "type": "STRING"}] + date_props
    return {
        "node_types": [{"label": t, "properties": props} for t in ENTITY_TYPES],
        "relationship_types": [{"label": rel_label(r), "description": d}
                               for r, d in RELATIONS.items() if r not in DATE_RELATIONS],
    }


def connect_m7(settings: Settings) -> Driver:
    if settings.neo4j_password is None:
        raise RuntimeError("NEO4J_PASSWORD is not set in .env")
    driver = GraphDatabase.driver(M7_NEO4J_URI, auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()))
    driver.verify_connectivity()
    return driver


def paragraph_text(chunk: dict) -> str:
    return f"{chunk['title']}\n{chunk['text']}"


async def build_paragraphs(chunks: list[dict], llm: CachedLibraryLLM, embedder: Embedder, driver: Driver,
                           concurrency: int = 8) -> list[dict]:
    """One library pipeline run per paragraph. Entity resolution is switched off inside the runs and run
    once afterwards with the library's default resolver (resolve_entities), which gives the same merges
    without 2,049 whole-graph passes. A paragraph whose extraction fails is recorded and writes no entities."""
    pipeline = SimpleKGPipeline(llm=llm, driver=driver, embedder=embedder, schema=library_schema(),
                                from_file=False, on_error="RAISE", perform_entity_resolution=False)
    sem = asyncio.Semaphore(concurrency)

    async def one(chunk: dict) -> dict:
        async with sem:
            try:
                await pipeline.run_async(text=paragraph_text(chunk), document_metadata={"chunk_id": chunk["id"]})
                return {"chunk_id": chunk["id"], "ok": True}
            except CostCapReached:
                raise
            except Exception as e:      # recorded per paragraph; one failure must not stop the build
                return {"chunk_id": chunk["id"], "ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}

    return list(await asyncio.gather(*(one(c) for c in chunks)))


def resolve_entities(driver: Driver) -> dict:
    from neo4j_graphrag.experimental.components.resolver import SinglePropertyExactMatchResolver

    stats = asyncio.run(SinglePropertyExactMatchResolver(driver=driver).run())
    return stats.model_dump()


def graph_counts(driver: Driver) -> dict:
    q = lambda cypher: driver.execute_query(cypher).records[0][0]
    return {
        "documents": q("MATCH (d:Document) RETURN count(d)"),
        "chunks": q("MATCH (c:Chunk) RETURN count(c)"),
        "entities": q("MATCH (e:__Entity__) RETURN count(e)"),
        "relations": q("MATCH (:__Entity__)-[r]->(:__Entity__) RETURN count(r)"),
        "entities_by_label": {r["l"]: r["n"] for r in driver.execute_query(
            "MATCH (e:__Entity__) UNWIND [x IN labels(e) WHERE NOT x STARTS WITH '__'] AS l "
            "RETURN l, count(*) AS n ORDER BY l").records},
        "relations_by_type": {r["t"]: r["n"] for r in driver.execute_query(
            "MATCH (:__Entity__)-[r]->(:__Entity__) RETURN type(r) AS t, count(*) AS n ORDER BY t").records},
    }


__all__ = ["CachedLibraryLLM", "CallRecord", "CostCapReached", "E5LibraryEmbedder", "asdict", "build_paragraphs",
           "connect_m7", "graph_counts", "library_schema", "paragraph_text", "rel_label", "resolve_entities"]
