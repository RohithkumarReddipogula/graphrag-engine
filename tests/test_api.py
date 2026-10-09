from fastapi.testclient import TestClient

from graphrag.api import MAX_QUESTION_CHARS, create_app


class FakePipeline:
    def __init__(self):
        self.asked = []

    def health(self, expected):
        return {"neo4j": "reachable", "graph_counts": expected, "graph_matches_m4": True, "models_loaded": True}

    def ask(self, question):
        self.asked.append(question)
        return {"answer": "Michael Mann", "unknown": False,
                "facts": [{"path": "Heat -director-> Michael Mann", "source_chunk_ids": ["Heat"]}],
                "paragraphs": [{"chunk_id": "Heat", "title": "Heat"}], "config": {"system": "graph_plus_chunks_g0.5"}}

    def entity(self, entity_id):
        return {"id": entity_id, "name": "Adam's Rib"} if entity_id == "m:Adam's Rib (1923 film)::0" else None


def client():
    fake = FakePipeline()
    return TestClient(create_app(pipeline=fake)), fake


def test_health_and_ask():
    c, fake = client()
    with c:
        assert c.get("/health").json()["graph_matches_m4"] is True
        r = c.post("/ask", json={"question": "  Who directed Heat?  "})
        assert r.status_code == 200 and r.json()["facts"][0]["source_chunk_ids"] == ["Heat"]
        assert fake.asked == ["Who directed Heat?"]


def test_ask_rejects_empty_and_too_long_questions():
    c, _ = client()
    with c:
        assert c.post("/ask", json={"question": ""}).status_code == 422
        assert c.post("/ask", json={"question": "x" * (MAX_QUESTION_CHARS + 1)}).status_code == 422


def test_entity_ids_with_colons_spaces_and_slashes_and_404():
    c, _ = client()
    with c:
        assert c.get("/entity/m:Adam's Rib (1923 film)::0").json()["name"] == "Adam's Rib"
        assert c.get("/entity/no such entity").status_code == 404


def test_api_is_read_only():
    c, _ = client()
    with c:
        routes = {(m, r.path) for r in c.app.routes for m in getattr(r, "methods", set())}
        assert {m for m, _ in routes} <= {"GET", "POST", "HEAD"}
        assert {p for m, p in routes if m == "POST"} == {"/ask"}
