"""Load the M4 graph into Neo4j and read it back (docs/PLAN.md, M4.1).

Same database as the M1 chunks. Loading is deterministic and idempotent: the entity graph (Entity nodes
and their MENTIONED_IN, REL and EXACT_TITLE edges) is deleted and rebuilt; Document and Chunk nodes are
never touched. Retrieval reads the graph back from Neo4j, so Neo4j is the source of truth.
"""

from neo4j import Driver

from graphrag.graph.build import DATE_PROPERTY, Entity, ExactTitle, GraphData, Rel

BATCH = 2000


def _batches(rows: list[dict]):
    for i in range(0, len(rows), BATCH):
        yield rows[i:i + BATCH]


def load_graph(driver: Driver, g: GraphData) -> dict[str, int]:
    driver.execute_query("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
    while driver.execute_query(
        "MATCH (e:Entity) WITH e LIMIT 5000 DETACH DELETE e RETURN count(e) AS n"
    ).records[0]["n"]:
        pass

    entity_rows = []
    for e in g.entities.values():
        row = {"id": e.id, "name": e.name, "aliases": e.aliases, "type": e.type, "is_page": e.is_page,
               "page_chunk_id": e.page_chunk_id, "mention_ids": e.mention_ids}
        for prop in DATE_PROPERTY.values():
            values = e.dates.get(prop, [])
            row[prop] = [v for v, _ in values]
            row[prop + "_sources"] = [c for _, c in values]
        entity_rows.append(row)
    for batch in _batches(entity_rows):
        driver.execute_query("UNWIND $rows AS r CREATE (e:Entity) SET e = r", rows=batch)

    mention_rows = [{"entity": e.id, "chunk": mid.split("::")[0], "mention_id": mid}
                    for e in g.entities.values() for mid in e.mention_ids]
    for batch in _batches(mention_rows):
        driver.execute_query(
            "UNWIND $rows AS r MATCH (e:Entity {id: r.entity}) MATCH (c:Chunk {id: r.chunk}) "
            "CREATE (e)-[:MENTIONED_IN {mention_id: r.mention_id}]->(c)", rows=batch)

    rel_rows = [{"s": r.source, "t": r.target, "type": r.type, "other_label": r.other_label,
                 "source_chunk_ids": list(r.source_chunk_ids), "n_sources": r.n_sources} for r in g.rels]
    for batch in _batches(rel_rows):
        driver.execute_query(
            "UNWIND $rows AS r MATCH (a:Entity {id: r.s}) MATCH (b:Entity {id: r.t}) "
            "CREATE (a)-[:REL {type: r.type, other_label: r.other_label, source_chunk_ids: r.source_chunk_ids, "
            "n_sources: r.n_sources}]->(b)", rows=batch)

    exact_rows = [{"s": x.source, "t": x.target, "mention_ids": list(x.mention_ids)} for x in g.exact_titles]
    for batch in _batches(exact_rows):
        driver.execute_query(
            "UNWIND $rows AS r MATCH (a:Entity {id: r.s}) MATCH (b:Entity {id: r.t}) "
            "CREATE (a)-[:EXACT_TITLE {mention_ids: r.mention_ids}]->(b)", rows=batch)

    counts = driver.execute_query(
        "MATCH (e:Entity) WITH count(e) AS entities "
        "OPTIONAL MATCH ()-[r:REL]->() WITH entities, count(r) AS rels "
        "OPTIONAL MATCH ()-[x:EXACT_TITLE]->() WITH entities, rels, count(x) AS exact "
        "OPTIONAL MATCH ()-[m:MENTIONED_IN]->() RETURN entities, rels, exact, count(m) AS mentioned_in").records[0]
    return dict(counts)


def fetch_graph(driver: Driver) -> GraphData:
    entities = {}
    for r in driver.execute_query("MATCH (e:Entity) RETURN e ORDER BY e.id").records:
        e = dict(r["e"])
        dates = {}
        for prop in DATE_PROPERTY.values():
            vals, srcs = e.get(prop) or [], e.get(prop + "_sources") or []
            if vals:
                dates[prop] = list(zip(vals, srcs))
        entities[e["id"]] = Entity(id=e["id"], name=e["name"], aliases=list(e["aliases"]), type=e["type"],
                                   is_page=e["is_page"], page_chunk_id=e.get("page_chunk_id"),
                                   mention_ids=list(e["mention_ids"]), dates=dates)
    rels = [Rel(source=r["s"], type=r["type"], other_label=r["other_label"], target=r["t"],
                source_chunk_ids=tuple(r["src"]))
            for r in driver.execute_query(
                "MATCH (a:Entity)-[x:REL]->(b:Entity) RETURN a.id AS s, x.type AS type, x.other_label AS other_label, "
                "b.id AS t, x.source_chunk_ids AS src ORDER BY s, type, coalesce(other_label, ''), t").records]
    exact = [ExactTitle(source=r["s"], target=r["t"], mention_ids=tuple(r["m"]))
             for r in driver.execute_query(
                 "MATCH (a:Entity)-[x:EXACT_TITLE]->(b:Entity) RETURN a.id AS s, b.id AS t, x.mention_ids AS m "
                 "ORDER BY s, t").records]
    return GraphData(entities=entities, rels=rels, exact_titles=exact, stats={})
