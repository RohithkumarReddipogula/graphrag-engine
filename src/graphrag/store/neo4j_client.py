"""Neo4j connection helper."""

from neo4j import Driver, GraphDatabase

from graphrag.config import Settings


def connect(settings: Settings) -> Driver:
    if settings.neo4j_password is None:
        raise RuntimeError("NEO4J_PASSWORD is not set in .env")
    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password.get_secret_value()),
    )
    driver.verify_connectivity()
    return driver
