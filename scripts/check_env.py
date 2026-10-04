"""M0 exit check: Neo4j is reachable and each pinned LLM answers one cached call.

Prints only pass/fail and model version strings. Never prints keys or .env contents.
"""

import sys

from graphrag.config import get_settings
from graphrag.llm.client import make_llm
from graphrag.store.neo4j_client import connect


def main() -> int:
    settings = get_settings()
    ok = True

    try:
        with connect(settings) as driver:
            version = driver.execute_query("CALL dbms.components() YIELD versions RETURN versions[0] AS v").records[0]["v"]
        print(f"[ok]   neo4j {version}")
    except Exception as exc:  # report and keep checking the rest
        ok = False
        print(f"[fail] neo4j: {type(exc).__name__}: {exc}")

    for role, model in [
        ("extractor/judge", settings.extractor_model),
        ("generator", settings.generator_model),
    ]:
        try:
            result = make_llm(model, settings).complete("Reply with the single word: ready")
            source = "cache" if result.cached else "live"
            print(f"[ok]   {role}: {model} -> {result.model_version} ({source}) said {result.text.strip()!r}")
        except Exception as exc:
            ok = False
            print(f"[fail] {role}: {model}: {type(exc).__name__}: {exc}")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
