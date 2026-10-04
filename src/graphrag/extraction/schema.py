"""Extraction schema: 6 coarse entity types and the fixed list of the 34 relations that occur in the
2WikiMultihopQA validation split, plus OTHER (stored, never scored). See docs/PLAN.md, M2."""

from typing import Literal

from pydantic import BaseModel, Field

ENTITY_TYPES = ("PERSON", "FILM", "PLACE", "ORG", "WORK", "OTHER")

# relation -> (subject, object) description given to the extractor. Direction follows Wikidata.
RELATIONS: dict[str, str] = {
    "director": "film or work -> person who directed it",
    "producer": "film or work -> person who produced it",
    "composer": "film or work -> person who composed its music",
    "performer": "song, album or work -> person or group who performs it",
    "creator": "work -> person who created it",
    "editor": "film or publication -> person who edited it",
    "presenter": "show -> person who presents it",
    "publisher": "work -> organisation that published it",
    "manufacturer": "product -> organisation that made it",
    "has part": "whole -> one of its parts",
    "country of origin": "film or work -> country it comes from",
    "publication date": "film or work -> date it was released or published",
    "date of birth": "person -> date of birth",
    "date of death": "person -> date of death",
    "place of birth": "person -> place of birth",
    "place of death": "person -> place of death",
    "place of burial": "person -> place of burial",
    "place of detention": "person -> place where they were held",
    "cause of death": "person -> cause of death",
    "country of citizenship": "person -> country of citizenship",
    "father": "person -> their father",
    "mother": "person -> their mother",
    "spouse": "person -> their spouse",
    "child": "person -> their child",
    "sibling": "person -> their sibling",
    "educated at": "person -> school or university they attended",
    "student of": "person -> their teacher",
    "doctoral advisor": "person -> their doctoral advisor",
    "employer": "person -> organisation they worked for",
    "occupation": "person -> their occupation",
    "award received": "person or work -> award it received",
    "founded by": "organisation -> its founder",
    "inception": "organisation or place -> date it was founded or created",
    "country": "place or organisation -> country it is in",
}
assert len(RELATIONS) == 34

# Relations whose object is a date literal; the graph stores them as node properties, not nodes.
DATE_RELATIONS = frozenset({"date of birth", "date of death", "publication date", "inception"})

RelationName = Literal[tuple(RELATIONS) + ("OTHER",)]  # type: ignore[valid-type]
EntityType = Literal[ENTITY_TYPES]  # type: ignore[valid-type]


class Entity(BaseModel):
    name: str = Field(min_length=1)
    type: EntityType
    description: str = Field(description="at most 12 words, from the paragraph")


class Relation(BaseModel):
    subject: str = Field(min_length=1)
    relation: RelationName
    object: str = Field(min_length=1)
    other_label: str | None = Field(default=None, description="short label, only when relation is OTHER")


class ParagraphExtraction(BaseModel):
    chunk_id: str
    entities: list[Entity]
    relations: list[Relation]


class BatchExtraction(BaseModel):
    paragraphs: list[ParagraphExtraction]


def strict_json_schema() -> dict:
    """BatchExtraction as a JSON schema that strict structured-output mode accepts: $refs inlined, every
    property listed as required (optional ones are nullable), and no additional properties."""
    raw = BatchExtraction.model_json_schema()
    defs = raw.pop("$defs", {})

    def fix(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return fix(dict(defs[node["$ref"].split("/")[-1]]))
            # Drop schema annotations ("title": "Entity", "default": null), not fields that happen to share the name.
            node = {k: fix(v) for k, v in node.items()
                    if not (k == "title" and isinstance(v, str)) and k != "default"}
            if node.get("type") == "object":
                node["required"] = list(node.get("properties", {}))
                node["additionalProperties"] = False
            if "anyOf" in node:   # Optional[str] -> {"type": ["string", "null"]}
                types = [a.get("type") for a in node["anyOf"]]
                if all(types):
                    rest = {k: v for k, v in node.items() if k != "anyOf"}
                    return {**rest, "type": types}
            return node
        if isinstance(node, list):
            return [fix(x) for x in node]
        return node

    return fix(raw)
