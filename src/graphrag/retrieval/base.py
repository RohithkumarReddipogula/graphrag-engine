"""Retriever interface shared by the baseline and (later) the graph retriever."""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Passage:
    chunk_id: str
    title: str
    text: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)   # per-stage scores, for analysis


class Retriever(Protocol):
    def retrieve(self, question: str, k: int) -> list[Passage]: ...


def passage_text(title: str, text: str, with_title: bool) -> str:
    """The text that gets indexed and reranked. Whether the title is included is a config choice."""
    return f"{title}. {text}" if with_title else text
