"""E5 encoder wrapper. E5 needs task prefixes: "query: " for questions, "passage: " for indexed text."""

from typing import Protocol, Sequence

import numpy as np

QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class SentenceEncoder(Protocol):
    def encode(self, sentences: list[str], **kwargs) -> np.ndarray: ...


class E5Encoder:
    def __init__(self, model_name: str, model: SentenceEncoder | None = None, batch_size: int = 32):
        if model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self._model = model
        self._batch_size = batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        # L2-normalised, so inner product equals cosine similarity.
        return np.asarray(
            self._model.encode(texts, batch_size=self._batch_size, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def encode_queries(self, questions: Sequence[str]) -> np.ndarray:
        return self._encode([QUERY_PREFIX + q for q in questions])

    def encode_passages(self, passages: Sequence[str]) -> np.ndarray:
        return self._encode([PASSAGE_PREFIX + p for p in passages])
