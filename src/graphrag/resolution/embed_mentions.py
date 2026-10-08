"""E5 embeddings of "name: description" for every mention, cached on disk by content hash.
E5's "query: " prefix is used on both sides, as E5 recommends for symmetric similarity."""

import hashlib
from pathlib import Path

import numpy as np

from graphrag.embed import E5Encoder
from graphrag.resolution.candidates import mention_text
from graphrag.resolution.mentions import Mention


def embed_mentions(mentions: list[Mention], model_name: str, cache_dir: Path) -> np.ndarray:
    texts = [mention_text(m) for m in mentions]
    key = hashlib.sha1((model_name + "\n" + "\n".join(texts)).encode("utf-8")).hexdigest()[:16]
    path = cache_dir / f"m3_mention_emb_{key}.npy"
    if path.exists():
        return np.load(path)
    emb = E5Encoder(model_name).encode_queries(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
