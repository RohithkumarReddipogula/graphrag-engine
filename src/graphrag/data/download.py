"""Pinned download of the 2WikiMultiHopQA validation split (the public test split has no answers)."""

import hashlib
from pathlib import Path

import httpx

REPO = "framolfese/2WikiMultihopQA"
REVISION = "fe713bfbd1afbca1a65246741a75890405d56a3a"
FILE = "data/validation-00000-of-00001.parquet"
SHA256 = "408e2dbb28edc6c8b9ca3ba0c94d4fc7bf17ffb923766593a3a7f546ab4cba59"
URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FILE}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_validation(raw_dir: Path) -> Path:
    """Download once into data/raw/ (gitignored) and verify the checksum every time."""
    path = raw_dir / "2wiki_validation.parquet"
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        with httpx.stream("GET", URL, follow_redirects=True, timeout=300) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        tmp.replace(path)
    actual = sha256_of(path)
    if actual != SHA256:
        raise RuntimeError(f"checksum mismatch for {path}: {actual} != {SHA256}")
    return path
