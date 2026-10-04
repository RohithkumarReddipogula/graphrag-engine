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


# Official 2Wiki release with Wikidata ids (answer_id, evidences_id) and id_aliases.json, used by the
# official evaluation script v1.1 to accept aliases and demonyms of the gold answer.
IDS_URL = "https://www.dropbox.com/s/7ep3h8unu2njfxv/data_ids.zip?dl=1"
IDS_SHA256 = "664013d34f169ed5223aedde0c8f013ac6be9568426966319240f722867c5b2d"


def fetch_ids(raw_dir: Path) -> Path:
    """Download data_ids.zip once (about 259 MB), verify it, and extract dev.json and id_aliases.json."""
    import zipfile

    zpath = raw_dir / "data_ids.zip"
    if not zpath.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        tmp = zpath.with_suffix(".part")
        with httpx.stream("GET", IDS_URL, follow_redirects=True, timeout=600) as resp:
            resp.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        tmp.replace(zpath)
    actual = sha256_of(zpath)
    if actual != IDS_SHA256:
        raise RuntimeError(f"checksum mismatch for {zpath}: {actual} != {IDS_SHA256}")
    out = raw_dir / "data_ids"
    if not (out / "id_aliases.json").exists() or not (out / "dev.json").exists():
        with zipfile.ZipFile(zpath) as z:
            z.extract("data_ids/dev.json", raw_dir)
            z.extract("data_ids/id_aliases.json", raw_dir)
    return out
