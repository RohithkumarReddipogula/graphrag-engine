"""Disk cache for LLM calls, keyed by (model, request). Re-running anything never pays twice for the same call."""

import hashlib
import json
from pathlib import Path
from typing import Any


def cache_key(model: str, request: dict[str, Any]) -> str:
    payload = json.dumps({"model": model, "request": request}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, model: str, key: str) -> Path:
        # Model names can contain "/" (for example "openai/gpt-oss-20b").
        safe_model = model.replace("/", "__")
        return self.root / safe_model / key[:2] / f"{key}.json"

    def get(self, model: str, key: str) -> dict[str, Any] | None:
        path = self._path(model, key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, model: str, key: str, value: dict[str, Any]) -> None:
        path = self._path(model, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first so an interrupted run never leaves a half-written entry.
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
