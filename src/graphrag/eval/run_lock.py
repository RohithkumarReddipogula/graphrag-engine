"""Run-once lock for the test split (docs/PLAN.md, M5.8).

The lock file is created atomically (it must not exist yet). It records the start time, the git commit
and a hash of every frozen setting. A run whose lock says "finished" can never run again. A run whose
lock says "running" (interrupted) may resume only with the same commit and settings hash; every LLM call
is cached, so resuming cannot change any answer.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class LockRefused(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def acquire(path: Path, commit: str, settings_hash: str) -> dict:
    """Create the lock, or resume an interrupted run with identical commit and settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"status": "running", "started_at": _now(), "commit": commit, "settings_hash": settings_hash}
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        existing = json.loads(path.read_text())
        if existing.get("status") == "finished":
            raise LockRefused(f"the test split was already run ({existing.get('finished_at')}); it runs only once")
        if existing.get("commit") != commit or existing.get("settings_hash") != settings_hash:
            raise LockRefused("an interrupted test run exists with a different commit or settings; refusing to resume")
        existing.setdefault("resumed_at", []).append(_now())
        path.write_text(json.dumps(existing, indent=1) + "\n")
        return existing
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(record, indent=1) + "\n")
    return record


def finish(path: Path, extra: dict) -> dict:
    record = json.loads(path.read_text())
    if record.get("status") != "running":
        raise LockRefused("lock is not in the running state")
    record.update(status="finished", finished_at=_now(), **extra)
    path.write_text(json.dumps(record, indent=1) + "\n")
    return record
