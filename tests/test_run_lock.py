import json

import pytest

from graphrag.eval.run_lock import LockRefused, acquire, finish


def test_first_run_creates_the_lock(tmp_path):
    lock = tmp_path / "m5" / "TEST_RUN.lock"
    rec = acquire(lock, "abc123", "h1")
    assert rec["status"] == "running" and json.loads(lock.read_text())["commit"] == "abc123"


def test_finished_run_can_never_run_again(tmp_path):
    lock = tmp_path / "TEST_RUN.lock"
    acquire(lock, "abc123", "h1")
    finish(lock, {"answers": 1875})
    with pytest.raises(LockRefused, match="only once"):
        acquire(lock, "abc123", "h1")


def test_interrupted_run_resumes_only_with_same_commit_and_settings(tmp_path):
    lock = tmp_path / "TEST_RUN.lock"
    acquire(lock, "abc123", "h1")
    assert acquire(lock, "abc123", "h1")["resumed_at"]
    with pytest.raises(LockRefused, match="different commit or settings"):
        acquire(lock, "other", "h1")
    with pytest.raises(LockRefused, match="different commit or settings"):
        acquire(lock, "abc123", "h2")
