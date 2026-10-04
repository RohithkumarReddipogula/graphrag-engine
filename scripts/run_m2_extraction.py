"""M2: extract entities and relations for the whole corpus (cached, resumable).

Usage: run_m2_extraction.py [--limit N]   (N = number of batches, for a pilot)
Writes results/m2/extractions.jsonl (one line per chunk), results/m2/extraction_failures.jsonl and
results/m2/extraction_run.json. A quota error stops the run; re-running resumes from the cache.
"""

import argparse
import json
import time

from graphrag.config import get_settings
from graphrag.extraction.extract import QuotaExhausted, RunStats, extract_batch, make_batches
from graphrag.llm.client import make_llm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    s = get_settings()
    chunks = [json.loads(line) for line in (s.data_dir / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
    batches = make_batches(chunks)
    todo = batches[: args.limit] if args.limit else batches
    llm = make_llm(s.extractor_model, s)
    stats = RunStats()
    rows, stopped = [], None
    t0 = time.time()
    for i, batch in enumerate(todo):
        stats.batches += 1
        try:
            parsed = extract_batch(llm, batch, stats)
        except QuotaExhausted as exc:
            stopped = f"quota exhausted at batch {i}: {exc}"
            stats.batches -= 1
            break
        if parsed:
            for p in parsed.paragraphs:
                rows.append({"chunk_id": p.chunk_id, "batch": i, **p.model_dump(exclude={"chunk_id"})})
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(todo)} batches, {stats.live_calls} live calls, {len(stats.failed)} failed", flush=True)

    out = s.results_dir / "m2"
    out.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: r["chunk_id"])
    with (out / "extractions.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out / "extraction_failures.jsonl").open("w", encoding="utf-8") as f:
        for r in stats.failed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    run = {
        "model": s.extractor_model,
        "batch_size": len(batches[0]),
        "batches_total": len(batches),
        "batches_attempted": stats.batches,
        "batches_ok": stats.ok,
        "batches_ok_after_retry": stats.retried,
        "batches_failed": len(stats.failed),
        "chunks_extracted": len(rows),
        "chunks_total": len(chunks),
        "live_calls_this_run": stats.live_calls,
        "complete": stats.batches == len(batches) and not stopped,
        "stopped": stopped,
        "seconds": round(time.time() - t0),
    }
    (out / "extraction_run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(json.dumps(run, indent=1))


if __name__ == "__main__":
    main()
