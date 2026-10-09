"""M5.5: judge the test answers with meta-llama/llama-3.3-70b-instruct on parasail/fp8 (approved
2026-10-09). Reads the committed generation results of the single test run; never re-generates.

Writes results/m5/judge_items.jsonl (one row per unique question and normalised answer) and
results/m5/judge_summary.json (judge accuracy per system with 95% CIs; "unknown" and "unsure" count as
incorrect; agreement with EM; the primary comparison on the judge metric as a secondary result).
"""

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from graphrag.config import get_settings
from graphrag.eval.bootstrap import mean_ci, paired_diff_ci
from graphrag.eval.judge import ITEMS_PER_CALL, build_items, item_id, judge_batch
from graphrag.eval.report import GROUPS, in_group
from graphrag.llm.client import make_llm
from graphrag.llm.spend import ledger_totals, write_balance

SYSTEMS = ["closed_book", "hybrid", "graph_only", "graph_plus_chunks_g0.5", "graph_plus_chunks_g0.5_no_exact_title"]
WORKERS = 4


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    s = get_settings()
    out = s.results_dir / "m5"
    lock = json.loads((out / "TEST_RUN.lock").read_text())
    assert lock["status"] == "finished", "the judge runs only on a finished test run"
    rows = {n: read_jsonl(out / f"generation_test_{n}.jsonl") for n in SYSTEMS}
    accepted = json.loads((s.data_dir / "answer_aliases.json").read_text(encoding="utf-8"))
    items = build_items(rows, accepted)
    batches = [list(items.values())[i:i + ITEMS_PER_CALL] for i in range(0, len(items), ITEMS_PER_CALL)]

    llm = make_llm(s.judge_model, s)
    spent_before = ledger_totals(s)["cost_usd"]
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda b: judge_batch(llm, b), batches))
    verdicts, calls, failed = {}, [], 0
    for dec, call_results, ok in results:
        verdicts.update(dec)
        calls += call_results
        failed += not ok

    with (out / "judge_items.jsonl").open("w", encoding="utf-8") as f:
        for iid, it in items.items():
            v = verdicts[iid]
            f.write(json.dumps({**it, "verdict": v.verdict, "reason": v.reason}, ensure_ascii=False) + "\n")

    def judged(r):
        return 0.0 if r["unknown"] else float(verdicts[item_id(r["id"], r["prediction"])].verdict == "correct")

    per_system = {}
    for n, rs in rows.items():
        groups = {}
        for g in GROUPS:
            sub = [r for r in rs if in_group(r, g)]
            groups[g] = {"judge_accuracy": mean_ci([judged(r) for r in sub])}
        answered = [r for r in rs if not r["unknown"]]
        agree = Counter((int(r["em"]), int(judged(r))) for r in answered)
        per_system[n] = {"groups": groups,
                         "unsure": sum(verdicts[item_id(r["id"], r["prediction"])].verdict == "unsure" for r in answered),
                         "em_vs_judge_on_answered": {"em1_judge_correct": agree[(1, 1)], "em1_judge_incorrect": agree[(1, 0)],
                                                     "em0_judge_correct": agree[(0, 1)], "em0_judge_incorrect": agree[(0, 0)]}}
    hyb = {r["id"]: r for r in rows["hybrid"]}
    gpc = rows["graph_plus_chunks_g0.5"]
    providers = sorted({c.provider for c in calls})
    balance = write_balance(s)
    summary = {
        "judge": {"model": s.judge_model, "endpoint": s.judge_provider, "temperature": 0.0, "items_per_call": ITEMS_PER_CALL,
                  "providers_seen": providers},
        "items": {"test_answers": sum(len(r) for r in rows.values()),
                  "unknown_answers_not_judged": sum(r["unknown"] for rs in rows.values() for r in rs),
                  "unique_items_judged": len(items), "verdicts": dict(Counter(v.verdict for v in verdicts.values()))},
        "calls": {"calls": len(calls), "live_calls": sum(not c.cached for c in calls), "batches_failed_twice": failed,
                  "input_tokens": sum(c.input_tokens for c in calls), "output_tokens": sum(c.output_tokens for c in calls),
                  "cost_usd": round(ledger_totals(s)["cost_usd"] - spent_before, 6), "seconds": round(time.time() - t0)},
        "systems": per_system,
        "primary_comparison_on_judge_metric": {
            "comparison": "graph_plus_chunks_g0.5 vs hybrid", "note": "secondary metric; EM stays primary",
            "paired_difference": paired_diff_ci([judged(r) for r in gpc], [judged(hyb[r["id"]]) for r in gpc])},
        "remaining_credit_usd": balance["remaining_usd"],
    }
    (out / "judge_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("judge", "items", "calls", "primary_comparison_on_judge_metric")}, indent=1))
    print({n: round(v["groups"]["all"]["judge_accuracy"]["mean"], 3) for n, v in per_system.items()})


if __name__ == "__main__":
    main()
