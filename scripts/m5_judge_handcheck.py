"""M5.6: the 30-item hand check of the judge (docs/PLAN.md, approved 2026-10-09).

Builds a blind sheet from results/m5/judge_items.jsonl: 10 seeded-random items where the judge says
correct but EM is 0, 10 where the judge says incorrect, 10 where judge and EM agree on correct (fewer if
a stratum is smaller; the rest filled from the others). Items are shuffled together. The sheet shows the
question, the gold answer with its aliases and the answer, but not the judge's verdict or the system;
those are in results/m5/judge_handcheck_key.json. The sheet is never overwritten.
"""

import json
import random

from graphrag.config import get_settings
from graphrag.eval.answers import score

SEED = 42
QUOTA = {"judge correct, EM 0": 10, "judge incorrect": 10, "judge correct, EM 1": 10}


def stratum(item: dict) -> str | None:
    em = score(item["prediction"], item["accepted"])["em"]
    if item["verdict"] == "correct":
        return "judge correct, EM 1" if em == 1.0 else "judge correct, EM 0"
    if item["verdict"] == "incorrect":
        return "judge incorrect"
    return None


def main() -> None:
    s = get_settings()
    out = s.results_dir / "m5"
    sheet, key_path = out / "judge_handcheck.md", out / "judge_handcheck_key.json"
    if sheet.exists():
        raise SystemExit(f"kept existing {sheet.name} (it may hold hand-check marks)")
    items = [json.loads(line) for line in (out / "judge_items.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(f"{SEED}:m5-judge-handcheck")
    pools = {k: sorted((it for it in items if stratum(it) == k), key=lambda it: it["item_id"]) for k in QUOTA}
    picked, short = [], 0
    for k, n in QUOTA.items():
        take = rng.sample(pools[k], min(n, len(pools[k])))
        short += n - len(take)
        picked += [(k, it) for it in take]
    if short:
        rest = [(stratum(it), it) for it in items if stratum(it) and all(it is not p for _, p in picked)]
        picked += rng.sample(sorted(rest, key=lambda x: x[1]["item_id"]), short)
    rng.shuffle(picked)

    lines = [
        "# M5 judge hand check (test answers)",
        "",
        f"{len(picked)} answers to grade against the gold answer, using the same rule the judge used:",
        '- "correct": the answer names the same entity or value as the gold answer (other wording, spellings,',
        "  aliases, abbreviations and date formats are fine);",
        '- "incorrect": a different entity or value, or less specific than the question asks for.',
        "",
        'Write "correct" or "incorrect" after each "Grade:", optionally followed by a short note. The judge\'s',
        "verdict and the system that gave the answer are not shown; they are in judge_handcheck_key.json.",
        "",
    ]
    for i, (_, it) in enumerate(picked, 1):
        lines += [f"## {i}. {it['question']}", "",
                  f"- Gold answer and accepted aliases: {'; '.join(it['accepted'])}",
                  f"- Answer to grade: {it['prediction']}",
                  "- Grade: ", ""]
    sheet.write_text("\n".join(lines), encoding="utf-8")
    key = [{"n": i, "item_id": it["item_id"], "stratum": k, "judge_verdict": it["verdict"], "judge_reason": it["reason"]}
           for i, (k, it) in enumerate(picked, 1)]
    key_path.write_text(json.dumps(key, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {sheet} ({len(picked)} items) and {key_path.name}")


if __name__ == "__main__":
    main()
