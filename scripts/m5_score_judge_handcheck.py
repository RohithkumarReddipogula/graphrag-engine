"""Score the M5 judge hand check (docs/PLAN.md, M5.6) against results/m5/judge_handcheck_key.json.
Writes results/m5/judge_handcheck_scores.json. Not a pass/fail bar: below 27 of 30 the judge metric is
reported with a warning, and EM stays the primary metric either way."""

import json
import re
from collections import defaultdict

from graphrag.config import get_settings

WARN_BELOW = 27
LABEL_NOTE = ("First pass drafted by Claude (a different model from the llama-3.3-70b judge and the "
              "gpt-oss-120b generator); reviewed by Rohith Kumar Reddipogula.")


def main() -> None:
    out = get_settings().results_dir / "m5"
    key = json.loads((out / "judge_handcheck_key.json").read_text())
    grades = [g.strip() for g in re.findall(r"(?m)^- Grade: *(.*)$", (out / "judge_handcheck.md").read_text(encoding="utf-8"))]
    assert len(grades) == len(key) and all(grades), "every item needs a grade"
    hand = []
    for i, g in enumerate(grades, 1):
        word = g.split()[0].lower().strip(".,;:")
        if word not in ("correct", "incorrect"):
            raise ValueError(f"item {i}: grade {g!r} does not start with correct or incorrect")
        hand.append(word)

    per = defaultdict(lambda: {"items": 0, "agree": 0})
    disagreements = []
    for k, h, g in zip(key, hand, grades):
        agree = (k["judge_verdict"] == h)
        per[k["stratum"]]["items"] += 1
        per[k["stratum"]]["agree"] += agree
        if not agree:
            disagreements.append({"n": k["n"], "stratum": k["stratum"], "judge": k["judge_verdict"],
                                  "judge_reason": k["judge_reason"], "hand": g})
    agree_total = sum(v["agree"] for v in per.values())
    scores = {
        "labelling": LABEL_NOTE,
        "items": len(key), "agreement": agree_total, "agreement_rate": round(agree_total / len(key), 4),
        "warn_below": WARN_BELOW, "warning": agree_total < WARN_BELOW,
        "by_stratum": dict(sorted(per.items())),
        "flagged_by_labeller": [k["n"] for k, g in zip(key, grades) if "FLAG" in g],
        "disagreements": disagreements,
    }
    (out / "judge_handcheck_scores.json").write_text(json.dumps(scores, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(scores, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
