"""QA result summaries. Every table reports EM, F1, the unknown (abstention) rate and accuracy on the
questions a system actually answered, next to each other."""

from graphrag.eval.bootstrap import mean_ci, paired_diff_ci

GROUPS = ["comparison", "inference", "compositional", "bridge_comparison", "multi_hop", "single_hop", "all"]


def in_group(row: dict, group: str) -> bool:
    if group == "all":
        return True
    if group == "multi_hop":
        return row["type"] != "single_hop"
    return row["type"] == group


def group_metrics(rows: list[dict]) -> dict:
    answered = [r for r in rows if not r["unknown"]]
    return {
        "em": mean_ci([r["em"] for r in rows]),
        "f1": mean_ci([r["f1"] for r in rows]),
        "unknown_rate": round(sum(r["unknown"] for r in rows) / len(rows), 4),
        "answered": {
            "n": len(answered),
            "em": round(sum(r["em"] for r in answered) / len(answered), 4) if answered else None,
            "f1": round(sum(r["f1"] for r in answered) / len(answered), 4) if answered else None,
        },
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    for g in GROUPS:
        sub = [r for r in rows if in_group(r, g)]
        if sub:
            out[g] = group_metrics(sub)
    return out


def build_summary(per_system: dict[str, list[dict]], header: dict, reference: str = "closed_book") -> dict:
    """header: split, generator, budget, scoring notes. The reference system defines the
    "reference-wrong" headline subset (questions it gets wrong) and the paired differences."""
    ref = {r["id"]: r for r in per_system[reference]}
    ref_wrong = {i for i, r in ref.items() if r["em"] == 0.0}
    summary = {
        **header,
        "ci": "95% percentile bootstrap, 10,000 resamples, seed 0",
        "abstention": "every system may answer 'unknown'; it scores 0. unknown_rate and accuracy on answered questions are reported next to EM",
        "systems": {},
        "closed_book_wrong_subset": {"definition": f"questions with {reference} EM = 0", "n": len(ref_wrong)},
    }
    for name, rows in per_system.items():
        entry = {
            "all_questions": summarize(rows),
            "closed_book_wrong": summarize([r for r in rows if r["id"] in ref_wrong]),
            "providers": sorted({r["provider"] for r in rows}),
        }
        if name != reference:
            ctx_rows = [r for r in rows if r["type"] != "single_hop"]
            entry["multi_hop_all_gold_in_context"] = round(sum(r["all_gold_in_context"] for r in ctx_rows) / len(ctx_rows), 4)
            entry["mean_context_tokens"] = round(sum(r["context_tokens"] for r in rows) / len(rows), 1)
            for metric in ("em", "f1"):
                entry[f"paired_diff_vs_{reference}_{metric}"] = paired_diff_ci(
                    [r[metric] for r in rows], [ref[r["id"]][metric] for r in rows]
                )
        summary["systems"][name] = entry
    return summary
