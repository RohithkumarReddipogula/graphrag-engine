"""Render the README's number blocks from committed files and insert them between marker comments
(<!-- BEGIN name --> / <!-- END name -->). Nothing in these blocks is typed by hand. Sources:
results/m5/test_summary.json, results/m5/judge_summary.json, results/m3/run1_flawed/resolution_run.json,
results/m3/resolution_run.json, results/m3/handcheck_scores.json, results/m5/spend_snapshot.json and
data/stats.json."""

import json
import re

from graphrag.config import ROOT

HEADLINE_SYSTEMS = [("closed_book", "Closed-book (no retrieval)"), ("hybrid", "Hybrid retriever (baseline)"),
                    ("graph_plus_chunks_g0.5", "GraphRAG: graph + chunks (pre-registered system)"),
                    ("graph_only", "Graph only")]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def ci(m, signed=False):
    fmt = "{:+.2f}" if signed else "{:.2f}"
    return f"{fmt.format(m['mean'])} [{fmt.format(m['ci_low'])}, {fmt.format(m['ci_high'])}]"


def render_headline() -> str:
    t = load("results/m5/test_summary.json")
    p, cbw = t["primary_result"], t["secondary_results"]["closed_book_wrong_subset"]
    rows = []
    for key, label in HEADLINE_SYSTEMS:
        a = t["systems"][key]["all_questions"]
        rows.append([label, ci(a["all"]["em"]), ci(a["multi_hop"]["em"]), f"{a['all']['unknown_rate']:.2f}"])
    n = p["paired_em_difference"]["n"]
    return "\n".join([
        f"Source: `results/m5/test_summary.json`. Test split, {n} questions, run once. EM = exact match "
        "(official 2Wiki scoring with aliases), 95% bootstrap CI. Unknown = the system abstained (scores 0).",
        "",
        table(["System", f"EM, all {n} [95% CI]", f"EM, multi-hop {t['systems']['hybrid']['all_questions']['multi_hop']['em']['n']} [95% CI]",
               "Unknown rate"], rows),
        "",
        f"Primary result (one, fixed before the run): paired EM difference, GraphRAG minus hybrid, over all {n} "
        f"test questions: **{ci(p['paired_em_difference'], True)}**. Pre-registered expectation (95% CI above 0): "
        f"**{'met' if p['expectation_met'] else 'not met'}**.",
        "",
        f"Pre-specified secondary result, the {cbw['n']} questions the model gets wrong without retrieval: "
        f"{ci(cbw['paired_em_difference'], True)}.",
    ])


def render_m3_run1() -> str:
    r1 = load("results/m3/run1_flawed/resolution_run.json")
    r2 = load("results/m3/resolution_run.json")
    hc = load("results/m3/handcheck_scores.json")
    b1, b2 = r1["bridge_link_recall"]["report"], r2["bridge_link_recall"]["report"]
    big = r1["clusters"]["largest"][0]["names"]
    return "\n".join([
        "Sources: `results/m3/run1_flawed/resolution_run.json`, `results/m3/resolution_run.json`, "
        "`results/m3/handcheck_scores.json`.",
        "",
        table(["", "Run 1 (flawed)", "Run 2 (fixed)"], [
            ["Bridge links found (report split; exact name matching finds "
             f"{b1['before_exact_name']} of {b1['links']})", f"{b1['after_m3']} of {b1['links']}", f"{b2['after_m3']} of {b2['links']}"],
            ["Merges refused by cannot-link", "rule did not exist", f"{r2['merges_rejected'].get('cannot_link', 0):,}"],
            ["Largest cluster", " + ".join(big), " + ".join(r2["clusters"]["largest"][0]["names"])],
            ["Hand-checked merges correct", "not checked (sheets deleted)",
             f"{hc['merge_precision']['correct']} of {hc['merge_precision']['checked']}"],
        ]),
    ])


def render_cost() -> str:
    snap = load("results/m5/spend_snapshot.json")
    t, j = load("results/m5/test_summary.json"), load("results/m5/judge_summary.json")
    return "\n".join([
        f"Source: `results/m5/spend_snapshot.json`, taken from the spend ledger `{snap['source']}` (one row per "
        "paid call, with provider, tokens and cost).",
        "",
        table(["Item", "Cost (USD)"], [
            ["All OpenRouter calls, M0 to M5", f"{snap['cost_usd']:.2f} ({snap['calls']:,} calls: "
             + ", ".join(f"{k} {v:,}" for k, v in snap["by_provider"].items()) + ")"],
            ["of which the single test run", f"{t['spend']['this_run_usd']:.3f}"],
            ["of which the test-answer judge", f"{j['calls']['cost_usd']:.3f}"],
        ]),
    ])


def render_data() -> str:
    st = load("data/stats.json")
    c = st["corpus"]
    return (f"Source: `data/stats.json`. Corpus: {c['chunks']:,} Wikipedia paragraphs (2WikiMultihopQA, HF revision "
            f"{st['source']['revision'][:7]}). Dev: {sum(st['multi_hop']['dev'].values())} multi-hop + "
            f"{st['single_hop']['dev']} single-hop questions. Test: {sum(st['multi_hop']['test'].values())} multi-hop + "
            f"{st['single_hop']['test']} single-hop questions.")


BLOCKS = {"m5-headline": render_headline, "m3-run1": render_m3_run1, "cost": render_cost, "data-stats": render_data}


if __name__ == "__main__":
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for name, fn in BLOCKS.items():
        text, n = re.subn(rf"(<!-- BEGIN {name} -->\n)(?:.*?\n)?(<!-- END {name} -->)",
                          lambda m: m.group(1) + fn() + "\n" + m.group(2), text, flags=re.S)
        assert n == 1, f"markers for {name} not found in README.md"
    readme.write_text(text, encoding="utf-8")
    print("README blocks rendered:", ", ".join(BLOCKS))
