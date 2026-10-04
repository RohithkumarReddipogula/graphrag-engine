"""Render tables from committed files and insert them into README.md between marker comments
(<!-- BEGIN name --> / <!-- END name -->). Numbers come from data/stats.json,
results/m1/retrieval_dev.json and results/m1/generation_dev.json; nothing is typed by hand."""

import json
import re

from graphrag.config import ROOT

SYSTEM_NAMES = {
    "closed_book": "Closed-book (no retrieval)",
    "hybrid": "Hybrid baseline (chosen: no title, alpha 0.9)",
    "hybrid_thesis": "Hybrid, thesis setting (title, alpha 0.7)",
}
TYPES = ["comparison", "inference", "compositional", "bridge_comparison"]


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def ci(m):
    return f"{m['mean']:.2f} [{m['ci_low']:.2f}, {m['ci_high']:.2f}]"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def retrieval_table(r):
    chosen = next(c for c in r["configs"] if all(c[k] == r["chosen"][k] for k in ("with_title", "fusion", "alpha")))
    rows = []
    for label, c in [("Chosen (no title, alpha 0.9)", chosen), ("Thesis setting (title, alpha 0.7)", r["thesis_setting"])]:
        for stage, name in [("pre_rerank", "before rerank"), ("post_rerank", "after rerank")]:
            m, s = c[stage]["multi_hop"], c[stage]["single_hop"]
            rows.append([f"{label}, {name}", f2(m["recall@5"]), f2(m["recall@10"]), f2(m["all_gold@10"]), f2(s["recall@2"])])
    return table(["Retriever", "Multi-hop recall@5", "Multi-hop recall@10", "Multi-hop all gold in top 10", "Single-hop recall@2"], rows)


def qa_table(g, subset):
    rows = []
    for name, e in g["systems"].items():
        for group, label in [("multi_hop", "multi-hop"), ("single_hop", "single-hop"), ("all", "all")]:
            m = e[subset].get(group)
            if not m:
                continue
            rows.append([SYSTEM_NAMES[name], label, m["em"]["n"], ci(m["em"]), f2(m["f1"]["mean"]),
                         f2(m["unknown_rate"]), m["answered"]["n"], f2(m["answered"]["em"])])
    return table(["System", "Questions", "n", "EM [95% CI]", "F1", "Unknown rate", "Answered n", "EM on answered"], rows)


def type_table(g):
    rows = []
    for name, e in g["systems"].items():
        for t in TYPES:
            m = e["all_questions"][t]
            rows.append([SYSTEM_NAMES[name], t, m["em"]["n"], ci(m["em"]), f2(m["unknown_rate"]), f2(m["answered"]["em"])])
    return table(["System", "Type", "n", "EM [95% CI]", "Unknown rate", "EM on answered"], rows)


def spread_line(r) -> str:
    vals = [c["post_rerank"]["multi_hop"]["all_gold@10"] for c in r["configs"]]
    return (f"Across all {len(vals)} retrieval configurations tried on dev, multi-hop all-gold-in-top-10 after "
            f"rerank ranges from {min(vals):.2f} to {max(vals):.2f}. With {r['questions']['multi_hop']} questions, one "
            "question is 0.01, so these differences are within noise.")


def render_data() -> str:
    st = json.loads((ROOT / "data/stats.json").read_text())
    mh, sh, c = st["multi_hop"], st["single_hop"], st["corpus"]
    hc = sh["hand_check"]
    rows = [
        ["Source", f"2WikiMultihopQA validation split ({st['validation_questions']:,} questions), HF revision {st['source']['revision'][:7]}"],
        ["Multi-hop questions", f"dev {sum(mh['dev'].values())} ({', '.join(f'{k} {v}' for k, v in mh['dev'].items())}); test {sum(mh['test'].values())}"],
        ["Single-hop questions", f"dev {sh['dev']}, test {sh['test']} (generated from evidence triples)"],
        ["Corpus", f"{c['chunks']:,} paragraphs, {c['words_total']:,} words, median {c['words_median']:g} words"],
        ["Distractors", f"{st['params']['distractors_per_question']} per question, pooled into one corpus"],
        ["Same paragraph in two tokenisations", f"{c['merged_spacing_variants']} titles, merged"],
        ["Single-hop hand check", f"{hc['errors']} errors in {hc['checked']} checked questions ({hc['error_rate']:.0%}); rules tightened afterwards"],
    ]
    return "Source: `data/stats.json`.\n\n" + table(["Item", "Value"], rows)


def render() -> str:
    r = json.loads((ROOT / "results/m1/retrieval_dev.json").read_text())
    g = json.loads((ROOT / "results/m1/generation_dev.json").read_text())
    h = g["systems"]["hybrid"]
    d = h["paired_diff_vs_closed_book_em"]
    t = g["paired_diff_hybrid_vs_thesis_em"]
    parts = [
        "Source: `results/m1/retrieval_dev.json`, dev split, "
        f"{r['questions']['multi_hop']} multi-hop + {r['questions']['single_hop']} single-hop questions.",
        "",
        retrieval_table(r),
        "",
        spread_line(r),
        "",
        "Source: `results/m1/generation_dev.json`, dev split. Unknown = the system abstained (scores 0).",
        "",
        "Table: all dev questions.",
        "",
        qa_table(g, "all_questions"),
        "",
        f"Table: headline subset, questions the closed-book model gets wrong (n = {g['closed_book_wrong_subset']['n']}).",
        "",
        qa_table(g, "closed_book_wrong"),
        "",
        "Table: multi-hop questions by type.",
        "",
        type_table(g),
        "",
        f"Paired difference in EM, hybrid baseline minus closed-book, all {d['n']} dev questions: "
        f"{d['mean']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}]. "
        f"Hybrid baseline minus thesis setting: {t['mean']:+.2f} [{t['ci_low']:+.2f}, {t['ci_high']:+.2f}].",
        "",
        f"All gold paragraphs inside the packed context (multi-hop, hybrid baseline): "
        f"{h['multi_hop_all_gold_in_context']:.2f}. Mean context size: {h['mean_context_tokens']:.0f} tokens.",
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for name, fn in [("data-stats", render_data), ("m1-tables", render)]:
        text, n = re.subn(rf"(<!-- BEGIN {name} -->\n)(?:.*?\n)?(<!-- END {name} -->)",
                          lambda m: m.group(1) + fn() + "\n" + m.group(2), text, flags=re.S)
        assert n == 1, f"markers for {name} not found in README.md"
    readme.write_text(text, encoding="utf-8")
    print("tables rendered into README.md")
