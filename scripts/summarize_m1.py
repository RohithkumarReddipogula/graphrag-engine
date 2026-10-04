"""Rebuild results/m1/generation_dev.json from the committed per-question files (no LLM calls).
The spend section of the existing summary is kept as it was recorded by the generation run."""

import json

from graphrag.config import get_settings
from graphrag.eval.bootstrap import paired_diff_ci
from graphrag.eval.report import build_summary

SYSTEMS = ["closed_book", "hybrid", "hybrid_thesis"]

if __name__ == "__main__":
    import importlib.util

    s = get_settings()
    out = s.results_dir / "m1" / "generation_dev.json"
    old = json.loads(out.read_text())
    spec = importlib.util.spec_from_file_location("gen", s.results_dir.parent / "scripts" / "run_m1_generation.py")
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    per_system = {
        n: [json.loads(line) for line in (s.results_dir / "m1" / f"generation_dev_{n}.jsonl").read_text().splitlines()]
        for n in SYSTEMS
    }
    summary = build_summary(per_system, gen.summary_header(s))
    h, t = per_system["hybrid"], {r["id"]: r for r in per_system["hybrid_thesis"]}
    summary["paired_diff_hybrid_vs_thesis_em"] = paired_diff_ci([r["em"] for r in h], [t[r["id"]]["em"] for r in h])
    summary["spend"] = old["spend"]
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print("rewrote", out)
