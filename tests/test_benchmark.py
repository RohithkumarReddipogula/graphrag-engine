import importlib.util
import re

from graphrag.config import ROOT


def test_benchmark_is_ascii_numbered_and_matches_the_results_files():
    text = (ROOT / "BENCHMARK.md").read_text(encoding="utf-8")
    assert text.isascii() and "—" not in text
    assert re.search(r"(?m)^## Contents\n\n1\. ", text)
    spec = importlib.util.spec_from_file_location("render_benchmark", ROOT / "scripts" / "render_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert text == mod.render(), "BENCHMARK.md is stale; run scripts/render_benchmark.py"
