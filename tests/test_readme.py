import importlib.util
import re

from graphrag.config import ROOT

README = (ROOT / "README.md").read_text(encoding="utf-8")


def _renderer():
    spec = importlib.util.spec_from_file_location("render_tables", ROOT / "scripts" / "render_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_readme_follows_writing_rules():
    assert README.isascii(), "README must be plain ASCII"
    assert "<details" not in README.lower()
    assert re.search(r"(?m)^## Contents\n\n1\. ", README), "numbered table of contents"


def test_readme_tables_match_committed_results():
    r = _renderer()
    for name, fn in [("data-stats", r.render_data), ("m1-tables", r.render)]:
        block = re.search(rf"<!-- BEGIN {name} -->\n(.*?)\n<!-- END {name} -->", README, re.S).group(1)
        assert block == fn(), f"README block {name} is stale; run scripts/render_tables.py"
