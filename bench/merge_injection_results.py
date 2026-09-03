"""Merge a correctness run and a (later) throughput-only run of bench/test_injection_modes.py.

    python bench/merge_injection_results.py <out_dir> <correctness_dir> <throughput_dir> [<throughput_dir> ...]

The correctness run's per-engine JSONs are copied with their (superseded) ``throughput``
case removed; each throughput run's JSONs are added as ``<model>__<engine>_throughput.json``
(a later directory overrides an earlier one for the same model/engine).  Then
``summary.json`` / ``summary.md`` are rebuilt for ``<out_dir>``.  Sources are left untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_injection_modes import markdown_table, summarize  # noqa: E402


def main(out: str, corr: str, *tps: str) -> None:
    c, o = Path(corr), Path(out)
    o.mkdir(parents=True, exist_ok=True)
    for f in c.glob("*.json"):
        if f.name == "summary.json":
            continue
        rec = json.loads(f.read_text())
        if "result" in rec and isinstance(rec["result"], dict):
            rec["result"].get("cases", {}).pop("throughput", None)
            rec["result"]["checks"] = [k for k in rec["result"].get("checks", []) if k["case"] != "throughput"]
            rec["result"]["source"] = f"{c.name}/{f.name}"
        (o / f.name).write_text(json.dumps(rec, indent=1))
    for tp in tps:
        t = Path(tp)
        for f in t.glob("*.json"):
            if f.name == "summary.json" or "hf_ref" in f.name:
                continue
            rec = json.loads(f.read_text())
            if "result" in rec and isinstance(rec["result"], dict):
                rec["result"]["source"] = f"{t.name}/{f.name}"
                rec["result"]["throughput_only"] = True
            (o / f"{f.stem}_throughput.json").write_text(json.dumps(rec, indent=1))  # later dirs override
    s = summarize(o)
    (o / "summary.json").write_text(json.dumps(s, indent=1))
    (o / "summary.md").write_text(markdown_table(s))
    print(f"{s['n_pass']}/{s['n_gated']} gated checks pass, {s['n_info']} not resolvable -> {o}")


if __name__ == "__main__":
    main(*sys.argv[1:])
