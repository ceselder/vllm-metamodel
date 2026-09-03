"""Fill the README's DeepSeek-V4 (multi-stream) test-matrix block from one or more
bench/results/dsv4_* directories (merged: all *.json with a result are summarised together).

    python bench/render_dsv4_readme.py bench/results/dsv4_run1_eager_<ts> bench/results/dsv4_run2_graphs_<ts> [bench/results/dsv4_run3_graphs_tp_<ts>]

Rewrites everything between ``<!-- INJECTION-DSV4:BEGIN -->`` and ``<!-- INJECTION-DSV4:END -->``
in README.md, writes the merged summary to bench/results/dsv4_final/ and renders
bench/dsv4_throughput.{png,pdf,_data.json}.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from test_injection_modes import markdown_table, summarize  # noqa: E402


def merge(dirs: list[str], out: Path) -> dict:
    """Copy every per-engine JSON into ``out``. A later ``*_mixed.json`` for the same engine
    supersedes the ``mixed`` / ``effect_check`` cases (and their checks) of an earlier full
    run of that engine (the first eager run's mixed check lacked the clean-vs-clean noise
    control); source dirs are left untouched."""
    out.mkdir(parents=True, exist_ok=True)
    recs: list[tuple[str, dict]] = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.json")):
            if f.name == "summary.json":
                continue
            rec = json.loads(f.read_text())
            if "result" in rec:
                rec["result"]["source"] = f"{Path(d).name}/{f.name}"
                if "throughput" in f.stem:
                    rec["result"]["throughput_only"] = True
            recs.append((f"{Path(d).name}__{f.name}", rec))
    superseded = {rec["result"]["engine"] for name, rec in recs if "result" in rec and name.endswith("_mixed.json")}
    for name, rec in recs:
        r = rec.get("result")
        if r and r["engine"] in superseded and not name.endswith("_mixed.json") and not r.get("throughput_only"):
            for case in ("mixed", "effect_check"):
                if r["cases"].pop(case, None) is not None:
                    r["checks"] = [c for c in r["checks"] if c["case"] != case]
                    r["superseded_cases"] = r.get("superseded_cases", []) + [case]
        (out / name).write_text(json.dumps(rec, indent=1))
    s = summarize(out)
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    (out / "summary.md").write_text(markdown_table(s))
    return s


def main(dirs: list[str]) -> None:
    out = HERE / "results" / "dsv4_final"
    if out.exists():
        shutil.rmtree(out)
    s = merge(dirs, out)
    runs = s["runs"]
    gpu = next((r["gpu"] for r in runs if r.get("gpu")), "GPU").replace("NVIDIA ", "")
    ver = next((r["versions"] for r in runs if r.get("versions")), {})
    rc = next((r["resolved_config"] for r in runs if r.get("resolved_config")), {})
    has_tp = any(r["case"].startswith("throughput_dsv4/") for r in s["rows"])
    fig = ""
    if has_tp:
        subprocess.run([sys.executable, str(HERE / "plot_injection.py"), str(out), "--out-dir", str(HERE), "--stem", "dsv4_throughput",
                        "--title", ("Embedding-stream injection on DeepSeek-V4-Flash (hyper-connection architecture) costs no decode time: "
                                    "decode-step time of embedding replacement and embedding add vs no steering, one distinct vector per request\\n"
                                    f"vLLM {ver.get('vllm')} · TP{rc.get('tensor_parallel_size')} · {gpu} · fp8 + fp4 experts · kv fp8_ds_mla · "
                                    f"decode-step time = (wall at 80 new tokens − wall at 40) / 40, error bars = min–max over interleaved repeats · "
                                    f"vllm-metamodel {ver.get('vllm_lens')}")], check=True)
        fig = "\n![DeepSeek-V4 throughput by injection mode](bench/dsv4_throughput.png)\n"
    block = f"""<!-- INJECTION-DSV4:BEGIN -->
`bench/test_injection_dsv4.py` on {gpu} (TP{rc.get('tensor_parallel_size')}, vLLM {ver.get('vllm')}, torch {ver.get('torch')}, vllm-lens {ver.get('vllm_lens')};
`kv_cache_dtype={rc.get('kv_cache_dtype')}`, `kernel_config.moe_backend={rc.get('moe_backend')}`, `max_num_batched_tokens={rc.get('max_num_batched_tokens')}` so prefill is chunked;
`hc_mult={rc.get('hc_mult')}`, `expert_dtype={rc.get('expert_dtype')}`): **{s['n_pass']}/{s['n_gated']} gated checks pass**
({'all' if s['all_pass'] else 'NOT all'}{f"; {s['n_info']} informational" if s.get('n_info') else ''}). Layer outputs on this architecture are a
deferred 4-stream fold, so the fork refuses layer-output steering / capture with a `ValueError` (engine alive) and everything
goes through the embedding stream (`EMBED_LAYER_INDEX`). The reference is the NLA session's own arithmetic
(`nla.utils.dsv4.scale_vector_to_alpha`, `alpha·v/‖v‖`) and its worker-side pre-hook (`nla.utils.dsv4_fast_hooks`),
run on the same engine; "bf16 rel err" in the table is the max relative error of the written marker row against the target.

{markdown_table(s)}
{fig}
Re-run: `MODAL_PROFILE=safety-sahan modal run bench/modal_bench_dsv4.py::run1` (eager correctness), `::run2` (CUDA graphs:
correctness + throughput), `::run3` (throughput repeat, interleaved, 3 repeats); then
`python bench/render_dsv4_readme.py bench/results/dsv4_run1_eager_<ts> bench/results/dsv4_run2_graphs_<ts> bench/results/dsv4_run3_graphs_tp_<ts>`.
<!-- INJECTION-DSV4:END -->"""
    readme = REPO / "README.md"
    txt = readme.read_text()
    i, j = txt.index("<!-- INJECTION-DSV4:BEGIN -->"), txt.index("<!-- INJECTION-DSV4:END -->") + len("<!-- INJECTION-DSV4:END -->")
    readme.write_text(txt[:i] + block + txt[j:])
    print(f"README DSv4 block updated from {dirs} -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
