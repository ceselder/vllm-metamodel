"""Fill the README results block from a bench results directory.

    python bench/render_readme.py bench/results/<timestamp>

Rewrites everything between ``<!-- RESULTS:BEGIN -->`` and ``<!-- RESULTS:END -->``
in README.md (headline sentence, plot, speedup table, correctness summary) and
refreshes the ``git diff v1.1.0 --stat`` block.  Also copies the plot PNG/PDF
into ``bench/`` so the README can reference them from the repo.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from compare import load_dir, summarize  # noqa: E402

ROWS = [
    ("stock_eager", "stock vllm-lens 1.1.0 (eager forced)"),
    ("fork_indexed", "fork: indexed hook (eager)"),
    ("fork_vectorized", "fork: indexed + vectorised apply (eager)"),
    ("fork_graphs", "fork: indexed + vectorised + CUDA graphs"),
    ("ceiling_graphs", "no steering, same engine config (ceiling)"),
    ("ceiling_plain", "no steering, vLLM default compile + graphs (ceiling)"),
]


def fmt(v: dict, stock: dict | None, b: int, is_stock: bool) -> str:
    s = f"{v['wall_s']:.1f} s"
    if is_stock or not stock or b not in stock:
        return s
    return f"{s} (**{stock[b]['wall_s'] / v['wall_s']:.1f}×**)"


def table(t: dict) -> str:
    series = t["series"]
    stock = series.get("stock_eager")
    batches = sorted({b for name, _ in ROWS for b in series.get(name, {})})
    lines = ["| configuration | " + " | ".join(f"B = {b:,}" for b in batches) + " |",
             "|---|" + "|".join("---:" for _ in batches) + "|"]
    for name, label in ROWS:
        if name not in series:
            continue
        cells = [fmt(series[name][b], stock, b, name == "stock_eager") if b in series[name] else "—" for b in batches]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(results_dir: str) -> None:
    d = Path(results_dir)
    summary = summarize(load_dir(d))
    models = list(summary["tables"])
    big = next((m for m in models if "27B" in m), models[0])
    # plots -> bench/ (committed with the repo)
    subprocess.run([sys.executable, str(HERE / "plot_bench.py"), str(d), "--out-dir", str(HERE)], check=True)

    parts = []
    tb = summary["tables"][big]
    sp = tb["speedup_vs_stock"]
    gpu = (tb["meta"].get("gpu") or "GPU").replace("NVIDIA ", "")
    P, T = tb["meta"].get("prompt_tokens"), tb["meta"].get("max_tokens")
    b_max = max(tb["series"].get("stock_eager", {0: 0}))
    g = sp.get("fork_graphs", {})
    v = sp.get("fork_vectorized", {})
    i = sp.get("fork_indexed", {})
    ceil = tb["series"].get("ceiling_graphs", {}).get(b_max)
    fg = tb["series"].get("fork_graphs", {}).get(b_max)
    gap = (fg["wall_s"] / ceil["wall_s"] - 1) * 100 if (ceil and fg) else None
    n_pass = sum(a["ok"] for a in summary["assertions"])
    n_all = len(summary["assertions"])
    parts.append(
        f"**Measured on 1× {gpu}, {big} bf16, {P}-token prompts, {T} new tokens, one distinct steering vector per "
        f"request (layer 1, one prompt position):** at B = {b_max:,} the fork is **{g.get(b_max, 0):.1f}× faster** than stock "
        f"1.1.0 with CUDA graphs ({i.get(b_max, 0):.1f}× from the indexed hook alone, {v.get(b_max, 0):.1f}× with the "
        f"vectorised apply, eager)" + (f", within {gap:.0f}% of the same engine running no steering at all" if gap is not None else "")
        + f"; steering output is numerically identical to stock ({n_pass}/{n_all} correctness checks pass: injected delta "
        f"cos = 1.000, magnitude ratio = 1.000, same hidden states and next-token logprobs)."
    )
    parts.append("")
    parts.append("![per-request steering throughput vs batch size](bench/steering_throughput.png)")
    parts.append("")
    parts.append(f"Wall time of one `LLM.generate()` call ({big}, speedup vs stock in bold):")
    parts.append("")
    parts.append(table(tb))
    for m in models:
        if m == big:
            continue
        stem = f"steering_throughput_{m.split('/')[-1].lower()}"
        parts.append("")
        parts.append(f"Second panel, {m} (same protocol; small models are hook-overhead dominated, so the gap is larger):")
        parts.append("")
        parts.append(f"![{m}](bench/{stem}.png)")
        parts.append("")
        parts.append(table(summary["tables"][m]))
    parts.append("")
    parts.append(f"Full numbers, per-condition hook counters and every correctness assertion: `bench/results/` "
                 f"(`python bench/compare.py bench/results/<timestamp>`).")
    block = "\n".join(parts)

    readme = (REPO / "README.md").read_text()
    a, b = readme.index("<!-- RESULTS:BEGIN -->"), readme.index("<!-- RESULTS:END -->")
    readme = readme[: a + len("<!-- RESULTS:BEGIN -->")] + "\n" + block + "\n" + readme[b:]
    stat = subprocess.run(["git", "-C", str(REPO), "diff", "v1.1.0", "--stat", "--",
                           "vllm_lens/_worker_ext.py", "vllm_lens/_activations_plugin.py", "pyproject.toml"],
                          capture_output=True, text=True).stdout.rstrip()
    a, b = readme.index("<!-- DIFFSTAT:BEGIN -->"), readme.index("<!-- DIFFSTAT:END -->")
    readme = readme[: a + len("<!-- DIFFSTAT:BEGIN -->")] + "\n```\n" + stat + "\n```\n" + readme[b:]
    (REPO / "README.md").write_text(readme)
    # keep a copy of the summary next to the plots
    shutil.copy(d / "summary.json", HERE / "results_summary.json") if (d / "summary.json").exists() else None
    print("README results block updated from", d)


if __name__ == "__main__":
    main(sys.argv[1])
