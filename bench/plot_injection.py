"""Plot the throughput part of the injection-mode test matrix (bench/test_injection_modes.py).

    python bench/plot_injection.py bench/results/injection_<timestamp> [--out-dir DIR] [--stem injection_throughput]

One small multiple per (model, engine): wall time of one ``LLM.generate()`` call
for B requests with 40 new tokens, three conditions side by side -- no steering,
Karvonen-style norm-matched add at layer 1, embedding replacement -- each request
with its own vector.  Series colours are the first three categorical slots of the
dataviz reference palette (blue / orange / aqua: validated all-pairs); every bar
is direct-labelled with its wall time (the aqua slot needs the relief).  Writes
``<stem>.png`` + ``<stem>.pdf`` + ``<stem>_data.json`` (exact plotted numbers).
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COND_ALL = [  # (key, label, colour) -- categorical slots in fixed order; only conditions present are drawn
    ("nosteer", "no steering (hooks installed, idle)", "#2a78d6"),
    ("karvonen_add", "norm-matched add at layer 1 (one vector per request)", "#eb6834"),
    ("embed_replace", "embedding replacement (one vector per request)", "#1baf7a"),
    ("embed_add", "norm-matched add on the embedding stream (one vector per request)", "#eb6834"),
]
INK, INK_SOFT, GRID, AXIS = "#0b0b0b", "#52514e", "#e8e7e3", "#c3c2b7"


def load(d: Path) -> dict:
    if (d / "summary.json").exists():
        s = json.loads((d / "summary.json").read_text())
    else:  # raw results dir: summarise on the fly
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from test_injection_modes import summarize

        s = summarize(d)
    panels: dict[tuple[str, str], dict] = {}
    for r in s["rows"]:
        if not (r["case"].startswith("throughput") and "/" in r["case"]):
            continue
        cond = r["case"].split("/", 1)[1]
        panels.setdefault((r["model"], r["engine"]), {}).setdefault(cond, {})[int(r["batch"])] = r
    return panels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stem", default="injection_throughput")
    ap.add_argument("--estimator", choices=["median", "min"], default="median",
                    help="decode-step estimate per bar: paired median over repeats (default) or the stall-robust "
                         "(min wall_2T - min wall_T)/T")
    ap.add_argument("--title", default=(
        "Decode runs at the no-steering speed with one distinct vector per request: on Qwen3.6-27B the decode-step time of norm-matched "
        "addition and embedding replacement is within ±1.7% of no steering, eager and with CUDA graphs\\n"
        "(on Qwen3-1.7B a 0.5–2 s call cannot resolve per-step differences: error bars = min–max over interleaved repeats; the CUDA-graph "
        "evidence there is the hook count, 3 vs 41 invocations)\\n"
        "decode-step time = (wall at 80 new tokens − wall at 40) / 40 · 27B: min of 2 repeats, 1.7B: paired median of 3 · 96-token prompt · "
        "bf16 · 1× B200 · vllm-metamodel 1.1.0.post2"))
    a = ap.parse_args()
    d = Path(a.results_dir)
    out_dir = Path(a.out_dir) if a.out_dir else d
    out_dir.mkdir(parents=True, exist_ok=True)
    panels = load(d)
    present = {c for p_ in panels.values() for c in p_}
    COND = [c for c in COND_ALL if c[0] in present]
    assert len({c[2] for c in COND}) == len(COND), "conditions present share a colour slot"
    order = sorted(panels, key=lambda k: ("27B" not in k[0], k[0], k[1] != "eager", k[1]))
    n = len(order)
    fig_w = max(4.3 * n + 0.8, 10.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 5.6), facecolor="white", squeeze=False)
    data_out: dict = {"conditions": [c[0] for c in COND], "panels": {}}
    width = 0.24 if len(COND) == 3 else 0.36
    for ax, key in zip(axes[0], order):
        model, engine = key
        by_cond = panels[key]
        batches = sorted({b for c in by_cond.values() for b in c})
        ax.set_facecolor("white")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color(AXIS)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(colors=INK_SOFT, labelsize=8.5)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        metric = "decode_step_ms" if all(v.get("decode_step_ms") is not None for c in by_cond.values() for v in c.values()) else "wall_s"
        if metric == "decode_step_ms" and a.estimator == "min" and all(v.get("decode_step_min_ms") is not None for c in by_cond.values() for v in c.values()):
            metric = "decode_step_min_ms"
        ymax = max(v[metric] for c in by_cond.values() for v in c.values())
        pdata: dict = {"batches": batches, "series": {}}
        T = 40
        spreads = []
        ctrl_range: list[tuple[float, float]] = []
        top = ymax
        for j, (ck, _label, colour) in enumerate(COND):
            xs = [i + (j - (len(COND) - 1) / 2) * (width + 0.03) for i in range(len(batches))]
            ys = [by_cond.get(ck, {}).get(b, {}).get(metric, float("nan")) for b in batches]
            ax.bar(xs, ys, width=width, color=colour, edgecolor="white", linewidth=1.5, zorder=3)
            for x, y, b in zip(xs, ys, batches):
                if y != y:
                    continue
                reps = by_cond.get(ck, {}).get(b, {}).get("repeats")
                lo = hi = None
                if reps and metric in ("decode_step_ms", "decode_step_min_ms"):  # min..max of the paired per-repeat estimates
                    ests = [(r["wall_2T_s"] - r["wall_s"]) / T * 1000.0 for r in reps]
                    lo, hi = min(ests), max(ests)
                    top = max(top, hi)
                    if ck == "nosteer":
                        ctrl_range.append((lo, hi))
                    ax.plot([x, x], [lo, hi], color=INK_SOFT, linewidth=1.2, zorder=4)
                    ax.plot([x - 0.05, x + 0.05], [lo, lo], color=INK_SOFT, linewidth=1.0, zorder=4)
                    ax.plot([x - 0.05, x + 0.05], [hi, hi], color=INK_SOFT, linewidth=1.0, zorder=4)
                    if ck == "nosteer":
                        spreads.append((hi - lo) / y)
                ax.text(x, (hi if hi is not None else y) + 0.012 * ymax, f"{y:.2f}", ha="center", va="bottom", fontsize=7.6, color=INK)
            pdata["series"][ck] = {str(b): {k: by_cond.get(ck, {}).get(b, {}).get(k) for k in
                                            ("wall_s", "wall_2T_s", "decode_step_ms", "prefill_plus_overhead_s", "tok_per_s", "hook_passes")}
                                   for b in batches}
        pdata["metric"] = metric
        ax.set_xticks(range(len(batches)))
        ax.set_xticklabels([f"B = {b:,}" for b in batches])
        top = min(top, 2.5 * ymax)  # error bars from JIT-stall repeats may exceed the axis; the bar labels stay readable
        ax.set_ylim(0, top * 1.22)
        ax.set_ylabel(("decode-step time (ms), from wall(80 tok) − wall(40 tok)" if metric in ("decode_step_ms", "decode_step_min_ms")
                       else "wall time of one generate() call (s)") if ax is axes[0][0] else "", color=INK_SOFT, fontsize=9)
        hp = by_cond.get("embed_replace", {}).get(max(batches), {}).get("hook_passes")
        hp_ns = by_cond.get("nosteer", {}).get(max(batches), {}).get("hook_passes")
        mode = "CUDA graphs" if engine.startswith("graphs") else "eager"
        sub = f"layer-0 pre-hook invocations per generate() call: {hp} (no-steering {hp_ns})"
        if ctrl_range:
            lo_all, hi_all = min(r[0] for r in ctrl_range), max(r[1] for r in ctrl_range)
            sub += (f"\ncontrol's per-repeat estimates span {lo_all:.0f}…{hi_all:.0f} ms; bars = "
                    + ("min-over-repeats (stall-robust)" if metric == "decode_step_min_ms" else "paired median"))
        ax.set_title(f"{model.split('/')[-1]} — {mode}\n{sub}", fontsize=8.0, color=INK, loc="left")
        data_out["panels"][f"{model}|{engine}"] = pdata
    total_label_chars = sum(len(l) for _k, l, _c in COND)
    ncol = len(COND) if total_label_chars * 0.085 < fig_w else max(1, len(COND) // 2)
    fig.legend(handles=[Patch(facecolor=c, label=l) for _k, l, c in COND], loc="lower center", ncol=ncol, frameon=False,
               fontsize=8.6, bbox_to_anchor=(0.5, 0.0))
    width_chars = int(fig_w * 11.5)
    title = "\n".join(textwrap.fill(line, width_chars) for line in a.title.replace("\\n", "\n").split("\n"))
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    n_title_lines = title.count("\n") + 1
    fig.tight_layout(rect=(0, 0.05 + 0.03 * (ncol < len(COND)), 1, 1 - 0.035 * n_title_lines - 0.02))
    fig.savefig(out_dir / f"{a.stem}.png", dpi=170, facecolor="white")
    fig.savefig(out_dir / f"{a.stem}.pdf", facecolor="white")
    (out_dir / f"{a.stem}_data.json").write_text(json.dumps(data_out, indent=1))
    print("wrote", out_dir / f"{a.stem}.png")


if __name__ == "__main__":
    main()
