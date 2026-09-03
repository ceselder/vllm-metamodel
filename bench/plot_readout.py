#!/usr/bin/env python
"""Plots for the hidden-state readout benchmark (PNG + PDF + exact data JSON).

    python bench/plot_readout.py bench/results/readout_<ts> [--out-dir DIR]

Figures (per model, stem suffix = model short name for the second model):
  readout_cost            horizontal bars: seconds per 1,024 texts at the largest batch for every
                          way of reading layer L, HF reference as a marked line (emphasis form)
  readout_vs_batch        wall time of one generate() call vs batch size (log-log), main methods
  generated_positions     stacked bars: reading generated positions eagerly vs generate-under-graphs
                          + re-encode (gen / re-encode segments)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_readout import LABELS, load_dir, summarize  # noqa: E402

# dataviz reference palette (light mode) -- fixed slot order, never cycled
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
DEEMPH = "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

COST_ROWS = [  # (series key, short label)
    ("stock_eager:cap_all", "stock vllm-lens 1.1.0 capture (all positions)"),
    ("fork_eager:cap_all_legacy", "fork, 1.1.0 capture path (per-request .cpu() + RPC)"),
    ("fork_eager:cap_all", "fork: gather capture, all positions, one RPC"),
    ("fork_eager:cap_last5", "fork: gather capture, last 5 positions"),
    ("fork_eager:read_all", "fork: in-engine cosine readout, all positions"),
    ("fork_eager:read_last5", "fork: in-engine cosine readout, last 5 positions"),
    ("fork_eager:exit_read_all", "fork: readout all positions + early exit"),
    ("fork_eager:exit_read_last5", "fork: readout last 5 + early exit  (recommended)"),
    ("fork_eager:nocap", "vLLM prefill, no hooks (ceiling)"),
]
BATCH_SERIES = [  # fixed slot order
    ("stock_eager:cap_all", "stock 1.1.0 capture"),
    ("fork_eager:cap_all", "fork gather capture, all positions"),
    ("fork_eager:cap_last5", "fork capture, last 5"),
    ("fork_eager:read_last5", "fork readout, last 5"),
    ("fork_eager:exit_read_last5", "fork readout + early exit"),
    ("fork_eager:nocap", "no hooks (ceiling)"),
]


def _short(model: str) -> str:
    return model.split("/")[-1]


def _save(fig, out_dir: Path, stem: str, data: dict) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=170, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    (out_dir / f"{stem}_data.json").write_text(json.dumps(data, indent=1))
    plt.close(fig)
    print("wrote", out_dir / f"{stem}.png")


def plot_cost(model: str, m: dict, out_dir: Path, stem: str) -> None:
    b = m["headline"]["batch"]
    rows = [(k, lab) for k, lab in COST_ROWS if k in m["series"] and b in m["series"][k]]
    vals = [m["series"][k][b]["per_1024_s"] for k, _ in rows]
    labels = [lab for _, lab in rows]
    hf = m.get("hf") or {}
    hf_ee = hf.get("score_s_early_exit")
    hf_full = hf.get("score_s_full")
    n_texts = m["meta"].get("n_texts") or 1024
    meta = m["meta"]
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 2.6))
    y = list(range(len(rows)))[::-1]
    colors = [SLOT[0] if "recommended" in lab else (SLOT[1] if k.startswith("stock") else DEEMPH) for k, lab in rows]
    ax.barh(y, vals, height=0.55, color=colors, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v * 1.03, yi, f"{v:.2f} s", va="center", ha="left", fontsize=9, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(min(vals) * 0.7, max(vals + [hf_ee or 0, hf_full or 0]) * 1.6)
    ax.set_xlabel(f"seconds per {n_texts:,} texts (prefill-only re-encode, max_tokens=1, B = {b:,}; log scale)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, len(rows) - 0.3 + 0.9)
    if hf_ee:
        ax.axvline(hf_ee, color=INK, lw=1.2, ls=(0, (4, 3)), zorder=4)
        ax.text(hf_ee, len(rows) - 0.3 + 0.85, f" HF bf16 early-exit forward, batch {hf.get('hf_batch')} (the trainer's read_resid today): {hf_ee:.2f} s",
                fontsize=8.5, color=INK, va="top", ha="left")
    if hf_full:
        ax.axvline(hf_full, color=MUTED, lw=1.0, ls=(0, (2, 3)), zorder=4)
        ax.text(hf_full, len(rows) - 0.3 + 0.35, f" HF full {hf.get('n_layers')} layers: {hf_full:.2f} s", fontsize=8.5, color=MUTED, va="top", ha="left")
    sp = None
    if "stock_eager:cap_all" in m["series"] and "fork_eager:exit_read_last5" in m["series"]:
        sp = m["series"]["stock_eager:cap_all"][b]["wall_s"] / m["series"]["fork_eager:exit_read_last5"][b]["wall_s"]
    title = (f"In-engine readout with early exit reads layer {meta.get('layer')} of {meta.get('n_layers')} for {n_texts:,} texts "
             + (f"{sp:.0f}× faster than stock vllm-lens capture" if sp else "far faster than stock vllm-lens capture")
             + (f" and {hf_ee / m['series']['fork_eager:exit_read_last5'][b]['per_1024_s']:.1f}× faster than the HF early-exit forward" if hf_ee and "fork_eager:exit_read_last5" in m["series"] else ""))
    ax.set_title(title + f"\n{_short(model)} bf16 · 1× {str(meta.get('gpu', '')).replace('NVIDIA ', '')} · {n_texts:,} texts of {meta.get('mean_len', 0) or 0:.0f} tokens on average · "
                 f"cosine of every position with a per-text direction, max over the last {meta.get('last_k')} · min of repeats", fontsize=10, loc="left")
    _save(fig, out_dir, stem, {"model": model, "batch": b, "rows": [{"key": k, "label": lab, "per_1024_s": v, "wall_s": m["series"][k][b]["wall_s"]} for (k, lab), v in zip(rows, vals)],
                               "hf_early_exit_s_per_1024": hf_ee, "hf_full_s_per_1024": hf_full, "hf_batch": hf.get("hf_batch"), "meta": meta})


def plot_vs_batch(models: dict, out_dir: Path, stem: str) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6.6 * n, 5.0), squeeze=False)
    data = {}
    for ax, (model, m) in zip(axes[0], models.items()):
        meta = m["meta"]
        series = [(k, lab) for k, lab in BATCH_SERIES if k in m["series"]]
        for si, (k, lab) in enumerate(series):
            per_b = m["series"][k]
            xs = sorted(per_b)
            ys = [per_b[b]["wall_s"] for b in xs]
            ax.plot(xs, ys, color=SLOT[si], lw=2, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=1.5, label=lab, solid_capstyle="round")
            if k.endswith(":cap_all") or k.endswith(":exit_read_last5"):  # label the two extremes only
                ax.annotate(f"{ys[-1]:.2f} s", (xs[-1], ys[-1]), textcoords="offset points", xytext=(7, 0), fontsize=8.5, color=INK2, va="center")
            data.setdefault(model, {})[k] = {"label": lab, "batch": xs, "wall_s": ys}
        hf = m.get("hf") or {}
        if hf.get("score_s_early_exit"):
            n_texts = meta.get("n_texts") or 1024
            xs = sorted({b for k, _ in series for b in m["series"][k]})
            ys = [hf["score_s_early_exit"] * b / n_texts for b in xs]
            ax.plot(xs, ys, color=INK, lw=1.2, ls=(0, (4, 3)), label=f"HF bf16 early-exit forward, batch {hf['hf_batch']} (scaled)")
            data[model]["hf_early_exit"] = {"batch": xs, "wall_s": ys}
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(sorted({b for k, _ in series for b in m["series"][k]}))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlim(min(ax.get_xticks()) / 1.3, max(ax.get_xticks()) * 1.6)
        ax.set_xlabel("texts per generate() call (B)")
        ax.set_ylabel("wall time of one call (s)")
        ax.set_title(f"{_short(model)} · layer {meta.get('layer')} of {meta.get('n_layers')}", fontsize=10, loc="left")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Reading one layer's residual stream out of vLLM: stock capture cost grows with every request, the fork's readout stays at the no-hook ceiling\n"
                 "prefill-only re-encode (max_tokens=1) of 96–136-token texts · bf16 · 1× B200 · lines = min over repeats", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out_dir, stem, data)


def plot_generated(models: dict, out_dir: Path, stem: str) -> None:
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7.6 * n, 4.8), squeeze=False)
    data = {}
    for ax, (model, m) in zip(axes[0], models.items()):
        gen = m.get("gen", {})
        bs = sorted({b for v in gen.values() for b in v})
        if not bs:
            continue
        b = max(bs)
        bars = []  # (label, gen part, reencode part)
        for key, lab in (("fork_eager:gen_nocap", "eager engine: generate only"),
                         ("fork_eager:gen_cap_all", "eager engine: generate + capture every generated position"),
                         ("fork_graphs:gen_nocap", "CUDA-graph engine: generate only"),
                         ("fork_graphs:gen_then_read", "CUDA-graph engine: generate, then re-encode with readout (last 5)"),
                         ("fork_graphs:gen_then_exit_read", "CUDA-graph engine: generate, then re-encode with readout + early exit"),
                         ("stock_eager:gen_cap_all", "stock 1.1.0 eager: generate + capture every position")):
            v = gen.get(key, {}).get(b)
            if v:
                if v.get("gen_s") is not None:
                    bars.append((lab, v["gen_s"], v["reencode_s"]))
                else:
                    bars.append((lab, v["wall_s"], 0.0))
        y = list(range(len(bars)))[::-1]
        ax.barh(y, [g for _, g, _ in bars], height=0.55, color=SLOT[0], label="generation (40 new tokens)", zorder=3)
        ax.barh(y, [r for _, _, r in bars], left=[g + (0.0 if r == 0 else 0) for _, g, r in bars], height=0.55, color=SLOT[1], label="re-encode pass (prefill-only)", zorder=3,
                edgecolor=SURFACE, linewidth=2)
        for yi, (lab, g, r) in zip(y, bars):
            ax.text(g + r + 0.05 * max(g + r for _, g, r in bars), yi, f"{g + r:.2f} s", va="center", fontsize=9, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([lab for lab, _, _ in bars], fontsize=9)
        ax.set_xlabel(f"wall time for B = {b} prompts, 40 new tokens each (s)")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(g + r for _, g, r in bars) * 1.25)
        ax.set_title(f"{_short(model)}", fontsize=10, loc="left")
        ax.legend(fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        data[model] = {"batch": b, "bars": [{"label": lab, "gen_s": g, "reencode_s": r} for lab, g, r in bars]}
    fig.suptitle("Reading generated positions: generate under CUDA graphs and re-encode beats capturing during eager decode\n"
                 "hooks cannot run inside replayed decode graphs, so generated-position capture forces an eager engine; a second prefill-only pass is cheaper", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out_dir, stem, data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    d = Path(a.results_dir)
    out_dir = Path(a.out_dir) if a.out_dir else d
    out_dir.mkdir(parents=True, exist_ok=True)
    s = summarize(load_dir(d))
    models = s["models"]
    big = next((k for k in models if "27B" in k), next(iter(models)))
    for model, m in models.items():
        suffix = "" if model == big else f"_{_short(model).lower()}"
        if m.get("headline"):
            plot_cost(model, m, out_dir, f"readout_cost{suffix}")
    plot_vs_batch(models, out_dir, "readout_vs_batch")
    plot_generated(models, out_dir, "generated_positions")


if __name__ == "__main__":
    main()
