"""Plot bench_steering.py results: throughput vs batch size and speedup vs stock.

    python bench/plot_bench.py bench/results/<timestamp> [--out-dir DIR] [--stem steering_throughput]

Writes ``<stem>.png`` + ``<stem>.pdf`` per model (one figure, two panels) and
``<stem>_data.json`` with the exact numbers plotted.  Colors are the validated
categorical slots from the dataviz reference palette (blue / aqua / orange);
the two no-steering ceilings are drawn as muted reference lines with direct
labels because they are bounds, not competing series.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from compare import load_dir, summarize  # noqa: E402

SERIES_STYLE = {
    # name: (label, color, marker) -- validated categorical order (dataviz reference palette)
    "stock_eager": (
        "stock vllm-lens 1.1.0 (per-layer key scan, eager forced)",
        "#2a78d6",
        "o",
    ),
    "fork_indexed": ("vllm-lens-port: indexed hook, eager", "#eda100", "s"),
    "fork_vectorized": (
        "vllm-lens-port: indexed + vectorised apply, eager",
        "#eda100",
        "^",
    ),
    "fork_graphs": (
        "vllm-lens-port: indexed + vectorised + CUDA graphs",
        "#eb6834",
        "D",
    ),
}
CEILING_STYLE = {
    "ceiling_plain": ("vLLM default (torch.compile + graphs), no steering", "#898781"),
    "ceiling_graphs": ("same engine config, no hooks", "#c3c2b7"),
}
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#e1e0d9"


def _style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.grid(True, which="major", axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _fmt_int(x, _pos):
    return f"{int(x):,}" if x >= 1 else f"{x:g}"


def plot_model(
    model: str, table: dict, out_dir: Path, stem: str, title_prefix: str
) -> dict:
    series = table["series"]
    speed = table["speedup_vs_stock"]
    meta = table["meta"]
    gpu = (meta.get("gpu") or "GPU").replace("NVIDIA ", "")
    P, T = meta.get("prompt_tokens"), meta.get("max_tokens")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0), facecolor="white")
    data = {
        "model": model,
        "gpu": gpu,
        "prompt_tokens": P,
        "max_tokens": T,
        "series": {},
        "speedup_vs_stock": {},
    }

    # --- panel 1: tokens/s vs batch ------------------------------------
    for name, (label, color) in CEILING_STYLE.items():
        if name not in series:
            continue
        bs = sorted(series[name])
        ys = [series[name][b]["tok_per_s"] for b in bs]
        ax1.plot(bs, ys, color=color, lw=1.6, zorder=1)
        ax1.annotate(
            label,
            (bs[-1], ys[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=7.5,
            color=INK_SOFT,
            va="center",
        )
        data["series"][name] = {
            "label": label,
            "batch": bs,
            "tok_per_s": ys,
            "wall_s": [series[name][b]["wall_s"] for b in bs],
        }
    for name, (label, color, marker) in SERIES_STYLE.items():
        if name not in series:
            continue
        bs = sorted(series[name])
        ys = [series[name][b]["tok_per_s"] for b in bs]
        ax1.plot(
            bs,
            ys,
            color=color,
            lw=2,
            marker=marker,
            ms=6,
            mec="white",
            mew=1.2,
            label=label,
            zorder=3,
        )
        data["series"][name] = {
            "label": label,
            "batch": bs,
            "tok_per_s": ys,
            "wall_s": [series[name][b]["wall_s"] for b in bs],
        }
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xticks(sorted({b for s in series.values() for b in s}))
    ax1.xaxis.set_major_formatter(FuncFormatter(_fmt_int))
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.yaxis.set_major_formatter(FuncFormatter(_fmt_int))
    ax1.yaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_xlabel(
        "requests per generate() call (one steering vector each)",
        color=INK_SOFT,
        fontsize=9.5,
    )
    ax1.set_ylabel("generated tokens / s", color=INK_SOFT, fontsize=9.5)
    ax1.set_title(
        "Generation throughput with per-request steering",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    ax1.legend(frameon=False, fontsize=8.2, loc="upper left")
    _style_axes(ax1)
    ax1.margins(x=0.08)
    ax1.set_xlim(right=ax1.get_xlim()[1] * 2.6)  # room for the ceiling labels

    # --- panel 2: speedup vs stock -------------------------------------
    ax2.axhline(1.0, color="#c3c2b7", lw=1.2, zorder=1)
    ax2.annotate(
        "stock 1.1.0 = 1×",
        (1, 1.0),
        xycoords=("axes fraction", "data"),
        xytext=(-4, 4),
        textcoords="offset points",
        fontsize=7.5,
        color=INK_SOFT,
        ha="right",
    )
    for name in ("fork_indexed", "fork_vectorized", "fork_graphs"):
        if name not in speed:
            continue
        label, color, marker = SERIES_STYLE[name]
        bs = sorted(speed[name])
        ys = [speed[name][b] for b in bs]
        ax2.plot(
            bs,
            ys,
            color=color,
            lw=2,
            marker=marker,
            ms=6,
            mec="white",
            mew=1.2,
            label=label,
            zorder=3,
        )
        if name == "fork_graphs":  # direct labels on the headline series only
            for b, y in zip(bs, ys):
                ax2.annotate(
                    f"{y:.1f}×",
                    (b, y),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.5,
                    color=INK,
                )
        data["speedup_vs_stock"][name] = {"batch": bs, "speedup": ys}
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sorted({b for s in speed.values() for b in s}) or [1])
    ax2.xaxis.set_major_formatter(FuncFormatter(_fmt_int))
    ax2.xaxis.set_minor_formatter(NullFormatter())
    ax2.set_ylim(bottom=0)
    ax2.set_xlabel("requests per generate() call", color=INK_SOFT, fontsize=9.5)
    ax2.set_ylabel("wall-time speedup vs stock 1.1.0 (×)", color=INK_SOFT, fontsize=9.5)
    ax2.set_title(
        "Speedup over stock vllm-lens 1.1.0", fontsize=10.5, color=INK, loc="left"
    )
    ax2.legend(frameon=False, fontsize=8.2, loc="upper left")
    _style_axes(ax2)
    ax2.margins(x=0.08)

    best = max((max(v.values()) for v in speed.values() if v), default=0)
    lo = min(
        (min(v.values()) for k, v in speed.items() if v and k == "fork_graphs"),
        default=0,
    )
    fig.suptitle(
        f"{title_prefix}{model} on 1× {gpu}: one steering vector per request (RL-rollout style) runs "
        f"{lo:.0f}–{best:.0f}× faster than stock vllm-lens 1.1.0 with the indexed hook, vectorised injection and CUDA graphs",
        fontsize=11.5,
        color=INK,
        x=0.01,
        ha="left",
        y=1.0,
    )
    fig.text(
        0.01,
        0.94,
        f"{P}-token prompt, {T} new tokens per request, bf16, one steering vector per request "
        f"(layer 1, one prompt position, distinct random unit vector per request). Ceilings = the same vLLM without any steering.",
        fontsize=8.5,
        color=INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=170, facecolor="white")
    fig.savefig(out_dir / f"{stem}.pdf", facecolor="white")
    plt.close(fig)
    (out_dir / f"{stem}_data.json").write_text(json.dumps(data, indent=1))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--stem", default="steering_throughput")
    a = ap.parse_args()
    d = Path(a.results_dir)
    out_dir = Path(a.out_dir) if a.out_dir else d
    summary = summarize(load_dir(d))
    for i, (model, table) in enumerate(summary["tables"].items()):
        stem = a.stem if i == 0 else f"{a.stem}_{model.split('/')[-1].lower()}"
        plot_model(model, table, out_dir, stem, "")
        print("wrote", out_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
