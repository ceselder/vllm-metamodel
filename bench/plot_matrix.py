"""Plots for the vLLM-version matrix and the LoRA merge-on-publish benchmark (PNG + PDF + data JSON).

    python bench/plot_matrix.py [--out-dir bench] [--lora-dirs bench/results/lora_*]

Reads bench/results/version_matrix.json (bench/summarize_matrix.py) and the LoRA result dirs.
Writes  vllm_versions_generation.{png,pdf,_data.json}, vllm_versions_readout.*,
        lora_decode_step.*, lora_publish_latency.*  (titles state the measured claim).
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
# validated categorical slots (light surface): blue, orange, aqua, yellow, magenta
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e4de"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def _save(fig, stem: Path, data: dict) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=160, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    Path(str(stem) + "_data.json").write_text(json.dumps(data, indent=1))
    plt.close(fig)
    print("wrote", stem.with_suffix(".png"))


def _vkey(v: str) -> tuple:
    return tuple(int(x) for x in v.split("."))


def _legend_below(fig, ax, ncol: int = 2) -> None:
    h, l = ax.get_legend_handles_labels()
    if h:
        fig.legend(h, l, frameon=False, fontsize=8.5, loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=ncol)


def _bars(ax, groups: list[str], series: dict[str, list[float | None]], colors: list[str], ylabel: str, fmt: str = "{:,.0f}") -> None:
    n = len(series)
    w = 0.8 / max(n, 1)
    for i, (name, vals) in enumerate(series.items()):
        xs = [g + (i - (n - 1) / 2) * w for g in range(len(groups))]
        ys = [v if v is not None else 0 for v in vals]
        bars = ax.bar(xs, ys, width=w * 0.92, color=colors[i], label=name, linewidth=0)
        for b, v in zip(bars, vals):
            if v is not None:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=7.5, color=INK2, rotation=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_versions(vm: list[dict], out: Path) -> None:
    models = sorted({m for a in vm for m in a["models"]}, key=lambda m: ("27B" not in m, m))
    versions = sorted({a["vllm"] for a in vm}, key=_vkey)
    rows = {"plain vLLM (its default runner)": "ceiling_plain", "plain vLLM, V1 model runner": "ceiling_plain_v1",
            "fork: per-request steering + CUDA graphs": "fork_graphs", "fork: per-request steering, eager": "fork_vectorized"}
    data: dict = {"batch": 1024, "models": {}}
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.3), squeeze=False)
    claims = []
    for ax, model in zip(axes[0], models):
        series: dict[str, list[float | None]] = {}
        vers_present = [v for v in versions if any(a["vllm"] == v and model in a["models"] for a in vm)]
        for label, key in rows.items():
            vals = []
            for v in vers_present:
                a = next(x for x in vm if x["vllm"] == v)
                s = a["models"].get(model, {}).get("steering", {}).get("series", {}).get(key, {})
                e = s.get(1024) or s.get("1024")
                vals.append(e["tok_per_s"] if e else None)
            if any(x is not None for x in vals):
                series[label] = vals
        data["models"][model] = {"versions": vers_present, "tok_per_s": series}
        _bars(ax, vers_present, series, C, "generated tokens / s at B = 1,024")
        ax.set_title(model.split("/")[-1], loc="left", fontsize=11, color=INK)
        ax.set_xlabel("vLLM version")
        p = series.get("plain vLLM (its default runner)", [])
        f = series.get("fork: per-request steering + CUDA graphs", [])
        if p and p[0] and p[-1] and f and f[-1]:
            claims.append(f"{model.split('/')[-1]}: plain vLLM {vers_present[-1]} is {p[-1]/p[0]:.2f}x {vers_present[0]}; fork+graphs at {100*f[-1]/p[-1]:.0f}% of plain")
    _legend_below(fig, axes[0][0])
    fig.suptitle("Newer vLLM generates faster on the same B200, and the fork's per-request steering stays close to plain vLLM on every version\n"
                 + " · ".join(claims), fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, -0.10, "96-token prompt, 40 new tokens, one distinct steering vector per request (layer 1, marker token), best of 2 repeats; "
             "plain = VLLM_LENS_DISABLE=1 with vLLM's default torch.compile + CUDA graphs", fontsize=8, color=INK2)
    fig.subplots_adjust(top=0.80, bottom=0.22)
    _save(fig, out / "vllm_versions_generation", data)

    # readout per 1,024 texts
    conds = {"no hooks (prefill ceiling)": "nocap", "capture last 5 positions": "cap_last5", "in-engine readout, last 5": "read_last5", "readout + early exit": "exit_read_last5"}
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.3), squeeze=False)
    data = {"per_1024_texts_s": {}}
    for ax, model in zip(axes[0], models):
        vers_present = [v for v in versions if any(a["vllm"] == v and model in a["models"] and a["models"][model].get("readout") for a in vm)]
        series = {}
        for label, key in conds.items():
            vals = []
            for v in vers_present:
                a = next(x for x in vm if x["vllm"] == v)
                ro = a["models"][model].get("readout", {})
                eng = ro.get("graphs") or ro.get("eager") or {}
                vals.append(eng.get("per_1024", {}).get(key))
            if any(x is not None for x in vals):
                series[label] = vals
        data["per_1024_texts_s"][model] = {"versions": vers_present, "series": series}
        _bars(ax, vers_present, series, C, "seconds per 1,024 texts (prefill only)", fmt="{:.2f}")
        ax.set_title(model.split("/")[-1], loc="left", fontsize=11, color=INK)
        ax.set_xlabel("vLLM version")
    _legend_below(fig, axes[0][0])
    fig.suptitle("Hidden-state readout with early exit stays the cheapest way to read layer L on every vLLM version tested\n"
                 "(1,024 texts of 96-136 tokens, layer 42/64 on the 27B and 18/28 on the 1.7B, CUDA-graph engine)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.80, bottom=0.22)
    _save(fig, out / "vllm_versions_readout", data)


def load_lora(dirs: list[str]) -> dict:
    """{vllm: {model: {stage: result}}}.  Runs of the same (version, model, stage) are merged: the newest
    run wins field by field (layout, correctness, ...), while throughput rows for (condition, batch) pairs,
    publish entries for (mode, source) pairs, and drift / ipc_push blocks the newest run does not have are
    kept from older runs (e.g. a B = 1,024 measurement or the cpu-mode publish from an earlier session)."""
    out: dict = {}
    for d in sorted(dirs):  # oldest -> newest
        m = re.search(r"lora_([\d.]+)_", Path(d).name)
        if not m:
            continue
        ver = m.group(1)
        for f in glob.glob(f"{d}/*.json"):
            j = json.loads(Path(f).read_text())
            res = j.get("result")
            if not res or not res.get("throughput"):
                continue
            slot = out.setdefault(ver, {}).setdefault(res["model"], {})
            prev = slot.get(res["stage"])
            if prev is None:
                slot[res["stage"]] = res
                continue
            merged = dict(res)
            have = {(r["condition"], int(r["batch"])) for r in res["throughput"]}
            merged["throughput"] = list(res["throughput"]) + [r for r in prev.get("throughput", []) if (r["condition"], int(r["batch"])) not in have]
            have_p = {(p_.get("mode"), p_.get("source"), p_.get("phase")) for p_ in res.get("publish", [])}
            merged["publish"] = list(res.get("publish", [])) + [p_ for p_ in prev.get("publish", []) if (p_.get("mode"), p_.get("source"), p_.get("phase")) not in have_p]
            for key in ("drift", "ipc_push"):
                if not res.get(key) or "error" in res.get(key, {}) or "skipped" in res.get(key, {}):
                    if prev.get(key):
                        merged[key] = prev[key]
            slot[res["stage"]] = merged
    return out


def plot_lora(lora: dict, out: Path) -> None:
    if not lora:
        return
    versions = sorted(lora, key=_vkey)
    models = sorted({m for v in lora.values() for m in v}, key=lambda m: ("27B" not in m, m))
    conds = [("nolora", "lora_engine", "LoRA-capable engine, no adapter"), ("lora", "lora_engine", "rank-64 LoRA on every request"),
             ("merged", "lora_engine", "adapter merged into the weights (LoRA-capable engine)"),
             ("plain", "plain_engine", "plain engine (enable_lora=False)"), ("merged", "plain_engine", "adapter merged, plain engine")]
    data: dict = {"decode_step_ms": {}}
    fig, axes = plt.subplots(len(models), 2, figsize=(12.4, 4.2 * len(models)), squeeze=False)
    claims = []
    for r, model in enumerate(models):
        for c, B in enumerate((512, 1024)):
            ax = axes[r][c]
            groups = [v for v in versions if model in lora[v]]
            series = {}
            for cond, stage, label in conds:
                vals = []
                for v in groups:
                    res = lora[v][model].get(stage)
                    rows = [row for row in (res or {}).get("throughput", []) if row["condition"] == cond and int(row["batch"]) == B]
                    best = None
                    if rows:  # robust: best 40-token wall minus best 1-token wall over repeats
                        T = (res or {}).get("max_tokens", 40)
                        best = (min(r["wall_s"] for r in rows) - min(r["wall_1tok_s"] for r in rows)) / max(T - 1, 1) * 1000.0
                        best = best if best > 0 else None
                    vals.append(best)
                if any(x is not None for x in vals):
                    series[label] = vals
            data["decode_step_ms"].setdefault(model, {})[str(B)] = {"versions": groups, "series": series}
            _bars(ax, groups, series, C, "decode step, ms (lower is better)", fmt="{:.1f}")
            ax.set_title(f"{model.split('/')[-1]}, B = {B:,}", loc="left", fontsize=11, color=INK)
            ax.set_xlabel("vLLM version")
            lo, me = series.get("rank-64 LoRA on every request", []), series.get("adapter merged, plain engine", [])
            if B == 512:
                for gi in range(len(groups) - 1, -1, -1):  # newest version with both measurements
                    if gi < len(lo) and gi < len(me) and lo[gi] and me[gi]:
                        claims.append(f"{model.split('/')[-1]}: LoRA {lo[gi]:.1f} ms -> merged {me[gi]:.1f} ms per decode step at B=512 ({100*(lo[gi]-me[gi])/lo[gi]:.0f}% less, vLLM {groups[gi]})")
                        break
    _legend_below(fig, axes[-1][0], ncol=3)
    fig.suptitle("Merging the LoRA into the weights removes the LoRA kernels from every decode step\n" + " · ".join(claims), fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.text(0.01, -0.06 / len(models), "decode step = (wall for 40 new tokens - wall for 1 token) / 39; one steering vector per request; CUDA graphs (FULL_DECODE_ONLY); rank 64, rsLoRA alpha 16, ||sBA||/||W|| = 0.5% per module",
             fontsize=8, color=INK2)
    fig.subplots_adjust(top=0.86 if len(models) > 1 else 0.78, bottom=0.26 / len(models), hspace=0.45)
    _save(fig, out / "lora_decode_step", data)

    # publish latency by mode
    fig, axes = plt.subplots(1, len(models), figsize=(6.2 * len(models), 4.3), squeeze=False)
    data = {"publish_s": {}}
    for ax, model in zip(axes[0], models):
        groups = [v for v in versions if model in lora[v]]
        series: dict[str, list[float | None]] = {}
        labels = {("gpu", "dir"): "merge, base copy on GPU (exact)", ("cpu", "dir"): "merge, base copy pinned on host (exact)",
                  ("none", "dir"): "merge, no base copy (subtract previous; drifts)", ("gpu", "pickled_tensors"): "merge, A/B shipped as pickled tensors (RPC total)"}
        for (mode, src), label in labels.items():
            vals = []
            for v in groups:
                pubs = [p for st in ("lora_engine", "plain_engine") for p in lora[v][model].get(st, {}).get("publish", [])]
                ps = [p.get("rpc_s") or p["publish_s"] for p in pubs if p.get("mode") == mode and p.get("source") == src and p.get("phase") == "latency" and p.get("adapter") in ("a1",)]
                vals.append(min(ps) if ps else None)
            if any(x is not None for x in vals):
                series[label] = vals
        ipc = []
        for v in groups:
            pe = lora[v][model].get("plain_engine", {}).get("ipc_push", {})
            ipc.append(pe.get("total_s"))
        if any(x is not None for x in ipc):
            series["EasyNLA-style: full merged matrices over CUDA IPC + load_weights"] = ipc
        data["publish_s"][model] = {"versions": groups, "series": series}
        _bars(ax, groups, series, C, "seconds per adapter publish", fmt="{:.2f}")
        ax.set_title(model.split("/")[-1], loc="left", fontsize=11, color=INK)
        ax.set_xlabel("vLLM version")
    _legend_below(fig, axes[0][0])
    fig.suptitle("Publishing a new adapter by in-place merge takes well under a second on the 27B when a base copy is kept on the GPU\n"
                 "(replace the previous adapter with the next: resolve names, W = round(W0 + s B A) per targeted matrix)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(top=0.80, bottom=0.30)
    _save(fig, out / "lora_publish_latency", data)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default=str(HERE))
    p.add_argument("--matrix", default=str(HERE / "results" / "version_matrix.json"))
    p.add_argument("--lora-dirs", nargs="*", default=sorted(glob.glob(str(HERE / "results" / "lora_*"))))
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vm = json.loads(Path(a.matrix).read_text())
    plot_versions(vm, out)
    plot_lora(load_lora(a.lora_dirs), out)


if __name__ == "__main__":
    main()
