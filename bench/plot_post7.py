#!/usr/bin/env python
"""Plots + data JSON for the 1.1.0.post7 benchmarks (prefix caching with steering, torch.compile-
compatible hooks, shared-memory transport).  Reads the result directories written by
``bench/modal_bench.py::prefix_cache`` / ``::compile_op`` / ``::shm`` and writes PNG + PDF + a
replot-ready ``*_data.json`` per figure into ``--out`` (default: bench/).

    python bench/plot_post7.py --prefix bench/results/prefix_cache_<ts> \
        --compile bench/results/compile_0.19.0_<ts> bench/results/compile_0.27.1_<ts> \
        --shm bench/results/shm_<ts> --out bench/ --report ~/shared/reports/vllm-metamodels/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# validated categorical palette (dataviz skill reference instance, light surface), fixed slot order
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
SHORT = {"Qwen/Qwen3.6-27B": "Qwen3.6-27B (hybrid GDN)", "Qwen/Qwen3-1.7B": "Qwen3-1.7B (dense attention)"}


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="semibold", pad=10)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _bars(ax, groups: list[str], series: list[tuple[str, str, list[float | None]]], unit: str = "s") -> None:
    """Grouped thin bars, 2px surface gaps, values as direct labels (text ink), legend for >= 2 series."""
    n = len(series)
    w = 0.8 / n
    for j, (name, color, vals) in enumerate(series):
        xs = [i - 0.4 + w * (j + 0.5) for i in range(len(groups))]
        ys = [v if v is not None else 0.0 for v in vals]
        bars = ax.bar(xs, ys, width=w * 0.92, color=color, label=name, linewidth=0, zorder=2)
        for b, v in zip(bars, vals):
            if v is None:
                continue
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}{unit}" if v < 100 else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK2, rotation=0)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=9, color=INK)
    top = max([v for _, _, vals in series for v in vals if v is not None] or [1.0])
    ax.set_ylim(0, top * 1.22)  # headroom for the value labels
    if n >= 2:  # legend below the axes so it never covers a bar
        ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, labelcolor=INK2)


def _save(fig, out_dirs: list[Path], stem: str, data: dict[str, Any]) -> None:
    fig.patch.set_facecolor(SURF)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{stem}.png", dpi=170)
        fig.savefig(d / f"{stem}.pdf")
        (d / f"{stem}_data.json").write_text(json.dumps(data, indent=1))
        rd = d / "data" / "post7"
        if (d / "data").exists():
            rd.mkdir(parents=True, exist_ok=True)
            (rd / f"{stem}.json").write_text(json.dumps(data, indent=1))
    plt.close(fig)


def _min_wall(res: dict, cond: str, B: int) -> float | None:
    ws = [r["wall_s"] for r in res.get("throughput", []) if r["condition"] == cond and int(r["batch"]) == B]
    return min(ws) if ws else None


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    rec = json.loads(path.read_text())
    return rec.get("result") if "result" in rec else rec


# ---------------------------------------------------------------------------


def plot_prefix(prefix_dir: Path, out: list[Path]) -> dict:
    data: dict[str, Any] = {"source": str(prefix_dir), "models": {}}
    models = [m for m in ("Qwen/Qwen3-1.7B", "Qwen/Qwen3.6-27B") if list(prefix_dir.glob(f"{m.replace('/', '__')}__steer_off_m*.json"))]
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 5.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        mk = m.replace("/", "__")
        off = next(iter(prefix_dir.glob(f"{mk}__steer_off_m*.json")), None)
        on90 = next(iter(prefix_dir.glob(f"{mk}__steer_on_m9*.json")), None)
        on70 = next((p for p in prefix_dir.glob(f"{mk}__steer_on_m*.json") if "payload" not in p.name and "_m9" not in p.name), None)
        pay = next(iter(prefix_dir.glob(f"{mk}__steer_on_payload_m*.json")), None)
        r_off, r_on, r_on70, r_pay = (_load(p) if p else None for p in (off, on90, on70, pay))
        groups, s_off, s_on, s_pay = [], [], [], []
        for B in (512, 1024):
            groups.append(f"B={B}")
            s_off.append(_min_wall(r_off, "steer3d", B) if r_off else None)
            s_on.append(_min_wall(r_on, "steer3d", B) if r_on else None)
            s_pay.append(_min_wall(r_pay, "steer3d", B) if r_pay else None)
        hits_on = (r_on or {}).get("cache_counters", {}).get("steer3d", {})
        series = [("prefix caching off", C1, s_off), ("on, marker 90/96 (template prefix shared)", C2, s_on),
                  ("on, marker 70/96, payload tags (identical rows share steered blocks)", C3, s_pay)]
        _bars(ax, groups, series)
        _style(ax, SHORT.get(m, m), "wall time per generate() call, 40 new tokens (s)")
        data["models"][m] = {
            "steer3d_wall_s": {"off": dict(zip(groups, s_off)), "on_m90_nonce": dict(zip(groups, s_on)),
                               "on_m70_nonce": {g: _min_wall(r_on70, "steer3d", B) for g, B in zip(groups, (512, 1024))} if r_on70 else None,
                               "on_m70_payload": dict(zip(groups, s_pay))},
            "nosteer_wall_s": {"off": {g: _min_wall(r_off, "nosteer", B) for g, B in zip(groups, (512, 1024))} if r_off else None,
                               "on_m90": {g: _min_wall(r_on, "nosteer", B) for g, B in zip(groups, (512, 1024))} if r_on else None},
            "prefix_cache_counters_on_m90_steer3d": hits_on,
            "rows_steered": {"on_m90": (r_on or {}).get("stats", {}).get("steer3d", {}).get("rows_steered"),
                             "on_m70_payload": (r_pay or {}).get("stats", {}).get("steer3d", {}).get("rows_steered")},
        }
    fig.suptitle("Prefix caching with per-request steering stays exact (steered blocks salted) and saves 7-15% on a dense-attention model;\n"
                 "the hybrid 27B gains nothing because vLLM cannot cache its recurrent (GatedDeltaNet) state", fontsize=10.5, color=INK, x=0.01, ha="left")
    _save(fig, out, "prefix_cache_steering", data)
    return data


def plot_compile(compile_dirs: list[Path], out: list[Path]) -> dict:
    data: dict[str, Any] = {"sources": [str(d) for d in compile_dirs], "versions": {}}
    cols = []
    for d in compile_dirs:
        ver = d.name.split("_")[1]
        for m in ("Qwen/Qwen3.6-27B", "Qwen/Qwen3-1.7B"):
            mk = m.replace("/", "__")
            if (d / f"{mk}__fork__compile.json").exists():
                cols.append((ver, m, d))
    if not cols:
        return data
    fig, axes = plt.subplots(1, len(cols), figsize=(4.8 * len(cols), 5.8), squeeze=False)
    for ax, (ver, m, d) in zip(axes[0], cols):
        mk = m.replace("/", "__")
        rg, rc, rp = (_load(d / f"{mk}__fork__{e}.json") for e in ("graphs", "compile", "plain"))
        groups = ["B=512", "B=1024"]
        s_g = [_min_wall(rg, "steer3d", B) if rg else None for B in (512, 1024)]
        s_c = [_min_wall(rc, "steer3d", B) if rc else None for B in (512, 1024)]
        s_p = [_min_wall(rp, "nosteer", B) if rp else None for B in (512, 1024)]
        _bars(ax, groups, [("hook engine: compile off + decode graphs (steering)", C1, s_g),
                           ("torch.compile on + custom-op hooks (steering)", C2, s_c),
                           ("plain vLLM, no hooks (compile + graphs)", C3, s_p)])
        _style(ax, f"{SHORT.get(m, m)}, vLLM {ver}", "wall time per generate() call, 40 new tokens (s)")
        probes = (rc or {}).get("probes", {}).get("steer3d", {})
        data["versions"].setdefault(ver, {})[m] = {
            "steer3d_wall_s": {"hooks_graphs": dict(zip(groups, s_g)), "compile_op": dict(zip(groups, s_c))},
            "nosteer_wall_s": {"hooks_graphs": {g: _min_wall(rg, "nosteer", B) for g, B in zip(groups, (512, 1024))} if rg else None,
                               "compile_op": {g: _min_wall(rc, "nosteer", B) for g, B in zip(groups, (512, 1024))} if rc else None,
                               "plain": dict(zip(groups, s_p))},
            "compile_probe": {k: probes.get(k) for k in ("cos_delta_vs_v", "norm_ratio", "max_other_row_abs_delta", "ok")},
            "compile_stats_steer3d": {k: (rc or {}).get("stats", {}).get("steer3d", {}).get(k) for k in ("rows_steered", "op_calls", "steps_planned", "errors")},
            "resolved": {e: (r or {}).get("resolved_config", {}).get("compilation_mode") for e, r in (("graphs", rg), ("compile", rc), ("plain", rp))},
            "engine_up_s": {e: (r or {}).get("engine_up_s") for e, r in (("graphs", rg), ("compile", rc), ("plain", rp))},
        }
        inj = _load(d / "injection" / f"{mk}__compile.json")
        if inj:
            data["versions"][ver][m]["injection_compile"] = {"n_checks": len(inj["checks"]), "n_pass": sum(1 for c in inj["checks"] if c["ok"]),
                                                             "n_unresolvable": sum(1 for c in inj["checks"] if c["ok"] is None),
                                                             "failed": [c["check"] for c in inj["checks"] if c["ok"] is False]}
    fig.suptitle("Keeping torch.compile on (hooks as a custom op) recovers most of the speed the compile-off hook engine gave up:\n"
                 "steered generation lands within 2% of plain vLLM on the 27B / vLLM 0.27.1", fontsize=10.5, color=INK, x=0.01, ha="left")
    _save(fig, out, "compile_op_steering", data)
    return data


def plot_shm(shm_dir: Path, out: list[Path]) -> dict:
    data: dict[str, Any] = {"source": str(shm_dir), "models": {}}
    models = [m for m in ("Qwen/Qwen3.6-27B", "Qwen/Qwen3-1.7B") if (shm_dir / f"{m.replace('/', '__')}__shm.json").exists()]
    if not models:
        return data
    fig, axes = plt.subplots(1, len(models), figsize=(5.8 * len(models), 5.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        r = _load(shm_dir / f"{m.replace('/', '__')}__shm.json") or {}
        groups = ["B=512", "B=1024"]
        series = []
        for tag, name, color in (("pickle", "pickled RPC (get_captured_states_many)", C1), ("shm_copy", "shared memory, copied out", C2), ("shm_view", "shared memory, zero-copy views", C3)):
            vals = []
            for B in (512, 1024):
                ws = [x["wall_s"] for x in r.get("capture", []) if x["transport"] == tag and int(x["batch"]) == B]
                vals.append(min(ws) if ws else None)
            series.append((name, color, vals))
        _bars(ax, groups, series)
        _style(ax, f"{SHORT.get(m, m)}, layer {r.get('layer')}", "wall time per generate() call, prefill only (s)")
        data["models"][m] = {
            "capture_wall_s": {s[0]: dict(zip(groups, s[2])) for s in series},
            "capture_worker_retrieval_s": {tag: {f"B={B}": min([x["retrieval_s"] for x in r.get("capture", []) if x["transport"] == tag and int(x["batch"]) == B] or [None]) for B in (512, 1024)}
                                           for tag in ("pickle", "shm_copy", "shm_view")},
            "capture_bytes": {f"B={B}": max([x["bytes"] for x in r.get("capture", []) if int(x["batch"]) == B] or [None]) for B in (512, 1024)},
            "set_steering_block_rpc_s": {f"{x['transport']} n={x['batch']}": {"min": x["rpc_s_min"], "median": x["rpc_s_median"], "bytes": x["bytes"]} for x in r.get("rpc", [])},
            "steer_generate_wall_s": {f"{x['transport']} B={x['batch']}": x["wall_s"] for x in r.get("vectors", [])},
            "hidden_dim": r.get("hidden_dim"),
        }
    fig.suptitle("Shipping captured activations through shared memory instead of pickled RPCs: 14% faster on the 27B (persistent arena),\n"
                 "35-42% on the 1.7B; all-position capture of 512 / 1,024 texts", fontsize=10.5, color=INK, x=0.01, ha="left")
    _save(fig, out, "shm_transport", data)
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="")
    p.add_argument("--compile", nargs="*", default=[])
    p.add_argument("--shm", default="")
    p.add_argument("--out", default=str(Path(__file__).resolve().parent))
    p.add_argument("--report", default="")
    a = p.parse_args()
    outs = [Path(a.out)] + ([Path(a.report)] if a.report else [])
    if a.prefix:
        plot_prefix(Path(a.prefix), outs)
    if a.compile:
        plot_compile([Path(c) for c in a.compile], outs)
    if a.shm:
        plot_shm(Path(a.shm), outs)
    print("plots written to", [str(o) for o in outs])


if __name__ == "__main__":
    main()
