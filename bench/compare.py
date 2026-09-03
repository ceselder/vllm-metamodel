"""Turn bench_steering.py JSON outputs into a speedup table + correctness assertions.

Usable standalone on a saved results directory:

    python bench/compare.py bench/results/<timestamp>

Series naming used everywhere (README, plots, report):
  stock_eager      stock vllm-lens 1.1.0, per-request steering (eager, forced by the plugin)
  fork_eager       vllm-lens-port, per-request steering, eager
  fork_graphs      vllm-lens-port, per-request steering, CUDA graphs (FULL_DECODE_ONLY)
  ceiling_graphs   vLLM, no steering / no hooks, same FULL_DECODE_ONLY engine config
  ceiling_eager    vLLM, no steering / no hooks, eager
  ceiling_plain    vLLM, no steering, its default compilation (torch.compile + CUDA graphs)
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

# (variant, engine, condition) -> series name
SERIES = {
    ("stock", "eager", "steer3d"): "stock_eager",
    ("fork", "eager", "steer3d_loop"): "fork_indexed",
    ("fork", "eager", "steer3d"): "fork_vectorized",
    ("fork", "graphs", "steer3d_loop"): "fork_graphs_indexed",
    ("fork", "graphs", "steer3d"): "fork_graphs",
    ("fork", "graphs", "nosteer"): "ceiling_graphs",
    ("fork", "eager", "nosteer"): "ceiling_eager",
    ("stock", "eager", "nosteer"): "ceiling_eager_stockimg",
    ("fork", "plain", "nosteer"): "ceiling_plain",
    ("stock", "eager", "steer2d"): "stock_eager_2d",
    ("fork", "eager", "steer2d"): "fork_eager_2d",
}

TOL = {
    "cos_min": 0.999,
    "ratio_band": (0.99, 1.01),
    "hidden_rel_max": 1e-3,  # max|h_a - h_b| / max|h_a| for the steered marker row (prefill is eager in every mode)
    "logprob_abs_max": 0.05,
    "delta_norm_rel_max": 2e-2,  # norm_match probe: per-position |steer - clean| stock vs fork
}


def _rows(res: dict) -> dict[str, dict[int, list[float]]]:
    """condition -> batch -> list of wall_s (per repeat)."""
    out: dict[str, dict[int, list[float]]] = {}
    for r in res.get("throughput", []):
        out.setdefault(r["condition"], {}).setdefault(int(r["batch"]), []).append(
            float(r["wall_s"])
        )
    return out


def _rel(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return math.inf
    scale = max(abs(x) for x in a[:n]) or 1.0
    return max(abs(x - y) for x, y in zip(a[:n], b[:n])) / scale


def summarize(all_results: dict[str, dict[str, dict[str, dict]]]) -> dict[str, Any]:
    """all_results[model][variant][engine] = {"returncode", "result": <bench json>, ...}."""
    tables: dict[str, Any] = {}
    assertions: list[dict[str, Any]] = []
    # headline model (the 27B) first: plots/README/report derive their order from this dict
    for model in sorted(all_results, key=lambda m: ("27B" not in m, m)):
        by_variant = all_results[model]
        series: dict[str, dict[int, dict[str, float]]] = {}
        probes: dict[str, dict] = {}
        meta: dict[str, Any] = {}
        for variant, by_engine in by_variant.items():
            for engine, rec in by_engine.items():
                res = rec.get("result")
                if not res:
                    assertions.append(
                        {
                            "model": model,
                            "check": f"{variant}/{engine} produced results",
                            "ok": False,
                            "detail": f"rc={rec.get('returncode')}",
                        }
                    )
                    continue
                meta.setdefault("gpu", res.get("gpu"))
                meta.setdefault("prompt_tokens", res.get("prompt_tokens"))
                meta.setdefault("max_tokens", res.get("max_tokens"))
                meta[f"{variant}_{engine}_resolved"] = res.get("resolved_config")
                meta[f"{variant}_{engine}_engine_up_s"] = res.get("engine_up_s")
                meta[f"{variant}_{engine}_versions"] = {
                    **res.get("versions", {}),
                    **res.get("variant", {}),
                }
                meta[f"{variant}_{engine}_stats"] = res.get("stats")
                for cond, by_b in _rows(res).items():
                    name = SERIES.get((variant, engine, cond))
                    if name is None:
                        continue
                    n_gen = res["max_tokens"]
                    series[name] = {
                        b: {
                            "wall_s": min(ws),
                            "tok_per_s": b * n_gen / min(ws),
                            "n_rep": len(ws),
                        }
                        for b, ws in sorted(by_b.items())
                    }
                for pname, p in res.get("probes", {}).items():
                    probes[f"{variant}_{engine}_{pname}"] = p

        # speedups vs stock
        speed: dict[str, dict[int, float]] = {}
        stock = series.get("stock_eager", {})
        for name in (
            "fork_indexed",
            "fork_vectorized",
            "fork_graphs_indexed",
            "fork_graphs",
            "ceiling_graphs",
            "ceiling_eager",
            "ceiling_plain",
        ):
            if name in series:
                speed[name] = {
                    b: stock[b]["wall_s"] / v["wall_s"]
                    for b, v in series[name].items()
                    if b in stock
                }
        if "stock_eager_2d" in series and "fork_eager_2d" in series:
            speed["fork_eager_2d"] = {
                b: series["stock_eager_2d"][b]["wall_s"] / v["wall_s"]
                for b, v in series["fork_eager_2d"].items()
                if b in series["stock_eager_2d"]
            }
        tables[model] = {"meta": meta, "series": series, "speedup_vs_stock": speed}

        # --- correctness assertions ---------------------------------------
        def add(check: str, ok: bool, detail: str) -> None:
            assertions.append(
                {"model": model, "check": check, "ok": bool(ok), "detail": detail}
            )

        p3 = {k: v for k, v in probes.items() if "_steer3d" in k}
        for k, p in p3.items():
            add(
                f"{k}: injected delta == vector (cos>{TOL['cos_min']}, ratio in {TOL['ratio_band']})",
                p["cos_delta_vs_v"] > TOL["cos_min"]
                and TOL["ratio_band"][0] < p["norm_ratio"] < TOL["ratio_band"][1],
                f"cos={p['cos_delta_vs_v']:.5f} ratio={p['norm_ratio']:.5f} other_rows={p['max_other_row_abs_delta']:.2e}",
            )
            add(
                f"{k}: rows other than the marker untouched",
                p["max_other_row_abs_delta"] == 0.0,
                f"max|delta|={p['max_other_row_abs_delta']:.2e}",
            )
        ref = p3.get("stock_eager_steer3d")
        if ref is not None:
            for k, p in p3.items():
                if k == "stock_eager_steer3d":
                    continue
                rel = _rel(ref["h_steer_marker"], p["h_steer_marker"])
                add(
                    f"{k} vs stock: steered hidden state identical (rel<{TOL['hidden_rel_max']})",
                    rel < TOL["hidden_rel_max"],
                    f"rel={rel:.2e}",
                )
                relc = _rel(ref["h_clean_marker"], p["h_clean_marker"])
                add(
                    f"{k} vs stock: clean hidden state identical",
                    relc < TOL["hidden_rel_max"],
                    f"rel={relc:.2e}",
                )
                if (
                    "h_steer_normmatch_marker" in ref
                    and "h_steer_normmatch_marker" in p
                ):
                    reln = _rel(
                        ref["h_steer_normmatch_marker"], p["h_steer_normmatch_marker"]
                    )
                    add(
                        f"{k} vs stock: norm_match=True steered hidden state identical (rel<{TOL['hidden_rel_max']})",
                        reln < TOL["hidden_rel_max"],
                        f"rel={reln:.2e}",
                    )
                a_lp, b_lp = ref["next_token_top20"], p["next_token_top20"]
                common = set(a_lp) & set(b_lp)
                mx = max((abs(a_lp[t] - b_lp[t]) for t in common), default=math.inf)
                add(
                    f"{k} vs stock: steered next-token argmax equal",
                    ref["next_token_argmax"] == p["next_token_argmax"],
                    f"{ref['next_token_argmax']} vs {p['next_token_argmax']}",
                )
                a_cl, b_cl = (
                    ref.get("next_token_top20_clean"),
                    p.get("next_token_top20_clean"),
                )
                if a_cl and b_cl:
                    # Two engine processes can pick different Triton autotune configs, so even the
                    # CLEAN prompt's logits differ slightly between them.  Steering is fine if the
                    # steered cross-engine difference is no larger than that noise floor.
                    common_c = set(a_cl) & set(b_cl)
                    noise = max(
                        (abs(a_cl[t] - b_cl[t]) for t in common_c), default=math.inf
                    )
                    tol = max(TOL["logprob_abs_max"], 1.5 * noise + 0.01)
                    add(
                        f"{k} vs stock: steered next-token top-20 logprobs within the engine-to-engine "
                        f"noise floor (max|d| <= max({TOL['logprob_abs_max']}, 1.5*clean noise+0.01))",
                        mx <= tol and len(common) >= 15,
                        f"steered max|d|={mx:.4f} over {len(common)} tokens; clean-prompt max|d|={noise:.4f} "
                        f"over {len(common_c)} tokens (tol {tol:.4f})",
                    )
                else:
                    add(
                        f"{k} vs stock: steered next-token top-20 logprobs (max|d|<{TOL['logprob_abs_max']})",
                        mx < TOL["logprob_abs_max"] and len(common) >= 15,
                        f"max|d|={mx:.4f} over {len(common)} shared tokens",
                    )
                same8 = ref["greedy8"] == p["greedy8"]
                add(
                    f"{k} vs stock: greedy 8-token steered continuation equal (informational)",
                    True,
                    f"{'equal' if same8 else 'DIFFERS: ' + str(ref['greedy8']) + ' vs ' + str(p['greedy8'])}",
                )
        r2 = probes.get("stock_eager_steer2d_normmatch")
        for f2name in (
            "fork_eager_steer2d_normmatch_loop",
            "fork_eager_steer2d_normmatch",
        ):
            f2 = probes.get(f2name)
            if not (r2 and f2):
                continue
            rel = _rel(r2["delta_norms"], f2["delta_norms"])
            add(
                f"{f2name} vs stock: per-position |delta| of norm_match broadcast steering (rel<{TOL['delta_norm_rel_max']})",
                rel < TOL["delta_norm_rel_max"],
                f"rel={rel:.2e} positions={len(r2['delta_norms'])}/{len(f2['delta_norms'])}",
            )
            relh = _rel(r2["h_steer_last"], f2["h_steer_last"])
            add(
                f"{f2name} vs stock: last generated-position hidden state identical",
                relh < TOL["hidden_rel_max"],
                f"rel={relh:.2e}",
            )
            add(
                f"{f2name} vs stock: generated tokens equal (informational)",
                True,
                "equal"
                if r2["tokens"] == f2["tokens"]
                else f"DIFFERS {r2['tokens']} vs {f2['tokens']}",
            )
        for k in [k for k in probes if "steer2d_normmatch" in k]:
            if probes[k].get("generated_rows_steered") is not None:
                add(
                    f"{k}: generated positions ARE steered in eager mode",
                    probes[k]["generated_rows_steered"],
                    "",
                )
        gg = probes.get("fork_graphs_graph_guard_2d")
        if gg is not None:
            add(
                "CUDA-graph mode refuses 2-D broadcast vectors (ValueError)",
                gg["raised"],
                gg.get("error", "")[:120],
            )
        for tag in ("fork_eager_stats", "fork_graphs_stats"):
            st = meta.get(tag)
            if not isinstance(st, dict):
                continue
            for cond in ("steer3d_loop", "steer3d", "steer2d"):
                s = st.get(cond)
                if not s:
                    continue
                add(
                    f"{tag}/{cond}: hook errors == 0",
                    s.get("errors", 0) == 0,
                    json.dumps(s),
                )
                if cond == "steer3d":
                    add(
                        f"{tag}/{cond}: every steered layer-step took the vectorised path",
                        s.get("vectorized_layer_steps", 0)
                        == s.get("steer_layer_steps", -1),
                        json.dumps(s),
                    )
                if cond == "steer3d_loop":
                    add(
                        f"{tag}/{cond}: vectorised path off",
                        s.get("vectorized_layer_steps", 0) == 0,
                        json.dumps(s),
                    )
                if tag == "fork_eager_stats" and cond.startswith("steer3d"):
                    add(
                        f"{tag}/{cond}: decode passes skipped by the idle fast path",
                        s.get("steps_fast_idle", 0) > 0,
                        json.dumps(s),
                    )

    text = []
    for a in assertions:
        text.append(
            f"[{'PASS' if a['ok'] else 'FAIL'}] {a['model']}: {a['check']}  {a['detail']}"
        )
    return {
        "tables": tables,
        "assertions": assertions,
        "assertions_text": text,
        "all_pass": all(a["ok"] for a in assertions),
        "tolerances": TOL,
    }


def load_dir(d: Path) -> dict:
    all_results: dict = {}
    for f in sorted(d.glob("*__*__*.json")):
        mtag, variant, engine = f.stem.rsplit("__", 2)  # model tag itself contains "__"
        all_results.setdefault(mtag.replace("__", "/"), {}).setdefault(variant, {})[
            engine
        ] = json.loads(f.read_text())
    return all_results


if __name__ == "__main__":
    d = Path(sys.argv[1])
    s = summarize(load_dir(d))
    (d / "summary.json").write_text(json.dumps(s, indent=1))
    for m, t in s["tables"].items():
        print(f"== {m} ==")
        for name, by_b in t["series"].items():
            print(
                f"  {name:22s} "
                + "  ".join(
                    f"B={b}: {v['wall_s']:.1f}s/{v['tok_per_s']:.0f}tok/s"
                    for b, v in by_b.items()
                )
            )
        for name, by_b in t["speedup_vs_stock"].items():
            print(
                f"  speedup {name:14s} "
                + "  ".join(f"B={b}: {x:.2f}x" for b, x in by_b.items())
            )
    print("\n".join(s["assertions_text"]))
    print("ALL PASS" if s["all_pass"] else "SOME CHECKS FAILED")
