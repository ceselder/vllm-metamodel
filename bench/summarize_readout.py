#!/usr/bin/env python
"""Summarise a ``bench/results/readout_<ts>/`` directory (bench_readout.py output).

    python bench/summarize_readout.py bench/results/readout_<ts>      # prints tables, writes summary.json + summary.md

Per model: one row per (engine, condition) with wall time per batch size, seconds per
1,024 texts, prompt tokens/s, engine vs RPC split, worker counters; the HF reference
timing; every correctness check; cross-engine agreement of the sampled rows (stock vs
fork vs HF).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from typing import Any

LABELS = {
    "stock_eager:cap_all": "stock vllm-lens 1.1.0: capture layer L, all positions",
    "fork_eager:cap_all_legacy": "fork, 1.1.0 capture path (per-request .cpu() + per-request RPC)",
    "fork_eager:cap_all": "fork: gather capture, all positions, one RPC",
    "fork_eager:cap_last5": "fork: gather capture, last 5 positions",
    "fork_eager:read_last5": "fork: in-engine cosine readout, last 5 positions",
    "fork_eager:read_all": "fork: in-engine cosine readout, all positions",
    "fork_eager:exit_read_last5": "fork: readout last 5 + early exit after layer L",
    "fork_eager:exit_cap_last5": "fork: capture last 5 + early exit after layer L",
    "fork_eager:exit_read_all": "fork: readout all positions + early exit",
    "fork_eager:nocap": "vLLM prefill, no hooks (ceiling, eager)",
    "fork_eager:nocap_hooked": "vLLM prefill, hooks installed but idle (eager)",
    "stock_eager:nocap": "vLLM prefill, no hooks (stock image, eager)",
    "fork_graphs:nocap": "vLLM prefill, no hooks (ceiling, CUDA-graph engine)",
    "fork_graphs:nocap_hooked": "vLLM prefill, hooks idle (CUDA-graph engine)",
    "fork_graphs:cap_all": "fork + CUDA-graph engine: gather capture, all positions",
    "fork_graphs:cap_all_legacy": "fork + CUDA-graph engine: 1.1.0 capture path",
    "fork_graphs:cap_last5": "fork + CUDA-graph engine: capture last 5",
    "fork_graphs:read_last5": "fork + CUDA-graph engine: readout last 5",
    "fork_graphs:read_all": "fork + CUDA-graph engine: readout all positions",
    "fork_graphs:exit_read_last5": "fork + CUDA-graph engine: readout last 5 + early exit",
    "fork_graphs:exit_cap_last5": "fork + CUDA-graph engine: capture last 5 + early exit",
    "fork_graphs:exit_read_all": "fork + CUDA-graph engine: readout all + early exit",
}
PREFILL_ORDER = [
    "stock_eager:cap_all", "fork_eager:cap_all_legacy", "fork_eager:cap_all", "fork_eager:cap_last5",
    "fork_eager:read_all", "fork_eager:read_last5", "fork_eager:exit_read_all", "fork_eager:exit_cap_last5",
    "fork_eager:exit_read_last5", "fork_eager:nocap_hooked", "fork_eager:nocap", "stock_eager:nocap",
    "fork_graphs:cap_all_legacy", "fork_graphs:cap_all", "fork_graphs:cap_last5", "fork_graphs:read_all",
    "fork_graphs:read_last5", "fork_graphs:exit_read_all", "fork_graphs:exit_cap_last5", "fork_graphs:exit_read_last5",
    "fork_graphs:nocap_hooked", "fork_graphs:nocap",
]


def load_dir(d: Path) -> dict[str, dict[str, dict]]:
    """model -> tag (hf | stock_eager | fork_eager | fork_graphs) -> result dict."""
    out: dict[str, dict[str, dict]] = {}
    for f in sorted(d.glob("*.json")):
        if f.name in ("summary.json",):
            continue
        model, tag = f.stem.rsplit("__", 1)
        rec = json.loads(f.read_text())
        if rec.get("returncode", 0) != 0 or "result" not in rec:
            print(f"[summarize] {f.name}: rc={rec.get('returncode')} (skipped)", file=sys.stderr)
            continue
        out.setdefault(model.replace("__", "/"), {})[tag] = rec["result"]
    return out


def _agg(rows: list[dict]) -> dict[str, Any]:
    """Min wall over repeats (rep >= 0) plus the matching split; means kept for reference."""
    timed = [r for r in rows if r.get("rep", 0) >= 0]
    if not timed:
        return {}
    best = min(timed, key=lambda r: r["wall_s"])
    return {
        "wall_s": best["wall_s"],
        "wall_mean_s": statistics.fmean(r["wall_s"] for r in timed),
        "n_rep": len(timed),
        "engine_s": best.get("engine_s"),
        "rpc_s": sum((best.get("rpc_s") or {}).values()) if best.get("rpc_s") is not None else None,
        "rpc_by_method": best.get("rpc_s"),
        "client_s": (best["wall_s"] - (best.get("engine_s") or 0) - sum((best.get("rpc_s") or {}).values())) if best.get("engine_s") is not None else None,
        "per_1024_s": best["per_1024_s"],
        "prompt_tokens": best.get("prompt_tokens"),
        "gen_tokens": best.get("gen_tokens"),
        "prompt_tok_per_s": best.get("prompt_tok_per_s"),
        "early_exits": (best.get("stats") or {}).get("early_exits"),
        "hook_capture_s": (best.get("stats") or {}).get("hook_capture_s"),
        "hook_readout_s": (best.get("stats") or {}).get("hook_readout_s"),
        "retrieval_s": (best.get("stats") or {}).get("retrieval_s"),
        "gen_s": best.get("gen_s"),
        "reencode_s": best.get("reencode_s"),
    }


def summarize(results: dict[str, dict[str, dict]]) -> dict[str, Any]:
    out: dict[str, Any] = {"models": {}}
    for model, tags in results.items():
        m: dict[str, Any] = {"meta": {}, "hf": None, "series": {}, "gen": {}, "checks": [], "engines": {}, "cross_engine": []}
        hf = tags.get("hf")
        if hf:
            m["hf"] = {k: hf.get(k) for k in ("score_s_early_exit", "score_s_full", "hf_batch", "attn_implementation", "hf_class",
                                              "n_layers", "n_tokens", "mean_len", "timings_s", "transformers", "load_s")}
            m["meta"].update(gpu=hf.get("gpu"), layer=hf.get("layer"), n_layers=hf.get("n_layers"), hidden_dim=hf.get("hidden_dim"),
                             n_texts=hf.get("n_texts"), n_tokens=hf.get("n_tokens"), mean_len=hf.get("mean_len"), last_k=hf.get("last_k"))
        for tag, res in tags.items():
            if tag == "hf":
                continue
            m["meta"].setdefault("gpu", res.get("gpu"))
            m["meta"].setdefault("layer", res.get("layer"))
            m["meta"].setdefault("hidden_dim", res.get("hidden_dim"))
            m["meta"].setdefault("n_texts", res.get("n_texts"))
            m["meta"].setdefault("last_k", res.get("last_k"))
            m["meta"].setdefault("n_layers", (res.get("resolved_config") or {}).get("num_layers"))
            m["engines"][tag] = {
                "variant": res.get("variant"), "versions": res.get("versions"), "resolved_config": res.get("resolved_config"),
                "engine_up_s": res.get("engine_up_s"), "capabilities": res.get("capabilities"), "final_stats": res.get("final_stats"),
                "gen_capture_positions": res.get("gen_capture_positions"),
            }
            by: dict[str, dict[int, list[dict]]] = {}
            for r in res.get("rows", []):
                by.setdefault(r["condition"], {}).setdefault(int(r["batch"]), []).append(r)
            for cond, per_b in by.items():
                key = f"{tag}:{cond}"
                agg = {b: _agg(rows) for b, rows in sorted(per_b.items())}
                agg = {b: v for b, v in agg.items() if v}
                if cond.startswith("gen"):
                    m["gen"][key] = agg
                else:
                    m["series"][key] = agg
            for c in res.get("checks", []):
                m["checks"].append({"engine": tag, **c})
            # sampled rows (first texts, last-k positions) for cross-engine agreement
            for r in res.get("rows", []):
                if r.get("sample_rows_lastk"):
                    m.setdefault("_samples", {})[f"{tag}:{r['condition']}"] = r["sample_rows_lastk"]
                if r.get("reward_sample"):
                    m.setdefault("_rewards", {})[f"{tag}:{r['condition']}"] = r["reward_sample"]
        # cross-engine agreement
        samples = m.pop("_samples", {})
        rewards = m.pop("_rewards", {})
        ref_rows = hf.get("rows_lastk_sample") if hf else None
        ref_reward = hf.get("reward_sample") if hf else None
        for key, rows in samples.items():
            others = {"hf": ref_rows} if ref_rows else {}
            for k2, r2 in samples.items():
                if k2 != key:
                    others[k2] = r2
            for name, other in others.items():
                if other is None:
                    continue
                diffs = []
                for a, b in zip(rows, other):
                    if a is None or b is None:
                        continue
                    n = min(len(a), len(b))
                    for pa, pb in zip(a[-n:], b[-n:]):
                        diffs.append(max(abs(x - y) for x, y in zip(pa, pb)))
                if diffs:
                    m["cross_engine"].append({"a": key, "b": name, "kind": "hidden rows (last-k, first texts)", "max_abs_diff": max(diffs), "n_rows": len(diffs)})
        for key, rw in rewards.items():
            if ref_reward:
                m["cross_engine"].append({"a": key, "b": "hf", "kind": "reward (max cos over last-k)", "max_abs_diff": max(abs(x - y) for x, y in zip(rw, ref_reward)), "n_rows": len(rw)})
        m["n_checks"] = len(m["checks"])
        m["n_pass"] = sum(1 for c in m["checks"] if c.get("ok"))
        # headline numbers at the largest common batch
        bs = [set(v) for v in m["series"].values()]
        if bs:
            common = set.intersection(*bs) if len(bs) > 1 else bs[0]
            b_max = max(common) if common else max(max(v) for v in m["series"].values())
            m["headline"] = {"batch": b_max}
            for key, per_b in m["series"].items():
                if b_max in per_b:
                    m["headline"][key] = per_b[b_max]["wall_s"]
            if m["hf"]:
                m["headline"]["hf_early_exit"] = m["hf"]["score_s_early_exit"] * b_max / (m["meta"].get("n_texts") or 1024)
                m["headline"]["hf_full"] = m["hf"]["score_s_full"] * b_max / (m["meta"].get("n_texts") or 1024)
        out["models"][model] = m
    return out


def fmt(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.3f} s" if x < 10 else f"{x:.1f} s"


def markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    for model, m in summary["models"].items():
        meta = m["meta"]
        lines.append(f"### {model} — layer {meta.get('layer')} of {meta.get('n_layers')}, {meta.get('n_texts')} texts, mean {meta.get('mean_len', 0) or 0:.0f} tokens, 1× {str(meta.get('gpu', '')).replace('NVIDIA ', '')}")
        batches = sorted({b for v in m["series"].values() for b in v})
        stock = m["series"].get("stock_eager:cap_all", {})
        head = "| method (prefill-only re-encode, max_tokens=1) | " + " | ".join(f"B = {b:,}" for b in batches) + " | s / 1,024 texts | vs stock |"
        lines.append(head)
        lines.append("|---|" + "---:|" * (len(batches) + 2))
        keys = [k for k in PREFILL_ORDER if k in m["series"]] + [k for k in m["series"] if k not in PREFILL_ORDER]
        for key in keys:
            per_b = m["series"][key]
            cells = [fmt(per_b[b]["wall_s"]) if b in per_b else "—" for b in batches]
            b_last = max(per_b)
            p1024 = per_b[b_last]["per_1024_s"]
            sp = (stock[b_last]["wall_s"] / per_b[b_last]["wall_s"]) if b_last in stock and key != "stock_eager:cap_all" else None
            lines.append(f"| {LABELS.get(key, key)} | " + " | ".join(cells) + f" | {p1024:.3f} s | " + (f"**{sp:.1f}×**" if sp else "—") + " |")
        if m["hf"]:
            hf = m["hf"]
            lines.append(f"| HF transformers bf16, early exit after layer {meta.get('layer')} (batch {hf['hf_batch']}, the trainer's `read_resid`) | " + " | ".join("—" for _ in batches) + f" | {hf['score_s_early_exit']:.3f} s | — |")
            lines.append(f"| HF transformers bf16, full {hf['n_layers']} layers (batch {hf['hf_batch']}) | " + " | ".join("—" for _ in batches) + f" | {hf['score_s_full']:.3f} s | — |")
        if m["gen"]:
            lines.append("")
            gb = sorted({b for v in m["gen"].values() for b in v})
            lines.append("| generation + reading generated positions (40 new tokens) | " + " | ".join(f"B = {b:,}" for b in gb) + " |")
            lines.append("|---|" + "---:|" * len(gb))
            for key, per_b in m["gen"].items():
                cells = []
                for b in gb:
                    v = per_b.get(b)
                    if not v:
                        cells.append("—")
                    elif v.get("gen_s") is not None:
                        cells.append(f"{v['wall_s']:.2f} s (gen {v['gen_s']:.2f} + re-encode {v['reencode_s']:.2f})")
                    else:
                        cells.append(f"{v['wall_s']:.2f} s")
                lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append(f"Correctness: {m['n_pass']}/{m['n_checks']} checks pass.")
        for c in m["checks"]:
            lines.append(f"- [{'PASS' if c.get('ok') else 'FAIL'}] {c['engine']} {c['condition']}: {c['check']} — {c.get('detail', '')}")
        for x in m["cross_engine"]:
            lines.append(f"- cross-engine {x['kind']}: {x['a']} vs {x['b']}: max |diff| = {x['max_abs_diff']:.4g} over {x['n_rows']} rows")
        lines.append("")
    return "\n".join(lines)


def readme_block(summary: dict[str, Any]) -> str:
    """Compact results block for the README (between the READOUT_RESULTS markers)."""
    lines: list[str] = []
    models = list(summary["models"])
    big = next((k for k in models if "27B" in k), models[0])
    for model in [big] + [m for m in models if m != big]:
        m = summary["models"][model]
        meta, hf = m["meta"], m.get("hf") or {}
        b = m.get("headline", {}).get("batch")
        if not b:
            continue
        gpu = str(meta.get("gpu", "")).replace("NVIDIA ", "")
        lines.append(f"**{model}** — layer {meta.get('layer')} of {meta.get('n_layers')}, {meta.get('n_texts'):,} texts of 96–136 tokens "
                     f"(mean {meta.get('mean_len', 0) or 0:.0f}), 1× {gpu}, wall time of one `generate()` call with B = {b:,} texts "
                     f"(prefill-only, `max_tokens=1`), min over repeats; HF = transformers bf16 on the same texts, batch {hf.get('hf_batch')}:")
        lines.append("")
        lines.append("| how layer L is read out | eager engine | CUDA-graph engine | vs stock |")
        lines.append("|---|---:|---:|---:|")
        stock = m["series"].get("stock_eager:cap_all", {}).get(b, {}).get("wall_s")
        rows = [
            ("stock vllm-lens 1.1.0 capture (`output_residual_stream=[L]`, all positions)", "stock_eager:cap_all", None),
            ("fork, 1.1.0 capture path (per-request `.cpu()` + per-request RPC)", "fork_eager:cap_all_legacy", "fork_graphs:cap_all_legacy"),
            ("fork gather capture, all positions, one RPC", "fork_eager:cap_all", "fork_graphs:cap_all"),
            ("fork gather capture, `capture_positions={\"last\": 5}`", "fork_eager:cap_last5", "fork_graphs:cap_last5"),
            ("fork `ReadoutVector` cosine, all positions", "fork_eager:read_all", "fork_graphs:read_all"),
            ("fork `ReadoutVector` cosine, last 5 positions", "fork_eager:read_last5", "fork_graphs:read_last5"),
            ("fork `ReadoutVector` last 5 **+ early exit**", "fork_eager:exit_read_last5", "fork_graphs:exit_read_last5"),
            ("fork capture last 5 + early exit", "fork_eager:exit_cap_last5", "fork_graphs:exit_cap_last5"),
            ("vLLM prefill with no hooks at all (ceiling)", "fork_eager:nocap", "fork_graphs:nocap"),
        ]
        for lab, ke, kg in rows:
            ve = m["series"].get(ke, {}).get(b, {}).get("wall_s")
            vg = m["series"].get(kg, {}).get(b, {}).get("wall_s") if kg else None
            best = min(x for x in (ve, vg) if x is not None) if (ve or vg) else None
            sp = f"**{stock / best:.1f}×**" if (stock and best and ke != "stock_eager:cap_all") else "—"
            lines.append(f"| {lab} | {fmt(ve)} | {fmt(vg)} | {sp} |")
        if hf:
            n_texts = meta.get("n_texts") or 1024
            hf_ee = hf["score_s_early_exit"] * b / n_texts
            hf_full = hf["score_s_full"] * b / n_texts
            lines.append(f"| HF transformers bf16, forward hook + early exit after layer {meta.get('layer')} (the trainer's `read_resid`, batch {hf.get('hf_batch')}) | {fmt(hf_ee)} | — | {('**' + f'{stock / hf_ee:.1f}×**') if stock else '—'} |")
            lines.append(f"| HF transformers bf16, all {hf.get('n_layers')} layers (batch {hf.get('hf_batch')}) | {fmt(hf_full)} | — | {('**' + f'{stock / hf_full:.1f}×**') if stock else '—'} |")
        lines.append("")
        gen = m.get("gen", {})
        gb = sorted({bb for v in gen.values() for bb in v})
        if gb:
            gbm = max(gb)
            lines.append(f"Reading *generated* positions, B = {gbm} prompts × 40 new tokens: ")
            parts = []
            for key, lab in (("stock_eager:gen_cap_all", "stock 1.1.0 eager generate + capture"),
                             ("fork_eager:gen_nocap", "eager generate, no capture"),
                             ("fork_eager:gen_cap_all", "eager generate + capture every generated position"),
                             ("fork_graphs:gen_nocap", "CUDA-graph generate, no capture"),
                             ("fork_graphs:gen_then_read", "CUDA-graph generate + re-encode with readout"),
                             ("fork_graphs:gen_then_exit_read", "CUDA-graph generate + re-encode with readout + early exit")):
                v = gen.get(key, {}).get(gbm)
                if v:
                    parts.append(f"{lab} **{v['wall_s']:.2f} s**" + (f" ({v['gen_s']:.2f} + {v['reencode_s']:.2f})" if v.get("gen_s") is not None else ""))
            lines.append("; ".join(parts) + ".")
            lines.append("")
        lines.append(f"Correctness: {m['n_pass']}/{m['n_checks']} checks against the HF reference pass "
                     "(captured rows: cosine ≥ 0.999 with HF's layer output at every compared position; readout values: reward = max cosine over the last 5 "
                     "positions within 2e-3 of HF's for every one of the texts; early-exit results identical to the non-exit ones).")
        lines.append("")
    return "\n".join(lines)


def update_readme(readme: Path, summary: dict[str, Any]) -> None:
    txt = readme.read_text()
    b0, b1 = "<!-- READOUT_RESULTS:BEGIN -->", "<!-- READOUT_RESULTS:END -->"
    i, j = txt.index(b0) + len(b0), txt.index(b1)
    readme.write_text(txt[:i] + "\n" + readme_block(summary) + txt[j:])


if __name__ == "__main__":
    d = Path(sys.argv[1])
    s = summarize(load_dir(d))
    (d / "summary.json").write_text(json.dumps(s, indent=1))
    md = markdown(s)
    (d / "summary.md").write_text(md)
    if "--readme" in sys.argv:
        update_readme(Path(sys.argv[sys.argv.index("--readme") + 1]), s)
        print("README updated")
    else:
        print(md)
