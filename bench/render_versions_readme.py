"""Fill the README's vLLM-version and LoRA blocks from the benchmark results.

    python bench/render_versions_readme.py            # uses bench/results/version_matrix.json + bench/results/lora_*

Rewrites everything between ``<!-- VERSIONS:BEGIN -->`` / ``<!-- VERSIONS:END -->`` (compatibility
matrix + apples-to-apples throughput per vLLM version) and ``<!-- LORA:BEGIN -->`` / ``<!-- LORA:END -->``
(LoRA decode overhead, publish latency, correctness / drift) in README.md, and copies the plots into
``bench/``.
"""

from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from plot_matrix import load_lora  # noqa: E402


def _vkey(v: str) -> tuple:
    return tuple(int(x) for x in v.split("."))


def _s(v) -> str:
    return "—" if v is None else f"{v['wall_s']:.2f} s ({v['tok_per_s']:,.0f} tok/s)"


def _intkeys(vm: list[dict]) -> list[dict]:
    for a in vm:
        for rec in a["models"].values():
            ser = rec.get("steering", {}).get("series", {})
            for name in list(ser):
                ser[name] = {int(b): v for b, v in ser[name].items()}
    return vm


def versions_block(vm: list[dict]) -> str:
    vm = _intkeys(vm)
    versions = sorted({a["vllm"] for a in vm}, key=_vkey)
    by = {a["vllm"]: a for a in vm}
    models = sorted({m for a in vm for m in a["models"]}, key=lambda m: ("27B" not in m, m))
    out = []
    out.append("| vLLM | torch | Qwen3.6-27B | Qwen3-1.7B | upstream vllm-lens on this vLLM |")
    out.append("|---|---|---|---|---|")
    for v in versions:
        cells = []
        torch = None
        stock_txt = []
        for m in ("Qwen/Qwen3.6-27B", "Qwen/Qwen3-1.7B"):
            rec = by[v]["models"].get(m)
            if not rec:
                cells.append("not run")
                continue
            torch = torch or (rec.get("versions", {}).get("installed") or {}).get("torch")
            st = rec["stages"]
            fork = [k for k in st if k.startswith("fork/") and k != "fork/cpu_tests"]
            ok = sum(1 for k in fork if st[k] == 0)
            inj = rec.get("injection", {})
            ro = rec.get("readout", {})
            ro_txt = ", ".join(f"{r['n_pass']}/{r['n_checks']}" for r in ro.values() if r["n_checks"]) or "n/a"
            cells.append(f"**{ok}/{len(fork)} stages ok**; CPU suites {'pass' if rec.get('cpu_tests', {}).get('rc') == 0 else 'FAIL'}; "
                         f"steering {rec['steering']['n_pass']}/{rec['steering']['n_checks']}, injection {inj.get('n_pass', '—')}/{inj.get('n_checks', '—')}, readout-vs-HF {ro_txt}")
            s_series = rec["steering"]["series"].get("stock_eager", {})
            if s_series and m.endswith("1.7B"):
                b = 512 if 512 in s_series else max(s_series)
                lens = by[v].get("stock_lens") or "1.1.0"
                stock_txt.append(f"{lens}: works, {s_series[b]['wall_s']:.0f} s per `generate()` at B={b}")
            if rec.get("stock110_rc") is not None and m.endswith("1.7B"):
                stock_txt.append("1.1.0: **silently captures nothing** (V2 model runner)" if rec["stock110_rc"] != 0 else "1.1.0: works")
        out.append(f"| **{v}** | {torch or '—'} | {cells[0]} | {cells[1]} | {'; '.join(stock_txt) or '—'} |")
    out.append("")
    out.append("Apples-to-apples generation throughput per vLLM version (same GPU type, prompts and settings; 96-token prompt, 40 new tokens, B = 512 / 1,024; one distinct steering vector per request for the fork rows; best of 2 repeats; each version in its own container, so treat ±10 % as noise):")
    out.append("")
    rows = [("ceiling_plain", "plain vLLM, its default (torch.compile + graphs; V2 model runner on ≥ 0.23)"),
            ("ceiling_plain_v1", "plain vLLM forced onto the V1 model runner (what the plugin uses)"),
            ("ceiling_graphs", "hook-compatible engine, no steering (compile off, decode graphs)"),
            ("fork_graphs", "**fork: one steering vector per request + CUDA graphs**"),
            ("fork_vectorized", "fork: one steering vector per request, eager"),
            ("stock_eager", "upstream vllm-lens, one vector per request (eager, forced)")]
    for m in models:
        vers_m = [v for v in versions if m in by[v]["models"]]
        out.append(f"**{m}**")
        out.append("")
        out.append("| configuration | " + " | ".join(f"vLLM {v}" for v in vers_m) + " |")
        out.append("|---|" + "|".join("---" for _ in vers_m) + "|")
        for key, label in rows:
            cells = []
            any_ = False
            for v in vers_m:
                s = by[v]["models"][m]["steering"]["series"].get(key, {})
                c512, c1024 = s.get(512), s.get(1024)
                any_ |= bool(c512 or c1024)
                cells.append(f"{_s(c512)} / {_s(c1024)}")
            if any_:
                out.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("")
        vers_r = [v for v in vers_m if by[v]["models"][m].get("readout")]
        if vers_r:
            out.append(f"Hidden-state readout on {m.split('/')[-1]} (seconds per 1,024 texts, prefill only, CUDA-graph engine): " + "; ".join(
                f"vLLM {v}: " + ", ".join(f"{k} {val:.2f} s" for k, val in ((by[v]['models'][m]['readout'].get('graphs') or by[v]['models'][m]['readout'].get('eager'))['per_1024']).items() if val)
                for v in vers_r))
            out.append("")
    return "\n".join(out)


def lora_block(lora: dict) -> str:
    out = []
    if not lora:
        return "(no LoRA results yet)"
    out.append("| model | vLLM | B | rank-64 LoRA on every request | LoRA-capable engine, no adapter | merged (same engine) | plain engine | **merged, plain engine** | `generate()` 40 tokens: LoRA → merged |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for ver in sorted(lora, key=_vkey):
        for model in sorted(lora[ver], key=lambda m: ("27B" not in m, m)):
            le, pe = lora[ver][model].get("lora_engine", {}), lora[ver][model].get("plain_engine", {})

            def best(res, cond, B, key="decode_step_ms"):
                vals = [r[key] for r in res.get("throughput", []) if r["condition"] == cond and int(r["batch"]) == B]
                return min(vals) if vals else None

            for B in (512, 1024):
                lo = best(le, "lora", B)

                def c(v):
                    if v is None:
                        return "—"
                    return f"{v:.1f} ms" + (f" ({100*(v-lo)/lo:+.0f}%)" if lo and v is not lo else "")

                wl, wm = best(le, "lora", B, "wall_s"), best(pe, "merged", B, "wall_s")
                out.append(f"| {model.split('/')[-1]} | {ver} | {B:,} | {c(lo)} | {c(best(le, 'nolora', B))} | {c(best(le, 'merged', B))} | {c(best(pe, 'plain', B))} | **{c(best(pe, 'merged', B))}** | "
                           f"{'—' if wl is None else f'{wl:.2f} s'} → {'—' if wm is None else f'{wm:.2f} s'} |")
    out.append("")
    out.append("Publish latency (replace the served adapter by the next one; the 27B has ~24 B LoRA-targeted parameters = 48 GB bf16, ~0.7 GB of A/B):")
    out.append("")
    out.append("| model | vLLM | `keep_base` | adapter source | worker time | RPC total | base copy |")
    out.append("|---|---|---|---|---:|---:|---:|")
    for ver in sorted(lora, key=_vkey):
        for model in sorted(lora[ver], key=lambda m: ("27B" not in m, m)):
            le, pe = lora[ver][model].get("lora_engine", {}), lora[ver][model].get("plain_engine", {})
            for p in le.get("publish", []):
                if p.get("phase") == "latency" and p.get("adapter") == "a1":
                    out.append(f"| {model.split('/')[-1]} | {ver} | {p['mode']} | {p['source']} | {p['publish_s']:.3f} s | {('—' if p.get('rpc_s') is None else f'{p[chr(114)+chr(112)+chr(99)+chr(95)+chr(115)]:.3f} s')} | {(p.get('base_bytes') or 0)/1e9:.1f} GB ({p.get('base_where') or '—'}) |")
            ipc = pe.get("ipc_push", {})
            if ipc.get("total_s") is not None:
                out.append(f"| {model.split('/')[-1]} | {ver} | EasyNLA-style full-matrix push (CUDA IPC → `load_weights`) | {ipc['bytes']/1e9:.1f} GB in {ipc['n_buckets']} buckets | {ipc['total_s']:.2f} s | {ipc['gb_per_s']:.0f} GB/s | — |")
    out.append("")
    out.append("Correctness (16 prompts, greedy 16 tokens, first-token top-20 log-probs):")
    out.append("")
    out.append("| model | vLLM | comparison | argmax equal | token agreement | max abs Δ log-prob | median |")
    out.append("|---|---|---|---:|---:|---:|---:|")
    for ver in sorted(lora, key=_vkey):
        for model in sorted(lora[ver], key=lambda m: ("27B" not in m, m)):
            le, pe = lora[ver][model].get("lora_engine", {}), lora[ver][model].get("plain_engine", {})
            corr = {**le.get("correctness", {}), **{f"plain engine: {k}": v for k, v in pe.get("correctness", {}).items()}}
            for k, v in corr.items():
                if isinstance(v, dict) and "argmax_equal" in v:
                    out.append(f"| {model.split('/')[-1]} | {ver} | {k} | {v['argmax_equal']}/{v['n']} | {v['mean_token_agreement']:.3f} | {v['max_abs_dlogprob_top20']:.3f} | {v['median_abs_dlogprob_top20']:.3f} |")
            dr = le.get("drift", {})
            if dr:
                ve, bs = dr.get("vs_exact_merge", {}), dr.get("base_after_subtract_unmerge", {})
                o = dr["outputs_vs_exact_merged"]
                out.append(f"| {model.split('/')[-1]} | {ver} | `keep_base=\"none\"`, {dr['n_publishes']} publishes, vs exact merge — weights: max {ve.get('max_diff_ulps', 0):.1f} bf16 spacings, rel-Frobenius {ve.get('rel_frobenius', 0):.1e}; base after subtract-unmerge rel-F {bs.get('rel_frobenius', 0):.1e} | {o['argmax_equal']}/{o['n']} | {o['mean_token_agreement']:.3f} | {o['max_abs_dlogprob_top20']:.3f} | {o['median_abs_dlogprob_top20']:.3f} |")
    return "\n".join(out)


def replace_block(text: str, tag: str, body: str) -> str:
    pat = re.compile(rf"(<!-- {tag}:BEGIN -->\n).*?(\n<!-- {tag}:END -->)", re.S)
    assert pat.search(text), f"missing {tag} markers"
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main() -> None:
    subprocess.run([sys.executable, str(HERE / "summarize_matrix.py"), *sorted(glob.glob(str(HERE / "results" / "matrix_*")))], check=True, capture_output=True)
    vm = json.loads((HERE / "results" / "version_matrix.json").read_text())
    lora = load_lora(sorted(glob.glob(str(HERE / "results" / "lora_*"))))
    subprocess.run([sys.executable, str(HERE / "plot_matrix.py"), "--out-dir", str(HERE)], check=True)
    readme = REPO / "README.md"
    text = readme.read_text()
    text = replace_block(text, "VERSIONS", versions_block(vm))
    text = replace_block(text, "LORA", lora_block(lora))
    readme.write_text(text)
    print("README blocks updated")


if __name__ == "__main__":
    main()
