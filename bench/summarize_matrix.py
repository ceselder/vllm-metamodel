"""Aggregate ``bench/results/matrix_<vllm>_<ts>/`` directories (``modal run bench/modal_bench.py::matrix``)
into one vLLM-version compatibility table.

    python bench/summarize_matrix.py bench/results/matrix_*            # -> bench/results/version_matrix.{json,md}

Per vLLM version x model: did every stage run (rc), CPU tests, steering correctness assertions
(compare.py), injection checks, readout checks, and the headline throughput rows -- plain vLLM
(default compile + graphs; on >= 0.23 also its V1 model runner), the fork with steering under CUDA
graphs / eager, stock vllm-lens -- at B in {512, 1024}, plus the prefill-only 1,024-text readout pass.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from compare import load_dir as load_steering_dir, summarize as summarize_steering  # noqa: E402

HEADLINE_B = (512, 1024)
READOUT_CONDS = ("nocap", "cap_all", "cap_last5", "read_last5", "exit_read_last5")


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return None


def _rows(res: dict) -> dict[str, dict[int, float]]:
    """condition -> batch -> best wall time (steering bench JSON)."""
    out: dict[str, dict[int, float]] = {}
    for r in res.get("throughput", []):
        b, w = int(r["batch"]), float(r["wall_s"])
        cur = out.setdefault(r["condition"], {}).get(b)
        out[r["condition"]][b] = w if cur is None else min(cur, w)
    return out


def summarize_dir(d: Path) -> dict[str, Any]:
    m = _load(d / "matrix.json") or {}
    ver = m.get("vllm") or re.match(r"matrix_([\d.]+)_", d.name).group(1)
    out: dict[str, Any] = {"vllm": ver, "dir": str(d), "stock_lens": m.get("stock_lens"), "gpu": m.get("gpu"), "models": {}}
    # steering (fork + stock) via compare.py, plus plain_v1 rows which compare.py does not know
    steer_dir = d / "steering"
    steering = summarize_steering(load_steering_dir(steer_dir)) if steer_dir.exists() else {"tables": {}, "assertions": []}
    for model, by_variant in (m.get("models") or {}).items():
        mk = model.replace("/", "__")
        rec: dict[str, Any] = {"stages": {}, "versions": {}}
        for variant, stages in by_variant.items():
            for tag, r in stages.items():
                if tag == "versions":
                    rec["versions"][variant] = r
                else:
                    rec["stages"][f"{variant}/{tag}"] = r.get("returncode")
        # CPU tests
        cpu = _load(d / f"{mk}__fork__cpu_tests.json")
        if cpu:
            mm = re.search(r"(\d+) passed", cpu.get("log_tail", "")) or re.search(r"(\d+) failed", cpu.get("log_tail", ""))
            rec["cpu_tests"] = {"rc": cpu.get("returncode"), "summary": (re.findall(r"=+ (.*?) =+\n?$", cpu.get("log_tail", "").strip()) or [""])[-1][:80] or (mm.group(0) if mm else "")}
        # steering series + assertions
        t = steering["tables"].get(model, {})
        series = {name: {int(b): v for b, v in by_b.items()} for name, by_b in t.get("series", {}).items()}
        for extra_tag, sname in (("plain_v1", "ceiling_plain_v1"), ("stock110", None)):
            pass
        pv1 = _load(steer_dir / f"{mk}__fork__plain_v1.json")
        if pv1 and pv1.get("result"):
            series["ceiling_plain_v1"] = {b: {"wall_s": w, "tok_per_s": b * pv1["result"]["max_tokens"] / w} for b, w in _rows(pv1["result"]).get("nosteer", {}).items()}
            rec["plain_runner"] = {"default": pv1["result"].get("model_runner_resolved")}
        pl = _load(steer_dir / f"{mk}__fork__plain.json")
        if pl and pl.get("result"):
            rec.setdefault("plain_runner", {})["plain"] = pl["result"].get("model_runner_resolved", "v1")
        s110 = _load(steer_dir / f"{mk}__stock110__eager.json")
        if s110 and s110.get("result"):
            rows = _rows(s110["result"])
            series["stock110_eager"] = {b: {"wall_s": w, "tok_per_s": b * s110["result"]["max_tokens"] / w} for b, w in rows.get("steer3d", {}).items()}
            probes = s110["result"].get("probes", {})
            rec["stock110_probe"] = probes.get("steer3d", {}).get("ok")
            rec["stock110_rc"] = s110.get("returncode")
        elif s110:
            rec["stock110_rc"] = s110.get("returncode")
            rec["stock110_error"] = (s110.get("log_tail") or "")[-400:]
        rec["steering"] = {"series": series, "n_pass": sum(a["ok"] for a in steering["assertions"] if a["model"] == model),
                           "n_checks": sum(1 for a in steering["assertions"] if a["model"] == model),
                           "failed": [a["check"][:90] + " :: " + a["detail"][:90] for a in steering["assertions"] if a["model"] == model and not a["ok"]]}
        for var in ("fork", "stock"):
            for eng in ("eager", "graphs", "plain"):
                j = _load(steer_dir / f"{mk}__{var}__{eng}.json")
                if j and j.get("result"):
                    rec["versions"].setdefault("resolved", {})[f"{var}_{eng}"] = j["result"].get("resolved_config")
                    rec["versions"]["installed"] = j["result"].get("versions")
                    rec["versions"]["variant_" + var] = j["result"].get("variant")
                    probes = j["result"].get("probes", {})
                    if "steer3d" in probes:
                        rec.setdefault("probes", {})[f"{var}_{eng}"] = {k: probes["steer3d"].get(k) for k in ("cos_delta_vs_v", "norm_ratio", "max_other_row_abs_delta", "ok")}
        # injection
        inj = _load(d / "injection" / "summary.json")
        if inj:
            checks = [c for c in inj.get("checks", []) if c.get("model") == model]
            rec["injection"] = {"n_pass": sum(1 for c in checks if c.get("ok")), "n_checks": sum(1 for c in checks if c.get("ok") is not None),
                                "failed": [f"{c['engine']} {c['case']}: {c['check']} {c['detail'][:80]}" for c in checks if c.get("ok") is False]}
        # readout
        ro: dict[str, Any] = {}
        for eng in ("eager", "graphs"):
            j = _load(d / "readout" / f"{mk}__fork_{eng}.json")
            if not j or not j.get("result"):
                continue
            res = j["result"]
            best: dict[str, dict[int, float]] = {}
            for r in res.get("rows", []):
                b = int(r["batch"])
                cur = best.setdefault(r["condition"], {}).get(b)
                best[r["condition"]][b] = min(cur, r["wall_s"]) if cur is not None else r["wall_s"]
            checks = res.get("checks", [])
            ro[eng] = {"per_1024": {c: best[c].get(1024) or best[c].get(max(best[c])) for c in READOUT_CONDS if c in best},
                       "n_pass": sum(1 for c in checks if c.get("ok")), "n_checks": sum(1 for c in checks if c.get("ok") is not None),
                       "failed": [str({k: v for k, v in c.items() if k in ("cond", "condition", "check", "detail")})[:120] for c in checks if c.get("ok") is False],
                       "capabilities": res.get("capabilities")}
        if ro:
            rec["readout"] = ro
        out["models"][model] = rec
    return out


def headline(rec: dict, name: str, b: int) -> str:
    v = rec.get("steering", {}).get("series", {}).get(name, {}).get(b)
    return f"{v['wall_s']:.2f} s ({v['tok_per_s']:,.0f} tok/s)" if v else "—"


def markdown(all_: list[dict]) -> str:
    lines = ["# vLLM version matrix (vllm-metamodels)", ""]
    models = sorted({m for a in all_ for m in a["models"]}, key=lambda m: ("27B" not in m, m))
    for model in models:
        lines += [f"## {model}", "", "| vLLM | fork stages | CPU tests | steering checks | injection checks | readout checks | plain vLLM (default) B=512 / 1024 | plain, V1 runner | fork steering + graphs B=512 / 1024 | fork steering eager B=512 | stock vllm-lens eager B=512 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in sorted(all_, key=lambda a: tuple(int(x) for x in a["vllm"].split("."))):
            rec = a["models"].get(model)
            if not rec:
                continue
            st = rec["stages"]
            fork_ok = sum(1 for k, rc in st.items() if k.startswith("fork/") and k != "fork/cpu_tests" and rc == 0)
            fork_n = sum(1 for k in st if k.startswith("fork/") and k != "fork/cpu_tests")
            cpu = rec.get("cpu_tests", {})
            ro = rec.get("readout", {})
            ro_txt = ", ".join(f"{e}: {r['n_pass']}/{r['n_checks']}" for e, r in ro.items()) or "—"
            stock = rec["steering"]["series"].get("stock_eager", {}).get(512)
            stock_txt = f"{stock['wall_s']:.1f} s" if stock else ("n/a" if rec.get("stock110_rc") is None else f"rc={rec.get('stock110_rc')}")
            lines.append(
                f"| {a['vllm']} | {fork_ok}/{fork_n} ok | {'ok' if cpu.get('rc') == 0 else cpu.get('summary', '—')} | "
                f"{rec['steering']['n_pass']}/{rec['steering']['n_checks']} | {rec.get('injection', {}).get('n_pass', '—')}/{rec.get('injection', {}).get('n_checks', '—')} | {ro_txt} | "
                f"{headline(rec, 'ceiling_plain', 512)} / {headline(rec, 'ceiling_plain', 1024)} | {headline(rec, 'ceiling_plain_v1', 512)} / {headline(rec, 'ceiling_plain_v1', 1024)} | "
                f"{headline(rec, 'fork_graphs', 512)} / {headline(rec, 'fork_graphs', 1024)} | {headline(rec, 'fork_vectorized', 512)} | {stock_txt} |"
            )
        lines.append("")
        for a in sorted(all_, key=lambda a: tuple(int(x) for x in a["vllm"].split("."))):
            rec = a["models"].get(model)
            if not rec:
                continue
            ro = rec.get("readout", {})
            for eng, r in ro.items():
                lines.append(f"- vLLM {a['vllm']} readout ({eng}), per 1,024 texts: " + ", ".join(f"{c} {v:.2f} s" for c, v in r["per_1024"].items() if v))
            for k in ("steering", "injection"):
                for f in rec.get(k, {}).get("failed", []):
                    lines.append(f"- vLLM {a['vllm']} {k} FAIL: {f}")
            for eng, r in ro.items():
                for f in r.get("failed", []):
                    lines.append(f"- vLLM {a['vllm']} readout ({eng}) FAIL: {f}")
            if rec.get("stock110_error"):
                lines.append(f"- vLLM {a['vllm']} stock vllm-lens 1.1.0: rc={rec.get('stock110_rc')} {rec['stock110_error'][-200:]!r}")
        lines.append("")
    return "\n".join(lines)


def main(dirs: list[str]) -> None:
    all_ = [summarize_dir(Path(d)) for d in dirs if (Path(d) / "matrix.json").exists()]
    # keep the latest dir per version
    latest: dict[str, dict] = {}
    for a in sorted(all_, key=lambda a: a["dir"]):
        latest[a["vllm"]] = a
    all_ = list(latest.values())
    out_json = HERE / "results" / "version_matrix.json"
    out_json.write_text(json.dumps(all_, indent=1))
    md = markdown(all_)
    (HERE / "results" / "version_matrix.md").write_text(md)
    print(md)
    print(f"[summarize_matrix] wrote {out_json} and version_matrix.md")


if __name__ == "__main__":
    main(sys.argv[1:])
