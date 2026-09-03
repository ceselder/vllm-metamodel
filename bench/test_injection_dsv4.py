#!/usr/bin/env python
"""GPU test matrix for vllm-metamodel on a hyper-connection (multi-stream) architecture:
DeepSeek-V4-Flash-0731 (mHC, hc_mult=4, fp8 + fp4 experts) on vLLM 0.27.1, TP4.

On mHC the residual stream at a layer boundary is a deferred fold of
``(x, residual[T, 4, D], post_mix, res_mix)``, so layer-output steering is undefined;
the headline path is EMBEDDING replacement (``EMBED_LAYER_INDEX``, ``mode="replace"``,
± ``norm_match``), which the fork applies in the layer-0 pre-hook on the plain ``[T, D]``
embedding tensor.  The matrix therefore checks:

  probe              hidden size / embedding norm, capabilities RPC reports multi_stream
  embed_replace      marker row == scale·v (bf16) / scale·‖e‖·v/‖v‖, other rows untouched,
                     rows_replaced == B, injection changes the next-token distribution;
                     PRESCALED variant (activations = alpha·v/‖v‖ fp32, scale 1) must be
                     bit-identical to the NLA session's reference arithmetic
                     (nla.utils.dsv4.scale_vector_to_alpha, imported from /repo if present)
  reference_impl     the NLA session's own worker-side pre-hook (nla.utils.dsv4_fast_hooks,
                     installed BEFORE ours so our layer -1 capture reads its write) must
                     produce the same bytes as ours
  mixed              embed-replace + NO-steer requests in one batch: steered rows correct,
                     unsteered rows and their log-probs identical to a clean batch
  chunked            marker in a NON-first prefill chunk (max_num_batched_tokens small)
  multi_stream_guard layer-output steering / capture must FAIL LOUDLY (ValueError before the
                     request reaches the engine; engine stays alive) -- never mis-inject
  throughput_dsv4    graphs: no-steer vs embed-replace vs embed-add, decode-step time from
                     wall(2T) - wall(T), interleaved repeats, hook-pass counts (graphs engage)

Output JSON has the same shape as bench/test_injection_modes.py so summarize() /
markdown_table() apply.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_injection_modes import _step_estimate, cos, log, prompt_ids, rel, unit_vectors  # noqa: E402

STEP_TOL, RESOLVABLE_SPREAD = 0.10, 0.10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash-0731")
    p.add_argument("--engine", choices=["eager", "graphs"], default="eager")
    p.add_argument("--out", required=True)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--kv-cache-dtype", default="fp8_ds_mla")
    p.add_argument("--moe-backend", default="deep_gemm")
    p.add_argument("--prompt-tokens", type=int, default=96)
    p.add_argument("--marker", type=int, default=10)
    p.add_argument("--chunk-marker", type=int, default=70)
    p.add_argument("--alpha", type=float, default=95.5, help="NLA injection alpha used by the DSv4 session")
    p.add_argument("--batches", default="64,512")
    p.add_argument("--tp-batches", default="512,1024")
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--max-num-seqs", type=int, default=1024)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096, help="small: prefill is chunked")
    p.add_argument("--tp-repeats", type=int, default=2)
    p.add_argument("--skip-throughput", action="store_true")
    p.add_argument("--only-throughput", action="store_true")
    p.add_argument("--skip-reference-impl", action="store_true")
    p.add_argument("--only-mixed", action="store_true", help="run only probe + mixed + effect_check (+ noise control)")
    return p.parse_args()


def ref_scale(v, alpha: float):
    """alpha * v / ||v|| in fp32 -- the NLA session's arithmetic (imported when /repo is present)."""
    try:
        from nla.utils.dsv4 import scale_vector_to_alpha  # type: ignore

        return scale_vector_to_alpha(v, alpha), "nla.utils.dsv4.scale_vector_to_alpha"
    except Exception:  # noqa: BLE001
        v = v.float()
        return v * (alpha / (v.norm() + 1e-9)), "inline alpha*v/(|v|+1e-9)"


def main() -> None:
    a = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if a.engine == "graphs":
        os.environ["VLLM_LENS_CUDA_GRAPHS"] = "1"
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    import vllm_lens
    from vllm_lens import EMBED_LAYER_INDEX, SteeringVector

    tok = AutoTokenizer.from_pretrained(a.model)
    P, T, M = a.prompt_tokens, a.max_tokens, a.marker
    ids = prompt_ids(tok, P)
    batches = [int(b) for b in a.batches.split(",") if b]
    tp_batches = [int(b) for b in a.tp_batches.split(",") if b]

    kw: dict[str, Any] = dict(
        model=a.model, tensor_parallel_size=a.tp, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
        max_model_len=P + 2 * T + 8, enable_prefix_caching=False, max_num_seqs=a.max_num_seqs,
        max_num_batched_tokens=a.max_num_batched_tokens, kv_cache_dtype=a.kv_cache_dtype,
        kernel_config={"moe_backend": a.moe_backend}, seed=0,
    )
    if a.engine == "eager":
        kw["enforce_eager"] = True
    else:
        kw["compilation_config"] = {"max_cudagraph_capture_size": min(a.max_num_seqs, 1024)}
    t0 = time.perf_counter()
    llm = LLM(**kw)
    up = time.perf_counter() - t0
    vc = llm.llm_engine.vllm_config
    cc = vc.compilation_config
    resolved = {
        "enforce_eager": bool(vc.model_config.enforce_eager),
        "compilation_mode": str(getattr(cc.mode, "name", cc.mode)),
        "cudagraph_mode": str(getattr(cc.cudagraph_mode, "name", cc.cudagraph_mode)),
        "n_capture_sizes": len(cc.cudagraph_capture_sizes or []),
        "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
        "max_num_seqs": vc.scheduler_config.max_num_seqs,
        "num_layers": vc.model_config.get_num_layers(vc.parallel_config),
        "tensor_parallel_size": vc.parallel_config.tensor_parallel_size,
        "kv_cache_dtype": str(vc.cache_config.cache_dtype),
        "moe_backend": a.moe_backend,
        "hc_mult": getattr(vc.model_config.hf_config, "hc_mult", None),
        "expert_dtype": getattr(vc.model_config.hf_config, "expert_dtype", None),
    }
    log(f"engine up {up:.0f}s vllm={vllm.__version__} vllm-lens={vllm_lens.__version__} {resolved}")

    result: dict[str, Any] = {
        "model": a.model, "engine": a.engine, "chunked": False, "multi_stream": True,
        "gpu": f"{torch.cuda.get_device_name(0)} x{a.tp}",
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__, "vllm_lens": vllm_lens.__version__},
        "resolved_config": resolved, "engine_up_s": up, "prompt_tokens": P, "marker": M, "alpha": a.alpha,
        "layer": EMBED_LAYER_INDEX, "hf_ref": False, "cases": {}, "checks": [],
    }
    checks: list[dict[str, Any]] = result["checks"]

    def check(case: str, name: str, ok: bool | None, detail: str) -> None:
        checks.append({"case": case, "check": name, "ok": ok, "detail": detail})
        log(f"  [{'PASS' if ok else ('n/a ' if ok is None else 'FAIL')}] {case}: {name}  {detail}")

    def dump() -> None:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)

    def stats() -> dict[str, int]:
        return llm.collective_rpc("steering_stats", args=(True,))[0]

    def sp(capture: bool, steer: list | None = None, max_tokens: int = 1, logprobs: int | None = None,
           temperature: float = 0.0, extra_more: dict | None = None) -> SamplingParams:
        extra: dict[str, Any] = {}
        if capture:
            extra["output_residual_stream"] = [EMBED_LAYER_INDEX]
        if steer:
            extra["apply_steering_vectors"] = steer
        if extra_more:
            extra.update(extra_more)
        return SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, logprobs=logprobs,
                              extra_args=extra or None)

    def gen(params):
        return llm.generate([{"prompt_token_ids": ids}] * len(params), params, use_tqdm=False)

    def emb(out):
        return out.activations["residual_stream"][0].float()  # (P, D) embedding stream (layer -1)

    def lp(out) -> dict[int, float]:
        return {int(t): float(v.logprob) for t, v in out.outputs[0].logprobs[0].items()}

    def subset(B: int) -> list[int]:
        return list(range(B)) if B <= 64 else sorted({i for i in range(0, B, max(1, B // 16))} | {B - 1})

    def sv_embed(u, scale: float, nm: bool, mode: str = "replace") -> SteeringVector:
        return SteeringVector(activations=u.reshape(1, 1, -1), layer_indices=[EMBED_LAYER_INDEX], scale=scale,
                              norm_match=nm, mode=mode, position_indices=[M])

    # ---- the NLA session's worker-side pre-hook, installed BEFORE ours (so our capture sees it) -----
    ref_impl = None
    if not a.skip_reference_impl and not a.only_throughput:
        try:
            from nla.utils.dsv4_fast_hooks import bound, fetch_nla_results, install_nla_hooks, set_nla_payload  # type: ignore

            r = llm.collective_rpc(bound(install_nla_hooks, 0, None))
            ref_impl = {"bound": bound, "set": set_nla_payload, "fetch": fetch_nla_results, "install": r}
            log(f"reference impl (nla.utils.dsv4_fast_hooks) installed on layer 0: {r}")
        except Exception as e:  # noqa: BLE001
            log(f"reference impl not available: {e!r}")
            result["reference_impl_error"] = repr(e)[:300]

    # ---- probe ---------------------------------------------------------------------------------------
    _ = stats()
    probe = gen([sp(True)])[0]
    E0 = emb(probe)
    D = E0.shape[-1]
    e_norm = float(E0[M].norm())
    caps = llm.collective_rpc("lens_capabilities")[0]
    result.update(hidden_dim=D, embed_marker_norm=e_norm, capabilities=caps)
    log(f"D={D} |embed[M]|={e_norm:.4f} caps={caps}")
    check("probe", "capabilities RPC reports a multi-stream (hyper-connection) architecture", bool(caps.get("multi_stream")), json.dumps(caps))
    ps = stats()
    check("probe", "embedding-stream capture pass: no hook errors", ps["errors"] == 0 and ps["embed_errors"] == 0, json.dumps(ps))
    U = unit_vectors(max(batches + tp_batches), D)
    _, ref_name = ref_scale(U[0], a.alpha)
    result["reference_arithmetic"] = ref_name

    clean_cache: dict[int, Any] = {}

    def _lp_maxdiff(l1: dict[int, float], l2: dict[int, float]) -> float:
        common = set(l1) & set(l2)
        return max((abs(l1[t] - l2[t]) for t in common), default=float("nan"))

    def clean(B: int):
        """Clean batch of B requests: embedding capture on the subset + top-20 log-probs on all, run TWICE
        so the engine's own batch-to-batch log-prob noise (fp8 MoE forward is not bit-deterministic) is
        known -- every log-prob comparison is judged against it."""
        if B not in clean_cache:
            S = subset(B)
            outs = gen([sp(i in S, None, logprobs=20) for i in range(B)])
            outs2 = gen([sp(False, None, logprobs=20) for i in range(B)])
            lps, lps2 = [lp(o) for o in outs], [lp(o) for o in outs2]
            noise = max(_lp_maxdiff(a_, b_) for a_, b_ in zip(lps, lps2))
            flips = sum(int(o.outputs[0].token_ids[0]) != int(o2.outputs[0].token_ids[0]) for o, o2 in zip(outs, outs2)) / B
            clean_cache[B] = {"emb": {i: emb(outs[i]) for i in S}, "lp": lps, "argmax": [int(o.outputs[0].token_ids[0]) for o in outs],
                              "noise": noise, "argmax_flip_noise": flips}
            log(f"  clean-vs-clean control B={B}: top-20 log-prob max|Δ| = {noise:.3f}, argmax flips {flips:.1%} (engine nondeterminism floor)")
            _ = stats()
        return clean_cache[B]

    def effect(outs, C, idxs) -> tuple[float, float]:
        """(mean over idxs of max|Δ top-20 logprob| vs clean, fraction whose argmax changed)."""
        ds, ch = [], 0
        for i in idxs:
            l_s, l_c = lp(outs[i]), C["lp"][i]
            common = set(l_s) & set(l_c)
            ds.append(max((abs(l_s[t] - l_c[t]) for t in common), default=float("nan")))
            ch += int(int(outs[i].outputs[0].token_ids[0]) != C["argmax"][i])
        return float(sum(ds) / len(ds)), ch / len(idxs)

    def check_rows(case, tag, S, C, outs, scale, nm, replaced, mode="replace"):
        r_row, c_row, others, exact = [], [], [], []
        for i in S:
            es, ec = emb(outs[i]), C["emb"][i]
            if replaced(i):
                if mode == "replace":
                    tgt = (U[i] * (scale * float(ec[M].norm()) if nm else scale)).to(torch.bfloat16).float()
                else:  # add
                    tgt = (ec[M] + (U[i] * (scale * float(ec[M].norm()) if nm else scale)).to(torch.bfloat16).float())
                r_row.append(rel(es[M], tgt))
                c_row.append(cos(es[M] - (0 if mode == "replace" else ec[M]), U[i]))
                o = es - ec
                o[M] = 0
                others.append(float(o.abs().max()))
            else:
                exact.append(float((es - ec).abs().max()))
        rec = {"n_checked": len(S), "scale": scale, "norm_match": nm,
               "max_rel_err_marker_row": max(r_row) if r_row else None, "min_cos_marker_row": min(c_row) if c_row else None,
               "max_other_embed_row_abs_delta": max(others) if others else None,
               "max_unsteered_request_abs_delta": max(exact) if exact else None}
        if r_row:
            what = ("scale·‖e‖·v/‖v‖" if nm else "scale·v") if mode == "replace" else ("e + scale·‖e‖·v/‖v‖" if nm else "e + scale·v")
            check(case, f"{tag}: embedding-stream marker row == {what} per request (bf16 rel<1e-2, cos>0.9999)",
                  max(r_row) < 1e-2 and min(c_row) > 0.9999, f"max rel={max(r_row):.2e} min cos={min(c_row):.6f} over {len(r_row)} requests")
            check(case, f"{tag}: every other embedding row of steered requests untouched", max(others) == 0.0, f"max|Δ|={max(others):.2e}")
        if exact:
            check(case, f"{tag}: unsteered requests' embedding stream identical to the clean batch", max(exact) == 0.0, f"max|Δ|={max(exact):.2e}")
        return rec

    # ---- embed_replace ------------------------------------------------------------------------------
    def case_embed(case: str, B: int, nm: bool, scale: float, marker: int | None = None) -> dict:
        nonlocal M
        if marker is not None:
            M = marker
        S = subset(B)
        C = clean(B)
        outs = gen([sp(i in S, [sv_embed(U[i], scale, nm)], logprobs=20) for i in range(B)])
        st = stats()
        rec = check_rows(case, f"B={B} norm_match={nm} scale={scale:.3g} marker={M}", S, C, outs, scale, nm, lambda i: True)
        eff, frac = effect(outs, C, range(B))
        rec.update(batch=B, marker=M, stats=st, logprob_effect_mean=eff, argmax_changed_frac=frac,
                   steps_planned=st["steps_planned"], clean_noise=C["noise"], clean_argmax_flip_noise=C["argmax_flip_noise"])
        check(case, f"B={B} norm_match={nm} scale={scale:.3g}: rows_replaced == B, embed passes ≥ 1, no errors",
              st["rows_replaced"] == B and st["embed_apply_steps"] >= 1 and st["errors"] == 0 and st["embed_errors"] == 0, json.dumps(st))
        check(case, f"B={B} norm_match={nm} scale={scale:.3g}: next-token distribution at the last prompt position vs clean (informational; "
                    f"marker is {P - 1 - M} tokens back)", None,
              f"mean max|Δ top-20 logprob| = {eff:.3f} (clean-vs-clean floor {C['noise']:.3f}); argmax changed {frac:.0%} (floor {C['argmax_flip_noise']:.0%})")
        result["cases"].setdefault(case, []).append(rec)
        dump()
        if marker is not None:
            M = a.marker
        return rec

    def case_embed_prescaled(B: int) -> dict:
        """activations = alpha·v/‖v‖ (fp32, the reference arithmetic), scale=1: the written bf16 row must be
        bit-identical to reference.to(bf16) -- exactly what the NLA session's hook writes."""
        case = "embed_replace_prescaled"
        S = subset(B)
        C = clean(B)
        vecs = [ref_scale(U[i], a.alpha)[0] for i in range(B)]
        outs = gen([sp(i in S, [SteeringVector(activations=vecs[i].reshape(1, 1, -1), layer_indices=[EMBED_LAYER_INDEX],
                                                scale=1.0, norm_match=False, mode="replace", position_indices=[M])], logprobs=20)
                    for i in range(B)])
        st = stats()
        diffs = [float((emb(outs[i])[M] - vecs[i].to(torch.bfloat16).float()).abs().max()) for i in S]
        others = []
        for i in S:
            o = emb(outs[i]) - C["emb"][i]
            o[M] = 0
            others.append(float(o.abs().max()))
        rec = {"batch": B, "scale": a.alpha, "norm_match": False, "n_checked": len(S), "ref_max_abs_diff": max(diffs),
               "max_other_embed_row_abs_delta": max(others), "stats": st, "rows": {i: emb(outs[i])[M].tolist() for i in S[:4]}}
        check(case, f"B={B}: marker row BIT-IDENTICAL to the reference arithmetic {ref_name}(v, alpha={a.alpha}).to(bf16)",
              max(diffs) == 0.0, f"max|Δ|={max(diffs):.2e} over {len(S)} requests; other rows max|Δ|={max(others):.2e}")
        check(case, f"B={B}: rows_replaced == B, no errors", st["rows_replaced"] == B and st["errors"] == 0, json.dumps(st))
        result["cases"].setdefault(case, []).append(rec)
        dump()
        return rec

    # ---- reference implementation on the same engine ------------------------------------------------
    def case_reference_impl(B: int, ours: dict) -> None:
        if ref_impl is None:
            return
        case = "reference_impl"
        S = subset(B)
        C = clean(B)
        payload = {i: {"vec": ref_scale(U[i], a.alpha)[0].cpu(), "marker": M, "capture_last": None} for i in range(B)}
        n = llm.collective_rpc(ref_impl["bound"](ref_impl["set"], payload))
        outs = gen([sp(i in S, None, logprobs=20, extra_more={"_nla_tag": i}) for i in range(B)])
        res = llm.collective_rpc(ref_impl["fetch"])[0]
        st = stats()
        applied = [res.get(i, {}).get("applied", 0) for i in range(B)]
        diffs_ref = [float((emb(outs[i])[M] - payload[i]["vec"].to(torch.bfloat16).float()).abs().max()) for i in S]
        diffs_ours = [float((emb(outs[i])[M] - torch.tensor(ours["rows"][i])).abs().max()) for i in S if i in ours["rows"]]
        others = []
        for i in S:
            o = emb(outs[i]) - C["emb"][i]
            o[M] = 0
            others.append(float(o.abs().max()))
        eff, frac = effect(outs, C, range(B))
        rec = {"batch": B, "scale": a.alpha, "n_checked": len(S), "applied_min": min(applied), "applied_max": max(applied),
               "ref_max_abs_diff": max(diffs_ref), "ref_impl_max_abs_diff": max(diffs_ours) if diffs_ours else None,
               "max_other_embed_row_abs_delta": max(others), "logprob_effect_mean": eff, "argmax_changed_frac": frac,
               "dispatch_err": res.get("_dispatch_err"), "payload_n": n}
        check(case, f"B={B}: NLA session's fast pre-hook applied exactly once per request", min(applied) == 1 == max(applied) and res.get("_dispatch_err") is None,
              f"applied ∈ [{min(applied)}, {max(applied)}], dispatch_err={res.get('_dispatch_err')}")
        check(case, f"B={B}: row written by the NLA session's hook == reference arithmetic (bf16), read back through our layer -1 capture",
              max(diffs_ref) == 0.0, f"max|Δ|={max(diffs_ref):.2e}")
        if diffs_ours:
            check(case, f"B={B}: NLA session's hook and vllm-metamodel embed-replace (prescaled) write IDENTICAL bytes", max(diffs_ours) == 0.0,
                  f"max|Δ|={max(diffs_ours):.2e} over {len(diffs_ours)} requests")
        check(case, f"B={B}: our hooks stayed passive (rows_replaced == 0) and captured cleanly", st["rows_replaced"] == 0 and st["errors"] == 0, json.dumps(st))
        result["cases"].setdefault(case, []).append(rec)
        dump()

    # ---- mixed: embed-replace + no-steer -------------------------------------------------------------
    def case_mixed(B: int) -> None:
        case = "mixed"
        S = subset(B)
        C = clean(B)
        outs = gen([sp(i in S, [sv_embed(U[i], a.alpha, False)] if i % 2 == 0 else None, logprobs=20) for i in range(B)])
        st = stats()
        rec = check_rows(case, f"B={B} (even = embed-replace, odd = no steering)", S, C, outs, a.alpha, False, lambda i: i % 2 == 0)
        odd = [i for i in range(B) if i % 2 == 1]
        d_odd = max(_lp_maxdiff(lp(outs[i]), C["lp"][i]) for i in odd)
        flips_odd = sum(int(outs[i].outputs[0].token_ids[0]) != C["argmax"][i] for i in odd) / len(odd)
        eff, frac = effect(outs, C, [i for i in range(B) if i % 2 == 0])
        tol = max(1e-3, 1.5 * C["noise"] + 0.01)
        rec.update(batch=B, stats=st, nosteer_logprob_maxdiff=d_odd, nosteer_argmax_flip=flips_odd, clean_noise=C["noise"],
                   clean_argmax_flip_noise=C["argmax_flip_noise"], logprob_effect_mean=eff, argmax_changed_frac=frac)
        check(case, f"B={B}: unsteered requests' next-token log-probs within the engine's clean-vs-clean noise floor "
                    f"(max|Δ| ≤ max(1e-3, 1.5·floor + 0.01)) and argmax flips ≤ floor + 1/{len(odd)}",
              d_odd <= tol and flips_odd <= C["argmax_flip_noise"] + 1.0 / len(odd),
              f"max|Δ|={d_odd:.3f} vs floor {C['noise']:.3f} (tol {tol:.3f}); argmax flips {flips_odd:.1%} vs floor {C['argmax_flip_noise']:.1%}; "
              f"steered requests: mean max|Δ|={eff:.3f}, argmax changed {frac:.0%}")
        check(case, f"B={B}: rows_replaced == B/2, no errors", st["rows_replaced"] == B // 2 and st["errors"] == 0, json.dumps(st))
        result["cases"].setdefault(case, []).append(rec)
        dump()

    # ---- effect probe: marker 3 tokens before the predicted position ---------------------------------
    def case_effect(B: int) -> None:
        case = "effect_check"
        M2 = P - 3
        set_marker(M2)
        try:
            S = subset(B)
            C = clean(B)
            outs = gen([sp(i in S, [sv_embed(U[i], a.alpha, False)], logprobs=20) for i in range(B)])
            st = stats()
            rec = check_rows(case, f"B={B} marker={M2} scale={a.alpha}", S, C, outs, a.alpha, False, lambda i: True)
            eff, frac = effect(outs, C, range(B))
            rec.update(batch=B, marker=M2, scale=a.alpha, stats=st, logprob_effect_mean=eff, argmax_changed_frac=frac, clean_noise=C["noise"],
                       clean_argmax_flip_noise=C["argmax_flip_noise"])
            check(case, f"B={B}: replacing the embedding {P - 1 - M2} tokens before the predicted position visibly changes the next-token "
                        f"distribution (argmax changed for > clean-vs-clean flip floor + 25% of requests)",
                  frac > C["argmax_flip_noise"] + 0.25,
                  f"argmax changed {frac:.0%} (floor {C['argmax_flip_noise']:.0%}); mean max|Δ top-20 logprob| {eff:.3f} (floor {C['noise']:.3f})")
            check(case, f"B={B}: rows_replaced == B, no errors", st["rows_replaced"] == B and st["errors"] == 0, json.dumps(st))
            result["cases"].setdefault(case, []).append(rec)
            dump()
        finally:
            set_marker(a.marker)

    # ---- multi-stream guard: layer-output steering / capture must fail loudly, engine alive ------------
    def case_guard() -> None:
        case = "multi_stream_guard"

        def alive() -> bool:
            try:
                o = gen([sp(False)])
                return len(o) == 1 and len(o[0].outputs[0].token_ids) == 1
            except Exception:  # noqa: BLE001
                return False

        attempts = [
            ("steer add at layer 1 (norm_match)", lambda: gen([sp(False, [SteeringVector(activations=U[0].reshape(1, 1, -1), layer_indices=[1],
                                                                                          scale=1.0, norm_match=True, position_indices=[M])])])),
            ("steer replace at layer 5", lambda: gen([sp(False, [SteeringVector(activations=U[0].reshape(1, 1, -1), layer_indices=[5], scale=1.0,
                                                                                mode="replace", position_indices=[M])])])),
            ("capture output_residual_stream=[1]", lambda: gen([SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": [1]})])),
            ("capture output_residual_stream=True", lambda: gen([SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": True})])),
        ]
        recs = []
        for name, fn in attempts:
            err = None
            try:
                fn()
                outcome = "SILENT (no error)"
            except ValueError as e:
                err, outcome = str(e)[:300], "ValueError"
            except Exception as e:  # noqa: BLE001
                err, outcome = repr(e)[:300], type(e).__name__
            ok_alive = alive()
            st = stats()
            recs.append({"attempt": name, "outcome": outcome, "error": err, "engine_alive_after": ok_alive, "stats": st})
            check(case, f"{name}: refused with a ValueError BEFORE reaching the engine (never silently mis-injected)",
                  outcome == "ValueError" and err is not None and "hyper-connection" in err, f"{outcome}: {err}")
            check(case, f"{name}: engine alive afterwards, no rows steered/replaced", ok_alive and st["rows_steered"] == 0 and st["rows_replaced"] == 0
                  and st["unsupported_layer_output"] == 0, f"alive={ok_alive} {json.dumps(st)}")
        result["cases"][case] = recs
        dump()

    # ---- throughput ----------------------------------------------------------------------------------
    def case_throughput() -> None:
        case = "throughput_dsv4"

        def gp(extra, n_tok: int):
            return SamplingParams(temperature=1.0, top_p=1.0, max_tokens=n_tok, min_tokens=n_tok, ignore_eos=True, extra_args=extra)

        conds = {
            "nosteer": lambda i: None,
            "embed_replace": lambda i: {"apply_steering_vectors": [sv_embed(U[i], a.alpha, False)]},
            "embed_add": lambda i: {"apply_steering_vectors": [sv_embed(U[i], 1.0, True, mode="add")]},
        }
        for cond, mk in conds.items():
            gen([gp(mk(i), T) for i in range(8)])
            _ = stats()
        rows: dict[str, dict[int, dict]] = {}
        for B in tp_batches:
            for rep_i in range(max(1, a.tp_repeats)):
                for cond, mk in conds.items():
                    t1 = time.perf_counter()
                    outs = gen([gp(mk(i), T) for i in range(B)])
                    w1 = time.perf_counter() - t1
                    st1 = stats()
                    n1 = sum(len(o.outputs[0].token_ids) for o in outs)
                    t1 = time.perf_counter()
                    gen([gp(mk(i), 2 * T) for i in range(B)])
                    w2 = time.perf_counter() - t1
                    st2 = stats()
                    r = rows.setdefault(cond, {}).setdefault(B, {"wall_s": math.inf, "wall_2T_s": math.inf, "repeats": []})
                    r["repeats"].append({"wall_s": w1, "wall_2T_s": w2})
                    r["wall_s"], r["wall_2T_s"] = min(r["wall_s"], w1), min(r["wall_2T_s"], w2)
                    r.update(tok_per_s=n1 / r["wall_s"], gen_tokens=n1, decode_step_ms=(r["wall_2T_s"] - r["wall_s"]) / T * 1000.0,
                             prefill_plus_overhead_s=2 * r["wall_s"] - r["wall_2T_s"],
                             hook_passes=st1["steps_fast_idle"] + st1["steps_planned"], hook_passes_2T=st2["steps_fast_idle"] + st2["steps_planned"],
                             stats=st1)
                    log(f"  rep{rep_i} {cond:14s} B={B:5d}: {w1:6.2f}s ({T} tok) {w2:6.2f}s ({2 * T} tok) -> best decode step {r['decode_step_ms']:.2f} ms, "
                        f"prefill+overhead {r['prefill_plus_overhead_s']:.2f}s, hook passes={r['hook_passes']}")
        n_prefill = math.ceil(max(tp_batches) * P / a.max_num_batched_tokens)
        result["cases"][case] = {"rows": rows, "T": T, "repeats": a.tp_repeats, "max_num_batched_tokens": a.max_num_batched_tokens,
                                 "baseline_series": None}
        for B in tp_batches:
            n_pre = math.ceil(B * P / a.max_num_batched_tokens)
            ns = rows["nosteer"][B]
            ns_ms, ns_sp = _step_estimate(ns, T)
            resolvable = ns_sp is None or ns_sp <= RESOLVABLE_SPREAD
            for name, key in (("embed-replace", "embed_replace"), ("embed-add (norm_match)", "embed_add")):
                r = rows[key][B]
                ms, _ = _step_estimate(r, T)
                detail = (f"{ms:.2f} ms vs {ns_ms:.2f} ms per decode step ({ms / ns_ms - 1:+.1%}); wall {T} tok {r['wall_s']:.2f}s vs {ns['wall_s']:.2f}s "
                          f"({r['wall_s'] / ns['wall_s'] - 1:+.1%}, {r['wall_s'] - ns['wall_s']:+.3f}s per call = shipping {B} vectors + prefill hook)")
                if resolvable:
                    check(case, f"B={B}: {name} decode-step time within {STEP_TOL:.0%} of no-steering (same engine, hooks installed)", ms <= (1 + STEP_TOL) * ns_ms, detail)
                else:
                    check(case, f"B={B}: {name} decode-step time vs no-steering -- NOT RESOLVABLE (control repeat spread {ns_sp:.0%}; informational)", None, detail)
            if a.engine == "graphs":
                er = rows["embed_replace"][B]
                check(case, f"B={B}: CUDA graphs engage with embed-replace requests (hook passes ≤ prefill passes {n_pre} + {T // 4}; eager would be ≈ {n_pre + T})",
                      er["hook_passes"] <= n_pre + T // 4, f"hook passes: embed={er['hook_passes']} add={rows['embed_add'][B]['hook_passes']} nosteer={ns['hook_passes']}")
        result["n_prefill_passes_max"] = n_prefill
        dump()

    def set_marker(m: int) -> None:
        nonlocal M
        M = m

    # ---- run -----------------------------------------------------------------------------------------
    if a.only_mixed:
        case_mixed(batches[0])
        case_effect(batches[0])
        result["all_pass"] = all(c["ok"] is not False for c in checks)
        dump()
        log(f"done (only-mixed): {sum(c['ok'] is True for c in checks)}/{sum(c['ok'] is not None for c in checks)} gated checks pass")
        return
    if not a.only_throughput:
        prescaled = {}
        for B in batches:
            case_embed("embed_replace", B, False, e_norm)
            case_embed("embed_replace", B, False, a.alpha)
            case_embed("embed_replace", B, True, 1.0)
            prescaled[B] = case_embed_prescaled(B)
        case_reference_impl(batches[0], prescaled[batches[0]])
        case_mixed(batches[0])
        case_effect(batches[0])
        for B in batches:
            rec = case_embed("chunked_m%d_embed_replace" % a.chunk_marker, B, False, a.alpha, marker=a.chunk_marker)
            n_pre = math.ceil(B * P / a.max_num_batched_tokens)
            check("chunked", f"B={B} marker {a.chunk_marker} (NON-first {a.max_num_batched_tokens}-token chunk for the split requests): prefill chunked "
                             f"(planned passes ≥ {n_pre}), rows_replaced == B", rec["steps_planned"] >= n_pre and rec["stats"]["rows_replaced"] == B,
                  f"planned passes={rec['steps_planned']} rows_replaced={rec['stats']['rows_replaced']} embed passes={rec['stats']['embed_apply_steps']}")
        case_guard()
    if not a.skip_throughput:
        case_throughput()
    result["all_pass"] = all(c["ok"] is not False for c in checks)
    dump()
    log(f"done: {sum(c['ok'] is True for c in checks)}/{sum(c['ok'] is not None for c in checks)} gated checks pass, "
        f"{sum(c['ok'] is None for c in checks)} informational")


if __name__ == "__main__":
    main()
