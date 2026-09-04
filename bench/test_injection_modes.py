#!/usr/bin/env python
"""GPU test matrix for the injection modes of vllm-metamodels (1×GPU, Modal).

Karvonen-style norm-matched addition (``norm_match=True, scale=coeff`` ==
``h + coeff·‖h‖·v/‖v‖`` at a decoder layer's output, checked against an HF
transformers reference that uses the exact ``mxf/inject.py`` hook), embedding
replacement (``EMBED_LAYER_INDEX`` + ``mode="replace"``), replacement at a
fused-residual layer output, mixed batches, chunked prefill with the marker in
a non-first chunk, and a throughput check that CUDA graphs still engage --
every case with a DISTINCT vector per request in the batch.

Stages (each its own subprocess; ``bench/modal_bench.py::test_injection`` drives them):

  --stage hf-ref    HF reference -> ``--out ref.pt``: layer-L output at the marker
                    with and without the mxf hook for the first N_HF unit vectors,
                    greedy continuations, next-token log-probs, embedding rows.
  --stage vllm      one vLLM engine (``--engine eager|graphs``, optionally
                    ``--chunked``) -> ``--out results.json`` with every number
                    and PASS/FAIL checks (``summarize`` turns a directory of these
                    into ``summary.json`` + a Markdown table).

Cases (vllm stage)
  karvonen_add   layer L, norm_match=True, scale=coeff in --coeffs, B in --batches
  layer_replace  mode="replace" at layer L (fused-residual: both halves rewritten)
  embed_replace  EMBED_LAYER_INDEX + mode="replace", norm_match off and on
  mixed          half the batch embed-replace, half karvonen-add at layer L
  chunked        (--chunked engine: max_num_batched_tokens small) karvonen_add +
                 embed_replace with the marker in a non-first prefill chunk
  throughput     nosteer / karvonen_add / embed_replace at B in --tp-batches,
                 40 new tokens, vs bench/results_summary.json (fork_graphs)
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
from bench_steering import PROMPT_TEXT  # noqa: E402

N_HF = 4  # vectors covered by the HF reference (first N_HF of the shared generator)
SEED = 1234


def log(msg: str) -> None:
    print(f"[inj {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_llm(LLM, kw: dict, log=print):
    """Build the engine, dropping engine kwargs this vLLM version does not know
    (``language_model_only`` appeared in 0.19, ``attention_backend`` values differ
    between releases) so one script runs on vLLM 0.16 .. 0.28.  Returns ``(llm, kw)``."""
    kw = dict(kw)
    for _ in range(4):
        try:
            return LLM(**kw), kw
        except (TypeError, ValueError, KeyError, RuntimeError) as e:
            msg = str(e)
            drop = next((k for k in ("language_model_only", "attention_backend", "gdn_prefill_backend")
                         if k in kw and (k in msg or (k == "attention_backend" and "backend" in msg.lower()))), None)
            if drop is None:
                raise
            log(f"engine arg {drop!r}={kw[drop]!r} rejected by this vLLM ({msg[:160]}); retrying without it")
            kw.pop(drop)
    raise RuntimeError("engine construction failed after dropping compatibility kwargs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--stage", choices=["hf-ref", "vllm"], required=True)
    p.add_argument("--engine", choices=["eager", "graphs"], default="eager")
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default="", help="hf-ref .pt (vllm stage)")
    p.add_argument("--baseline", default="", help="bench/results_summary.json for the throughput check")
    p.add_argument("--prompt-tokens", type=int, default=96)
    p.add_argument("--marker", type=int, default=10)
    p.add_argument("--inject-layer", type=int, default=1)
    p.add_argument("--coeffs", default="1.0,4.0")
    p.add_argument("--batches", default="64,512")
    p.add_argument("--tp-batches", default="512,1024")
    p.add_argument("--max-tokens", type=int, default=40, help="throughput: new tokens per request")
    p.add_argument("--max-num-seqs", type=int, default=1024)
    p.add_argument("--chunked", action="store_true", help="small max_num_batched_tokens engine; marker in a later chunk")
    p.add_argument("--chunk-tokens", type=int, default=64)
    p.add_argument("--chunk-batch", type=int, default=16)
    p.add_argument("--chunk-marker", type=int, default=70,
                   help="chunked engine: second marker that lies in a NON-first prefill chunk (> --chunk-tokens)")
    p.add_argument("--skip-throughput", action="store_true")
    p.add_argument("--only-throughput", action="store_true", help="standard engine: run only the throughput case")
    p.add_argument("--tp-repeats", type=int, default=2, help="throughput: repeats per measurement (min is kept)")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--attention-backend", default="")
    p.add_argument("--language-model-only", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def prompt_ids(tok, n: int) -> list[int]:
    ids = tok(PROMPT_TEXT, add_special_tokens=False)["input_ids"]
    while len(ids) < n:
        ids = ids + ids
    return [int(t) for t in ids[:n]]


def unit_vectors(n: int, d: int):
    import torch

    g = torch.Generator().manual_seed(SEED)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


def cos(a, b) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0))


def rel(a, b) -> float:
    """max|a-b| / max|b| (b = reference)."""
    den = float(b.float().abs().max()) or 1.0
    return float((a.float() - b.float()).abs().max()) / den


def find_layers(model):
    m = model
    for path in (
        "model.language_model.layers",
        "model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "model.decoder.layers",
    ):
        cur = m
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


# ---------------------------------------------------------------------------
# stage 1: HF reference (exact mxf/inject.py hook)
# ---------------------------------------------------------------------------


def make_inject_hook(vecs, positions, coeff, device, dtype, mode="add"):
    """Verbatim copy of maemm-pub/mxf/inject.py::make_inject_hook (the trainer's hook)."""
    import torch

    if len(vecs) != len(positions):
        raise ValueError(f"{len(vecs)} vector rows != {len(positions)} position rows")
    counts = [len(p) for p in positions]
    if any(v.shape[0] != n for v, n in zip(vecs, counts)):
        raise ValueError("each vector row must have one vector per marker position")
    normed = torch.nn.functional.normalize(torch.cat(vecs).to(device, dtype), dim=-1)
    rows = torch.repeat_interleave(torch.arange(len(vecs), device=device), torch.tensor(counts, device=device))
    cols = torch.tensor([p for row in positions for p in row], device=device)

    def hook(_module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[1] <= 1:  # decode step (KV-cache): marker already injected at prefill
            return out
        if h.shape[0] != len(vecs):
            raise RuntimeError(f"inject batch {h.shape[0]} != {len(vecs)} vector rows")
        base = h[rows, cols]
        if mode == "add":
            scale = base.norm(dim=-1, keepdim=True) * coeff
            h[rows, cols] = base + (normed * scale).to(h.dtype).detach()
        elif mode == "replace":
            h = h.clone()
            h[rows, cols] = (normed * coeff).to(h.dtype).detach()
        else:
            raise ValueError(f"unknown injection mode: {mode}")
        return (h, *out[1:]) if isinstance(out, tuple) else h

    return hook


def hf_ref(a: argparse.Namespace) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = prompt_ids(tok, a.prompt_tokens)
    cfg = AutoConfig.from_pretrained(a.model)
    cls = AutoModelForImageTextToText if getattr(cfg, "text_config", None) is not None else AutoModelForCausalLM
    t0 = time.perf_counter()
    try:
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    except (ValueError, ImportError) as e:  # no `accelerate`: load on CPU, then move
        log(f"device_map load failed ({str(e)[:80]}...); loading on CPU then .to('cuda')")
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    log(f"HF {cls.__name__} up in {time.perf_counter() - t0:.0f}s")
    layers = find_layers(model)
    L, M = a.inject_layer, a.marker
    inp = torch.tensor([ids] * N_HF, device="cuda")
    attn = torch.ones_like(inp)
    coeffs = [float(c) for c in a.coeffs.split(",") if c]

    cap: dict[str, Any] = {}

    def cap_out(_m, _i, out):
        cap["L"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()

    def cap_in(_m, args, kwargs):
        hs = kwargs.get("hidden_states")
        if hs is None:
            hs = next(t for t in args if isinstance(t, torch.Tensor) and t.dim() == 3)
        cap["E"] = hs.detach().float().cpu()

    h_in = layers[0].register_forward_pre_hook(cap_in, with_kwargs=True)
    with torch.no_grad():
        h_out = layers[L].register_forward_hook(cap_out)
        out = model(input_ids=inp, attention_mask=attn)
        h_out.remove()
        lp_clean = torch.log_softmax(out.logits[:, -1].float(), dim=-1).cpu()
        h_clean = cap["L"]  # (B, T, D)
        embed = cap["E"]
        D = h_clean.shape[-1]
        U = unit_vectors(N_HF, D)
        vecs = [U[i : i + 1] for i in range(N_HF)]
        greedy_clean = model.generate(
            inp, attention_mask=attn, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id
        )[:, inp.shape[1] :].cpu().tolist()
        ref: dict[str, Any] = {
            "model": a.model,
            "layer": L,
            "marker": M,
            "prompt_ids": ids,
            "hidden_dim": D,
            "n_hf": N_HF,
            "coeffs": coeffs,
            "hf_class": cls.__name__,
            "embed_marker": embed[:, M].clone(),  # (B, D) embedding stream at the marker
            "embed_norm": float(embed[0, M].norm()),
            "h_clean_marker": h_clean[:, M].clone(),  # (B, D) full residual stream after layer L
            "h_clean_marker_norm": float(h_clean[0, M].norm()),
            "logprobs_clean": lp_clean.half(),  # (B, V)
            "greedy8_clean": greedy_clean,
            "per_coeff": {},
        }
        for coeff in coeffs:
            hook = make_inject_hook(vecs, [[M]] * N_HF, coeff, "cuda", torch.bfloat16)
            hh = layers[L].register_forward_hook(hook)
            hc = layers[L].register_forward_hook(cap_out)  # registered after -> sees the injected output
            try:
                out_s = model(input_ids=inp, attention_mask=attn)
                h_steer = cap["L"]
                lp_s = torch.log_softmax(out_s.logits[:, -1].float(), dim=-1).cpu()
                try:
                    greedy = model.generate(
                        inp, attention_mask=attn, max_new_tokens=8, do_sample=False,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                    )[:, inp.shape[1] :].cpu().tolist()
                    gerr = None
                except Exception as e:  # noqa: BLE001
                    greedy, gerr = None, repr(e)[:300]
            finally:
                hh.remove()
                hc.remove()
            delta = h_steer[:, M] - h_clean[:, M]
            ratios = [float(delta[i].norm() / (coeff * h_clean[i, M].norm())) for i in range(N_HF)]
            coss = [cos(delta[i], U[i]) for i in range(N_HF)]
            other = (h_steer - h_clean).clone()
            other[:, M] = 0
            ref["per_coeff"][str(coeff)] = {
                "h_steer_marker": h_steer[:, M].clone(),
                "delta": delta.clone(),
                "cos_delta_vs_v": coss,
                "ratio_delta_over_coeff_hnorm": ratios,
                "max_other_row_abs_delta": float(other.abs().max()),
                "logprobs_steer": lp_s.half(),
                "greedy8": greedy,
                "greedy_error": gerr,
            }
            log(f"HF coeff={coeff}: cos={min(coss):.5f} ratio={ratios} other={float(other.abs().max()):.2e} greedy={greedy}")
    h_in.remove()
    torch.save(ref, a.out)
    log(f"wrote {a.out}")


# ---------------------------------------------------------------------------
# stage 2: vLLM engine + cases
# ---------------------------------------------------------------------------


def vllm_stage(a: argparse.Namespace) -> None:
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
    P = a.prompt_tokens
    ids = prompt_ids(tok, P)
    L, M, T = a.inject_layer, a.marker, a.max_tokens
    coeffs = [float(c) for c in a.coeffs.split(",") if c]
    batches = [int(b) for b in a.batches.split(",") if b]
    tp_batches = [int(b) for b in a.tp_batches.split(",") if b]
    ref = torch.load(a.ref, weights_only=False) if a.ref and os.path.exists(a.ref) else None
    if ref is not None:
        assert ref["prompt_ids"] == ids and ref["marker"] == M and ref["layer"] == L, "HF ref does not match this run"

    kw: dict[str, Any] = dict(
        model=a.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=a.gpu_mem,
        enable_prefix_caching=False,
        dtype="bfloat16",
        seed=0,
    )
    if a.chunked:
        mns = a.chunk_batch
        kw.update(
            max_num_seqs=mns,
            max_num_batched_tokens=a.chunk_tokens,
            max_model_len=P + 16,
            enable_chunked_prefill=True,
        )
    else:
        mns = a.max_num_seqs
        kw.update(
            max_num_seqs=mns,
            max_num_batched_tokens=max(8192, mns * (P + 8)),
            max_model_len=P + 2 * T + 8,  # throughput measures T and 2T new tokens
        )
    if a.attention_backend:
        kw["attention_backend"] = a.attention_backend
    if a.language_model_only:
        kw["language_model_only"] = True
    if a.engine == "eager":
        kw["enforce_eager"] = True
    else:
        kw["compilation_config"] = {"max_cudagraph_capture_size": min(mns, 1024)}

    t0 = time.perf_counter()
    llm, kw = make_llm(LLM, kw, log)
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
    }
    log(f"engine up {up:.0f}s vllm={vllm.__version__} vllm-lens={vllm_lens.__version__} {resolved}")

    result: dict[str, Any] = {
        "model": a.model,
        "engine": a.engine,
        "chunked": a.chunked,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__, "vllm_lens": vllm_lens.__version__},
        "resolved_config": resolved,
        "engine_up_s": up,
        "prompt_tokens": P,
        "marker": M,
        "layer": L,
        "hf_ref": bool(ref is not None),
        "cases": {},
        "checks": [],
    }
    checks: list[dict[str, Any]] = result["checks"]

    def check(case: str, name: str, ok: bool, detail: str) -> None:
        checks.append({"case": case, "check": name, "ok": bool(ok), "detail": detail})
        log(f"  [{'PASS' if ok else 'FAIL'}] {case}: {name}  {detail}")

    def dump() -> None:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)

    def stats() -> dict[str, int]:
        return llm.collective_rpc("steering_stats", args=(True,))[0]

    def sp(layers: list[int] | None, steer: list | None = None, max_tokens: int = 1, logprobs: int | None = None,
           temperature: float = 0.0) -> SamplingParams:
        extra: dict[str, Any] = {}
        if layers:
            extra["output_residual_stream"] = layers
        if steer:
            extra["apply_steering_vectors"] = steer
        return SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_tokens, logprobs=logprobs,
                              extra_args=extra or None)

    def gen(params: list[SamplingParams]):
        return llm.generate([{"prompt_token_ids": ids}] * len(params), params, use_tqdm=False)

    def acts(out):
        return out.activations["residual_stream"].float()  # (n_requested_layers, P, D), layers ascending

    def subset(B: int) -> list[int]:
        return list(range(B)) if B <= 64 else sorted({i for i in range(0, B, max(1, B // 16))} | {B - 1})

    def sv_add(u, coeff: float) -> SteeringVector:
        return SteeringVector(activations=u.reshape(1, 1, -1), layer_indices=[L], scale=coeff, norm_match=True,
                              position_indices=[M])

    def sv_embed(u, scale: float, nm: bool) -> SteeringVector:
        return SteeringVector(activations=u.reshape(1, 1, -1), layer_indices=[EMBED_LAYER_INDEX], scale=scale,
                              norm_match=nm, mode="replace", position_indices=[M])

    def sv_layer_replace(u, scale: float) -> SteeringVector:
        return SteeringVector(activations=u.reshape(1, 1, -1), layer_indices=[L], scale=scale, norm_match=False,
                              mode="replace", position_indices=[M])

    CAP = [EMBED_LAYER_INDEX, 0, L]  # -> acts index 0: embedding stream, 1: layer 0 out, 2: layer L out
    if L == 0:
        CAP = [EMBED_LAYER_INDEX, 0]

    # ---- probe: hidden size, clean norms, embed-capture sanity vs HF ----------------------
    _ = stats()
    probe = gen([sp(CAP)])[0]
    A0 = acts(probe)
    D = A0.shape[-1]
    e_norm = float(A0[0, M].norm())
    h_norm = float(A0[-1, M].norm())
    U = unit_vectors(max(batches + tp_batches + [a.chunk_batch]), D)
    result["hidden_dim"] = D
    result["embed_marker_norm"] = e_norm
    result["h_full_marker_norm_layer_L"] = h_norm
    log(f"D={D} |embed[M]|={e_norm:.3f} |h_L[M]|={h_norm:.3f}")
    if ref is not None:
        check("probe", "embedding stream capture (layer -1) == HF embedding row at the marker (rel<1e-2)",
              rel(A0[0, M], ref["embed_marker"][0]) < 1e-2, f"rel={rel(A0[0, M], ref['embed_marker'][0]):.2e}")
        r_h = rel(A0[-1, M], ref["h_clean_marker"][0])
        check("probe", "clean layer-L residual stream vs HF (engine-vs-HF kernel noise, informational, rel<5e-2)",
              r_h < 5e-2, f"rel={r_h:.2e} |h|_vllm={h_norm:.3f} |h|_hf={ref['h_clean_marker_norm']:.3f}")
    ps = stats()
    check("probe", "capture pass: no hook errors", ps["errors"] == 0 and ps["embed_errors"] == 0, json.dumps(ps))

    clean_cache: dict[int, Any] = {}

    def clean(B: int):
        """Clean capture at CAP for the subset rows of a B-request batch (cached per B)."""
        if B not in clean_cache:
            S = subset(B)
            outs = gen([sp(CAP) if i in S else sp(None) for i in range(B)])
            clean_cache[B] = {i: acts(outs[i]) for i in S}
            _ = stats()
        return clean_cache[B]

    # ---- case A: Karvonen-style norm-matched add at layer L -------------------------------
    def case_karvonen(case: str, B: int, coeff: float, with_hf: bool) -> None:
        S = subset(B)
        C = clean(B)
        outs = gen([sp(CAP if i in S else None, [sv_add(U[i], coeff)]) for i in range(B)])
        st = stats()
        coss, ratios, others, pre_other = [], [], [], []
        hf_cos, hf_ratio = [], []
        for i in S:
            hc, hs = C[i][-1], acts(outs[i])[-1]
            delta = hs[M] - hc[M]
            coss.append(cos(delta, U[i]))
            ratios.append(float(delta.norm() / (coeff * hc[M].norm())))
            o = hs - hc
            o[M] = 0
            others.append(float(o.abs().max()))
            e_o = float((acts(outs[i])[0] - C[i][0]).abs().max())  # embedding stream must be untouched
            pre_other.append(e_o)
            if with_hf and ref is not None and i < N_HF:
                d_hf = ref["per_coeff"][str(coeff)]["delta"][i]
                hf_cos.append(cos(delta, d_hf))
                hf_ratio.append(float(delta.norm() / d_hf.norm()))
        rec = {
            "batch": B, "coeff": coeff, "n_checked": len(S),
            "min_cos_delta_vs_v": min(coss), "max_abs_ratio_minus_1": max(abs(r - 1) for r in ratios),
            "ratios": ratios[:8], "max_other_row_abs_delta": max(others), "max_embed_stream_abs_delta": max(pre_other),
            "stats": st,
        }
        check(case, f"B={B} coeff={coeff}: injected delta == coeff·‖h_full‖·unit(v) per request (cos≥0.999, ratio within 1%)",
              min(coss) >= 0.999 and max(abs(r - 1) for r in ratios) <= 0.01,
              f"min cos={min(coss):.5f} ratio∈[{min(ratios):.4f},{max(ratios):.4f}] over {len(S)} requests")
        check(case, f"B={B} coeff={coeff}: rows other than the marker untouched at layer L", max(others) == 0.0,
              f"max|Δ|={max(others):.2e}")
        check(case, f"B={B} coeff={coeff}: hook errors == 0, every steered layer-step vectorised",
              st["errors"] == 0 and st["vectorized_layer_steps"] == st["steer_layer_steps"] > 0, json.dumps(st))
        if hf_cos:
            rec["hf_cos"] = hf_cos
            rec["hf_ratio"] = hf_ratio
            check(case, f"B={B} coeff={coeff}: delta == HF mxf/inject.py delta (cos≥0.999, norm ratio within 1%)",
                  min(hf_cos) >= 0.999 and max(abs(r - 1) for r in hf_ratio) <= 0.01,
                  f"cos={min(hf_cos):.5f} ratio∈[{min(hf_ratio):.4f},{max(hf_ratio):.4f}] (n={len(hf_cos)})")
        result["cases"].setdefault(case, []).append(rec)
        dump()

    def case_karvonen_generation(coeff: float) -> None:
        """Greedy continuation + next-token log-probs vs the HF-hooked reference (first N_HF vectors)."""
        if ref is None:
            return
        case = "karvonen_add_generation"
        rc = ref["per_coeff"][str(coeff)]
        outs_c = gen([sp(None, None, max_tokens=8, logprobs=20) for _ in range(N_HF)])
        outs_s = gen([sp(None, [sv_add(U[i], coeff)], max_tokens=8, logprobs=20) for i in range(N_HF)])
        _ = stats()

        def lp(out):
            return {int(t): float(v.logprob) for t, v in out.outputs[0].logprobs[0].items()}

        noise = max(abs(v - float(ref["logprobs_clean"][i, t])) for i in range(N_HF) for t, v in lp(outs_c[i]).items())
        mx = max(abs(v - float(rc["logprobs_steer"][i, t])) for i in range(N_HF) for t, v in lp(outs_s[i]).items())
        tol = max(0.05, 1.5 * noise + 0.01)
        vllm_g = [[int(t) for t in outs_s[i].outputs[0].token_ids] for i in range(N_HF)]
        vllm_gc = [[int(t) for t in outs_c[i].outputs[0].token_ids] for i in range(N_HF)]
        same = rc["greedy8"] is not None and vllm_g == rc["greedy8"]
        same_c = vllm_gc == ref["greedy8_clean"]
        n_same = sum(1 for i in range(N_HF) if rc["greedy8"] and vllm_g[i] == rc["greedy8"][i]) if rc["greedy8"] else None
        rec = {"coeff": coeff, "clean_logprob_noise_vs_hf": noise, "steered_logprob_maxdiff_vs_hf": mx, "tol": tol,
               "greedy8_vllm": vllm_g, "greedy8_hf": rc["greedy8"], "greedy8_clean_vllm": vllm_gc,
               "greedy8_clean_hf": ref["greedy8_clean"], "greedy_all_equal": same, "n_greedy_equal": n_same,
               "hf_greedy_error": rc.get("greedy_error")}
        check(case, f"coeff={coeff}: steered next-token top-20 log-probs within the vLLM-vs-HF noise floor "
                    f"(max|d| ≤ max(0.05, 1.5·clean noise + 0.01))", mx <= tol,
              f"steered max|d|={mx:.4f}, clean-prompt noise={noise:.4f}, tol={tol:.4f}")
        hf_g = rc["greedy8"]
        detail = "equal" if same else f"{n_same}/{N_HF} equal; vllm={vllm_g} hf={hf_g}"
        if not same_c:
            detail += f"; NOTE clean greedy also differs: vllm={vllm_gc} hf={ref['greedy8_clean']}"
        check(case, f"coeff={coeff}: greedy 8-token steered continuation == HF-hooked reference (informational)", True, detail)
        result["cases"].setdefault(case, []).append(rec)
        dump()

    # ---- case B: embedding replacement -----------------------------------------------------
    def check_embed_rows(case: str, tag: str, S, C, outs, scale: float, nm: bool, expect_replaced) -> dict:
        """expect_replaced(i) -> bool: whether request i had an embed-replace vector."""
        r_row, c_row, others, l0_pre, l0_mark, lL_pre, lL_mark = [], [], [], [], [], [], []
        for i in S:
            A, Ac = acts(outs[i]), C[i]
            es, ec = A[0], Ac[0]
            if expect_replaced(i):
                tgt = (U[i] * (scale * float(ec[M].norm()) if nm else scale)).to(torch.bfloat16).float()
                r_row.append(rel(es[M], tgt))
                c_row.append(cos(es[M], U[i]))
            o = es - ec
            o[M] = 0
            others.append(float(o.abs().max()) if expect_replaced(i) else float((es - ec).abs().max()))
            if len(CAP) == 3:
                l0, l0c, lL, lLc = A[1], Ac[1], A[2], Ac[2]
                l0_pre.append(float((l0[:M] - l0c[:M]).abs().max()))
                lL_pre.append(float((lL[:M] - lLc[:M]).abs().max()))
                l0_mark.append(float((l0[M] - l0c[M]).abs().max()))
                lL_mark.append(float((lL[M] - lLc[M]).abs().max()))
        rec = {"n_checked": len(S), "scale": scale, "norm_match": nm,
               "max_rel_err_marker_row": max(r_row) if r_row else None, "min_cos_marker_row": min(c_row) if c_row else None,
               "max_other_embed_row_abs_delta": max(others),
               "max_layer0_pre_marker_abs_delta": max(l0_pre) if l0_pre else None,
               "min_layer0_marker_abs_delta": min(l0_mark) if l0_mark else None,
               "max_layerL_pre_marker_abs_delta": max(lL_pre) if lL_pre else None,
               "min_layerL_marker_abs_delta": min(lL_mark) if lL_mark else None}
        if r_row:
            check(case, f"{tag}: embedding-stream marker row == {'scale·‖e‖·v/‖v‖' if nm else 'scale·v'} per request "
                        f"(bf16 rel<1e-2, cos>0.9999)", max(r_row) < 1e-2 and min(c_row) > 0.9999,
                  f"max rel={max(r_row):.2e} min cos={min(c_row):.6f} over {len(r_row)} requests")
        check(case, f"{tag}: every other embedding row untouched", max(others) == 0.0, f"max|Δ|={max(others):.2e}")
        if l0_pre:
            check(case, f"{tag}: downstream layers differ from clean only causally (positions < marker identical at layers 0 and L; marker row changed)",
                  max(l0_pre) == 0.0 and max(lL_pre) == 0.0 and (min(l0_mark) > 0 if any(expect_replaced(i) for i in S) else True),
                  f"pre-marker max|Δ| L0={max(l0_pre):.2e} LL={max(lL_pre):.2e}; marker |Δ| L0≥{min(l0_mark):.3e}")
        return rec

    def case_embed(case: str, B: int, nm: bool) -> None:
        S = subset(B)
        C = clean(B)
        scale = 1.0 if nm else e_norm
        outs = gen([sp(CAP if i in S else None, [sv_embed(U[i], scale, nm)]) for i in range(B)])
        st = stats()
        rec = check_embed_rows(case, f"B={B} norm_match={nm}", S, C, outs, scale, nm, lambda i: True)
        rec.update(batch=B, stats=st)
        check(case, f"B={B} norm_match={nm}: rows_replaced == B, embed passes ≥ 1, no errors",
              st["rows_replaced"] == B and st["embed_apply_steps"] >= 1 and st["errors"] == 0 and st["embed_errors"] == 0,
              json.dumps(st))
        result["cases"].setdefault(case, []).append(rec)
        dump()

    # ---- case B': replace at a regular (fused-residual) layer output -------------------------
    def case_layer_replace(B: int) -> None:
        case = "layer_replace"
        S = subset(B)
        C = clean(B)
        scale = h_norm
        outs = gen([sp(CAP if i in S else None, [sv_layer_replace(U[i], scale)]) for i in range(B)])
        st = stats()
        rr, cc_, others, emb = [], [], [], []
        for i in S:
            A, Ac = acts(outs[i]), C[i]
            tgt = (U[i] * scale).to(torch.bfloat16).float()
            rr.append(rel(A[-1][M], tgt))
            cc_.append(cos(A[-1][M], U[i]))
            o = A[-1] - Ac[-1]
            o[M] = 0
            others.append(float(o.abs().max()))
            emb.append(float((A[0] - Ac[0]).abs().max()))
        rec = {"batch": B, "scale": scale, "max_rel_err_marker_row": max(rr), "min_cos_marker_row": min(cc_),
               "max_other_row_abs_delta": max(others), "max_embed_stream_abs_delta": max(emb), "stats": st}
        check(case, f"B={B}: FULL residual stream at layer L marker == scale·v (both fused halves rewritten; bf16 rel<1e-2, cos>0.9999)",
              max(rr) < 1e-2 and min(cc_) > 0.9999, f"max rel={max(rr):.2e} min cos={min(cc_):.6f}")
        check(case, f"B={B}: other rows and the embedding stream untouched", max(others) == 0.0 and max(emb) == 0.0,
              f"max|Δ| rows={max(others):.2e} embed={max(emb):.2e}")
        check(case, f"B={B}: vectorised replace path, no errors",
              st["errors"] == 0 and st["vectorized_layer_steps"] == st["steer_layer_steps"] > 0, json.dumps(st))
        result["cases"].setdefault(case, []).append(rec)
        dump()

    # ---- case C: mixed batch -------------------------------------------------------------------
    def case_mixed(B: int, coeff: float = 1.0) -> None:
        case = "mixed"
        S = subset(B)
        C = clean(B)
        scale = e_norm
        params = [sp(CAP if i in S else None, [sv_embed(U[i], scale, False) if i % 2 == 0 else sv_add(U[i], coeff)])
                  for i in range(B)]
        outs = gen(params)
        st = stats()
        rec = check_embed_rows(case, f"B={B} even requests (embed replace)", [i for i in S if i % 2 == 0], C, outs, scale,
                               False, lambda i: True)
        coss, ratios, others, emb = [], [], [], []
        for i in S:
            if i % 2 == 0:
                continue
            A, Ac = acts(outs[i]), C[i]
            delta = A[-1][M] - Ac[-1][M]
            coss.append(cos(delta, U[i]))
            ratios.append(float(delta.norm() / (coeff * Ac[-1][M].norm())))
            o = A[-1] - Ac[-1]
            o[M] = 0
            others.append(float(o.abs().max()))
            emb.append(float((A[0] - Ac[0]).abs().max()))
        rec.update(batch=B, coeff=coeff, add_min_cos=min(coss), add_max_abs_ratio_minus_1=max(abs(r - 1) for r in ratios),
                   add_max_other_row_abs_delta=max(others), add_max_embed_stream_abs_delta=max(emb), stats=st)
        check(case, f"B={B} odd requests (karvonen add): delta == coeff·‖h‖·unit(v), embedding stream untouched, other rows untouched",
              min(coss) >= 0.999 and max(abs(r - 1) for r in ratios) <= 0.01 and max(others) == 0.0 and max(emb) == 0.0,
              f"min cos={min(coss):.5f} ratio∈[{min(ratios):.4f},{max(ratios):.4f}] other={max(others):.1e} embed={max(emb):.1e}")
        check(case, f"B={B}: rows_replaced == B/2, rows_steered == B/2, vectorised, no errors",
              st["rows_replaced"] == B // 2 and st["rows_steered"] == B // 2 and st["errors"] == 0
              and st["vectorized_layer_steps"] == st["steer_layer_steps"] > 0, json.dumps(st))
        result["cases"].setdefault(case, []).append(rec)
        dump()

    # ---- case E: throughput (graphs must still engage) ------------------------------------------
    def case_throughput() -> None:
        """Wall time at T and 2T new tokens per condition; the difference isolates the
        DECODE-step time (graph replays) from prefill + the per-call cost of shipping B
        vectors (pickle / RPC / pydantic, O(B) and model-size independent)."""
        case = "throughput"
        base = None
        if a.baseline and os.path.exists(a.baseline):
            try:
                base = json.load(open(a.baseline))["tables"].get(a.model, {}).get("series", {})
            except Exception:  # noqa: BLE001
                base = None

        def gp(extra, n_tok: int):
            return SamplingParams(temperature=1.0, top_p=1.0, max_tokens=n_tok, min_tokens=n_tok, ignore_eos=True, extra_args=extra)

        conds = {
            "nosteer": lambda i: None,
            "karvonen_add": lambda i: {"apply_steering_vectors": [sv_add(U[i], 1.0)]},
            "embed_replace": lambda i: {"apply_steering_vectors": [sv_embed(U[i], e_norm, False)]},
        }

        def timed(mk, B: int, n_tok: int) -> tuple[float, int, dict]:
            best, n_gen, st = math.inf, 0, {}
            for _ in range(max(1, a.tp_repeats)):
                t1 = time.perf_counter()
                outs = gen([gp(mk(i), n_tok) for i in range(B)])
                wall = time.perf_counter() - t1
                st = stats()
                n_gen = sum(len(o.outputs[0].token_ids) for o in outs)
                best = min(best, wall)
            return best, n_gen, st

        rows: dict[str, dict[int, dict]] = {}
        for cond, mk in conds.items():
            gen([gp(mk(i), T) for i in range(8)])  # warm-up
            _ = stats()
        # conditions INTERLEAVED per batch size and repeat, so GPU-clock / thermal drift
        # affects all three alike; min over repeats is kept.
        for B in tp_batches:
            for rep in range(max(1, a.tp_repeats)):
                for cond, mk in conds.items():
                    t1 = time.perf_counter()
                    outs = gen([gp(mk(i), T) for i in range(B)])
                    w1 = time.perf_counter() - t1
                    st1 = stats()
                    n1 = sum(len(o.outputs[0].token_ids) for o in outs)
                    t1 = time.perf_counter()
                    outs2 = gen([gp(mk(i), 2 * T) for i in range(B)])
                    w2 = time.perf_counter() - t1
                    st2 = stats()
                    r = rows.setdefault(cond, {}).setdefault(B, {"wall_s": math.inf, "wall_2T_s": math.inf, "repeats": []})
                    r["repeats"].append({"wall_s": w1, "wall_2T_s": w2})
                    r["wall_s"] = min(r["wall_s"], w1)
                    r["wall_2T_s"] = min(r["wall_2T_s"], w2)
                    r.update(tok_per_s=n1 / r["wall_s"], gen_tokens=n1, gen_tokens_2T=sum(len(o.outputs[0].token_ids) for o in outs2),
                             decode_step_ms=(r["wall_2T_s"] - r["wall_s"]) / T * 1000.0,
                             prefill_plus_overhead_s=2 * r["wall_s"] - r["wall_2T_s"],
                             hook_passes=st1["steps_fast_idle"] + st1["steps_planned"],
                             hook_passes_2T=st2["steps_fast_idle"] + st2["steps_planned"], stats=st1)
                    log(f"  rep{rep} {cond:14s} B={B:5d}: {w1:6.2f}s ({T} tok) {w2:6.2f}s ({2 * T} tok) -> best decode step "
                        f"{r['decode_step_ms']:.2f} ms, prefill+overhead {r['prefill_plus_overhead_s']:.2f}s, hook passes={r['hook_passes']}")
        result["cases"][case] = {"rows": rows, "T": T, "repeats": a.tp_repeats,
                                 "baseline_series": {k: base.get(k) for k in ("fork_graphs", "ceiling_graphs", "fork_vectorized", "ceiling_eager")} if base else None}
        for c in throughput_checks(result["cases"][case], a.engine):
            checks.append(c)
            log(f"  [{'PASS' if c['ok'] else ('n/a ' if c['ok'] is None else 'FAIL')}] {case}: {c['check']}  {c['detail']}")
        dump()

    def set_marker(m: int) -> None:
        nonlocal M
        M = m

    # ---- run -----------------------------------------------------------------------------------
    if a.chunked:
        B = a.chunk_batch
        result["markers"] = [a.marker, a.chunk_marker]
        min_passes = math.ceil(B * P / a.chunk_tokens) // 2
        for m in (a.marker, a.chunk_marker):
            set_marker(m)
            where = "FIRST" if m < a.chunk_tokens else "NON-first"
            tag = f"chunked_m{m}"
            case_karvonen(f"{tag}_karvonen_add", B, coeffs[0], with_hf=False)
            st_k = result["cases"][f"{tag}_karvonen_add"][-1]["stats"]
            case_embed(f"{tag}_embed_replace", B, nm=False)
            st_e = result["cases"][f"{tag}_embed_replace"][-1]["stats"]
            check("chunked", f"marker {m} ({where} {a.chunk_tokens}-token chunk of a {P}-token prompt): prefill really was chunked "
                             f"(planned passes ≥ {min_passes} for {B}×{P} tokens), injection landed exactly once",
                  st_k["steps_planned"] >= min_passes and st_e["steps_planned"] >= min_passes
                  and st_e["rows_replaced"] == B and st_k["rows_steered"] == B,
                  f"planned passes: add={st_k['steps_planned']} embed={st_e['steps_planned']}; rows_steered={st_k['rows_steered']} rows_replaced={st_e['rows_replaced']}")
        result["all_pass"] = all(c["ok"] for c in checks)
        dump()
        log(f"chunked engine done: {sum(c['ok'] for c in checks)}/{len(checks)} checks pass")
        return

    if not a.only_throughput:
        for coeff in coeffs:
            for B in batches:
                case_karvonen("karvonen_add", B, coeff, with_hf=(B == batches[0]))
            case_karvonen_generation(coeff)
        case_layer_replace(batches[0])
        for nm in (False, True):
            for B in batches:
                case_embed("embed_replace", B, nm)
        case_mixed(batches[0])
    if not a.skip_throughput:
        case_throughput()
    result["all_pass"] = all(c["ok"] for c in checks)
    dump()
    log(f"done: {sum(c['ok'] for c in checks)}/{len(checks)} checks pass")


# ---------------------------------------------------------------------------
# throughput checks: a deterministic function of the STORED raw numbers, so the
# criterion can be refined offline without re-measuring (used at run time for the
# log and by ``summarize`` as the authoritative version)
# ---------------------------------------------------------------------------

STEP_TOL = 0.10  # decode-step time: steer <= (1 + STEP_TOL) * no-steering
WALL_TOL = 0.10  # total wall incl. per-call vector shipping: steer/no-steer <= max(recorded, 1) + WALL_TOL
RESOLVABLE_SPREAD = 0.10  # a relative gate is evaluated only if the control's own repeat spread is below this


def _step_estimate(r: dict, T: int) -> tuple[float, float | None]:
    """(decode-step ms, relative repeat spread or None). With per-repeat data the
    PAIRED median of wall_2T - wall_T is used (robust to drift); else min(2T)-min(T)."""
    reps = r.get("repeats")
    if reps:
        diffs = sorted(x["wall_2T_s"] - x["wall_s"] for x in reps)
        n = len(diffs)
        med = diffs[n // 2] if n % 2 else 0.5 * (diffs[n // 2 - 1] + diffs[n // 2])
        spread = (max(diffs) - min(diffs)) / (sum(diffs) / n)
        return med / T * 1000.0, spread
    return float(r["decode_step_ms"]), None


def _wall_spread(r: dict) -> float | None:
    reps = r.get("repeats")
    if not reps:
        return None
    ws = [x["wall_s"] for x in reps]
    return (max(ws) - min(ws)) / (sum(ws) / len(ws))


def throughput_checks(tp: dict, engine: str) -> list[dict[str, Any]]:
    """Checks for one engine's throughput case. ``ok`` is True / False, or None when the
    gate is NOT RESOLVABLE at this model's scale (control repeat spread > RESOLVABLE_SPREAD)
    -- such rows are counted separately, never as passes."""
    rows, T = tp["rows"], int(tp.get("T", 40))
    base = tp.get("baseline_series") or {}
    out: list[dict[str, Any]] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        out.append({"case": "throughput", "check": name, "ok": ok, "detail": detail})

    for B in sorted(rows["nosteer"], key=int):
        B = int(B)
        ns = rows["nosteer"][B] if B in rows["nosteer"] else rows["nosteer"][str(B)]
        ns_ms, ns_spread = _step_estimate(ns, T)
        step_ok = ns_spread is None or ns_spread <= RESOLVABLE_SPREAD
        for name, key in (("embed-replace", "embed_replace"), ("karvonen-add (norm_match)", "karvonen_add")):
            r = rows[key][B] if B in rows[key] else rows[key][str(B)]
            ms, _sp = _step_estimate(r, T)
            detail = (f"{ms:.2f} ms vs {ns_ms:.2f} ms per decode step ({ms / ns_ms - 1:+.1%}); wall {T} tok "
                      f"{r['wall_s']:.2f}s vs {ns['wall_s']:.2f}s ({r['wall_s'] / ns['wall_s'] - 1:+.1%}, {r['wall_s'] - ns['wall_s']:+.3f}s "
                      f"per call = shipping {B} vectors + prefill hook)")
            if step_ok:
                add(f"B={B}: {name} decode-step time within {STEP_TOL:.0%} of no-steering (same engine, hooks installed)",
                    ms <= (1 + STEP_TOL) * ns_ms, detail)
            else:
                add(f"B={B}: {name} decode-step time vs no-steering -- NOT RESOLVABLE at this scale "
                    f"(control's repeat spread of wall_2T-wall_T = {ns_spread:.0%} > {RESOLVABLE_SPREAD:.0%}; informational)", None, detail)
        er = rows["embed_replace"][B] if B in rows["embed_replace"] else rows["embed_replace"][str(B)]
        ka = rows["karvonen_add"][B] if B in rows["karvonen_add"] else rows["karvonen_add"][str(B)]
        if engine.startswith("graphs"):
            add(f"B={B}: CUDA graphs engage with embed-replace requests (hooks ran in <= {T // 4} of {T + 1} forward passes; eager would be ~{T + 1})",
                er["hook_passes"] <= T // 4, f"hook passes: embed={er['hook_passes']} add={ka['hook_passes']} nosteer={ns['hook_passes']}")
        fk, ck = ("fork_graphs", "ceiling_graphs") if engine.startswith("graphs") else ("fork_vectorized", "ceiling_eager")
        if base.get(fk) and base.get(ck) and str(B) in base[fk] and str(B) in base[ck]:
            rec_f, rec_c = base[fk][str(B)]["wall_s"], base[ck][str(B)]["wall_s"]
            rec_ratio, now_ratio = rec_f / rec_c, ka["wall_s"] / ns["wall_s"]
            thr = max(rec_ratio, 1.0) + WALL_TOL
            detail = (f"steer/no-steer wall now {now_ratio:.3f} vs recorded {rec_ratio:.3f} ({fk}/{ck}), gate <= {thr:.3f}; "
                      f"absolute (different Modal instance): karvonen {ka['wall_s']:.2f}s vs recorded {rec_f:.2f}s ({ka['wall_s'] / rec_f - 1:+.1%}), "
                      f"no-steering {ns['wall_s']:.2f}s vs recorded ceiling {rec_c:.2f}s ({ns['wall_s'] / rec_c - 1:+.1%})")
            wsp = _wall_spread(ns)
            if wsp is None or wsp <= RESOLVABLE_SPREAD:
                add(f"B={B}: no regression of the steering path vs bench/results_summary.json: steer/no-steer wall ratio "
                    f"<= max(recorded, 1) + {WALL_TOL:.0%} (container-independent)", now_ratio <= thr, detail)
            else:
                add(f"B={B}: steer/no-steer wall ratio vs recorded -- NOT RESOLVABLE (control wall repeat spread {wsp:.0%}; informational)",
                    None, detail)
    return out


def throughput_checks_dsv4(tp: dict, engine: str) -> list[dict[str, Any]]:
    """DeepSeek-V4 variant: conditions nosteer / embed_replace / embed_add, no recorded baseline, prefill
    chunked at ``max_num_batched_tokens``.  The 284B fp8/fp4 engine shows sporadic multi-second stalls on
    ANY condition (Triton / DeepGEMM JIT for new shapes), so the gate uses the stall-robust estimate
    ``min(wall_2T) - min(wall_T)`` over repeats; the paired-median estimate and the control's repeat
    spread are reported alongside."""
    rows, T = tp["rows"], int(tp.get("T", 40))
    P = int(tp.get("_prompt_tokens", 96))
    mnbt = int(tp.get("max_num_batched_tokens", 4096))
    out: list[dict[str, Any]] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        out.append({"case": "throughput_dsv4", "check": name, "ok": ok, "detail": detail})

    def get(cond: str, B: int) -> dict:
        d = rows[cond]
        return d[B] if B in d else d[str(B)]

    for B in sorted({int(b) for b in rows["nosteer"]}):
        ns = get("nosteer", B)
        ns_min = (ns["wall_2T_s"] - ns["wall_s"]) / T * 1000.0
        ns_med, ns_sp = _step_estimate(ns, T)
        for name, key in (("embed-replace", "embed_replace"), ("embed-add (norm_match)", "embed_add")):
            if key not in rows:
                continue
            r = get(key, B)
            r_min = (r["wall_2T_s"] - r["wall_s"]) / T * 1000.0
            r_med, _ = _step_estimate(r, T)
            reps = r.get("repeats") or []
            detail = (f"stall-robust (min over repeats): {r_min:.2f} ms vs {ns_min:.2f} ms per decode step ({r_min / ns_min - 1:+.1%}); "
                      f"paired-median: {r_med:.2f} vs {ns_med:.2f} ms (control spread {ns_sp:.0%}); wall {T} tok {r['wall_s']:.2f}s vs "
                      f"{ns['wall_s']:.2f}s ({r['wall_s'] / ns['wall_s'] - 1:+.1%}, {r['wall_s'] - ns['wall_s']:+.3f}s per call = shipping {B} vectors + "
                      f"prefill hook); repeats (wall_T, wall_2T): {[(round(x['wall_s'], 2), round(x['wall_2T_s'], 2)) for x in reps]}"
                      if ns_sp is not None else "")
            add(f"B={B}: {name} decode-step time within {STEP_TOL:.0%} of no-steering (same engine, hooks installed; min-over-repeats estimate)",
                r_min <= (1 + STEP_TOL) * ns_min, detail)
        if engine.startswith("graphs"):
            n_pre = math.ceil(B * P / mnbt)
            er = get("embed_replace", B)
            add(f"B={B}: CUDA graphs engage with embed-replace requests (hook passes <= prefill passes {n_pre} + {T // 4}; eager would be ~{n_pre + T})",
                er["hook_passes"] <= n_pre + T // 4,
                f"hook passes: embed={er['hook_passes']} add={get('embed_add', B)['hook_passes'] if 'embed_add' in rows else '-'} nosteer={ns['hook_passes']}")
    return out


THROUGHPUT_CHECKERS: dict[str, Any] = {"throughput": throughput_checks, "throughput_dsv4": throughput_checks_dsv4}


def effect_checks(recs: list[dict]) -> list[dict[str, Any]]:
    """Offline re-evaluation of the DSv4 ``effect_check`` case from its stored metrics: the criterion is a
    next-token distribution shift well above the engine's clean-vs-clean floor (greedy argmax need not flip
    -- the token 1 position before the prediction dominates it)."""
    out = []
    for r in recs:
        eff, noise = r.get("logprob_effect_mean"), r.get("clean_noise", 0.0) or 0.0
        if eff is None:
            continue
        out.append({"case": "effect_check", "ok": eff > 10 * noise + 0.05,
                    "check": f"B={r.get('batch')}: replacing the embedding shortly before the predicted position shifts the next-token "
                             f"distribution well above the clean-vs-clean floor (mean max|Δ top-20 logprob| > 10·floor + 0.05)",
                    "detail": f"mean max|Δ top-20 logprob| {eff:.3f} (floor {noise:.3f}); greedy argmax changed {r.get('argmax_changed_frac', 0):.0%}"})
        st = r.get("stats") or {}
        if st:
            out.append({"case": "effect_check", "ok": st.get("rows_replaced") == r.get("batch") and st.get("errors", 0) == 0,
                        "check": f"B={r.get('batch')}: rows_replaced == B, no errors", "detail": json.dumps(st)})
    return out


CASE_RECHECKERS: dict[str, Any] = {"effect_check": effect_checks}
"""list-type cases whose pass/fail is recomputed offline from stored metrics (criterion refinements
without re-measuring); the marker-row / untouched-rows checks of such cases are kept as stored."""
"""case name -> function(case_record, engine) recomputing its checks offline;
other throughput-like cases keep the checks stored at run time."""


# ---------------------------------------------------------------------------
# summary over a results directory (no GPU / vLLM needed)
# ---------------------------------------------------------------------------


def summarize(d: Path) -> dict[str, Any]:
    recs = []
    for f in sorted(d.glob("*.json")):
        if f.name == "summary.json":
            continue
        r = json.loads(f.read_text())
        if "result" in r:
            r = r["result"]
        if "cases" not in r:
            continue
        recs.append(r)
    checks = []
    for r in recs:
        eng = r["engine"] + (" +chunked" if r.get("chunked") else "")
        own = [c for c in r["checks"] if c["case"] not in THROUGHPUT_CHECKERS or c["case"] not in r["cases"]]
        for rcase, fn in CASE_RECHECKERS.items():
            if rcase in r["cases"]:
                keep_words = ("marker row", "untouched", "identical to the clean")
                own = [c for c in own if c["case"] != rcase or any(w in c["check"] for w in keep_words)]
                own += fn(r["cases"][rcase])
        for tcase, fn in THROUGHPUT_CHECKERS.items():
            if tcase in r["cases"]:
                r["cases"][tcase]["_prompt_tokens"] = r.get("prompt_tokens", 96)
                own += fn(r["cases"][tcase], r["engine"])  # authoritative, from raw numbers
        checks += [dict(c, model=r["model"], engine=eng) for c in own]
    rows = []
    for r in recs:
        eng = r["engine"] + (" +chunked" if r.get("chunked") else "")
        for case, recs_c in r["cases"].items():
            if isinstance(recs_c, dict) and "rows" in recs_c:  # throughput-like case
                for cond, by_b in recs_c["rows"].items():
                    for B, v in by_b.items():
                        rows.append({"model": r["model"], "engine": eng, "case": f"{case}/{cond}", "batch": int(B),
                                     "wall_s": v["wall_s"], "tok_per_s": v["tok_per_s"], "hook_passes": v["hook_passes"],
                                     "decode_step_ms": _step_estimate(v, int(recs_c.get("T", 40)))[0] if v.get("decode_step_ms") is not None else None,
                                     "decode_step_spread": _step_estimate(v, int(recs_c.get("T", 40)))[1] if v.get("decode_step_ms") is not None else None,
                                     "decode_step_min_ms": v.get("decode_step_ms"),  # (min wall_2T - min wall_T)/T: robust to sporadic stalls
                                     "wall_2T_s": v.get("wall_2T_s"),
                                     "prefill_plus_overhead_s": v.get("prefill_plus_overhead_s"), "T": recs_c.get("T"),
                                     "repeats": v.get("repeats")})
                continue
            for rec in recs_c:
                row = {"model": r["model"], "engine": eng, "case": case}
                for k in ("batch", "coeff", "norm_match", "scale", "n_checked", "min_cos_delta_vs_v", "max_abs_ratio_minus_1",
                          "max_other_row_abs_delta", "max_embed_stream_abs_delta", "max_rel_err_marker_row", "min_cos_marker_row",
                          "max_other_embed_row_abs_delta", "max_layer0_pre_marker_abs_delta", "max_layerL_pre_marker_abs_delta",
                          "min_layer0_marker_abs_delta", "hf_cos", "hf_ratio", "clean_logprob_noise_vs_hf",
                          "steered_logprob_maxdiff_vs_hf", "greedy_all_equal", "n_greedy_equal", "add_min_cos",
                          "add_max_abs_ratio_minus_1", "ref_max_abs_diff", "ref_impl_max_abs_diff", "logprob_effect_mean",
                          "argmax_changed_frac", "nosteer_logprob_maxdiff", "n_split_requests", "steps_planned"):
                    if k in rec:
                        v = rec[k]
                        row[k] = (min(v) if k.endswith("cos") else max(v)) if isinstance(v, list) and v and isinstance(v[0], float) else v
                rows.append(row)
    n_ok = sum(c["ok"] is True for c in checks)
    n_fail = sum(c["ok"] is False for c in checks)
    n_info = sum(c["ok"] is None for c in checks)
    return {"rows": rows, "checks": checks, "n_pass": n_ok, "n_fail": n_fail, "n_info": n_info, "n_gated": n_ok + n_fail,
            "n_checks": len(checks), "all_pass": n_fail == 0,
            "runs": [{"model": r["model"], "engine": r["engine"], "chunked": r.get("chunked"), "gpu": r.get("gpu"),
                      "throughput_only": r.get("throughput_only", False), "source": r.get("source"),
                      "versions": r.get("versions"), "resolved_config": r.get("resolved_config"), "engine_up_s": r.get("engine_up_s"),
                      "hf_ref": r.get("hf_ref"), "hidden_dim": r.get("hidden_dim"), "embed_marker_norm": r.get("embed_marker_norm"),
                      "h_full_marker_norm_layer_L": r.get("h_full_marker_norm_layer_L")} for r in recs]}


MATRIX_HEADER = ["model", "engine", "case", "B", "coeff / scale", "cos(Δ, v)", "‖Δ‖/(c·‖h‖) − 1 or bf16 rel err",
                 "other rows max|Δ|", "vs HF reference", "checks"]
TP_HEADER = ["model", "engine", "condition", "B", "wall, 40 new tok (s)", "wall, 80 new tok (s)", "decode step (ms)",
             "prefill + per-call overhead (s)", "tok/s", "hook passes", "checks"]


def _fmt(x, fmt):
    return "—" if x is None else fmt.format(x)


def matrix_rows(summary: dict[str, Any]) -> tuple[list[list[str]], list[list[str]]]:
    """(correctness rows, throughput rows) as lists of cell strings; one row per
    (model, engine, case, batch) with the key numbers and its PASS count."""
    by_key: dict[tuple, list] = {}
    for c in summary["checks"]:
        by_key.setdefault((c["model"], c["engine"], c["case"]), []).append(c)
    out: list[list[str]] = []
    for r in summary["rows"]:
        if r["case"].startswith("throughput") and "/" in r["case"]:
            continue
        cs = by_key.get((r["model"], r["engine"], r["case"]), [])
        b = r.get("batch")
        cs_b = [c for c in cs if b is None or f"B={b}" in c["check"] or "B=" not in c["check"]]
        if r.get("coeff") is not None and any("coeff=" in c["check"] for c in cs_b):
            cs_b = [c for c in cs_b if f"coeff={r['coeff']}" in c["check"] or "coeff=" not in c["check"]]
        if r.get("norm_match") is not None and any("norm_match=" in c["check"] for c in cs_b):
            cs_b = [c for c in cs_b if f"norm_match={r['norm_match']}" in c["check"] or "norm_match=" not in c["check"]]
        n_ok, n = sum(c["ok"] is True for c in cs_b), sum(c["ok"] is not None for c in cs_b)
        cosv = r.get("min_cos_delta_vs_v", r.get("add_min_cos", r.get("min_cos_marker_row")))
        ratio = r.get("max_abs_ratio_minus_1", r.get("add_max_abs_ratio_minus_1", r.get("max_rel_err_marker_row")))
        other = r.get("max_other_row_abs_delta", r.get("max_other_embed_row_abs_delta"))
        if r.get("hf_cos") is not None:
            hf = f"cos {r['hf_cos']:.4f}, norm ratio {r['hf_ratio']:.4f}"
        elif r.get("steered_logprob_maxdiff_vs_hf") is not None:
            hf = (f"next-token logprob max diff {r['steered_logprob_maxdiff_vs_hf']:.3f} (clean-prompt noise "
                  f"{r['clean_logprob_noise_vs_hf']:.3f}); greedy-8 equal {r.get('n_greedy_equal')}/{N_HF}")
        else:
            hf = "—"
        if r.get("coeff") is not None:
            cscale = f"coeff {r['coeff']:.1f}"
        elif r.get("scale") is not None:
            cscale = f"scale {r['scale']:.2f}" + (" (norm_match)" if r.get("norm_match") else "")
        else:
            cscale = "—"
        out.append([r["model"].split("/")[-1], r["engine"], r["case"], _fmt(b, "{}"), cscale, _fmt(cosv, "{:.5f}"),
                    _fmt(ratio, "{:.1e}"), _fmt(other, "{:.1e}"), hf, f"{n_ok}/{n}" + ("" if n_ok == n else " FAIL")])
    tp = []
    for r in summary["rows"]:
        if not (r["case"].startswith("throughput") and "/" in r["case"]):
            continue
        cond = r["case"].split("/")[1]
        label = {"embed_replace": "embed-replace", "karvonen_add": "karvonen-add", "embed_add": "embed-add"}.get(cond)
        tcase = r["case"].split("/")[0]
        cs = [c for c in by_key.get((r["model"], r["engine"], tcase), []) if f"B={r['batch']}" in c["check"]
              and (label is None or label in c["check"] or "CUDA graphs" in c["check"] and cond == "embed_replace"
                   or "regression" in c["check"] and cond == "karvonen_add" or "wall ratio" in c["check"] and cond == "karvonen_add")]
        n_ok, n_f, n_i = sum(c["ok"] is True for c in cs), sum(c["ok"] is False for c in cs), sum(c["ok"] is None for c in cs)
        ctxt = "—" if cond == "nosteer" else f"{n_ok}/{n_ok + n_f}" + (f" (+{n_i} n/a)" if n_i else "") + (" FAIL" if n_f else "")
        tp.append([r["model"].split("/")[-1], r["engine"], cond, str(r["batch"]), f"{r['wall_s']:.2f}",
                   _fmt(r.get("wall_2T_s"), "{:.2f}"), _fmt(r.get("decode_step_ms"), "{:.2f}"), _fmt(r.get("prefill_plus_overhead_s"), "{:.2f}"),
                   f"{r['tok_per_s']:,.0f}", str(r["hook_passes"]), ctxt])
    return out, tp


def markdown_table(summary: dict[str, Any]) -> str:
    """Compact Markdown matrix: one row per (model, engine, case, batch) with the key numbers."""
    def esc(c: str) -> str:
        return c.replace("|", "\\|")

    rows, tp = matrix_rows(summary)
    lines = ["| " + " | ".join(esc(h) for h in MATRIX_HEADER) + " |", "|---|---|---|---:|---|---:|---:|---:|---|---|"]
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    if tp:
        lines += ["", "| " + " | ".join(TP_HEADER) + " |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        lines += ["| " + " | ".join(r) + " |" for r in tp]
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) == 2 and Path(sys.argv[1]).is_dir():  # summarize a results dir
        d = Path(sys.argv[1])
        s = summarize(d)
        (d / "summary.json").write_text(json.dumps(s, indent=1))
        (d / "summary.md").write_text(markdown_table(s))
        print(markdown_table(s))
        for c in s["checks"]:
            print(f"[{'PASS' if c['ok'] else ('n/a ' if c['ok'] is None else 'FAIL')}] {c['model']} {c['engine']} {c['case']}: {c['check']}  {c['detail'][:160]}")
        print(f"{s['n_pass']}/{s['n_gated']} gated checks pass, {s['n_info']} not resolvable" + (" -- ALL PASS" if s["all_pass"] else " -- SOME FAILED"))
        sys.exit(0)
    a = parse_args()
    if a.stage == "hf-ref":
        hf_ref(a)
    else:
        vllm_stage(a)
