#!/usr/bin/env python
"""LoRA decode overhead vs merge-on-publish for RL rollouts (vllm-metamodels).

The trainer generates with the CURRENT policy as a rank-64 LoRA on every request (vLLM
LoRA kernels on every layer of every decode step) and re-publishes the adapter after every
optimizer step.  This script measures, on ONE engine configuration (``--engine graphs``
= ``VLLM_LENS_CUDA_GRAPHS=1``), with a distinct steering vector per request as in the
rollout workload:

  nolora   LoRA-capable engine (``enable_lora=True``), no LoRARequest        (stage lora_engine)
  lora     the same engine with a rank-64 LoRA on every request              (stage lora_engine)
  merged   ``lens_merge_lora``: adapter merged INTO the base weights in place, no LoRARequest
           (both stages; in stage plain_engine the engine has ``enable_lora=False`` so its CUDA
           graphs contain no LoRA kernels at all)
  plain    plain engine (``enable_lora=False``), no adapter                   (stage plain_engine)

for B in --sizes, --max-tokens new tokens and a 1-token call (decode-step time =
(wall_T - wall_1) / (T - 1)), plus:

  * publish latency of the three merge modes (keep_base = gpu | cpu | none), adapter from a
    PEFT directory and as pickled tensors;
  * correctness: greedy continuations + first-token top-20 log-probs, LoRA path vs merged
    (same adapter), unmerge restores the base bit-exactly (weight fingerprints);
  * drift: --n-publishes successive publishes in keep_base="none" mode (subtract previous,
    add new) vs an exact single-rounding merge, in bf16 ulps / relative Frobenius, and the
    drift of the base after a subtract-unmerge;
  * option (b), EasyNLA-style: push FULL merged weight matrices through CUDA-IPC handles into
    ``model.load_weights`` (timing only; synthetic data, runs last).

The adapter is synthetic (``vllm_lens._lora_merge.synth_adapter``): rank --rank, rsLoRA alpha 16,
random A/B scaled so ||s B A||_F / ||W||_F = --rel-norm per module, over every LoRA-capable
linear layer the served model exposes (the layout comes from the worker: ``lens_lora_layout``).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import pickle
import sys
import time
from typing import Any

PROMPT_TEXT = (
    "The history of computing is a story of abstraction: from relays and vacuum tubes to "
    "transistors, from machine code to compilers, from single programs to operating systems that "
    "share one machine among many users. Each layer hides the one below it, and each hiding "
    "makes the next invention possible. Consider the humble cache, a small fast memory that "
    "remembers what was recently used so that the slow memory behind it is rarely consulted. "
    "Nobody who writes a spreadsheet thinks about caches, yet the spreadsheet would be unusable "
    "without them. The same pattern repeats in networks, in databases, and in the software that "
    "trains and serves large language models, where batching many requests into one pass over "
    "the weights is the abstraction that turns an impossibly expensive computation into a "
    "service. This document describes one such system in detail, beginning with its "
    "scheduler, which decides at every step which requests advance and by how many tokens."
)
GRAPH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def log(msg: str) -> None:
    print(f"[lora {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def make_llm(LLM, kw: dict, log=print):
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
    p.add_argument("--engine", choices=["eager", "graphs"], default="graphs")
    p.add_argument("--stage", choices=["lora_engine", "plain_engine"], default="lora_engine")
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default="/tmp/lora_bench_ref.json", help="lora_engine writes, plain_engine reads (correctness across stages)")
    p.add_argument("--sizes", default="512,1024")
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--prompt-tokens", type=int, default=96)
    p.add_argument("--marker", type=int, default=10)
    p.add_argument("--inject-layer", type=int, default=1)
    p.add_argument("--rank", type=int, default=64)
    p.add_argument("--rel-norm", type=float, default=0.005, help="||s B A||_F / ||W||_F per module of the synthetic adapter")
    p.add_argument("--n-publishes", type=int, default=30)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--n-check", type=int, default=16, help="prompts for the correctness comparison")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--attention-backend", default="")
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument("--skip-ipc", action="store_true", help="skip the option-(b) CUDA-IPC full-matrix push emulation")
    p.add_argument("--skip-cpu-mode", action="store_true")
    p.add_argument("--keep-base", default="auto", help="keep_base for the throughput 'merged' condition: auto|gpu|cpu (auto = gpu if the copy fits)")
    p.add_argument("--exclude-modules", default="", help="comma list of HF module-name suffixes to leave OUT of the synthetic adapter (e.g. in_proj_qkvz,in_proj_ba,out_proj)")
    p.add_argument("--work", default="/tmp/lora_bench")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    max_num_seqs = max(sizes)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if a.engine == "graphs":
        os.environ["VLLM_LENS_CUDA_GRAPHS"] = "1"

    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from vllm_lens import SteeringVector
    from vllm_lens._lora_merge import save_adapter, scaling_from_config, synth_adapter
    from vllm_lens.metamodel import lora_status, merge_lora, unmerge_lora

    try:
        lens_ver = importlib.metadata.version("vllm-lens")
    except importlib.metadata.PackageNotFoundError:
        lens_ver = "missing"
    log(f"vllm {vllm.__version__} | vllm-lens {lens_ver} | torch {torch.__version__} | engine={a.engine} stage={a.stage}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(PROMPT_TEXT, add_special_tokens=False)["input_ids"]
    while len(ids) < a.prompt_tokens:
        ids = ids + ids
    prompt_ids = [int(t) for t in ids[: a.prompt_tokens]]
    P, T = a.prompt_tokens, a.max_tokens
    max_len = P + T + 8
    lora_engine = a.stage == "lora_engine"

    kw: dict[str, Any] = dict(
        model=a.model, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_mem, max_model_len=max_len,
        enable_prefix_caching=False, max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max(8192, max_num_seqs * (P + 8)), dtype="bfloat16", seed=a.seed,
    )
    if a.attention_backend:
        kw["attention_backend"] = a.attention_backend
    if a.language_model_only:
        kw["language_model_only"] = True
    if lora_engine:
        kw.update(enable_lora=True, max_loras=2, max_lora_rank=a.rank)
    if a.engine == "eager":
        kw["enforce_eager"] = True
    else:
        kw["compilation_config"] = {"cudagraph_capture_sizes": sorted({s for s in GRAPH_SIZES + sizes if s <= max_num_seqs})}
    t0 = time.perf_counter()
    llm, kw = make_llm(LLM, kw, log)
    engine_up_s = time.perf_counter() - t0
    vc = llm.llm_engine.vllm_config
    cc = vc.compilation_config
    resolved = {
        "enforce_eager": bool(vc.model_config.enforce_eager),
        "compilation_mode": str(getattr(cc.mode, "name", cc.mode)),
        "cudagraph_mode": str(getattr(cc.cudagraph_mode, "name", cc.cudagraph_mode)),
        "max_num_seqs": vc.scheduler_config.max_num_seqs,
        "enable_lora": bool(getattr(vc, "lora_config", None)),
        "num_layers": vc.model_config.get_num_layers(vc.parallel_config),
        "model_runner": "v2" if getattr(vc, "use_v2_model_runner", False) else "v1",
    }
    log(f"engine up in {engine_up_s:.0f}s | {resolved}")

    result: dict[str, Any] = {
        "model": a.model, "engine": a.engine, "stage": a.stage,
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__, "vllm_lens": lens_ver},
        "gpu": torch.cuda.get_device_name(0), "engine_kwargs": {k: (v if not isinstance(v, dict) else dict(v)) for k, v in kw.items()},
        "resolved_config": resolved, "engine_up_s": engine_up_s, "prompt_tokens": P, "max_tokens": T,
        "rank": a.rank, "rel_norm": a.rel_norm, "sizes": sizes,
        "throughput": [], "publish": [], "correctness": {}, "drift": {}, "ipc_push": {}, "stats": {},
    }

    def dump() -> None:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)

    def rpc(name: str, *args: Any) -> Any:
        r = llm.collective_rpc(name, args=args)
        return r[0] if isinstance(r, (list, tuple)) else r

    # ---------------- adapter synthesis from the served model's own layout ----------------
    layout = rpc("lens_lora_layout", True)
    result["layout_engine"] = [{"vllm_name": t["vllm_name"], "kind": t["kind"], "shape": t["shape"], "subs": [s["hf_name"] for s in t["subs"]]} for t in layout]
    if not lora_engine and os.path.exists(a.ref):
        # the plain engine fuses in_proj_qkvz = [q|k|v|z] (HF: in_proj_qkv + in_proj_z, split known only from the
        # adapter); use the LoRA engine's layout (saved by stage 1) so BOTH stages synthesise the SAME adapter
        try:
            with open(a.ref) as f:
                ref_layout = json.load(f).get("layout_full")
            if ref_layout:
                layout = ref_layout
                log(f"using the LoRA-engine stage's layout for adapter synthesis ({len(layout)} params)")
        except Exception as e:  # noqa: BLE001
            log(f"could not reuse the LoRA-engine layout: {e!r}")
    if any(s.get("out") is None for t in layout for s in t["subs"]):
        raise SystemExit("layout has deferred slices (out=None) and no LoRA-engine layout to synthesise from; run the lora_engine stage first")
    if a.exclude_modules:
        excl = tuple(x.strip() for x in a.exclude_modules.split(",") if x.strip())
        for t in layout:
            t["subs"] = [s for s in t["subs"] if not s["hf_name"].endswith(excl)]
        layout = [t for t in layout if t["subs"]]
        log(f"excluding modules ending in {excl}: {len(layout)} vLLM params remain")
    result["exclude_modules"] = a.exclude_modules
    result["layout_all"] = [{"vllm_name": t["vllm_name"], "kind": t["kind"], "shape": t["shape"], "subs": [s["hf_name"] for s in t["subs"]]} for t in layout]
    result["layout_from_ref"] = layout is not None and not lora_engine and os.path.exists(a.ref)
    n_hf = sum(len(t["subs"]) for t in layout)
    targeted_params = sum(math.prod(t["shape"]) for t in layout)
    lora_params = sum(a.rank * (s["in"] + s["out"]) for t in layout for s in t["subs"])
    result["layout"] = {
        "n_vllm_params": len(layout), "n_hf_modules": n_hf, "targeted_params": targeted_params,
        "targeted_bytes_bf16": targeted_params * 2, "lora_params": lora_params, "kinds": sorted({t["kind"] for t in layout}),
        "example": layout[0] if layout else None,
    }
    log(f"layout: {len(layout)} vLLM params / {n_hf} HF modules, {targeted_params/1e9:.2f}B targeted params, {lora_params/1e6:.0f}M LoRA params")
    os.makedirs(a.work, exist_ok=True)

    def adapter(seed: int) -> tuple[dict, dict]:
        return synth_adapter(layout, a.rank, a.rel_norm, seed=1000 + seed)

    t1 = time.perf_counter()
    a0_t, a0_cfg = adapter(0)
    a0_path = os.path.join(a.work, "a0")
    save_adapter(a0_path, a0_t, a0_cfg)
    a1_t, a1_cfg = adapter(1)
    a1_path = os.path.join(a.work, "a1")
    save_adapter(a1_path, a1_t, a1_cfg)
    scaling = scaling_from_config(a0_cfg)
    result["layout"]["synth_s"] = time.perf_counter() - t1
    result["layout"]["scaling"] = scaling
    log(f"adapters a0/a1 synthesised + saved in {time.perf_counter() - t1:.1f}s (scaling {scaling:.3f})")
    del a0_t

    # ---------------- steering vectors (one per request, like the trainer) ----------------
    g = torch.Generator().manual_seed(1234)

    def capture_clean_norm() -> float:
        sp = SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": [a.inject_layer]})
        out = llm.generate([{"prompt_token_ids": prompt_ids}], [sp], use_tqdm=False)[0]
        act = out.activations["residual_stream"].float()
        return float(act[0, a.marker].norm())

    hnorm = capture_clean_norm()
    D = int(rpc("lens_lora_layout", False)[0]["subs"][0]["in"]) if layout else 0
    vecs = torch.nn.functional.normalize(torch.randn(max(sizes + [a.n_check]), D, generator=g), dim=-1)
    log(f"|h_marker| at layer {a.inject_layer} = {hnorm:.2f}; hidden {D}")

    def extra(i: int) -> dict:
        return {"apply_steering_vectors": [SteeringVector(
            activations=(vecs[i] * hnorm).view(1, 1, D), layer_indices=[a.inject_layer], scale=1.0,
            norm_match=False, position_indices=[a.marker])]}

    def params(B: int, T: int, greedy: bool = False, logprobs: int | None = None) -> list:
        return [SamplingParams(temperature=0.0 if greedy else 1.0, top_p=1.0, max_tokens=T, min_tokens=T,
                               ignore_eos=True, logprobs=logprobs, extra_args=extra(i)) for i in range(B)]

    lora_a0 = LoRARequest(lora_name="a0", lora_int_id=1, lora_path=a0_path)

    def gen(B: int, T: int, lora_req, greedy: bool = False, logprobs: int | None = None):
        prompts = [{"prompt_token_ids": prompt_ids} for _ in range(B)]
        t = time.perf_counter()
        outs = llm.generate(prompts, params(B, T, greedy, logprobs), lora_request=lora_req, use_tqdm=False)
        return time.perf_counter() - t, outs

    def stats(tag: str) -> None:
        try:
            result["stats"][tag] = rpc("steering_stats", True)
        except Exception as e:  # noqa: BLE001
            result["stats"][tag] = {"error": repr(e)}

    def run_condition(cond: str, lora_req) -> None:
        gen(min(8, max_num_seqs), T, lora_req)  # warm-up (graphs, LoRA slot load)
        stats(f"warmup_{cond}")
        for B in sizes:
            for rep in range(a.repeats):
                wall_T, outs = gen(B, T, lora_req)
                wall_1, _ = gen(B, 1, lora_req)
                n_gen = sum(len(o.token_ids) for out in outs for o in out.outputs)
                row = {"condition": cond, "batch": B, "rep": rep, "wall_s": wall_T, "wall_1tok_s": wall_1,
                       "decode_step_ms": (wall_T - wall_1) / max(T - 1, 1) * 1000.0, "gen_tokens": n_gen,
                       "tok_per_s": n_gen / wall_T, "lora_status": lora_status(llm)}
                result["throughput"].append(row)
                log(f"{cond:8s} B={B:5d} rep{rep}: {wall_T:6.2f}s ({n_gen / wall_T:8.0f} tok/s) | prefill+1tok {wall_1:5.2f}s | decode step {row['decode_step_ms']:6.1f} ms")
                dump()
        stats(cond)

    def check_outputs(tag: str, lora_req) -> dict:
        _, outs = gen(a.n_check, 16, lora_req, greedy=True, logprobs=20)
        return {
            "tag": tag,
            "tokens": [[int(t) for t in o.outputs[0].token_ids] for o in outs],
            "top20": [{str(tid): float(lp.logprob) for tid, lp in o.outputs[0].logprobs[0].items()} for o in outs],
        }

    def compare(x: dict, y: dict) -> dict:
        n = len(x["tokens"])
        argmax_eq = sum(int(xt[0] == yt[0]) for xt, yt in zip(x["tokens"], y["tokens"]))
        tok_agree = [sum(int(p == q) for p, q in zip(xt, yt)) / max(len(xt), 1) for xt, yt in zip(x["tokens"], y["tokens"])]
        maxd = []
        for xt, yt in zip(x["top20"], y["top20"]):
            common = set(xt) & set(yt)
            maxd.append(max((abs(xt[t] - yt[t]) for t in common), default=float("inf")))
        return {"a": x["tag"], "b": y["tag"], "n": n, "argmax_equal": argmax_eq, "mean_token_agreement": sum(tok_agree) / n,
                "min_token_agreement": min(tok_agree), "max_abs_dlogprob_top20": max(maxd), "median_abs_dlogprob_top20": sorted(maxd)[n // 2]}

    fp0 = rpc("lens_weight_fingerprint")
    result["correctness"]["fingerprint_params"] = len(fp0)

    def fp_equal(fp) -> dict:
        diff = max((abs(fp[k][1] - fp0[k][1]) / max(abs(fp0[k][1]), 1e-30) for k in fp0), default=0.0)
        return {"equal": all(fp[k] == fp0[k] for k in fp0), "max_rel_sumsq_diff": diff}

    # ---------------- throughput conditions ----------------
    ref: dict[str, Any] = {}
    if lora_engine:
        run_condition("nolora", None)
        run_condition("lora", lora_a0)
        ref["nolora"] = check_outputs("nolora", None)
        ref["lora"] = check_outputs("lora", lora_a0)
    else:
        run_condition("plain", None)
        ref["plain"] = check_outputs("plain", None)

    m = merge_lora(llm, a0_path, keep_base=a.keep_base)
    result["publish"].append({**m, "source": "dir", "adapter": "a0", "phase": "throughput"})
    result["merge_mode_throughput"] = m["mode"]
    log(f"merged a0 ({m['mode']} mode) in {m['publish_s']:.2f}s: {m['n_params']} params, base copy {m['base_bytes']/1e9:.1f} GB on {m.get('base_where')}")
    run_condition("merged", None)
    ref["merged"] = check_outputs("merged", None)
    u = unmerge_lora(llm)
    fp = fp_equal(rpc("lens_weight_fingerprint"))
    result["correctness"]["unmerge_gpu_restores_base"] = {**u, **fp}
    log(f"unmerge ({u['how']}) in {u['unmerge_s']:.2f}s -> base restored exactly: {fp['equal']}")
    after = check_outputs("nolora_after_unmerge" if lora_engine else "plain_after_unmerge", None)
    base_tag = "nolora" if lora_engine else "plain"
    result["correctness"]["after_unmerge_vs_before"] = compare(ref[base_tag], after)
    if lora_engine:
        result["correctness"]["lora_vs_merged"] = compare(ref["lora"], ref["merged"])
        result["correctness"]["nolora_vs_lora_control"] = compare(ref["nolora"], ref["lora"])
        ref["layout_names"] = sorted(t["vllm_name"] for t in layout)
        ref["layout_full"] = layout
        with open(a.ref, "w") as f:
            json.dump(ref, f)
    else:
        try:
            with open(a.ref) as f:
                ref_prev = json.load(f)
            result["correctness"]["merged_plain_vs_merged_lora_engine"] = compare(ref_prev["merged"], ref["merged"])
            result["correctness"]["merged_plain_vs_lora_path"] = compare(ref_prev["lora"], ref["merged"])
            result["correctness"]["plain_vs_nolora_lora_engine"] = compare(ref_prev["nolora"], ref["plain"])
            mine = sorted(t["vllm_name"] for t in layout)
            theirs = ref_prev.get("layout_names", [])
            result["layout_diff_vs_lora_engine"] = {"only_lora_engine": sorted(set(theirs) - set(mine))[:20], "only_plain_engine": sorted(set(mine) - set(theirs))[:20]}
            log(f"layout diff vs LoRA engine: {result['layout_diff_vs_lora_engine']}")
        except Exception as e:  # noqa: BLE001
            result["correctness"]["cross_stage_error"] = repr(e)
    for k, v in result["correctness"].items():
        if isinstance(v, dict) and "argmax_equal" in v:
            log(f"correctness {k}: argmax {v['argmax_equal']}/{v['n']}, token agreement {v['mean_token_agreement']:.3f}, max|dlogprob| {v['max_abs_dlogprob_top20']:.4f}")
    dump()

    # ---------------- publish latency: modes x sources ----------------
    exact_mode = result["merge_mode_throughput"] if result.get("merge_mode_throughput") in ("gpu", "cpu") else "gpu"
    modes = [exact_mode] + ([] if (a.skip_cpu_mode or not lora_engine or exact_mode == "cpu") else ["cpu"]) + (["none"] if lora_engine else [])
    a1_payload = pickle.dumps({k: v for k, v in a1_t.items()}, protocol=pickle.HIGHEST_PROTOCOL)
    result["publish_payload_bytes"] = len(a1_payload)
    for mode in modes:
        for adapter_tag, path in (("a0", a0_path), ("a1", a1_path), ("a0", a0_path)):
            m = merge_lora(llm, path, keep_base=mode)
            result["publish"].append({**m, "source": "dir", "adapter": adapter_tag, "phase": "latency"})
            log(f"publish {adapter_tag} mode={mode} from dir: {m['publish_s']:.3f}s (resolve {m['resolve_s']:.3f} snapshot {m['snapshot_s']:.3f} apply {m['apply_s']:.3f})")
        t = time.perf_counter()
        m = rpc("lens_merge_lora", None, a1_payload, scaling, mode)
        m["rpc_s"] = time.perf_counter() - t
        result["publish"].append({**m, "source": "pickled_tensors", "adapter": "a1", "phase": "latency"})
        log(f"publish a1 mode={mode} pickled tensors ({len(a1_payload)/1e6:.0f} MB): rpc {m['rpc_s']:.3f}s (apply {m['apply_s']:.3f})")
        how = "subtract" if mode == "none" else "auto"
        u = rpc("lens_unmerge_lora", True, how)
        fp = fp_equal(rpc("lens_weight_fingerprint"))
        result["correctness"][f"unmerge_after_{mode}_mode"] = {**u, **fp}
        log(f"unmerge ({u['how']}, release) after {mode} mode: base exact {fp['equal']} (max rel sumsq diff {fp['max_rel_sumsq_diff']:.2e})")
        dump()

    # ---------------- drift: n publishes in keep_base="none" mode vs exact ----------------
    if lora_engine and a.n_publishes > 0:
        merge_lora(llm, a0_path, keep_base=exact_mode)  # snapshot W0 (gpu if it fits, else pinned host)
        unmerge_lora(llm)  # copies retained
        t_all = time.perf_counter()
        pub_s = []
        for k in range(1, a.n_publishes + 1):
            ak_t, _ = adapter(k)
            payload = pickle.dumps(ak_t, protocol=pickle.HIGHEST_PROTOCOL)
            del ak_t
            m = rpc("lens_merge_lora", None, payload, scaling, "none")
            pub_s.append(m["publish_s"])
            if k % 10 == 0 or k == 1:
                log(f"drift test: publish {k}/{a.n_publishes} in none mode ({m['publish_s']:.3f}s)")
        m = rpc("lens_merge_lora", a0_path, None, None, "none")
        drift = rpc("lens_lora_compare_exact", a0_path)
        drifted = check_outputs("merged_none_drifted", None)
        result["drift"] = {"n_publishes": a.n_publishes, "publish_s_mean": sum(pub_s) / len(pub_s), "total_s": time.perf_counter() - t_all,
                           "vs_exact_merge": drift, "outputs_vs_exact_merged": compare(ref["merged"], drifted)}
        log(f"drift after {a.n_publishes} none-mode publishes: max {drift['max_diff_ulps']:.2f} ulp, rel Frobenius {drift['rel_frobenius']:.2e}, "
            f"{drift['frac_changed']*100:.2f}% elements differ; outputs vs exact: {result['drift']['outputs_vs_exact_merged']}")
        u = rpc("lens_unmerge_lora", False, "subtract")
        base_drift = rpc("lens_lora_compare_exact", None)
        result["drift"]["base_after_subtract_unmerge"] = {**u, **base_drift}
        log(f"base after subtract-unmerge: max {base_drift['max_diff_ulps']:.2f} ulp, rel Frobenius {base_drift['rel_frobenius']:.2e}, {base_drift['frac_changed']*100:.2f}% changed")
        # exact restore
        merge_lora(llm, a0_path, keep_base=exact_mode)
        u = rpc("lens_unmerge_lora", True, "copy")
        fp = fp_equal(rpc("lens_weight_fingerprint"))
        result["drift"]["exact_restore"] = {**u, **fp}
        log(f"exact restore via copy: {fp['equal']}")
        dump()

    # ---------------- option (b): EasyNLA-style full-matrix push over CUDA IPC (timing only) ----------------
    # Only on the plain engine: with enable_lora=True vLLM wraps the linears (qkv_proj.base_layer.weight)
    # and the model's own load_weights no longer finds "qkv_proj.weight" (KeyError) -- a vLLM limitation
    # that also applies to EasyNLA-style syncs into a LoRA-capable engine.
    if not a.skip_ipc and lora_engine:
        result["ipc_push"] = {"skipped": "load_weights cannot target LoRA-wrapped linears (enable_lora=True); see plain_engine stage"}
        dump()
    if not a.skip_ipc and not lora_engine:
        try:
            _ipc_push(a, layout, rpc, result, torch)
        except Exception as e:  # noqa: BLE001 -- timing-only emulation; never lose the run over it
            result["ipc_push"] = {**result.get("ipc_push", {}), "error": repr(e)[:400]}
            log(f"option (b) IPC push failed: {e!r}")
        dump()
    log("done")


def _ipc_push(a, layout, rpc, result, torch) -> None:
    """Option (b): full merged matrices, one CUDA-IPC handle per tensor, layer by layer (EasyNLA's
    pattern), into ``model.load_weights`` on the worker.  Synthetic data -> weights are garbage
    afterwards, so this runs last."""
    import pickle as _pickle

    from torch.multiprocessing.reductions import reduce_tensor

    dev = torch.device("cuda:0")
    buckets: dict[str, list[tuple[str, list[int]]]] = {}
    for t in layout:
        for s in t["subs"]:
            key = s["hf_name"].split(".layers.")[1].split(".")[0] if ".layers." in s["hf_name"] else "_other"
            buckets.setdefault(key, []).append((s["hf_name"] + ".weight", [s["out"], s["in"]]))
    t_total = 0.0
    per = []
    loaded_total = 0
    n_bytes = 0
    for key in sorted(buckets, key=lambda k: (k == "_other", int(k) if k.isdigit() else -1)):
        tensors = [(name, torch.empty(shape, dtype=torch.bfloat16, device=dev).uniform_(-0.02, 0.02)) for name, shape in buckets[key]]
        torch.cuda.synchronize()
        n_bytes += sum(tn.numel() * 2 for _, tn in tensors)
        t0 = time.perf_counter()
        handles = _pickle.dumps([(name, reduce_tensor(tn)) for name, tn in tensors], protocol=_pickle.HIGHEST_PROTOCOL)
        r = rpc("lens_load_weights_ipc", handles)
        dt = time.perf_counter() - t0
        t_total += dt
        loaded_total += int(r.get("loaded") or 0)
        per.append({"bucket": key, "n": r["n"], "loaded": r.get("loaded"), "s": dt, "rebuild_s": r["rebuild_s"], "load_s": r["load_s"]})
        del tensors
    result["ipc_push"] = {"total_s": t_total, "bytes": n_bytes, "gb_per_s": n_bytes / 1e9 / max(t_total, 1e-9), "n_buckets": len(per),
                          "loaded_total": loaded_total, "per_bucket": per[:8]}
    log(f"option (b) IPC push of {n_bytes/1e9:.1f} GB merged matrices in {len(per)} buckets: {t_total:.2f}s ({loaded_total} params matched by load_weights)")
    name, shape = buckets[sorted(buckets)[0]][0]
    cpu_t = torch.empty(shape, dtype=torch.bfloat16).uniform_(-0.02, 0.02)
    t0 = time.perf_counter()
    r = rpc("lens_load_weights_ipc", _pickle.dumps([(name, cpu_t)], protocol=_pickle.HIGHEST_PROTOCOL))
    result["ipc_push"]["cpu_pickle_one_tensor"] = {"name": name, "bytes": cpu_t.numel() * 2, "s": time.perf_counter() - t0, **r}
    log(f"option (b) via CPU pickle: {cpu_t.numel()*2/1e6:.0f} MB tensor in {result['ipc_push']['cpu_pickle_one_tensor']['s']:.2f}s")


if __name__ == "__main__":
    main()
