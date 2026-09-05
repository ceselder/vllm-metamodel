#!/usr/bin/env python
"""Reproducible per-request steering benchmark for vllm-lens (RL-rollout style).

The workload is the "meta-model" / activation-oracle rollout setting: every
request in a batch carries its OWN steering vector, applied at ONE layer on
ONE prompt position (the request's marker token), batches of 8 .. 2048
requests, ~40 generated tokens each.

Runs in ONE process with ONE vLLM engine configuration (``--engine``) and
measures generation throughput vs batch size for the conditions that engine
supports, plus fixed correctness probes whose outputs are saved so a driver
(``bench/modal_bench.py`` / ``bench/compare.py``) can assert that the
installed vllm-lens (stock 1.1.0 or vllm-lens-metamodel, any apply mode) steers
identically.

Engine modes
  eager   plugin active, ``enforce_eager`` (what stock 1.1.0 always forces)
  graphs  vllm-lens-metamodel only: ``VLLM_LENS_CUDA_GRAPHS=1`` -> compilation mode
          NONE + ``cudagraph_mode=FULL_DECODE_ONLY`` (the plugin fills these in)
  compile vllm-metamodels post7: ``VLLM_LENS_CUDA_GRAPHS=1 VLLM_LENS_COMPILE=1`` -> vLLM's
          torch.compile stays on, hooks run as the custom op, decode-only full graphs
  plain   ``VLLM_LENS_DISABLE=1``: vLLM with its default compilation
          (torch.compile + CUDA graphs), no hooks -- the no-steering ceiling

Conditions (fork engines run all that apply; stock runs nosteer + steer3d + steer2d)
  nosteer      B plain requests, hooks never installed -> ceiling for this engine config
  steer3d_loop fork: indexed hook, per-row ``_apply_steering`` loop, one ``set_steering_data_many`` RPC
  steer3d      fork: indexed hook, vectorised ``index_add_`` apply, one ``set_steering_block`` RPC
               stock: its per-layer ``startswith`` scan + one RPC per request
  steer2d      eager only: one (1, D) broadcast vector per request at ``--mid-layer``
               (classic "steer every token"; sizes capped by ``--sizes-2d``)

Every request uses the same ``--prompt-tokens``-token prompt (prefix caching is
off) and generates exactly ``--max-tokens`` tokens.  Output: one JSON file.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
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
GRAPH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def log(msg: str) -> None:
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True)
    p.add_argument("--engine", choices=["eager", "graphs", "compile", "plain"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--sizes",
        default="8,32,128,512,1024,2048",
        help="batch sizes (requests per generate call)",
    )
    p.add_argument(
        "--sizes-2d",
        default="8,32,128,512",
        help="batch sizes for the broadcast (steer2d) condition",
    )
    p.add_argument("--max-tokens", type=int, default=40)
    p.add_argument("--prompt-tokens", type=int, default=96)
    p.add_argument("--inject-layer", type=int, default=1)
    p.add_argument(
        "--marker", type=int, default=10, help="prompt position steered by steer3d"
    )
    p.add_argument(
        "--mid-layer",
        type=int,
        default=-1,
        help="layer for steer2d (-1 = n_layers // 2)",
    )
    p.add_argument("--max-num-seqs", type=int, default=0, help="0 = max(sizes)")
    p.add_argument(
        "--max-capture-size",
        type=int,
        default=0,
        help="largest CUDA-graph capture size (0 = max_num_seqs); larger batches run without a graph",
    )
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument(
        "--attention-backend",
        default="",
        help="e.g. TRITON_ATTN (empty = vLLM default)",
    )
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument(
        "--conditions", default="", help="override: comma list of conditions"
    )
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--enable-lora",
        action="store_true",
        help="enable LoRA slots (rank 64, 2 slots) as an RL rollout engine would",
    )
    p.add_argument(
        "--no-packed-decode",
        action="store_true",
        help="VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0 (GDN models)",
    )
    p.add_argument(
        "--model-runner",
        choices=["default", "v1", "v2"],
        default="default",
        help="vLLM >= 0.23: force the V1 or V2 GPU model runner (VLLM_USE_V2_MODEL_RUNNER); "
        "'default' leaves vLLM's choice (the plugin itself forces V1 whenever it is active)",
    )
    p.add_argument(
        "--capture-mode",
        choices=["list", "max"],
        default="list",
        help="CUDA-graph sizes: explicit list (GRAPH_SIZES + batch sizes) or vLLM defaults up to max_cudagraph_capture_size",
    )
    p.add_argument(
        "--prefix-caching",
        action="store_true",
        help="enable_prefix_caching=True (post7: steered requests share the template prefix; steered blocks salted)",
    )
    p.add_argument(
        "--cache-salt",
        default="",
        help="extra_args['lens_cache_salt'] for steered requests: '' = nonce (default), 'payload' = identical (prompt, vector) rows share",
    )
    return p.parse_args()


def installed_variant() -> dict[str, Any]:
    """Which vllm-lens is installed: stock 1.1.0 or vllm-lens-metamodel."""
    try:
        ver = importlib.metadata.version("vllm-lens")
    except importlib.metadata.PackageNotFoundError:
        ver = "missing"
    try:
        from vllm_lens import _worker_ext as W

        is_fork = hasattr(W.HiddenStatesExtension, "set_steering_block")
    except Exception:  # noqa: BLE001
        is_fork = False
    return {"vllm_lens_version": ver, "variant": "fork" if is_fork else "stock"}


def main() -> None:
    a = parse_args()
    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    sizes_2d = [int(s) for s in a.sizes_2d.split(",") if s.strip()]
    max_num_seqs = a.max_num_seqs or max(sizes)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Read when the plugin registers / when the engine config is built -> set before importing vLLM.
    if a.engine == "plain":
        os.environ["VLLM_LENS_DISABLE"] = "1"
    elif a.engine == "graphs":
        os.environ["VLLM_LENS_CUDA_GRAPHS"] = "1"
    elif a.engine == "compile":
        os.environ["VLLM_LENS_CUDA_GRAPHS"] = "1"
        os.environ["VLLM_LENS_COMPILE"] = "1"
    if a.no_packed_decode:
        os.environ["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"] = "0"
    if a.model_runner != "default":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if a.model_runner == "v2" else "0"

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    variant = installed_variant()
    is_fork = variant["variant"] == "fork"
    log(
        f"vllm {vllm.__version__} | vllm-lens {variant} | torch {torch.__version__} | engine={a.engine}"
    )
    if a.engine in ("graphs", "compile", "plain") and not is_fork:
        sys.exit(
            f"--engine {a.engine} needs vllm-lens-metamodel (stock 1.1.0 forces enforce_eager, has no disable switch)"
        )

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(PROMPT_TEXT, add_special_tokens=False)["input_ids"]
    while len(ids) < a.prompt_tokens:
        ids = ids + ids
    prompt_ids = [int(t) for t in ids[: a.prompt_tokens]]
    P, T = a.prompt_tokens, a.max_tokens
    max_len = P + T + 8

    kw: dict[str, Any] = dict(
        model=a.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=max_len,
        enable_prefix_caching=bool(a.prefix_caching),
        max_num_seqs=max_num_seqs,
        # never chunk a prompt: token budget covers every sequence's full prompt at once
        max_num_batched_tokens=max(8192, max_num_seqs * (P + 8)),
        dtype="bfloat16",
        seed=a.seed,
    )
    if a.prefix_caching:
        kw["disable_log_stats"] = False  # prefix-cache hit counters via llm.get_metrics()
    if a.attention_backend:
        kw["attention_backend"] = a.attention_backend
    if a.language_model_only:
        kw["language_model_only"] = True
    if a.enable_lora:
        kw.update(enable_lora=True, max_loras=2, max_lora_rank=64)
    max_capture = a.max_capture_size or max_num_seqs
    capture_sizes = sorted(
        {s for s in GRAPH_SIZES + sizes if s <= min(max_num_seqs, max_capture)}
    )
    cc_kw = (
        {"max_cudagraph_capture_size": min(max_num_seqs, max_capture)}
        if a.capture_mode == "max"
        else {"cudagraph_capture_sizes": capture_sizes}
    )
    if a.engine == "eager":
        kw["enforce_eager"] = True
    elif a.engine in ("graphs", "compile"):
        # mode / cudagraph_mode deliberately NOT given: the plugin must fill in
        # mode=NONE + FULL_DECODE_ONLY itself (VLLM_LENS_CUDA_GRAPHS=1), or keep vLLM's
        # compile mode + FULL_DECODE_ONLY (VLLM_LENS_COMPILE=1).
        kw["compilation_config"] = dict(cc_kw)
    else:  # plain: vLLM defaults (torch.compile + CUDA graphs), capture sizes matched to the batches
        kw["compilation_config"] = dict(cc_kw)

    t0 = time.perf_counter()
    llm, kw = make_llm(LLM, kw, log)
    engine_up_s = time.perf_counter() - t0
    resolved: dict[str, Any] = {}
    try:
        vc = llm.llm_engine.vllm_config
        cc = vc.compilation_config
        resolved = {
            "enforce_eager": bool(vc.model_config.enforce_eager),
            "compilation_mode": str(getattr(cc.mode, "name", cc.mode)),
            "cudagraph_mode": str(
                getattr(cc.cudagraph_mode, "name", cc.cudagraph_mode)
            ),
            "cudagraph_capture_sizes": list(cc.cudagraph_capture_sizes or []),
            "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
            "max_num_seqs": vc.scheduler_config.max_num_seqs,
            "num_layers": vc.model_config.get_num_layers(vc.parallel_config),
            "enable_prefix_caching": bool(vc.cache_config.enable_prefix_caching),
        }
    except Exception as e:  # noqa: BLE001
        resolved = {"error": repr(e)}
    if is_fork and a.engine != "plain":
        try:
            from vllm_lens.metamodel import capabilities as _caps

            resolved["lens_capabilities"] = {k: v for k, v in _caps(llm).items() if k in ("compile_op", "prompt_only", "prefix_caching", "kv_salt_active", "early_exit")}
        except Exception as e:  # noqa: BLE001
            resolved["lens_capabilities"] = {"error": repr(e)}
    log(f"engine up in {engine_up_s:.0f}s | resolved {resolved}")
    n_layers = int(resolved.get("num_layers") or 0)
    mid_layer = a.mid_layer if a.mid_layer >= 0 else max(1, n_layers // 2)

    result: dict[str, Any] = {
        "model": a.model,
        "engine": a.engine,
        "variant": variant,
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__},
        "gpu": torch.cuda.get_device_name(0),
        "engine_kwargs": {k: v for k, v in kw.items()},
        "resolved_config": resolved,
        "engine_up_s": engine_up_s,
        "prompt_tokens": P,
        "max_tokens": T,
        "inject_layer": a.inject_layer,
        "marker": a.marker,
        "mid_layer": mid_layer,
        "sizes": sizes,
        "sizes_2d": sizes_2d,
        "max_capture_size": max_capture,
        "capture_mode": a.capture_mode,
        "enable_lora": a.enable_lora,
        "prefix_caching": bool(a.prefix_caching),
        "cache_salt": a.cache_salt or "nonce",
        "packed_decode": not a.no_packed_decode,
        "model_runner": a.model_runner,
        "model_runner_resolved": "v2" if getattr(getattr(llm.llm_engine, "vllm_config", None), "use_v2_model_runner", False) else "v1",
        "throughput": [],
        "probes": {},
        "stats": {},
    }

    def dump() -> None:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)

    def stats(tag: str) -> None:
        if a.prefix_caching:
            try:
                c = {m.name: m.value for m in llm.get_metrics() if m.name in ("vllm:prefix_cache_queries", "vllm:prefix_cache_hits")}
                result.setdefault("cache_counters", {})[tag] = c
            except Exception as e:  # noqa: BLE001
                result.setdefault("cache_counters", {})[tag] = {"error": repr(e)}
        if not is_fork or a.engine == "plain":
            return
        try:
            result["stats"][tag] = llm.collective_rpc("steering_stats", args=(True,))[0]
        except Exception as e:  # noqa: BLE001
            result["stats"][tag] = {"error": repr(e)}

    def set_mode(vectorized: bool, block_rpc: bool) -> None:
        """fork only: apply mode (worker RPC) + RPC packing (plugin, this process)."""
        if not is_fork or a.engine == "plain":
            return
        llm.collective_rpc("set_vectorized", args=(bool(vectorized),))
        os.environ["VLLM_LENS_BLOCK_RPC"] = "1" if block_rpc else "0"

    def gen_params(extra: dict | None) -> SamplingParams:
        return SamplingParams(
            temperature=1.0,
            top_p=1.0,
            max_tokens=T,
            min_tokens=T,
            ignore_eos=True,
            extra_args=extra,
        )

    def timed_generate(cond: str, B: int, make_extra, rep: int) -> None:
        params = [gen_params(make_extra(i)) for i in range(B)]
        prompts = [{"prompt_token_ids": prompt_ids} for _ in range(B)]
        t1 = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        wall = time.perf_counter() - t1
        n_gen = sum(len(o.token_ids) for out in outs for o in out.outputs)
        row = {
            "condition": cond,
            "batch": B,
            "rep": rep,
            "wall_s": wall,
            "gen_tokens": n_gen,
            "tok_per_s": n_gen / wall,
            "seq_per_s": B / wall,
            "prompt_tokens_total": B * P,
            "sample": tok.decode(
                outs[0].outputs[0].token_ids, skip_special_tokens=True
            )[:80],
        }
        result["throughput"].append(row)
        log(
            f"{cond:13s} B={B:5d} rep{rep}: {wall:7.2f}s  {row['tok_per_s']:8.0f} tok/s  {row['seq_per_s']:6.1f} seq/s"
        )
        dump()

    def run_condition(cond: str, make_extra, cond_sizes: list[int]) -> None:
        if not a.no_warmup:
            wb = min(8, max(cond_sizes))
            llm.generate(
                [{"prompt_token_ids": prompt_ids}] * wb,
                [gen_params(make_extra(i)) for i in range(wb)],
                use_tqdm=False,
            )
        stats(f"warmup_{cond}")
        for B in cond_sizes:
            for rep in range(a.repeats):
                timed_generate(cond, B, make_extra, rep)
        stats(cond)

    if a.conditions:
        conds = [c for c in a.conditions.split(",") if c]
    elif a.engine == "plain":
        conds = ["nosteer"]
    elif not is_fork:
        conds = ["nosteer", "steer3d", "steer2d"]
    else:
        conds = ["nosteer", "steer3d_loop", "steer3d"] + (
            ["steer2d"] if a.engine == "eager" else []
        )

    # ---------------- ceiling: no steering, hooks not installed ----------------
    if "nosteer" in conds:
        run_condition("nosteer", lambda i: None, sizes)
    if a.engine == "plain":
        dump()
        log("plain engine done")
        return

    # ---------------- probes (install hooks, measure |h_marker|) ----------------
    from vllm_lens import SteeringVector

    def capture(
        layers: list[int],
        steer: list | None,
        max_tokens: int = 1,
        logprobs: int | None = None,
    ):
        extra: dict[str, Any] = {"output_residual_stream": layers}
        if steer is not None:
            extra["apply_steering_vectors"] = steer
        sp = SamplingParams(
            temperature=0.0, max_tokens=max_tokens, logprobs=logprobs, extra_args=extra
        )
        out = llm.generate([{"prompt_token_ids": prompt_ids}], [sp], use_tqdm=False)[0]
        act = getattr(out, "activations", None)
        assert act is not None and "residual_stream" in act, (
            "capture returned nothing -- hooks not live?"
        )
        return out, act["residual_stream"].float()

    def topk(out) -> dict[str, float]:
        lp = out.outputs[0].logprobs
        return {str(tid): float(v.logprob) for tid, v in lp[0].items()} if lp else {}

    g = torch.Generator().manual_seed(1234)
    _, h_clean = capture([a.inject_layer], None)
    D = h_clean.shape[-1]
    hnorm = h_clean[0, a.marker].norm().item()
    unit = torch.nn.functional.normalize(torch.randn(D, generator=g), dim=0)
    unit2 = torch.nn.functional.normalize(torch.randn(D, generator=g), dim=0)
    probe_vec = SteeringVector(
        activations=(unit * hnorm).view(1, 1, D),
        layer_indices=[a.inject_layer],
        scale=1.0,
        norm_match=False,
        position_indices=[a.marker],
    )
    probe_vec_nm = SteeringVector(
        activations=unit2.view(1, 1, D),
        layer_indices=[a.inject_layer],
        scale=0.8,
        norm_match=True,
        position_indices=[a.marker],
    )

    def probe_3d(tag: str) -> None:
        out_s, h_steer = capture([a.inject_layer], [probe_vec], logprobs=20)
        out_c, _ = capture([a.inject_layer], None, logprobs=20)
        delta = h_steer[0, a.marker] - h_clean[0, a.marker]
        other = h_steer[0] - h_clean[0]
        other[a.marker] = 0
        cos = torch.nn.functional.cosine_similarity(delta, unit, dim=0).item()
        ratio = (delta.norm() / hnorm).item()
        greedy = llm.generate(
            [{"prompt_token_ids": prompt_ids}],
            [
                SamplingParams(
                    temperature=0.0,
                    max_tokens=8,
                    extra_args={"apply_steering_vectors": [probe_vec]},
                )
            ],
            use_tqdm=False,
        )[0]
        _, h_nm = capture([a.inject_layer], [probe_vec_nm])
        result["probes"][tag] = {
            "hidden_dim": D,
            "hnorm": hnorm,
            "cos_delta_vs_v": cos,
            "norm_ratio": ratio,
            "max_other_row_abs_delta": other.abs().max().item(),
            "h_clean_marker": h_clean[0, a.marker].tolist(),
            "h_steer_marker": h_steer[0, a.marker].tolist(),
            "h_steer_normmatch_marker": h_nm[0, a.marker].tolist(),
            "next_token_top20": topk(out_s),
            "next_token_top20_clean": topk(out_c),
            "next_token_argmax": int(out_s.outputs[0].token_ids[0]),
            "greedy8": [int(t) for t in greedy.outputs[0].token_ids],
            "ok": cos > 0.99 and 0.95 < ratio < 1.05,
        }
        log(
            f"probe {tag}: |h|={hnorm:.1f} cos={cos:.4f} ratio={ratio:.4f} other={other.abs().max().item():.2e} "
            f"-> {'OK' if result['probes'][tag]['ok'] else 'FAIL'}"
        )

    def probe_2d(tag: str) -> None:
        vec2 = SteeringVector(
            activations=unit2.view(1, D),
            layer_indices=[mid_layer],
            scale=0.5,
            norm_match=True,
        )
        _, c2 = capture([mid_layer], None, max_tokens=4)
        o2, s2 = capture([mid_layer], [vec2], max_tokens=4)
        n = min(c2.shape[1], s2.shape[1])
        dn = (s2[0, :n] - c2[0, :n]).norm(dim=-1)
        result["probes"][tag] = {
            "layer": mid_layer,
            "positions": int(s2.shape[1]),
            "delta_norms": dn.tolist(),
            "h_steer_last": s2[0, -1].tolist(),
            "clean_norms": c2[0, :n].norm(dim=-1).tolist(),
            "tokens": [int(t) for t in o2.outputs[0].token_ids],
            "generated_rows_steered": bool((dn[P:] > 0).all().item())
            if n > P
            else None,
        }
        log(
            f"probe {tag}: positions={s2.shape[1]} mean|delta|={dn.mean():.3f} gen rows steered={result['probes'][tag]['generated_rows_steered']}"
        )

    modes = (
        [("steer3d", True, True)]
        if not is_fork
        else [("steer3d_loop", False, False), ("steer3d", True, True)]
    )
    for tag, vec_on, block_on in modes:
        set_mode(vec_on, block_on)
        probe_3d(tag)
        if a.engine == "eager":
            probe_2d(tag.replace("steer3d", "steer2d_normmatch"))
        stats(f"probes_{tag}")
    if a.engine in ("graphs", "compile"):
        # CUDA-graph mode must refuse broadcast vectors instead of applying them inconsistently
        vec2 = SteeringVector(
            activations=torch.randn(1, D, generator=g),
            layer_indices=[mid_layer],
            scale=0.5,
        )
        try:
            llm.generate(
                [{"prompt_token_ids": prompt_ids}],
                [
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=2,
                        extra_args={"apply_steering_vectors": [vec2]},
                    )
                ],
                use_tqdm=False,
            )
            result["probes"]["graph_guard_2d"] = {"raised": False}
        except Exception as e:  # noqa: BLE001
            result["probes"]["graph_guard_2d"] = {"raised": True, "error": str(e)[:300]}
        log(
            f"probe graph_guard_2d: raised={result['probes']['graph_guard_2d']['raised']}"
        )
    dump()

    # ---------------- per-request steering throughput ----------------
    vecs = torch.nn.functional.normalize(
        torch.randn(max(sizes + sizes_2d), D, generator=g), dim=-1
    )

    def extra3d(i: int) -> dict:
        e = {
            "apply_steering_vectors": [
                SteeringVector(
                    activations=(vecs[i] * hnorm).view(1, 1, D),
                    layer_indices=[a.inject_layer],
                    scale=1.0,
                    norm_match=False,
                    position_indices=[a.marker],
                )
            ]
        }
        if a.cache_salt:
            e["lens_cache_salt"] = a.cache_salt
        return e

    def extra2d(i: int) -> dict:
        return {
            "apply_steering_vectors": [
                SteeringVector(
                    activations=(vecs[i] * 4.0).view(1, D),
                    layer_indices=[mid_layer],
                    scale=1.0,
                    norm_match=False,
                )
            ]
        }

    if "steer3d_loop" in conds:
        set_mode(False, False)
        run_condition("steer3d_loop", extra3d, sizes)
    if "steer3d" in conds:
        set_mode(True, True)
        run_condition("steer3d", extra3d, sizes)
    if "steer2d" in conds:
        set_mode(True, True)
        run_condition("steer2d", extra2d, sizes_2d)
    dump()
    log("done")


if __name__ == "__main__":
    main()
