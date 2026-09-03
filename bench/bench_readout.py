#!/usr/bin/env python
"""Hidden-state readout benchmark: how fast can vLLM hand back (a projection of)
the layer-L residual stream for a batch of texts?

The workload is the RL-trainer *reward pass*: re-encode N texts (96-136 tokens)
through the CLEAN base model, read layer L at every token, take a per-token
cosine with a per-request target direction and the max over the last 5 tokens.

Stages (one process each; the Modal driver ``bench/modal_bench.py::readout``
chains them in one container so the weights stay in the page cache):

  --stage hf      HF transformers bf16 reference: the trainer's path (forward hook at
                  layer L raising a sentinel = early exit, batch 128) -> timing + reference
                  hidden states / cosines saved to --ref (torch.save)
  --stage vllm    one vLLM engine (--engine eager|graphs; eager on stock vllm-lens 1.1.0
                  measures the stock capture path) running the conditions below

Conditions (prefill-only = max_tokens=1; B in --sizes):
  nocap            no hooks installed (engine ceiling), then nocap_hooked (hooks live, idle)
  cap_all          output_residual_stream=[L], every position (1.1.0 semantics; on the fork
                   the fast gather path)      -- stock runs only this + nocap + gen*
  cap_all_legacy   fork: the per-request .cpu() slice + per-request RPC path (VLLM_LENS_FAST_CAPTURE=0)
  cap_last5        fork: capture_positions={"last": 5}
  read_last5       fork: ReadoutVector (cosine with the request's direction) at the last 5 positions
  read_all         fork: ReadoutVector at every position
  exit_read_last5  fork: read_last5 + lens_early_exit (layers > L never run)
  exit_cap_last5   fork: cap_last5 + lens_early_exit
  exit_read_all    fork: read_all + lens_early_exit
Generation conditions (B in --gen-sizes, --gen-tokens new tokens):
  gen_nocap        plain generation
  gen_cap_all      generation + capture of every position (eager: prompt + generated; graphs:
                   prompt only -- the documented rule)
  gen_then_read    gen_nocap followed by a re-encode readout pass of prompt+generated (the
                   trainer's flow), reported as the sum and as its parts
  gen_then_exit_read  same with early exit on the re-encode pass

Every vLLM condition stores wall time, the time inside the engine's generate
(``_original_llm_generate``), and RPC time by method; fork conditions also store
the worker's hook counters.  Correctness vs --ref: max |diff| and cosine of the
captured rows / readout values against the HF reference.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
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
    "scheduler, which decides at every step which requests advance and by how many tokens. "
    "A scheduler that is too eager starves the long requests; one that is too fair wastes the "
    "accelerator on tiny batches. Between those failure modes lies a narrow band of policies "
    "that keep the pipeline full without letting any single request wait for very long."
)
GRAPH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
N_SAMPLE = 8  # texts whose raw rows are kept in the JSON for cross-engine checks
N_FULL_REF = 16  # texts with full-sequence HF hidden states in the ref


def log(msg: str) -> None:
    print(f"[bench {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--stage", choices=["hf", "vllm"], required=True)
    p.add_argument("--engine", choices=["eager", "graphs"], default="eager")
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default="", help="hf stage: write here; vllm stage: read for correctness")
    p.add_argument("--layer", type=int, required=True, help="layer whose output is read (e.g. 42 of 64)")
    p.add_argument("--n-texts", type=int, default=1024)
    p.add_argument("--min-len", type=int, default=96)
    p.add_argument("--max-len", type=int, default=136)
    p.add_argument("--last-k", type=int, default=5)
    p.add_argument("--sizes", default="64,512,1024")
    p.add_argument("--gen-sizes", default="64,512")
    p.add_argument("--gen-tokens", type=int, default=40)
    p.add_argument("--hf-batch", type=int, default=128)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--conditions", default="", help="override: comma list")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--attention-backend", default="")
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument("--enable-lora", action="store_true", help="LoRA slots on (rollout-engine config)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# shared data: deterministic variable-length texts + one unit direction per text
# ---------------------------------------------------------------------------


def hidden_size(model: str) -> int:
    """hidden_size from config.json (read directly: the stock image's transformers predates Qwen3.5/3.6)."""
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(model, "config.json")) as f:
        cfg = json.load(f)
    tc = cfg.get("text_config") or {}
    return int(tc.get("hidden_size") or cfg["hidden_size"])


def make_texts(tok, n: int, min_len: int, max_len: int) -> list[list[int]]:
    base = tok(PROMPT_TEXT, add_special_tokens=False)["input_ids"]
    while len(base) < max_len + 256:
        base = base + base
    out = []
    span = max_len - min_len + 1
    for i in range(n):
        L = min_len + (i * 7) % span
        off = (i * 13) % 200
        out.append([int(t) for t in base[off : off + L]])
    return out


def make_dirs(n: int, d: int):
    import torch

    g = torch.Generator().manual_seed(1234)
    return torch.nn.functional.normalize(torch.randn(n, d, generator=g), dim=-1)


def last_k_positions(length: int, k: int) -> list[int]:
    return list(range(max(0, length - k), length))


# ---------------------------------------------------------------------------
# stage: HF reference (the trainer's read_resid path)
# ---------------------------------------------------------------------------


def find_layers(model):
    for path in ("model.language_model.layers", "model.layers", "language_model.model.layers", "model.decoder.layers"):
        cur, ok = model, True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok:
            return cur
    raise AttributeError(f"cannot find decoder layers on {type(model).__name__}")


class _Stop(Exception):
    pass


def hf_stage(a: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    texts = make_texts(tok, a.n_texts, a.min_len, a.max_len)
    D = hidden_size(a.model)
    dirs = make_dirs(a.n_texts, D)
    cfg = AutoConfig.from_pretrained(a.model)
    cls = AutoModelForImageTextToText if getattr(cfg, "text_config", None) is not None else AutoModelForCausalLM
    t0 = time.perf_counter()
    try:
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()
    except (ValueError, ImportError) as e:
        log(f"device_map load failed ({str(e)[:80]}); loading on CPU then .to('cuda')")
        model = cls.from_pretrained(a.model, dtype=torch.bfloat16).to("cuda").eval()
    load_s = time.perf_counter() - t0
    layers = find_layers(model)
    L = a.layer
    n_layers = len(layers)
    log(f"HF {cls.__name__} up in {load_s:.0f}s | {n_layers} layers | attn={getattr(model.config, '_attn_implementation', '?')}")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    captured: dict[str, torch.Tensor] = {}

    def cap_stop(_m, _i, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    def cap_pass(_m, _i, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    def batches(ids_list: list[list[int]], bs: int):
        for s in range(0, len(ids_list), bs):
            chunk = ids_list[s : s + bs]
            T = max(len(x) for x in chunk)
            inp = torch.full((len(chunk), T), pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), T), dtype=torch.long)
            for j, x in enumerate(chunk):
                inp[j, : len(x)] = torch.tensor(x)
                mask[j, : len(x)] = 1
            yield s, inp, mask, [len(x) for x in chunk]

    @torch.no_grad()
    def score_pass(early_exit: bool, keep_rows: bool = False):
        """The trainer's score(): re-encode, read layer L, cosine with dirs, max over last k.
        Returns (reward[n], cos_lastk[n, k], rows_lastk[n, k, D] or None, cos_max_all[n], full_rows)."""
        hook = layers[L].register_forward_hook(cap_stop if early_exit else cap_pass)
        reward = torch.zeros(len(texts))
        cos_lastk = torch.zeros(len(texts), a.last_k)
        cos_max_all = torch.zeros(len(texts))
        rows_lastk = torch.zeros(len(texts), a.last_k, D) if keep_rows else None
        full_rows: list[torch.Tensor] = []
        try:
            for s, inp, mask, lens in batches(texts, a.hf_batch):
                inp, mask = inp.cuda(non_blocking=True), mask.cuda(non_blocking=True)
                try:
                    model(input_ids=inp, attention_mask=mask)
                except _Stop:
                    pass
                h = captured.pop("h")  # [b, T, D] bf16 (full residual stream after layer L)
                d = dirs[s : s + h.shape[0]].cuda()
                cos = torch.einsum("btd,bd->bt", F.normalize(h.float(), dim=-1), d)  # [b, T]
                keep = mask.bool()
                cos_max_all[s : s + h.shape[0]] = cos.masked_fill(~keep, -1.0).max(1).values.cpu()
                for j, n in enumerate(lens):
                    pos = last_k_positions(n, a.last_k)
                    cos_lastk[s + j, -len(pos) :] = cos[j, pos].cpu()
                    reward[s + j] = cos[j, pos].max().cpu()
                    if keep_rows:
                        rows_lastk[s + j, -len(pos) :] = h[j, pos].float().cpu()
                    if keep_rows and s + j < N_FULL_REF:
                        full_rows.append(h[j, :n].float().cpu())
        finally:
            hook.remove()
        return reward, cos_lastk, rows_lastk, cos_max_all, full_rows

    # warm-up (kernel selection, allocator), then timed repeats
    score_pass(True)
    torch.cuda.synchronize()
    timings: dict[str, list[float]] = {"early_exit": [], "full": []}
    for _ in range(max(1, a.repeats)):
        t1 = time.perf_counter()
        score_pass(True)
        torch.cuda.synchronize()
        timings["early_exit"].append(time.perf_counter() - t1)
    for _ in range(max(1, a.repeats)):
        t1 = time.perf_counter()
        score_pass(False)
        torch.cuda.synchronize()
        timings["full"].append(time.perf_counter() - t1)
    reward, cos_lastk, rows_lastk, cos_max_all, full_rows = score_pass(True, keep_rows=True)
    n_tok = sum(len(x) for x in texts)
    log(
        f"HF score pass over {len(texts)} texts ({n_tok} tokens): early-exit@{L} "
        f"{min(timings['early_exit']):.3f}s (min of {a.repeats}) | full {n_layers} layers {min(timings['full']):.3f}s"
    )
    result = {
        "stage": "hf",
        "model": a.model,
        "layer": L,
        "n_layers": n_layers,
        "hidden_dim": D,
        "hf_class": cls.__name__,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "gpu": torch.cuda.get_device_name(0),
        "load_s": load_s,
        "n_texts": len(texts),
        "n_tokens": n_tok,
        "mean_len": n_tok / len(texts),
        "hf_batch": a.hf_batch,
        "last_k": a.last_k,
        "timings_s": timings,
        "score_s_early_exit": min(timings["early_exit"]),
        "score_s_full": min(timings["full"]),
        "reward_sample": reward[:N_SAMPLE].tolist(),
        "cos_lastk_sample": cos_lastk[:N_SAMPLE].tolist(),
        "rows_lastk_sample": rows_lastk[:N_SAMPLE].tolist(),
        "text_lens": [len(x) for x in texts],
    }
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    if a.ref:
        torch.save(
            {
                "model": a.model,
                "layer": L,
                "texts": texts,
                "dirs": dirs,
                "reward": reward,
                "cos_lastk": cos_lastk,
                "rows_lastk": rows_lastk,
                "cos_max_all": cos_max_all,
                "full_rows": full_rows,
                "last_k": a.last_k,
            },
            a.ref,
        )
        log(f"ref saved -> {a.ref}")
    log("hf stage done")


# ---------------------------------------------------------------------------
# stage: vLLM
# ---------------------------------------------------------------------------


def installed_variant() -> dict[str, Any]:
    try:
        ver = importlib.metadata.version("vllm-lens")
    except importlib.metadata.PackageNotFoundError:
        ver = "missing"
    try:
        from vllm_lens import _worker_ext as W

        is_fork = hasattr(W.HiddenStatesExtension, "set_steering_block")
        has_readout = hasattr(W.HiddenStatesExtension, "set_readout_block")
    except Exception:  # noqa: BLE001
        is_fork = has_readout = False
    return {"vllm_lens_version": ver, "variant": "fork" if is_fork else "stock", "readout": has_readout}


def vllm_stage(a: argparse.Namespace) -> None:
    sizes = [int(s) for s in a.sizes.split(",") if s.strip()]
    gen_sizes = [int(s) for s in a.gen_sizes.split(",") if s.strip()]
    max_num_seqs = max(sizes + gen_sizes)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if a.engine == "graphs":
        os.environ["VLLM_LENS_CUDA_GRAPHS"] = "1"

    import torch
    import torch.nn.functional as F
    import vllm
    from vllm import LLM, SamplingParams

    variant = installed_variant()
    is_fork, has_readout = variant["variant"] == "fork", variant["readout"]
    log(f"vllm {vllm.__version__} | vllm-lens {variant} | torch {torch.__version__} | engine={a.engine}")
    if a.engine == "graphs" and not is_fork:
        sys.exit("--engine graphs needs vllm-metamodel (stock forces enforce_eager)")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    texts = make_texts(tok, a.n_texts, a.min_len, a.max_len)
    D = hidden_size(a.model)
    dirs = make_dirs(a.n_texts, D)
    L = a.layer
    max_len = a.max_len + a.gen_tokens + 8

    kw: dict[str, Any] = dict(
        model=a.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=a.gpu_mem,
        max_model_len=max_len,
        enable_prefix_caching=False,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max(8192, max_num_seqs * max_len),  # whole batch prefills in one wave
        dtype="bfloat16",
        seed=a.seed,
    )
    if a.attention_backend:
        kw["attention_backend"] = a.attention_backend
    if a.language_model_only:
        kw["language_model_only"] = True
    if a.enable_lora:
        kw.update(enable_lora=True, max_loras=2, max_lora_rank=64)
    if a.engine == "eager":
        kw["enforce_eager"] = True
    else:
        capture_sizes = sorted({s for s in GRAPH_SIZES + sizes + gen_sizes if s <= max_num_seqs})
        kw["compilation_config"] = {"cudagraph_capture_sizes": capture_sizes}

    t0 = time.perf_counter()
    llm = LLM(**kw)
    engine_up_s = time.perf_counter() - t0
    vc = llm.llm_engine.vllm_config
    cc = vc.compilation_config
    resolved = {
        "enforce_eager": bool(vc.model_config.enforce_eager),
        "compilation_mode": str(getattr(cc.mode, "name", cc.mode)),
        "cudagraph_mode": str(getattr(cc.cudagraph_mode, "name", cc.cudagraph_mode)),
        "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
        "max_num_seqs": vc.scheduler_config.max_num_seqs,
        "num_layers": vc.model_config.get_num_layers(vc.parallel_config),
        "enable_prefix_caching": bool(vc.cache_config.enable_prefix_caching),
    }
    log(f"engine up in {engine_up_s:.0f}s | {resolved}")

    # ---- instrumentation: RPC time by method + time inside the engine's generate ----
    import vllm_lens._activations_plugin as P

    rpc_time: dict[str, float] = {}
    rpc_calls: dict[str, int] = {}
    orig_rpc = llm.collective_rpc

    def timed_rpc(method, *args, **kwargs):
        t = time.perf_counter()
        try:
            return orig_rpc(method, *args, **kwargs)
        finally:
            name = method if isinstance(method, str) else getattr(method, "__name__", str(method))
            rpc_time[name] = rpc_time.get(name, 0.0) + time.perf_counter() - t
            rpc_calls[name] = rpc_calls.get(name, 0) + 1

    llm.collective_rpc = timed_rpc  # type: ignore[assignment]
    engine_time = {"s": 0.0}
    orig_gen = P._original_llm_generate

    def timed_gen(*args, **kwargs):
        t = time.perf_counter()
        try:
            return orig_gen(*args, **kwargs)
        finally:
            engine_time["s"] += time.perf_counter() - t

    P._original_llm_generate = timed_gen

    ref = None
    if a.ref and os.path.exists(a.ref):
        ref = torch.load(a.ref, weights_only=False)
        if ref.get("model") != a.model or ref.get("layer") != L:
            log(f"ref mismatch ({ref.get('model')} L{ref.get('layer')}), ignoring")
            ref = None
        else:
            assert ref["texts"] == texts, "ref texts differ from this run's texts"

    result: dict[str, Any] = {
        "stage": "vllm",
        "model": a.model,
        "engine": a.engine,
        "variant": variant,
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__},
        "gpu": torch.cuda.get_device_name(0),
        "engine_kwargs": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in kw.items()},
        "resolved_config": resolved,
        "engine_up_s": engine_up_s,
        "layer": L,
        "hidden_dim": D,
        "n_texts": len(texts),
        "text_lens": [len(x) for x in texts],
        "last_k": a.last_k,
        "sizes": sizes,
        "gen_sizes": gen_sizes,
        "gen_tokens": a.gen_tokens,
        "has_ref": ref is not None,
        "rows": [],
        "checks": [],
        "stats": {},
    }

    def dump() -> None:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)

    def worker_stats(reset: bool = True) -> dict[str, Any]:
        if not is_fork:
            return {}
        try:
            return orig_rpc("steering_stats", args=(reset,))[0]
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

    def set_fast_capture(on: bool) -> None:
        os.environ["VLLM_LENS_FAST_CAPTURE"] = "1" if on else "0"
        if is_fork and has_readout:
            orig_rpc("set_fast_capture", args=(bool(on),))

    # ---- request builders -----------------------------------------------------
    ReadoutVector = None
    if has_readout:
        from vllm_lens import ReadoutVector  # type: ignore[no-redef]

    def sp(max_tokens: int, extra: dict | None) -> SamplingParams:
        if max_tokens > 1:
            return SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_tokens, min_tokens=max_tokens, ignore_eos=True, extra_args=extra)
        return SamplingParams(temperature=0.0, max_tokens=1, extra_args=extra)

    def readout_extra(i: int, positions, early_exit: bool) -> dict:
        extra: dict[str, Any] = {
            "apply_readout_vectors": [ReadoutVector(activations=dirs[i].view(1, D), layer_indices=[L], positions=positions, metric="cos")]
        }
        if early_exit:
            extra["lens_early_exit"] = True
        return extra

    def capture_extra(positions, early_exit: bool) -> dict:
        extra: dict[str, Any] = {"output_residual_stream": [L]}
        if positions is not None:
            extra["capture_positions"] = positions
        if early_exit:
            extra["lens_early_exit"] = True
        return extra

    COND: dict[str, tuple[int, Any]] = {
        # name -> (max_tokens, extra builder (i) -> dict | None)
        "nocap": (1, lambda i: None),
        "nocap_hooked": (1, lambda i: None),
        "cap_all": (1, lambda i: capture_extra(None, False)),
        "cap_all_legacy": (1, lambda i: capture_extra(None, False)),
        "cap_last5": (1, lambda i: capture_extra({"last": a.last_k}, False)),
        "read_last5": (1, lambda i: readout_extra(i, {"last": a.last_k}, False)),
        "read_all": (1, lambda i: readout_extra(i, "all", False)),
        "exit_read_last5": (1, lambda i: readout_extra(i, {"last": a.last_k}, True)),
        "exit_cap_last5": (1, lambda i: capture_extra({"last": a.last_k}, True)),
        "exit_read_all": (1, lambda i: readout_extra(i, "all", True)),
        "gen_nocap": (a.gen_tokens, lambda i: None),
        "gen_cap_all": (a.gen_tokens, lambda i: capture_extra(None, False)),
    }

    def run_generate(cond: str, B: int, ids_list: list[list[int]], rep: int, tag: str | None = None) -> tuple[dict, list]:
        max_tokens, mk = COND[cond]
        params = [sp(max_tokens, mk(i)) for i in range(B)]
        prompts = [{"prompt_token_ids": ids_list[i]} for i in range(B)]
        rpc_time.clear()
        rpc_calls.clear()
        engine_time["s"] = 0.0
        t1 = time.perf_counter()
        outs = llm.generate(prompts, params, use_tqdm=False)
        wall = time.perf_counter() - t1
        n_prompt = sum(len(ids_list[i]) for i in range(B))
        n_gen = sum(len(o.token_ids) for out in outs for o in out.outputs)
        row = {
            "condition": tag or cond,
            "batch": B,
            "rep": rep,
            "wall_s": wall,
            "engine_s": engine_time["s"],
            "rpc_s": dict(rpc_time),
            "rpc_calls": dict(rpc_calls),
            "prompt_tokens": n_prompt,
            "gen_tokens": n_gen,
            "per_1024_s": wall * 1024 / B,
            "prompt_tok_per_s": n_prompt / wall,
            "stats": worker_stats(True),
        }
        return row, outs

    def check_capture(cond: str, outs: list, ids_list: list[list[int]], full: bool) -> None:
        if ref is None:
            return
        import math

        diffs, coss, pos_ok = [], [], True
        n_rows_ref = min(len(outs), N_FULL_REF if full else len(texts))
        for i in range(n_rows_ref):
            act = getattr(outs[i], "activations", None)
            if act is None:
                result["checks"].append({"condition": cond, "check": "activations present", "ok": False, "detail": f"text {i}: none"})
                return
            rs = act["residual_stream"][0].float()  # [n_pos, D]
            n = len(ids_list[i])
            if full:
                pos = act.get("positions", list(range(rs.shape[0])))
                exp = ref["full_rows"][i]
                if list(pos) != list(range(n)) or rs.shape[0] != n:
                    pos_ok = False
                    continue
                d = (rs - exp).abs().max().item()
                c = F.cosine_similarity(rs.reshape(-1), exp.reshape(-1), dim=0).item()
            else:
                pos = act.get("positions")
                want = last_k_positions(n, a.last_k)
                if pos is None or list(pos) != want or rs.shape[0] != len(want):
                    pos_ok = False
                    continue
                exp = ref["rows_lastk"][i, -len(want) :]
                d = (rs - exp).abs().max().item()
                c = F.cosine_similarity(rs.reshape(-1), exp.reshape(-1), dim=0).item()
            diffs.append(d)
            coss.append(c)
        ok = pos_ok and bool(diffs) and min(coss) > 0.999 and not any(math.isnan(x) for x in diffs)
        result["checks"].append(
            {
                "condition": cond,
                "check": ("full-sequence" if full else f"last-{a.last_k}") + " rows vs HF layer output",
                "ok": ok,
                "n": len(diffs),
                "max_abs_diff": max(diffs) if diffs else None,
                "min_cos": min(coss) if coss else None,
                "mean_cos": statistics.fmean(coss) if coss else None,
                "positions_ok": pos_ok,
                "detail": f"n={len(diffs)} max|d|={max(diffs) if diffs else float('nan'):.4f} min cos={min(coss) if coss else float('nan'):.6f} positions_ok={pos_ok}",
            }
        )
        log(f"check {cond}: {result['checks'][-1]['detail']} -> {'OK' if ok else 'FAIL'}")

    def check_readout(cond: str, outs: list, ids_list: list[list[int]], last_k: bool) -> None:
        if ref is None:
            return
        dv, dr, pos_ok = [], [], True
        for i, o in enumerate(outs):
            ro = getattr(o, "readout", None)
            if not ro:
                result["checks"].append({"condition": cond, "check": "readout present", "ok": False, "detail": f"text {i}: none"})
                return
            vals = ro[0]["values"][0]
            n = len(ids_list[i])
            if last_k:
                want = last_k_positions(n, a.last_k)
                if list(ro[0]["positions"]) != want:
                    pos_ok = False
                    continue
                exp = ref["cos_lastk"][i, -len(want) :]
                dv.append((vals - exp).abs().max().item())
                dr.append(abs(vals.max().item() - ref["reward"][i].item()))
            else:
                if list(ro[0]["positions"]) != list(range(n)):
                    pos_ok = False
                    continue
                dr.append(abs(vals.max().item() - ref["cos_max_all"][i].item()))
        ok = pos_ok and bool(dr) and max(dr) < 0.02
        result["checks"].append(
            {
                "condition": cond,
                "check": ("last-k cosines + reward" if last_k else "max-over-all-positions cosine") + " vs HF",
                "ok": ok,
                "n": len(dr),
                "max_abs_diff_values": max(dv) if dv else None,
                "max_abs_diff_reward": max(dr) if dr else None,
                "mean_abs_diff_reward": statistics.fmean(dr) if dr else None,
                "positions_ok": pos_ok,
                "detail": f"n={len(dr)} max|d reward|={max(dr) if dr else float('nan'):.5f}"
                + (f" max|d cos|={max(dv):.5f}" if dv else "")
                + f" positions_ok={pos_ok}",
            }
        )
        log(f"check {cond}: {result['checks'][-1]['detail']} -> {'OK' if ok else 'FAIL'}")

    def sample_rows(outs: list, ids_list: list[list[int]]) -> list:
        """Last-k rows of the first N_SAMPLE texts (for cross-engine comparisons)."""
        out = []
        for i in range(min(N_SAMPLE, len(outs))):
            act = getattr(outs[i], "activations", None)
            if act is None:
                out.append(None)
                continue
            rs = act["residual_stream"][0].float()
            pos = act.get("positions", list(range(rs.shape[0])))
            want = last_k_positions(len(ids_list[i]), a.last_k)
            sel = [list(pos).index(p) for p in want if p in list(pos)]
            out.append(rs[sel].tolist())
        return out

    def run_condition(cond: str, cond_sizes: list[int]) -> None:
        warm_b = min(64, max(cond_sizes))
        run_generate(cond, warm_b, texts, -1)
        worker_stats(True)
        for B in cond_sizes:
            for rep in range(a.repeats if COND[cond][0] == 1 else 1):
                row, outs = run_generate(cond, B, texts, rep)
                result["rows"].append(row)
                log(
                    f"{cond:16s} B={B:5d} rep{rep}: {row['wall_s']:7.3f}s (engine {row['engine_s']:.3f}s, rpc {sum(row['rpc_s'].values()):.3f}s) "
                    f"= {row['per_1024_s']:.3f}s/1024 texts, {row['prompt_tok_per_s']:.0f} prompt tok/s"
                    + (f" | early_exits={row['stats'].get('early_exits')}" if row["stats"] else "")
                )
                if rep == 0 and B == max(cond_sizes):
                    if cond.startswith(("cap_", "exit_cap")):
                        check_capture(cond, outs, texts, full=cond in ("cap_all", "cap_all_legacy"))
                        row["sample_rows_lastk"] = sample_rows(outs, texts)
                    elif cond.startswith(("read_", "exit_read")):
                        check_readout(cond, outs, texts, last_k=cond.endswith("last5"))
                        row["reward_sample"] = [o.readout[0]["values"][0].max().item() for o in outs[:N_SAMPLE]]
                dump()

    if a.conditions:
        conds = [c for c in a.conditions.split(",") if c]
    elif not is_fork:
        conds = ["nocap", "cap_all", "gen_nocap", "gen_cap_all"]
    elif not has_readout:
        conds = ["nocap", "nocap_hooked", "cap_all", "gen_nocap", "gen_cap_all"]
    else:
        conds = [
            "nocap", "nocap_hooked", "cap_all_legacy", "cap_all", "cap_last5", "read_last5", "read_all",
            "exit_read_last5", "exit_cap_last5", "exit_read_all", "gen_nocap", "gen_cap_all",
            "gen_then_read", "gen_then_exit_read",
        ]

    if "nocap" in conds:  # before any hook is installed
        run_condition("nocap", sizes)
    if is_fork:
        llm.collective_rpc("install_hooks")
        caps = orig_rpc("lens_capabilities")[0]
        result["capabilities"] = caps
        log(f"capabilities: {caps}")
        if not caps.get("early_exit", False):
            conds = [c for c in conds if not c.startswith("exit_") and c != "gen_then_exit_read"]
            log(f"early exit unavailable ({caps.get('early_exit_reason')}), skipping exit conditions")
    if "nocap_hooked" in conds and is_fork:
        run_condition("nocap_hooked", sizes)
    for cond in conds:
        if cond in ("nocap", "nocap_hooked") or cond.startswith("gen"):
            continue
        if cond == "cap_all_legacy":
            set_fast_capture(False)
            run_condition(cond, sizes)
            set_fast_capture(True)
        else:
            run_condition(cond, sizes)
    if "gen_nocap" in conds:
        run_condition("gen_nocap", gen_sizes)
    if "gen_cap_all" in conds:
        run_condition("gen_cap_all", gen_sizes)
        # semantics probe: which positions came back?
        row, outs = run_generate("gen_cap_all", 4, texts, 0)
        got = [getattr(o, "activations", {}).get("residual_stream").shape[1] if getattr(o, "activations", None) else None for o in outs]
        result["gen_capture_positions"] = {"prompt_lens": [len(texts[i]) for i in range(4)], "captured_positions": got, "gen_tokens": a.gen_tokens}
        log(f"gen_cap_all positions: prompts {result['gen_capture_positions']['prompt_lens']} -> captured {got}")
    for cond, exit_ in (("gen_then_read", False), ("gen_then_exit_read", True)):
        if cond not in conds:
            continue
        for B in gen_sizes:
            row_g, outs = run_generate("gen_nocap", B, texts, 0, tag=f"{cond}:gen")
            full_ids = [texts[i] + list(outs[i].outputs[0].token_ids) for i in range(B)]
            # re-encode prompt+generated with a readout at the last k positions (the trainer's reward pass)
            COND["_reencode"] = (1, lambda i: readout_extra(i, {"last": a.last_k}, exit_))
            run_generate("_reencode", min(64, B), full_ids, -1)  # warm-up
            row_r, outs_r = run_generate("_reencode", B, full_ids, 0, tag=f"{cond}:reencode")
            del COND["_reencode"]
            for r in (row_g, row_r):
                result["rows"].append(r)
            total = row_g["wall_s"] + row_r["wall_s"]
            result["rows"].append(
                {"condition": cond, "batch": B, "rep": 0, "wall_s": total, "gen_s": row_g["wall_s"], "reencode_s": row_r["wall_s"],
                 "per_1024_s": total * 1024 / B, "prompt_tokens": row_g["prompt_tokens"], "gen_tokens": row_g["gen_tokens"],
                 "reencode_tokens": row_r["prompt_tokens"], "stats": row_r["stats"]}
            )
            log(f"{cond:16s} B={B:5d}: gen {row_g['wall_s']:.3f}s + re-encode {row_r['wall_s']:.3f}s = {total:.3f}s")
            dump()
    result["final_stats"] = worker_stats(False)
    dump()
    log("vllm stage done")


def main() -> None:
    a = parse_args()
    if a.stage == "hf":
        hf_stage(a)
    else:
        vllm_stage(a)


if __name__ == "__main__":
    main()
