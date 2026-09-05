#!/usr/bin/env python
"""Same-host transport micro-benchmark (vllm-metamodels 1.1.0.post7, ``VLLM_LENS_SHM``).

Measures, on ONE engine, the two bulk payloads the fork moves between the client process and
the engine-core / worker process:

  capture   ``output_residual_stream=[L]`` for B texts, all positions (the ~1.2 GB case for
            1,024 texts of the 27B): wall time of ``generate()`` and of the retrieval RPC alone,
            with the pickled ``get_captured_states_many`` vs the shared-memory descriptor
            (``get_captured_states_shm``; copy-out and zero-copy views).
  vectors   ``set_steering_block`` with one [n, hidden] block (n = B): RPC round-trip pickled vs
            shared memory, repeated; plus ``generate()`` wall for the steering condition.

Output: one JSON with every timing; ``bench/modal_bench.py::shm`` drives it.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_steering import PROMPT_TEXT, make_llm  # noqa: E402


def log(msg: str) -> None:
    print(f"[shm {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--sizes", default="512,1024")
    p.add_argument("--min-len", type=int, default=96)
    p.add_argument("--max-len", type=int, default=136)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--rpc-repeats", type=int, default=10)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--attention-backend", default="")
    p.add_argument("--language-model-only", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["VLLM_LENS_CUDA_GRAPHS"] = "1"
    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    import vllm_lens
    from vllm_lens import SteeringVector
    from vllm_lens import _shm

    sizes = [int(s) for s in a.sizes.split(",") if s]
    B_max = max(sizes)
    tok = AutoTokenizer.from_pretrained(a.model)
    base = tok(PROMPT_TEXT, add_special_tokens=False)["input_ids"]
    while len(base) < a.max_len + B_max:
        base = base + base
    g = torch.Generator().manual_seed(a.seed)
    lens = torch.randint(a.min_len, a.max_len + 1, (B_max,), generator=g).tolist()
    texts = [[int(t) for t in base[i : i + n]] for i, n in enumerate(lens)]  # all different, 96..136 tokens

    kw: dict[str, Any] = dict(model=a.model, tensor_parallel_size=1, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_len + 48,
                              enable_prefix_caching=False, max_num_seqs=B_max, max_num_batched_tokens=max(8192, B_max * (a.max_len + 8)),
                              dtype="bfloat16", seed=a.seed, compilation_config={"max_cudagraph_capture_size": min(B_max, 1024)})
    if a.attention_backend:
        kw["attention_backend"] = a.attention_backend
    if a.language_model_only:
        kw["language_model_only"] = True
    t0 = time.perf_counter()
    llm, kw = make_llm(LLM, kw, log)
    up = time.perf_counter() - t0
    log(f"engine up {up:.0f}s vllm={vllm.__version__} vllm-lens={vllm_lens.__version__}")
    result: dict[str, Any] = {"model": a.model, "layer": a.layer, "gpu": torch.cuda.get_device_name(0), "engine_up_s": up,
                              "versions": {"vllm": vllm.__version__, "torch": torch.__version__, "vllm_lens": vllm_lens.__version__},
                              "sizes": sizes, "capture": [], "vectors": [], "rpc": []}

    def dump() -> None:
        json.dump(result, open(a.out, "w"), indent=1)

    def stats() -> dict:
        return llm.collective_rpc("steering_stats", args=(True,))[0]

    def cap_params(B: int):
        return [SamplingParams(temperature=0.0, max_tokens=1, extra_args={"output_residual_stream": [a.layer]}) for _ in range(B)]

    def prompts(B: int):
        return [{"prompt_token_ids": texts[i]} for i in range(B)]

    # ---- capture: pickled vs shm copy vs shm view --------------------------------------------
    probe = llm.generate(prompts(1), cap_params(1), use_tqdm=False)[0]
    D = probe.activations["residual_stream"].shape[-1]
    result["hidden_dim"] = D
    for mode in ("", "1", "view"):
        os.environ["VLLM_LENS_SHM"] = mode
        tag = {"": "pickle", "1": "shm_copy", "view": "shm_view"}[mode]
        for B in sizes:
            _ = stats()
            for rep in range(a.repeats):
                t1 = time.perf_counter()
                outs = llm.generate(prompts(B), cap_params(B), use_tqdm=False)
                wall = time.perf_counter() - t1
                st = stats()
                n_bytes = sum(o.activations["residual_stream"].numel() * o.activations["residual_stream"].element_size() for o in outs)
                ok = all(o.activations["residual_stream"].shape[1] == len(texts[i]) for i, o in enumerate(outs))
                row = {"transport": tag, "batch": B, "rep": rep, "wall_s": wall, "retrieval_s": st.get("retrieval_s"),
                       "hook_capture_s": st.get("hook_capture_s"), "bytes": n_bytes, "shapes_ok": ok,
                       "sample_norm": float(outs[0].activations["residual_stream"][0, -1].float().norm())}
                result["capture"].append(row)
                log(f"capture {tag:9s} B={B:5d} rep{rep}: wall {wall:6.2f}s  worker retrieval {st.get('retrieval_s', 0):5.2f}s  {n_bytes / 1e9:.2f} GB  ok={ok}")
                del outs
                dump()
    os.environ["VLLM_LENS_SHM"] = ""

    # ---- vectors: set_steering_block RPC round trip, pickled vs shm -----------------------------
    vecs = torch.nn.functional.normalize(torch.randn(B_max, D, generator=g), dim=-1).to(torch.bfloat16)
    for B in sizes:
        block = {"keys": [f"_bench_{i}" for i in range(B)], "vecs": vecs[:B].contiguous(), "layers": [a.layer] * B, "positions": [10] * B,
                 "scales": [1.0] * B, "norm_match": [False] * B, "modes": ["add"] * B}
        for tag in ("pickle", "shm"):
            ts_ = []
            for _ in range(a.rpc_repeats):
                t1 = time.perf_counter()
                if tag == "shm":
                    desc = _shm.put({"vecs": block["vecs"]}, tag="bench")
                    llm.collective_rpc("set_steering_block", args=(pickle.dumps({**block, "vecs": None, "shm": desc}),))
                else:
                    llm.collective_rpc("set_steering_block", args=(pickle.dumps(block),))
                ts_.append(time.perf_counter() - t1)
                llm.collective_rpc("clear_steering_data_many", args=(block["keys"],))
            row = {"transport": tag, "batch": B, "bytes": vecs[:B].numel() * 2, "rpc_s_min": min(ts_), "rpc_s_median": sorted(ts_)[len(ts_) // 2], "n": len(ts_)}
            result["rpc"].append(row)
            log(f"set_steering_block {tag:6s} n={B:5d}: min {min(ts_) * 1e3:6.1f} ms median {row['rpc_s_median'] * 1e3:6.1f} ms ({row['bytes'] / 1e6:.1f} MB)")
        dump()

    # ---- steering generate() wall with the block through pickle vs shm --------------------------
    hn = float(probe.activations["residual_stream"][0, 10].float().norm())
    for mode in ("", "1"):
        os.environ["VLLM_LENS_SHM"] = mode
        tag = "pickle" if mode == "" else "shm"
        for B in sizes:
            params = [SamplingParams(temperature=1.0, max_tokens=40, min_tokens=40, ignore_eos=True, extra_args={"apply_steering_vectors": [
                SteeringVector(activations=(vecs[i].float() * hn).view(1, 1, D), layer_indices=[a.layer], position_indices=[10])]}) for i in range(B)]
            for rep in range(a.repeats):
                t1 = time.perf_counter()
                outs = llm.generate(prompts(B), params, use_tqdm=False)
                wall = time.perf_counter() - t1
                result["vectors"].append({"transport": tag, "batch": B, "rep": rep, "wall_s": wall, "gen_tokens": sum(len(o.outputs[0].token_ids) for o in outs)})
                log(f"steer gen {tag:6s} B={B:5d} rep{rep}: wall {wall:6.2f}s")
            dump()
    os.environ["VLLM_LENS_SHM"] = ""
    dump()
    log("done")


if __name__ == "__main__":
    main()
