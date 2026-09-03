#!/usr/bin/env python
"""Engine-init bisect for CUDA-graph failures: build a vLLM engine with one named
configuration, generate 8 tokens for 8 requests, print OK or the full traceback.

    python bench/diag_engine.py --model Qwen/Qwen3.6-27B --config lora_maxcap
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

CONFIGS = {
    # rl_disagg's exact known-good kwargs (LoRA slots on, max capture size, no explicit list)
    "lora_maxcap": dict(lora=True, capture="max", packed=True, graphs=True),
    "nolora_maxcap": dict(lora=False, capture="max", packed=True, graphs=True),
    "nolora_list": dict(lora=False, capture="list", packed=True, graphs=True),
    "nolora_maxcap_nopacked": dict(
        lora=False, capture="max", packed=False, graphs=True
    ),
    "nolora_list_nopacked": dict(lora=False, capture="list", packed=False, graphs=True),
    "plain_maxcap": dict(lora=False, capture="max", packed=True, graphs="plain"),
    "plain_maxcap_nopacked": dict(
        lora=False, capture="max", packed=False, graphs="plain"
    ),
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", choices=sorted(CONFIGS), required=True)
    p.add_argument("--max-num-seqs", type=int, default=1024)
    p.add_argument("--attention-backend", default="TRITON_ATTN")
    a = p.parse_args()
    cfg = CONFIGS[a.config]
    if not cfg["packed"]:
        os.environ["VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE"] = "0"
    if cfg["graphs"] == "plain":
        os.environ["VLLM_LENS_DISABLE"] = "1"
    elif cfg["graphs"]:
        os.environ["VLLM_LENS_CUDA_GRAPHS"] = "1"
    import torch
    from vllm import LLM, SamplingParams

    kw = dict(
        model=a.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        max_model_len=144,
        attention_backend=a.attention_backend,
        language_model_only=True,
        enable_prefix_caching=False,
        max_num_seqs=a.max_num_seqs,
        max_num_batched_tokens=max(8192, a.max_num_seqs * 144),
        seed=0,
        dtype="bfloat16",
    )
    if cfg["lora"]:
        kw.update(enable_lora=True, max_loras=2, max_lora_rank=64)
    cc: dict = {}
    if cfg["capture"] == "max":
        cc["max_cudagraph_capture_size"] = a.max_num_seqs
    else:
        cc["cudagraph_capture_sizes"] = [
            s
            for s in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
            if s <= a.max_num_seqs
        ]
    if cfg["graphs"] is True:
        cc.update(mode=0, cudagraph_mode="FULL_DECODE_ONLY")
    kw["compilation_config"] = cc
    print(f"[diag] config={a.config} kw={kw}", flush=True)
    import triton

    print(
        f"[diag] torch {torch.__version__} triton {triton.__version__} gpu {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    t0 = time.time()
    try:
        llm = LLM(**kw)
        outs = llm.generate(
            [{"prompt_token_ids": list(range(100, 140))}] * 8,
            [SamplingParams(temperature=0.0, max_tokens=8)] * 8,
            use_tqdm=False,
        )
        vc = llm.llm_engine.vllm_config
        print(
            f"[diag] OK {a.config} in {time.time() - t0:.0f}s | mode={vc.compilation_config.mode} "
            f"cudagraph_mode={vc.compilation_config.cudagraph_mode} sizes={vc.compilation_config.cudagraph_capture_sizes[-3:]} "
            f"| sample={outs[0].outputs[0].token_ids}",
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        print(f"[diag] FAIL {a.config} in {time.time() - t0:.0f}s", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
