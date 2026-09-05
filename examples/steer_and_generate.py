#!/usr/bin/env python
"""Per-request steering with vllm-metamodels: one vector per prompt, injected at one prompt
position, generation under CUDA graphs.

    VLLM_LENS_CUDA_GRAPHS=1 python examples/steer_and_generate.py --model Qwen/Qwen3-1.7B

Every request carries its own ``SteeringVector``; the fork indexes them, applies all of a
layer's vectors with one ``index_add_`` in the (eager) prefill pass and lets decode run as
CUDA-graph replays.  ``norm_match=True, scale=coeff`` is the activation-oracle injection
``h + coeff * |h| * v/|v|`` on the full residual stream.
"""

from __future__ import annotations

import argparse
import os

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--layer", type=int, default=1, help="decoder layer whose OUTPUT receives the vector")
    p.add_argument("--marker", type=int, default=10, help="absolute prompt position that receives the vector")
    p.add_argument("--coeff", type=float, default=4.0, help="norm-matched strength: |delta| = coeff * |h|")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--max-tokens", type=int, default=32)
    a = p.parse_args()
    os.environ.setdefault("VLLM_LENS_CUDA_GRAPHS", "1")  # must be set before vLLM is imported
    # post7 options (also before vLLM is imported):
    #   VLLM_LENS_COMPILE=1   keep vLLM's torch.compile; the hooks run as a custom op inside the graph
    #   VLLM_LENS_SHM=1       captured activations come back through shared memory (or "view" for zero-copy)
    # Prefix caching may stay on (enable_prefix_caching=True): steered KV blocks are salted per request;
    # add extra_args["lens_cache_salt"] = "payload" to let identical (prompt, vector) rows share them.

    from vllm import LLM, SamplingParams

    from vllm_lens import SteeringVector

    llm = LLM(model=a.model, enable_prefix_caching=False, max_model_len=512, max_num_seqs=a.batch, dtype="bfloat16")
    tok = llm.get_tokenizer()
    prompt = tok("The most important thing to understand about this topic is that", add_special_tokens=False)["input_ids"]
    assert a.marker < len(prompt), "the marker must be a prompt position"
    hidden = llm.llm_engine.vllm_config.model_config.get_hidden_size()

    # one distinct random unit direction per request (in practice: your meta-model's activation)
    dirs = torch.nn.functional.normalize(torch.randn(a.batch, hidden), dim=-1)
    params = [
        SamplingParams(
            temperature=0.7,
            max_tokens=a.max_tokens,
            extra_args={
                "apply_steering_vectors": [
                    SteeringVector(
                        activations=dirs[i].view(1, 1, hidden),  # (n_layers, n_positions, hidden)
                        layer_indices=[a.layer],
                        position_indices=[a.marker],
                        norm_match=True,
                        scale=a.coeff,
                    )
                ]
            },
        )
        for i in range(a.batch)
    ]
    outs = llm.generate([{"prompt_token_ids": prompt}] * a.batch, params, use_tqdm=False)
    for o in outs[:4]:
        print(repr(o.outputs[0].text[:120]))
    stats = llm.collective_rpc("steering_stats")[0]
    print(f"hook stats: {stats['rows_steered']} rows steered in {stats['steer_layer_steps']} layer-steps, "
          f"{stats['steps_fast_idle']} decode passes skipped by the idle fast path, errors={stats['errors']}")


if __name__ == "__main__":
    main()
