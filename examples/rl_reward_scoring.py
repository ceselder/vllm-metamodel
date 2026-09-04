#!/usr/bin/env python
"""RL-rollout loop on ONE engine: generate with per-request steering under the current policy
(a LoRA adapter, or that adapter merged into the weights), then score every rollout with a
per-request direction read from the CLEAN base model at layer L (early exit: layers > L never run).

    VLLM_LENS_CUDA_GRAPHS=1 python examples/rl_reward_scoring.py --model Qwen/Qwen3-1.7B [--adapter DIR] [--merge]

Without ``--adapter`` the base model generates.  With ``--adapter DIR`` (a PEFT LoRA directory):
  * default: vLLM serves it as a LoRA (``LoRARequest`` on every request) -- the scoring pass
    passes ``lora_request=None`` and therefore reads the clean base;
  * ``--merge``: ``merge_lora`` folds the adapter into the base weights on the worker (no LoRA
    kernels during decode), ``unmerge_lora`` restores the clean base before scoring -- both are
    device copies when a base copy is kept (``keep_base="gpu"``).
"""

from __future__ import annotations

import argparse
import os
import time

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--adapter", default="", help="PEFT LoRA directory (adapter_config.json + adapter_model.safetensors)")
    p.add_argument("--merge", action="store_true", help="merge the adapter into the weights instead of serving it as a LoRA")
    p.add_argument("--inject-layer", type=int, default=1)
    p.add_argument("--read-layer", type=int, default=18)
    p.add_argument("--marker", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=40)
    a = p.parse_args()
    os.environ.setdefault("VLLM_LENS_CUDA_GRAPHS", "1")

    from vllm import LLM, SamplingParams

    from vllm_lens import SteeringVector
    from vllm_lens.metamodel import lora_status, merge_lora, readout_max, unmerge_lora

    kw = dict(model=a.model, enable_prefix_caching=False, max_model_len=512, max_num_seqs=a.batch, dtype="bfloat16")
    if a.adapter and not a.merge:
        kw.update(enable_lora=True, max_loras=1, max_lora_rank=64)
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    hidden = llm.llm_engine.vllm_config.model_config.get_hidden_size()
    prompt = tok("Describe the concept in one sentence:", add_special_tokens=False)["input_ids"]
    prompt = (prompt * 4)[: max(a.marker + 1, 24)]

    lora_request = None
    if a.adapter:
        if a.merge:
            info = merge_lora(llm, a.adapter, keep_base="gpu")
            print(f"merged adapter into {info['n_params']} weights in {info['publish_s']:.2f}s (mode {info['mode']}, base copy {info['base_bytes']/1e9:.1f} GB)")
        else:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(lora_name="policy", lora_int_id=1, lora_path=a.adapter)

    # 1. rollouts: one injected direction per prompt (the concept the meta-model should describe)
    concepts = torch.nn.functional.normalize(torch.randn(a.batch, hidden), dim=-1)
    params = [
        SamplingParams(temperature=1.0, max_tokens=a.max_tokens, extra_args={"apply_steering_vectors": [
            SteeringVector(activations=concepts[i].view(1, 1, hidden), layer_indices=[a.inject_layer],
                           position_indices=[a.marker], norm_match=True, scale=4.0)]})
        for i in range(a.batch)
    ]
    t0 = time.perf_counter()
    outs = llm.generate([{"prompt_token_ids": prompt}] * a.batch, params, lora_request=lora_request, use_tqdm=False)
    t_gen = time.perf_counter() - t0
    rollouts = [list(prompt) + list(o.outputs[0].token_ids) for o in outs]

    # 2. reward: cosine between the clean base model's layer-L state on the rollout and the concept
    #    direction, max over the last 5 tokens -- computed in the worker, only scalars come back
    if a.adapter and a.merge:
        unmerge_lora(llm)  # clean base again (a device copy with keep_base="gpu")
    t0 = time.perf_counter()
    reward = readout_max(llm, rollouts, concepts, layer=a.read_layer, positions={"last": 5}, lora_request=None)
    t_score = time.perf_counter() - t0
    if a.adapter and a.merge:
        merge_lora(llm, a.adapter, keep_base="gpu")  # back to the policy for the next step
    print(f"generated {a.batch} rollouts in {t_gen:.2f}s, scored them (early exit after layer {a.read_layer}) in {t_score:.2f}s")
    print(f"reward: mean {reward.mean():.4f}, min {reward.min():.4f}, max {reward.max():.4f}")
    if a.adapter and a.merge:
        print("lora status:", lora_status(llm))


if __name__ == "__main__":
    main()
