# vllm-metamodel

<p align="center">
  <img width="280" alt="vllm-metamodel logo" src="https://github.com/user-attachments/assets/64ee65fe-a436-47ae-acd8-c66c73310775" />
  <img width="540" alt="steering throughput vs batch size" src="https://github.com/user-attachments/assets/9b39dd5c-c870-4b56-86e7-a91bd0ed186e" />
</p>

**A drop-in replacement for [vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens) for meta-model workloads (Activation Oracles, MAEMMs, LoRAcles, NLAs) but 3-59x faster**

Meta-models need you to inject an activation (or soft token) into the model at some position. Because vllm doesn't support this, we generally use a library called [vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens), a vllm plugin that allows for steering residual stream.

However, vllm-lens does not go brr. At large batch sizes it is up to ~40x slower than standard vllm, and the reason is dumb: its forward hook runs on every single forward pass (i.e. every generated token, for every hooked layer), and on every pass it does a python for loop over every request in the batch, string-matching the request id against every registered steering key, doing two gpu syncs per request to find its rows, and cloning the layer output. It does this whether or not any vector actually applies on that step.~~I~~ Fable rewrote the hook to index the steering configs in a hashmap, build one plan per forward pass from host-side buffers (no gpu syncs), skip idle steps on a single check, and apply all vectors with one index_add_ instead of a loop. 

**Anyway, this makes generation like 40x faster and should just be merged into vllm-lens but whatever**

This change also allows you to use cuda-graphs after prefill, since meta-models generally only inject once at one token index during prefill, we can get near vllm-level performance with them applied for meta-models! yippee!

```bash
pip install git+https://github.com/ceselder/vllm-metamodel
```

## Features

Everything below is opt-in per request, keeps the upstream vllm-lens API, and (unless noted) works with CUDA graphs on.

| feature | how | section |
|---|---|---|
| **Fast per-request steering** — one vector per prompt, batches of 1000s | same `SteeringVector` API; indexed hook + one `index_add_` per layer-step + one block RPC per `generate()` | [Comparisons](#comparisons) |
| **CUDA graphs with hooks** | `VLLM_LENS_CUDA_GRAPHS=1` → decode runs as graph replays, prompt positions stay hookable | [CUDA graphs](#cuda-graphs) |
| **Karvonen-style injection** `h + coeff·‖h‖·unit(v)` on the full residual stream | `SteeringVector(norm_match=True, scale=coeff)` | [Karvonen-style injection](#karvonen-style-norm-matched-injection-norm_matchtrue-scalecoeff) |
| **Embedding replacement** (NLA-style, hyper-connection models) | `mode="replace"`, `layer_indices=[EMBED_LAYER_INDEX]` | [Embedding replacement](#embedding-replacement-nla--metamodels-on-hyper-connection-architectures) |
| **Cheap hidden-state capture** — only the layers/positions you ask for | `extra_args={"output_residual_stream": [L], "capture_positions": {"last": k}}` | [Fast hidden-state readout](#fast-hidden-state-readout-110post4) |
| **In-engine readout (scalars, not tensors)** — cosine / dot with a per-request direction | `ReadoutVector(...)` in `extra_args["apply_readout_vectors"]` | [Fast hidden-state readout](#fast-hidden-state-readout-110post4) |
| **Early exit** — stop the forward pass after layer L | `extra_args["lens_early_exit"] = True` with `max_tokens=1` | [Early exit](#early-exit) |
| **One-call reward scoring** | `vllm_lens.metamodel.readout_scores(llm, token_ids, directions, layer=42)` | [One-call scoring](#one-call-scoring-vllm_lensmetamodel) |
| Capability introspection / kill switches | `llm.collective_rpc("lens_capabilities")`, `VLLM_LENS_DISABLE=1`, `VLLM_LENS_FAST_CAPTURE=0`, `VLLM_LENS_EARLY_EXIT=0`, `VLLM_LENS_BLOCK_RPC=0` | [API reference](#api-reference) |


# Cleaned up claudeslop information below

## Comparisons
The comparisons I did are for the exact situation for the model I'm training for my upcoming paper: MAEMMs

Measured on 1× B200, Qwen/Qwen3.6-27B bf16, 96-token prompts, generating 40 new tokens, one distinct steering vector per request (layer 1, one prompt position): at B = 2,048 the fork is **37.8× faster** than stock 1.1.0 with CUDA graphs (36.3× from the indexed hook alone, 36.8× with the vectorised apply, eager), coming within +-5% of not doing steering at all.

![per-request steering throughput vs batch size](bench/steering_throughput.png)

Wall time of one `LLM.generate()` call (Qwen/Qwen3.6-27B, speedup vs stock in bold):

| configuration | B = 8 | B = 32 | B = 128 | B = 512 | B = 1,024 | B = 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| stock vllm-lens 1.1.0 (eager forced) | 2.0 s | 3.6 s | 12.3 s | 76.3 s | 230.2 s | 777.6 s |
| fork: indexed/vectorized steering hook (eager) | 1.3 s (**1.5×**) | 1.4 s (**2.5×**) | 2.0 s (**6.1×**) | 5.8 s (**13.1×**) | 10.5 s (**21.8×**) | 21.1 s (**36.8×**) |
| fork: indexed/vectorized steering hook + CUDA graphs | 0.6 s (**3.3×**) | 0.8 s (**4.5×**) | 1.6 s (**7.9×**) | 5.5 s (**13.9×**) | 10.5 s (**21.9×**) | 20.5 s (**37.8×**) |
| vLLM baseline (compile + graphs) (ceiling) | 0.5 s (**3.7×**) | 0.7 s (**4.9×**) | 1.5 s (**8.4×**) | 5.3 s (**14.5×**) | 10.1 s (**22.8×**) | 19.7 s (**39.5×**) |

Eager rows ran with `max_num_seqs=2048` (B = 2,048 in one scheduler wave); the CUDA-graph rows with `max_num_seqs=1024` and vLLM's default capture ladder (`max_cudagraph_capture_size=1024`), so B = 2,048 is two waves there -- see the vLLM caveat below (packed GDN decode kernel grid limit). Decode at these sizes is compute-bound and scales linearly (2,048 takes 2x the 1,024 time in every configuration).

The gap is even bigger on smaller models, where the hook is actually the majority of overhead in a lot of cases (Qwen/Qwen3-1.7B, same experiment)

![Qwen/Qwen3-1.7B](bench/steering_throughput_qwen3-1.7b.png)

| configuration | B = 8 | B = 32 | B = 128 | B = 512 | B = 1,024 |
|---|---:|---:|---:|---:|---:|
| stock vllm-lens 1.1.0 (eager forced) | 0.6 s | 1.3 s | 5.0 s | 31.5 s | 96.8 s |
| fork: indexed hook (eager) | 0.7 s (**0.9×**) | 0.7 s (**1.9×**) | 0.9 s (**5.7×**) | 1.4 s (**22.4×**) | 2.7 s (**35.2×**) |
| fork: indexed/vectorized steering hook apply (eager) | 0.7 s (**0.8×**) | 0.7 s (**1.9×**) | 0.9 s (**5.5×**) | 1.4 s (**22.5×**) | 2.3 s (**43.0×**) |
| fork: indexed/vectorized steering hook + CUDA graphs | 0.2 s (**2.9×**) | 0.3 s (**5.2×**) | 0.4 s (**12.0×**) | 0.8 s (**37.5×**) | 1.6 s (**59.3×**) |
| vLLM  baseline (compile + graphs) (ceiling) | 0.1 s (**4.8×**) | 0.2 s (**7.4×**) | 0.3 s (**16.8×**) | 0.8 s (**38.9×**) | 1.7 s (**56.3×**) |

Full numbers, per-condition hook counters and every correctness assertion: `bench/results/` (`python bench/compare.py bench/results/<timestamp>`).
<!-- RESULTS:END -->

Changelog: [CHANGELOG.md](CHANGELOG.md) (current: **1.1.0.post4** — fast hidden-state readout: gather capture, `ReadoutVector` projections, early exit).

## Embedding replacement (NLA / metamodels on hyper-connection architectures)

Layer-output steering is undefined on architectures whose decoder layers emit
multi-stream outputs (hyper-connections, e.g. **DeepSeek-V4**: every layer
boundary carries a 4-stream residual stack — there is no single tensor to add
a vector to). NLA-style metamodels also *specify* their injection at the
embedding: *replace the marker token's embedding with α·v/‖v‖*.

This fork supports both, via `mode="replace"` and the `EMBED_LAYER_INDEX`
sentinel:

```python
from vllm_lens import SteeringVector, EMBED_LAYER_INDEX

# NLA injection: overwrite the marker token's embedding with alpha * v/||v||
v = activation / activation.norm()
sv = SteeringVector(
    activations=v.reshape(1, 1, -1),      # (n_layers=1, n_positions=1, hidden)
    layer_indices=[EMBED_LAYER_INDEX],    # the embedding stream, not a layer output
    position_indices=[marker_pos],        # absolute prompt position of the marker
    mode="replace",                       # overwrite, don't add
    scale=alpha,                          # e.g. p75 of the layer's activation norms
)
```

Semantics:
- `mode="replace"` overwrites the hidden row with `scale * v`
  (`norm_match=True` instead writes `scale * ‖h_orig‖ · v/‖v‖`). Requires 3-D
  (position-specific) activations — broadcast replacement is rejected.
- `EMBED_LAYER_INDEX` targets the hidden states *entering* decoder layer 0
  (the embedding output). It is applied by the first layer's **pre-hook**, so
  it never touches the (possibly multi-stream) layer outputs and works on any
  architecture. The pre-hook is registered with `with_kwargs=True` and finds
  the hidden states among positional *and* keyword inputs (Qwen3.5/3.6's
  `Qwen3NextModel` calls `layer(positions=…, hidden_states=…, residual=…)` by
  keyword, Qwen2/Llama positionally); it requires exactly one
  `[≥ total_tokens, hidden]` floating candidate and otherwise raises
  `EmbedInjectionError` out of the forward pass (counted in `embed_errors`) —
  never a silent skip.
- `mode="replace"` on a regular layer index of a fused-residual model (Qwen,
  Llama, Gemma, …: layers return `(hidden_states, residual)`) rewrites **both
  halves** (`hidden_states[row] = scale·v`, `residual[row] = 0`) so the full
  residual stream equals `scale·v` exactly; vectorised (`index_copy_`).
- `output_residual_stream=[EMBED_LAYER_INDEX, …]` captures the (post-injection)
  embedding stream as layer -1; `output_residual_stream=True` is unchanged.
- Injection happens during prefill only (markers are prompt positions), so
  **decode-only CUDA graphs stay legal** — same performance story as the
  indexed steering hook (measured: hooks run on 2 of 41 forward passes at
  B=1,024, wall time equal to no steering). Chunked prefill is handled: the
  replacement lands in whichever chunk contains the marker's absolute position
  (tested with the marker in a first and in a non-first 64-token chunk).

This is the injection mode used to RL-train the DeepSeek-V4-Flash NLA
(embedding replacement at TP8 on the native fp8 engine, measured within ~5%
of injection-free vLLM throughput with decode CUDA graphs).

## Karvonen-style (norm-matched) injection: `norm_match=True, scale=coeff`

Activation-oracle / MAEMM trainers inject `h' = h + coeff · ‖h‖ · v/‖v‖` at a
decoder layer's output, with an HF forward hook on the decoder layer
(reference: `mxf/inject.py::make_inject_hook`). In this fork that is exactly

```python
sv = SteeringVector(
    activations=v.reshape(1, 1, -1),   # any magnitude: only the direction is used
    layer_indices=[layer],
    position_indices=[marker_pos],
    norm_match=True,                   # scale to ‖h‖ of the FULL residual stream at that position
    scale=coeff,                       # h' = h + coeff · ‖h‖ · v/‖v‖
)
```

**Behaviour change vs vllm-lens 1.1.0 / fork post1.** vLLM's decoder layers on
Qwen / Llama / Gemma / Mistral return `(hidden_states, residual)` and the
residual stream is their *sum*. 1.1.0 measured `‖·‖` on `hidden_states` alone
(the MLP-delta half), so `norm_match=True` injected `coeff · ‖hidden_states‖ ·
v/‖v‖` — on Qwen3.6-27B layer 1 a magnitude ratio of **0.123** (≈ 8× too weak,
and a different factor per model and layer). Upstream fixed this in 1.2.0
(#7); 1.1.0.post2 ports the fix into the sequential, vectorised and
CUDA-graph paths, so `norm_match=True, scale=coeff` now matches the HF hook
to cos ≥ 0.99998 and magnitude ratio within 5e-4 of `coeff` (test matrix
below, both models, eager and graphs, one distinct vector per request). If you relied on the old behaviour, your
effective coefficient changes; trainers that worked around it by passing an
absolute vector with `norm_match=False` are unaffected.

## Injection modes: GPU test matrix

<!-- INJECTION:BEGIN -->
`bench/test_injection_modes.py` on 1× B200 (vLLM 0.19.0, torch 2.10.0+cu128, vllm-lens 1.1.0.post2),
models Qwen/Qwen3.6-27B, Qwen/Qwen3-1.7B; engines eager, eager +chunked, graphs, graphs +chunked; **278/278 gated checks pass**
(all; 10 throughput gates on the 1.7B — decode-step time and the eager wall ratio — are below the measurement resolution of a 0.5–2 s call (control repeat spread 13–28%) and are reported as informational, not as passes). Every request in a batch carries its own unit vector; B is the number of requests
per `generate()` call; the marker is prompt position 10 (chunked engines also test position 70, i.e. a non-first 64-token
chunk). "cos(Δ, v)" / "‖Δ‖/(c·‖h‖) − 1" are the worst request of the batch; "other rows" is the max absolute change of any
non-marker row of the captured layer (0 = bit-identical). The HF reference is the same model in transformers with the
trainer's exact hook (`mxf/inject.py`, `h + coeff·‖h‖·v/‖v‖` on the decoder-layer output); the log-prob noise floor is
the vLLM-vs-HF difference on the clean prompt.

| model | engine | case | B | coeff / scale | cos(Δ, v) | ‖Δ‖/(c·‖h‖) − 1 or bf16 rel err | other rows max\|Δ\| | vs HF reference | checks |
|---|---|---|---:|---|---:|---:|---:|---|---|
| Qwen3-1.7B | eager | karvonen_add | 64 | coeff 1.0 | 0.99999 | 3.7e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0020 | 4/4 |
| Qwen3-1.7B | eager | karvonen_add | 512 | coeff 1.0 | 0.99999 | 2.8e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | eager | karvonen_add | 64 | coeff 4.0 | 0.99999 | 3.1e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0018 | 4/4 |
| Qwen3-1.7B | eager | karvonen_add | 512 | coeff 4.0 | 0.99999 | 1.5e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | eager | karvonen_add_generation | — | coeff 1.0 | — | — | — | next-token logprob max diff 0.135 (clean-prompt noise 0.239); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | eager | karvonen_add_generation | — | coeff 4.0 | — | — | — | next-token logprob max diff 0.237 (clean-prompt noise 0.239); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | eager | layer_replace | 64 | scale 23.16 | 1.00000 | 7.8e-03 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | eager | embed_replace | 64 | scale 1.38 | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | eager | embed_replace | 512 | scale 1.38 | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | eager | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | eager | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | eager | mixed | 64 | coeff 1.0 | 0.99999 | 3.7e-04 | 0.0e+00 | — | 5/5 |
| Qwen3-1.7B | eager +chunked | chunked_m10_karvonen_add | 16 | coeff 1.0 | 0.99999 | 3.1e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | eager +chunked | chunked_m10_embed_replace | 16 | scale 1.38 | 1.00000 | 5.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | eager +chunked | chunked_m70_karvonen_add | 16 | coeff 1.0 | 0.99999 | 4.3e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | eager +chunked | chunked_m70_embed_replace | 16 | scale 1.38 | 1.00000 | 5.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs | karvonen_add | 64 | coeff 1.0 | 0.99999 | 3.7e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0020 | 4/4 |
| Qwen3-1.7B | graphs | karvonen_add | 512 | coeff 1.0 | 0.99999 | 2.8e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | graphs | karvonen_add | 64 | coeff 4.0 | 0.99999 | 3.1e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0018 | 4/4 |
| Qwen3-1.7B | graphs | karvonen_add | 512 | coeff 4.0 | 0.99999 | 1.5e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | graphs | karvonen_add_generation | — | coeff 1.0 | — | — | — | next-token logprob max diff 0.135 (clean-prompt noise 0.239); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | graphs | karvonen_add_generation | — | coeff 4.0 | — | — | — | next-token logprob max diff 0.237 (clean-prompt noise 0.239); greedy-8 equal 4/4 | 2/2 |
| Qwen3-1.7B | graphs | layer_replace | 64 | scale 23.16 | 1.00000 | 7.8e-03 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | graphs | embed_replace | 64 | scale 1.38 | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs | embed_replace | 512 | scale 1.38 | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 7.6e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 7.8e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs | mixed | 64 | coeff 1.0 | 0.99999 | 3.7e-04 | 0.0e+00 | — | 5/5 |
| Qwen3-1.7B | graphs +chunked | chunked_m10_karvonen_add | 16 | coeff 1.0 | 0.99999 | 3.1e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | graphs +chunked | chunked_m10_embed_replace | 16 | scale 1.38 | 1.00000 | 5.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3-1.7B | graphs +chunked | chunked_m70_karvonen_add | 16 | coeff 1.0 | 0.99999 | 4.3e-04 | 0.0e+00 | — | 3/3 |
| Qwen3-1.7B | graphs +chunked | chunked_m70_embed_replace | 16 | scale 1.38 | 1.00000 | 5.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager | karvonen_add | 64 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0001 | 4/4 |
| Qwen3.6-27B | eager | karvonen_add | 512 | coeff 1.0 | 0.99999 | 1.7e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | eager | karvonen_add | 64 | coeff 4.0 | 0.99999 | 2.4e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0002 | 4/4 |
| Qwen3.6-27B | eager | karvonen_add | 512 | coeff 4.0 | 0.99999 | 1.7e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | eager | karvonen_add_generation | — | coeff 1.0 | — | — | — | next-token logprob max diff 0.103 (clean-prompt noise 0.253); greedy-8 equal 3/4 | 2/2 |
| Qwen3.6-27B | eager | karvonen_add_generation | — | coeff 4.0 | — | — | — | next-token logprob max diff 0.121 (clean-prompt noise 0.253); greedy-8 equal 3/4 | 2/2 |
| Qwen3.6-27B | eager | layer_replace | 64 | scale 13.19 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | eager | embed_replace | 64 | scale 0.97 | 1.00000 | 7.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager | embed_replace | 512 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager | mixed | 64 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | — | 5/5 |
| Qwen3.6-27B | eager +chunked | chunked_m10_karvonen_add | 16 | coeff 1.0 | 0.99999 | 1.4e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | eager +chunked | chunked_m10_embed_replace | 16 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | eager +chunked | chunked_m70_karvonen_add | 16 | coeff 1.0 | 0.99999 | 2.3e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | eager +chunked | chunked_m70_embed_replace | 16 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs | karvonen_add | 64 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0001 | 4/4 |
| Qwen3.6-27B | graphs | karvonen_add | 512 | coeff 1.0 | 0.99999 | 1.7e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | graphs | karvonen_add | 64 | coeff 4.0 | 0.99999 | 2.4e-04 | 0.0e+00 | cos 1.0000, norm ratio 1.0002 | 4/4 |
| Qwen3.6-27B | graphs | karvonen_add | 512 | coeff 4.0 | 0.99999 | 1.7e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | graphs | karvonen_add_generation | — | coeff 1.0 | — | — | — | next-token logprob max diff 0.103 (clean-prompt noise 0.253); greedy-8 equal 3/4 | 2/2 |
| Qwen3.6-27B | graphs | karvonen_add_generation | — | coeff 4.0 | — | — | — | next-token logprob max diff 0.121 (clean-prompt noise 0.253); greedy-8 equal 3/4 | 2/2 |
| Qwen3.6-27B | graphs | layer_replace | 64 | scale 13.19 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | graphs | embed_replace | 64 | scale 0.97 | 1.00000 | 7.1e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs | embed_replace | 512 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs | mixed | 64 | coeff 1.0 | 0.99999 | 1.9e-04 | 0.0e+00 | — | 5/5 |
| Qwen3.6-27B | graphs +chunked | chunked_m10_karvonen_add | 16 | coeff 1.0 | 0.99999 | 1.4e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | graphs +chunked | chunked_m10_embed_replace | 16 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |
| Qwen3.6-27B | graphs +chunked | chunked_m70_karvonen_add | 16 | coeff 1.0 | 0.99999 | 2.3e-04 | 0.0e+00 | — | 3/3 |
| Qwen3.6-27B | graphs +chunked | chunked_m70_embed_replace | 16 | scale 0.97 | 1.00000 | 5.3e-03 | 0.0e+00 | — | 4/4 |

| model | engine | condition | B | wall, 40 new tok (s) | wall, 80 new tok (s) | decode step (ms) | prefill + per-call overhead (s) | tok/s | hook passes | checks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen3-1.7B | eager | nosteer | 512 | 0.65 | 1.13 | 11.14 | 0.17 | 31,354 | 41 | — |
| Qwen3-1.7B | eager | nosteer | 1024 | 1.08 | 1.82 | 18.43 | 0.35 | 37,810 | 42 | — |
| Qwen3-1.7B | eager | karvonen_add | 512 | 0.66 | 1.14 | 12.32 | 0.18 | 31,044 | 41 | 0/0 (+2 n/a) |
| Qwen3-1.7B | eager | karvonen_add | 1024 | 1.15 | 2.16 | 26.77 | 0.13 | 35,676 | 41 | 0/0 (+2 n/a) |
| Qwen3-1.7B | eager | embed_replace | 512 | 0.67 | 1.16 | 12.20 | 0.19 | 30,377 | 41 | 0/0 (+1 n/a) |
| Qwen3-1.7B | eager | embed_replace | 1024 | 1.12 | 1.87 | 18.84 | 0.36 | 36,621 | 41 | 0/0 (+1 n/a) |
| Qwen3-1.7B | graphs | nosteer | 512 | 0.47 | 0.73 | 6.42 | 0.21 | 43,702 | 2 | — |
| Qwen3-1.7B | graphs | nosteer | 1024 | 0.90 | 1.48 | 13.67 | 0.32 | 45,548 | 3 | — |
| Qwen3-1.7B | graphs | karvonen_add | 512 | 0.50 | 0.75 | 6.51 | 0.24 | 41,224 | 2 | 1/1 (+1 n/a) |
| Qwen3-1.7B | graphs | karvonen_add | 1024 | 0.96 | 1.39 | 12.79 | 0.52 | 42,882 | 4 | 1/1 (+1 n/a) |
| Qwen3-1.7B | graphs | embed_replace | 512 | 0.49 | 0.74 | 6.78 | 0.24 | 41,641 | 2 | 1/1 (+1 n/a) |
| Qwen3-1.7B | graphs | embed_replace | 1024 | 0.92 | 1.40 | 17.16 | 0.44 | 44,518 | 3 | 1/1 (+1 n/a) |
| Qwen3.6-27B | eager | nosteer | 512 | 7.70 | 12.48 | 119.42 | 2.92 | 2,660 | 82 | — |
| Qwen3.6-27B | eager | nosteer | 1024 | 13.27 | 20.69 | 185.38 | 5.86 | 3,086 | 123 | — |
| Qwen3.6-27B | eager | karvonen_add | 512 | 7.74 | 12.52 | 119.34 | 2.97 | 2,645 | 82 | 2/2 |
| Qwen3.6-27B | eager | karvonen_add | 1024 | 13.31 | 20.79 | 187.08 | 5.82 | 3,078 | 123 | 2/2 |
| Qwen3.6-27B | eager | embed_replace | 512 | 7.75 | 12.56 | 120.30 | 2.93 | 2,644 | 82 | 1/1 |
| Qwen3.6-27B | eager | embed_replace | 1024 | 13.31 | 20.78 | 186.74 | 5.84 | 3,078 | 123 | 1/1 |
| Qwen3.6-27B | graphs | nosteer | 512 | 5.66 | 8.46 | 69.88 | 2.87 | 3,617 | 4 | — |
| Qwen3.6-27B | graphs | nosteer | 1024 | 10.81 | 16.00 | 129.61 | 5.63 | 3,789 | 6 | — |
| Qwen3.6-27B | graphs | karvonen_add | 512 | 5.72 | 8.55 | 70.79 | 2.89 | 3,582 | 4 | 2/2 |
| Qwen3.6-27B | graphs | karvonen_add | 1024 | 10.84 | 16.10 | 131.44 | 5.58 | 3,779 | 6 | 2/2 |
| Qwen3.6-27B | graphs | embed_replace | 512 | 5.72 | 8.56 | 71.04 | 2.88 | 3,580 | 4 | 2/2 |
| Qwen3.6-27B | graphs | embed_replace | 1024 | 10.95 | 16.07 | 128.03 | 5.82 | 3,742 | 6 | 2/2 |

![steering throughput by injection mode](bench/injection_throughput.png)

Re-run: `MODAL_PROFILE=safety-sahan modal run bench/modal_bench.py::test_injection` (both models, eager + graphs, chunked
engines, HF reference; ~25 min of B200), then `python bench/test_injection_modes.py bench/results/injection_<ts>` for the
summary and `python bench/render_injection_readme.py bench/results/injection_<ts>` for this block. Single engine by hand:
`python bench/test_injection_modes.py --model Qwen/Qwen3-1.7B --stage hf-ref --out ref.pt` then
`VLLM_LENS_CUDA_GRAPHS=1 python bench/test_injection_modes.py --model Qwen/Qwen3-1.7B --stage vllm --engine graphs --ref ref.pt --out graphs.json`.
<!-- INJECTION:END -->

## Hyper-connection architectures (DeepSeek-V4-Flash, vLLM 0.27.1, TP4)

DeepSeek-V4 (mHC, `hc_mult=4`) keeps four residual streams between layers; in vLLM each
decoder layer returns `(x, residual[T, 4, D], post_mix, res_mix)` and the true stream is a
deferred fold computed by the *next* layer. There is no single `[tokens, hidden]` tensor at
a layer boundary, so **layer-output steering and capture are undefined there** — 1.1.0.post3
detects this (`hc_mult > 1`, override `VLLM_LENS_MULTI_STREAM=0/1`) and refuses
`layer_indices=[k]` / `output_residual_stream=[k]` with a `ValueError` before the request
reaches the engine (engine stays alive); a runtime `UnsupportedLayerOutputError` backstop
is raised — never swallowed — if a multi-stream tuple ever reaches the layer hook. The
embedding entering layer 0 is a plain `[T, D]` tensor, so `EMBED_LAYER_INDEX` injection
(replace or add, ± `norm_match`) and `output_residual_stream=[EMBED_LAYER_INDEX]` work
unchanged. post3 also stops depending on `query_start_loc` being present on the attention
metadata (vLLM ≥ 0.27 moved it; the runner's host buffers are used), which is what made
vllm-lens 1.2.1's hooks silently no-op on vLLM 0.27, and sets
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` for its own pickled RPC payloads.

**Engine finding (vLLM 0.27.1 DeepSeek-V4, not the fork).** With CUDA graphs on, the next-token
top-20 log-probs of an *unsteered* request depend on what else is in its prefill batch: a
batch identical to the reference is bit-exact (clean-vs-clean 0.000, prefix caching off,
`num_cached_tokens = 0`), but odd rows co-batched with even rows that merely carry a
*different marker token* — no hooks anywhere — shift by up to 1.016, the same value seen with
the fork's embed-replace on the even rows, and the greedy argmax never flips. The eager engine
shows no such dependence (0.000 in the same experiment). The fork's own write is verified inert
for those rows (their embedding stream is bit-identical to clean) and adds nothing beyond the
hook-free control; for RL the actionable part is that per-request log-probs on this engine are
batch-dependent at the O(1)-in-the-tail level under CUDA graphs.

<!-- INJECTION-DSV4:BEGIN -->
`bench/test_injection_dsv4.py` on B200 x4 (TP4, vLLM 0.27.1, torch 2.13.0+cu130, vllm-lens 1.1.0.post3;
`kv_cache_dtype=fp8_ds_mla`, `kernel_config.moe_backend=deep_gemm`, `max_num_batched_tokens=4096` so prefill is chunked;
`hc_mult=4`, `expert_dtype=fp4`): **128/128 gated checks pass**
(all; 14 informational). Layer outputs on this architecture are a
deferred 4-stream fold, so the fork refuses layer-output steering / capture with a `ValueError` (engine alive) and everything
goes through the embedding stream (`EMBED_LAYER_INDEX`). The reference is the NLA session's own arithmetic
(`nla.utils.dsv4.scale_vector_to_alpha`, `alpha·v/‖v‖`) and its worker-side pre-hook (`nla.utils.dsv4_fast_hooks`),
run on the same engine; "bf16 rel err" in the table is the max relative error of the written marker row against the target.

| model | engine | case | B | coeff / scale | cos(Δ, v) | ‖Δ‖/(c·‖h‖) − 1 or bf16 rel err | other rows max\|Δ\| | vs HF reference | checks |
|---|---|---|---:|---|---:|---:|---:|---|---|
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 64 | scale 1.53 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 2.9e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 512 | scale 1.53 | 1.00000 | 5.9e-03 | 0.0e+00 | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 3.0e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace_prescaled | 64 | scale 95.50 | — | — | 0.0e+00 | — | 2/2 |
| DeepSeek-V4-Flash-0731 | eager | embed_replace_prescaled | 512 | scale 95.50 | — | — | 0.0e+00 | — | 2/2 |
| DeepSeek-V4-Flash-0731 | eager | reference_impl | 64 | scale 95.50 | — | — | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | chunked_m70_embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | chunked_m70_embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | eager | mixed | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 5/5 |
| DeepSeek-V4-Flash-0731 | eager | effect_check | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 64 | scale 1.53 | 1.00000 | 6.3e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 64 | scale 1.00 (norm_match) | 1.00000 | 2.9e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | scale 1.53 | 1.00000 | 5.9e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | scale 1.00 (norm_match) | 1.00000 | 3.0e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace_prescaled | 64 | scale 95.50 | — | — | 0.0e+00 | — | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace_prescaled | 512 | scale 95.50 | — | — | 0.0e+00 | — | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | reference_impl | 64 | scale 95.50 | — | — | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | graphs | chunked_m70_embed_replace | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | chunked_m70_embed_replace | 512 | scale 95.50 | 1.00000 | 6.1e-03 | 0.0e+00 | — | 3/3 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | multi_stream_guard | — | — | — | — | — | — | 8/8 |
| DeepSeek-V4-Flash-0731 | graphs | mixed | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 6/6 |
| DeepSeek-V4-Flash-0731 | graphs | effect_check | 64 | scale 95.50 | 1.00000 | 6.5e-03 | 0.0e+00 | — | 4/4 |
| DeepSeek-V4-Flash-0731 | graphs | batch_composition | 64 | — | — | — | — | — | 1/1 |
| DeepSeek-V4-Flash-0731 | graphs | batch_composition | 64 | — | — | — | — | — | 1/1 |
| DeepSeek-V4-Flash-0731 | graphs | batch_composition | 64 | — | — | — | — | — | 1/1 |
| DeepSeek-V4-Flash-0731 | graphs | batch_composition | 64 | — | — | — | — | — | 1/1 |
| DeepSeek-V4-Flash-0731 | graphs | batch_composition | 64 | — | — | — | — | — | 1/1 |

| model | engine | condition | B | wall, 40 new tok (s) | wall, 80 new tok (s) | decode step (ms) | prefill + per-call overhead (s) | tok/s | hook passes | checks |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| DeepSeek-V4-Flash-0731 | graphs | nosteer | 512 | 2.85 | 3.84 | -39.18 | 1.86 | 7,197 | 15 | — |
| DeepSeek-V4-Flash-0731 | graphs | nosteer | 1024 | 5.23 | 6.75 | 40.12 | 3.71 | 7,834 | 29 | — |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 512 | 2.92 | 3.89 | 24.34 | 1.94 | 7,019 | 14 | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | embed_replace | 1024 | 5.39 | 6.88 | 37.17 | 3.90 | 7,601 | 29 | 2/2 |
| DeepSeek-V4-Flash-0731 | graphs | embed_add | 512 | 2.92 | 3.89 | 26.55 | 1.94 | 7,025 | 14 | 1/1 |
| DeepSeek-V4-Flash-0731 | graphs | embed_add | 1024 | 5.45 | 6.80 | 34.57 | 4.09 | 7,522 | 29 | 1/1 |

![DeepSeek-V4 throughput by injection mode](bench/dsv4_throughput.png)

Re-run: `MODAL_PROFILE=safety-sahan modal run bench/modal_bench_dsv4.py::run1` (eager correctness), `::run2` (CUDA graphs:
correctness + throughput), `::run3` (throughput repeat, interleaved, 3 repeats); then
`python bench/render_dsv4_readme.py bench/results/dsv4_run1_eager_<ts> bench/results/dsv4_run2_graphs_<ts> bench/results/dsv4_run3_graphs_tp_<ts>`.
<!-- INJECTION-DSV4:END -->

## Early exit

`extra_args["lens_early_exit"] = True` tells the engine that a request only needs the residual
stream up to the deepest layer it reads (its `output_residual_stream` layers and/or its
`ReadoutVector.layer_indices`) — so the remaining decoder layers, the final norm and the
LM head can be skipped. On Qwen3.6-27B reading layer 42 of 64 that removes ~34 % of the
prefill FLOPs: 1,024 texts in **4.7 s** instead of 6.6 s for a plain no-hook prefill.

How it works: when *every* request in a forward pass is an early-exit request, the hook on
the deepest requested layer raises a sentinel after it has captured / projected its rows;
a wrapper around the model runner's forward catches it and returns a zero placeholder for
the logits. Nothing is written to the KV cache beyond that layer for those rows.

Rules:

- `max_tokens=1` only (it is a scoring pass, not a generation); the one sampled token is
  meaningless — ignore it.
- The engine must run with **`enable_prefix_caching=False`** (skipped layers would leave stale
  KV blocks that a later request could reuse), pipeline parallel size 1, and no auxiliary
  hidden-state outputs (e.g. EAGLE). `llm.collective_rpc("lens_capabilities")[0]["early_exit"]`
  reports whether the engine qualifies (`"early_exit_reason"` says why not); the plugin
  refuses non-qualifying requests with a `ValueError` *before* they reach the engine.
- A pass that mixes generating requests with early-exit requests simply runs to the end —
  correct, just without the saving. So an RL rollout engine can score between generation
  calls with no mode switch (use `lora_request=None` to read the clean base model).
- Works identically on eager and CUDA-graph engines (prefill passes run eagerly either way).
- `VLLM_LENS_EARLY_EXIT=0` disables it globally.

```python
sp = SamplingParams(max_tokens=1, extra_args={
    "apply_readout_vectors": [ReadoutVector(activations=d.view(1, 5120), layer_indices=[42],
                                            positions={"last": 5}, metric="cos")],
    "lens_early_exit": True,
})
out = llm.generate([{"prompt_token_ids": ids}], sp, lora_request=None)[0]
reward = out.readout[0]["values"].max()     # layers 43..63 never ran
```

## One-call scoring (`vllm_lens.metamodel`)

```python
from vllm_lens.metamodel import readout_scores, readout_max, capabilities

values, positions = readout_scores(llm, token_ids, directions, layer=42,
                                   positions={"last": 5}, metric="cos", early_exit=True,
                                   lora_request=None)   # clean base on a LoRA engine
# values: float32 [n, n_layers, n_pos] (NaN-padded for short texts); positions: per-text absolute positions
rewards = readout_max(llm, token_ids, directions, layer=42)   # [n] = max over the window
capabilities(llm)   # {"early_exit": True, "multi_stream": False, ...}
```

`readout_scores` builds one prefill-only request per text (`max_tokens=1`, one `ReadoutVector`
each, `lens_early_exit` when the engine supports it — otherwise it warns and runs the full
model), issues a single `generate()` call, and stacks the scalars. `metric="dot"` with
`bias` gives SAE-feature pre-activations.

## API reference

Per-request keys in `SamplingParams.extra_args` (offline) / `vllm_xargs` (HTTP):

| key | type | meaning |
|---|---|---|
| `apply_steering_vectors` | `list[SteeringVector]` | upstream API; `mode="add"` (default) or `"replace"`, `norm_match`, `scale`, `layer_indices` (may contain `EMBED_LAYER_INDEX`), `position_indices` |
| `output_residual_stream` | `True` or `list[int]` | capture the residual stream at these layers (`True` = all; not allowed with early exit) |
| `capture_positions` | `"all"` \| `{"last": k}` \| `list[int]` | which positions to capture (default `"all"`); negative ints count from the end of the prompt |
| `apply_readout_vectors` | `list[ReadoutVector]` | in-engine projection; result on `output.readout` |
| `lens_early_exit` | `bool` | stop after the deepest requested layer (see [Early exit](#early-exit)) |

`ReadoutVector(activations=[n_layers, hidden], layer_indices, positions="all"|{"last": k}|[...], metric="cos"|"dot", bias=0.0)` →
`output.readout[i] = {"values": Tensor[n_layers, n_pos] float32, "positions": [int], "layers": [int]}`.

Environment switches (read when the engine starts): `VLLM_LENS_CUDA_GRAPHS=1` (enable graphs;
prompt-position steering/capture only), `VLLM_LENS_DISABLE=1` (plugin off), `VLLM_LENS_FAST_CAPTURE=0`
(upstream capture path), `VLLM_LENS_EARLY_EXIT=0`, `VLLM_LENS_BLOCK_RPC=0` (per-request RPCs),
`VLLM_LENS_MULTI_STREAM=0|1` (override hyper-connection detection).

## Fast hidden-state readout (1.1.0.post4)

The other half of a meta-model training loop is *reading* the residual stream back out:
an RL reward or eval re-encodes every generated text through the clean base model and
looks at layer L (for us: layer 42 of 64 on Qwen3.6-27B, a per-token cosine with a
per-request target direction, max over the last 5 tokens).  Stock vllm-lens can do this
(`output_residual_stream=[42]`) but pays for it three times: every one of the 64 layer
hooks loops over the batch, the captured layer does one blocking `.cpu()` per request,
and retrieval is one zstd-compressed RPC per request carrying the whole `[T, 5120]`
tensor.  1.1.0.post4 adds three things, all opt-in per request and all working under
CUDA graphs (they only ever touch prompt positions):

```python
from vllm_lens import ReadoutVector

# 1. capture only what you need: one gather + one pinned async copy per layer-step,
#    one RPC per generate() call
SamplingParams(max_tokens=1, extra_args={"output_residual_stream": [42],
                                         "capture_positions": {"last": 5}})   # or [0, -1], or "all"
out.activations["residual_stream"]  # [1, 5, 5120] bf16;  out.activations["positions"] -> [131, ..., 135]

# 2. projection in the worker: only float32 scalars leave the GPU
SamplingParams(max_tokens=1, extra_args={"apply_readout_vectors": [
    ReadoutVector(activations=direction.view(1, 5120), layer_indices=[42],
                  positions={"last": 5}, metric="cos")]})          # or metric="dot", bias=b (SAE features)
out.readout[0]["values"]     # [1, 5] float32 cosines;  out.readout[0]["positions"]
reward = out.readout[0]["values"].max()

# 3. early exit: layers 43..63 never run when every request in the pass is readout-only
SamplingParams(max_tokens=1, extra_args={"apply_readout_vectors": [...], "lens_early_exit": True})
```

Early exit needs `enable_prefix_caching=False` (skipped layers would leave stale KV blocks
a later request could reuse) and PP = 1; `llm.collective_rpc("lens_capabilities")[0]["early_exit"]`
tells you whether the engine allows it, and the plugin refuses (clear `ValueError`, engine stays
alive) otherwise, or when `max_tokens != 1`.  The sampled token of an early-exit request is
garbage — ignore it.  A batch that mixes generating requests with readout requests simply
runs to the end (no exit), so an RL rollout engine can score with `lora_request=None`
between generation calls without any mode switch.

![seconds per 1,024 texts for every way of reading layer 42 out of vLLM, Qwen3.6-27B](bench/readout_cost.png)

![wall time vs batch size for the main readout methods, both models](bench/readout_vs_batch.png)

<!-- READOUT_RESULTS:BEGIN -->
**Qwen/Qwen3.6-27B** — layer 42 of 64, 1,024 texts of 96–136 tokens (mean 116), 1× B200, wall time of one `generate()` call with B = 1,024 texts (prefill-only, `max_tokens=1`), min over repeats; HF = transformers bf16 on the same texts, batch 128:

| how layer L is read out | eager engine | CUDA-graph engine | vs stock |
|---|---:|---:|---:|
| stock vllm-lens 1.1.0 capture (`output_residual_stream=[L]`, all positions) | 12.4 s | — | — |
| fork, 1.1.0 capture path (per-request `.cpu()` + per-request RPC) | 12.9 s | 12.5 s | **1.0×** |
| fork gather capture, all positions, one RPC | 11.1 s | 10.3 s | **1.2×** |
| fork gather capture, `capture_positions={"last": 5}` | 7.271 s | 6.869 s | **1.8×** |
| fork `ReadoutVector` cosine, all positions | 7.242 s | 6.806 s | **1.8×** |
| fork `ReadoutVector` cosine, last 5 positions | 7.177 s | 6.751 s | **1.8×** |
| fork `ReadoutVector` last 5 **+ early exit** | 4.955 s | 4.672 s | **2.6×** |
| fork capture last 5 + early exit | 5.066 s | 4.773 s | **2.6×** |
| vLLM prefill with no hooks at all (ceiling) | 6.986 s | 6.617 s | **1.9×** |
| HF transformers bf16, forward hook + early exit after layer 42 (the trainer's `read_resid`, batch 128) | 7.001 s | — | **1.8×** |
| HF transformers bf16, all 64 layers (batch 128) | 10.6 s | — | **1.2×** |

Reading *generated* positions, B = 512 prompts × 40 new tokens: 
stock 1.1.0 eager generate + capture **13.53 s**; eager generate, no capture **8.14 s**; eager generate + capture every generated position **10.93 s**; CUDA-graph generate, no capture **6.23 s**; CUDA-graph generate + re-encode with readout **10.72 s** (6.23 + 4.49); CUDA-graph generate + re-encode with readout + early exit **9.35 s** (6.25 + 3.10).

Correctness vs the HF reference (transformers bf16, fla kernels, same token ids): 13/18 gated checks pass. Readout rewards (max cosine over the last 5 positions) agree with HF within 0.0072 on every one of the 1,024 texts, on both engines, with and without early exit. Captured rows vs HF, per text (5 rows flattened): fork_eager cap_last5: min cos 0.9894 over 1024 texts (median 0.99995, 98.6 % above 0.999, worst text 480 of length 135) [fla ref]. The fork's last-5 rows are bit-identical to its own all-positions capture, and every early-exit result is bit-identical to the non-exit one; the residual disagreement with HF is bf16 divergence in the deep layers (the same magnitude separates HF's fla kernel from its torch fallback, and vLLM's eager from its CUDA-graph engine).

**Qwen/Qwen3-1.7B** — layer 18 of 28, 1,024 texts of 96–136 tokens (mean 116), 1× B200, wall time of one `generate()` call with B = 1,024 texts (prefill-only, `max_tokens=1`), min over repeats; HF = transformers bf16 on the same texts, batch 128:

| how layer L is read out | eager engine | CUDA-graph engine | vs stock |
|---|---:|---:|---:|
| stock vllm-lens 1.1.0 capture (`output_residual_stream=[L]`, all positions) | 2.579 s | — | — |
| fork, 1.1.0 capture path (per-request `.cpu()` + per-request RPC) | 3.836 s | 3.495 s | **0.7×** |
| fork gather capture, all positions, one RPC | 2.335 s | 2.350 s | **1.1×** |
| fork gather capture, `capture_positions={"last": 5}` | 0.746 s | 0.670 s | **3.8×** |
| fork `ReadoutVector` cosine, all positions | 0.681 s | 0.664 s | **3.9×** |
| fork `ReadoutVector` cosine, last 5 positions | 0.681 s | 0.637 s | **4.0×** |
| fork `ReadoutVector` last 5 **+ early exit** | 0.544 s | 0.533 s | **4.8×** |
| fork capture last 5 + early exit | 0.558 s | 0.555 s | **4.6×** |
| vLLM prefill with no hooks at all (ceiling) | 0.615 s | 0.501 s | **5.1×** |
| HF transformers bf16, forward hook + early exit after layer 18 (the trainer's `read_resid`, batch 128) | 0.814 s | — | **3.2×** |
| HF transformers bf16, all 28 layers (batch 128) | 1.193 s | — | **2.2×** |

Reading *generated* positions, B = 512 prompts × 40 new tokens: 
stock 1.1.0 eager generate + capture **3.06 s**; eager generate, no capture **1.11 s**; eager generate + capture every generated position **2.43 s**; CUDA-graph generate, no capture **0.59 s**; CUDA-graph generate + re-encode with readout **1.03 s** (0.59 + 0.44); CUDA-graph generate + re-encode with readout + early exit **0.94 s** (0.58 + 0.36).

Correctness vs the HF reference (transformers bf16, same token ids): 16/16 gated checks pass. Readout rewards (max cosine over the last 5 positions) agree with HF within 0.0013 on every one of the 1,024 texts, on both engines, with and without early exit. Captured rows vs HF, per text (5 rows flattened): fork_eager cap_all_legacy: min cos 1.0000 over 16 texts [torch-fallback ref]; fork_eager cap_all: min cos 1.0000 over 16 texts [torch-fallback ref]; fork_eager cap_last5: min cos 0.9998 over 1024 texts [torch-fallback ref]; fork_eager exit_cap_last5: min cos 0.9998 over 1024 texts [torch-fallback ref]; fork_graphs cap_all_legacy: min cos 1.0000 over 16 texts [torch-fallback ref]; fork_graphs cap_all: min cos 1.0000 over 16 texts [torch-fallback ref]; fork_graphs cap_last5: min cos 0.9998 over 1024 texts [torch-fallback ref]; fork_graphs exit_cap_last5: min cos 0.9998 over 1024 texts [torch-fallback ref]. The fork's last-5 rows are bit-identical to its own all-positions capture, and every early-exit result is bit-identical to the non-exit one; the residual disagreement with HF is bf16 divergence in the deep layers (the same magnitude separates HF's fla kernel from its torch fallback, and vLLM's eager from its CUDA-graph engine).
<!-- READOUT_RESULTS:END -->

![reading generated positions: eager capture during decode vs generate under CUDA graphs + re-encode](bench/generated_positions.png)

**CUDA graphs and generated positions.** Hooks do not run inside replayed decode graphs,
so with `VLLM_LENS_CUDA_GRAPHS=1` capture and readout see *prompt* positions only (a
prefill-only `max_tokens=1` request is therefore fully graph-compatible, and early exit
works there too since prefill batches run eagerly).  To read *generated* positions you have
two options: run the whole engine eagerly and capture during decode (`gen_cap_all` rows
above), or generate under graphs and re-encode prompt+completion in a second, prefill-only
pass (`gen_then_read` rows) — the second is cheaper as soon as the batch is large, and it is
exactly the "clean base model, LoRA off" pass an RL reward wants anyway.

## Overview of changes
| | stock 1.1.0 | vllm-metamodel |
|---|---|---|
| resolve a request's vectors | `startswith` over all keys, every layer, every step | dict lookups, once per request, cached |
| per-step bookkeeping | 2 device syncs per request per layer | one plan per forward pass from host-side buffers |
| decode steps with nothing to do | full scan + clone of every layer output | pre-hook flags the pass idle; hooks return on one check (or the decode runs inside a CUDA graph, no Python at all) |
| applying the vectors | one Python iteration + one kernel per row | one `index_add_` for all rows of a layer (norm-match batched) |
| shipping vectors to the worker | one RPC per request | one `[n, d]` block RPC per `generate()` call |
| CUDA graphs | impossible (`enforce_eager` forced) | opt-in: `VLLM_LENS_CUDA_GRAPHS=1` → decode graphs, prompt-position steering |
| injection modes | add a vector to a decoder layer's output | add **or replace** (`mode="replace"`), at a layer output or at the embedding stream (`EMBED_LAYER_INDEX`) — the NLA-style injection, and the only well-defined one on multi-stream architectures |
| `norm_match=True` reference norm | the `hidden_states` half of a fused-residual layer output (MLP delta; ≈ 0.12× the stream norm on Qwen3.6-27B) | the **full residual stream** `hidden_states + residual` (upstream #7 port) — `scale=coeff` is exactly `h + coeff·‖h‖·v/‖v‖` |
| hidden-states lookup for the layer-0 pre-hook | — | positional **and keyword** inputs, exactly one candidate required, hard error (`EmbedInjectionError`) on a miss |
| multi-stream (hyper-connection) layer outputs | `output[0] + output[1]` broadcasts silently / steering mis-injects into the deferred fold | detected (`hc_mult`), layer-output steering & capture refused with `ValueError`; `UnsupportedLayerOutputError` backstop; embedding stream fully supported |
| vLLM ≥ 0.27 | — | pass plan from the runner's host buffers even when the attention metadata has no `query_start_loc`; pickled RPC opt-in set |
| reading hidden states out | blocking `.cpu()` per request per layer-step, one zstd RPC per request, all positions | one gather + one pinned async copy per layer-step, one RPC per `generate()`, `capture_positions` (`{"last": k}`, lists) |
| projections / rewards | move `[T, 5120]` tensors to the host, compute there | `ReadoutVector`: cosine / dot (+ bias) computed in the worker, float32 scalars returned (`output.readout`) |
| layers past the read layer | always computed | `lens_early_exit`: forward pass stops after the deepest requested layer when every request in the pass is readout-only |

### What changed (small, upstreamable diff)
Two library files carry the change against upstream `v1.1.0` (`_worker_ext.py`,
`_activations_plugin.py`), plus the `SteeringVector.mode` field / `EMBED_LAYER_INDEX`
constant and the `ReadoutVector` model / position-spec helpers in `_helpers/types.py`
(and their exports; `_serialize.py` passes non-tensor entries such as `positions` through). Upstream's functions stay in place
(`_get_layers`, `_find_steering_configs`, `norm_match`, the capture / state methods);
`_apply_steering` gains the `residual` argument of upstream #7 and the `mode="replace"`
branch, and the body of `_hook_inner` is replaced. `git diff v1.1.0 --stat -- pyproject.toml vllm_lens ':!vllm_lens/tests'`:

<!-- DIFFSTAT:BEGIN -->
```
 pyproject.toml                   |    7 +-
 vllm_lens/__init__.py            |   12 +-
 vllm_lens/_activations_plugin.py |  520 ++++++++++-
 vllm_lens/_helpers/_serialize.py |   10 +-
 vllm_lens/_helpers/types.py      |  153 +++-
 vllm_lens/_worker_ext.py         | 1878 +++++++++++++++++++++++++++++++++++---
 6 files changed, 2414 insertions(+), 166 deletions(-)
```
<!-- DIFFSTAT:END -->

`vllm_lens/_worker_ext.py`
- `_SteerEntry` / `_index_configs` / `_prefix_keys` / `_resolve_entries` — per-key summary built at `set_steering_data` time; a request's keys are found by dict lookups on its `_steering_id` and on the `-`-boundary prefixes of its internal id (exactly the set the old scan matched: vLLM ids are `"{external}-{8 hex}"`; multi-key matches keep insertion order).
- `_ReqPlan` (per-request, cached, invalidated on any set/clear) and `_StepPlan` (one per forward pass, built by the first layer hook from `runner.query_start_loc.np` / `input_batch.num_computed_tokens_cpu`, no device syncs). A row is scheduled for a layer only when one of its vectors can touch a position computed in this pass — chunked prefill steers the marker in exactly the chunk that computes it.
- `_make_pre_hook` / `_step_is_idle` — pre-hook on the first decoder layer; idle passes (uniform decode, no broadcast vectors, all positional vectors behind every row, no capture in flight) cost one flag check per layer.
- `_apply_layer_vectorized` — stack all (row, vector) pairs of a layer/pass, one `index_add_` (replace rows: one `index_copy_` + `index_fill_` on the fused residual half); falls back to upstream's `_apply_steering` when a row would receive several vectors or a broadcast vector covers a multi-token chunk. Bit-identical for `norm_match=False`.
- `_apply_steering` / `_apply_layer_vectorized` take the fused `residual` half: `norm_match` references the full stream `target + residual` (upstream #7); `mode="replace"` rewrites both halves. `_hook_inner` clones the residual only on layer-steps that contain a replace row.
- `_make_pre_hook(ext, layer_idx)` (`with_kwargs=True`) / `_find_hidden_states_arg` / `_apply_embed` — the layer-0 pre-hook now also builds the pass plan, applies `EMBED_LAYER_INDEX` vectors to the hidden states entering layer 0 (one `index_copy_` / `index_add_`), captures the embedding stream when layer -1 is requested, and raises `EmbedInjectionError` if the input tensor cannot be identified unambiguously.
- New RPCs: `set_steering_block` (per-entry `modes`, `EMBED_LAYER_INDEX` allowed), `set_steering_data_many`, `clear_steering_data_many`, `set_vectorized`, `steering_stats` (`rows_replaced`, `embed_apply_steps`, `embed_errors`). Existing `set_steering_data` / `clear_steering_data` also maintain the index.

`vllm_lens/_activations_plugin.py`
- `_patched_create_engine_config` forces `enforce_eager` only unless `VLLM_LENS_CUDA_GRAPHS=1` (`_configure_cuda_graphs` then sets compilation mode `NONE` + `cudagraph_mode=FULL_DECODE_ONLY` unless you passed a compatible config).
- Offline `LLM.generate`: one `set_steering_block` RPC for the call's single-position vectors (+ one `set_steering_data_many` for anything else) instead of one RPC per request; clears in a `finally`.
- `VLLM_LENS_DISABLE=1` no-op switch (as in upstream 1.2.0); `_check_graph_mode_request` fails fast on 2-D vectors under CUDA graphs.
- (post4) `apply_readout_vectors` popped per request → one `set_readout_block` RPC (+ `set_readout_data_many` for multi-layer / multi-vector requests); results fetched for the whole call in one `get_readouts_many` RPC and attached as `output.readout`; captured states fetched in one `get_captured_states_many` RPC (`VLLM_LENS_FAST_CAPTURE=0` restores per-request retrieval); `_check_readout_request` rejects early-exit requests the engine cannot honour (`lens_capabilities()["early_exit"]`, `max_tokens != 1`, `output_residual_stream=True`) before submission.

`vllm_lens/_worker_ext.py` (post4)
- `_select_positions` / `_capture_gather` / `_HostBlock` / `_flush_host_blocks` — per layer-step one `index_select` over every capturing row's requested positions (`capture_positions`: all / last-k / explicit), one pinned asynchronous device→host copy with a CUDA event, split per request lazily at retrieval; `_pop_activations` adds `"positions"`.
- `_ReadEntry` / `_readout_layer` — per-request directions live in one float32 `[n, hidden]` device block; the hook gathers the selected rows, computes cosine / dot (+ bias) in float32 in chunks of 8,192 rows and ships only the scalars.
- `_EarlyExit` / `_wrap_model_forward` / `_early_exit_supported` — `_StepPlan.exit_layer` is set when every request in the pass is an eligible readout-only request; the hook at that layer raises, the wrapped `model_runner._model_forward` returns a zero placeholder. Refused unless PP = 1, `enable_prefix_caching=False`, no aux hidden-state outputs.

### CUDA graphs

```python
import os
os.environ["VLLM_LENS_CUDA_GRAPHS"] = "1"          # before the engine is built
llm = LLM(model, compilation_config={"cudagraph_capture_sizes": [8, 64, 512, 1024]})  # optional
```

With `cudagraph_mode=FULL_DECODE_ONLY` (and no `torch.compile`, so the hooks still fire),
uniform-decode batches replay CUDA graphs — no Python runs, decode is exactly as fast as
without the plugin — while every batch that contains prompt tokens runs eagerly with the
hooks live. Steering and capture are therefore **prompt-position only** in this mode: the
worker never touches generated positions (even on mixed batches that happen to run
eagerly, so results do not depend on batch composition), 2-D broadcast vectors are refused
with a `ValueError`, and `output_residual_stream` returns prompt positions only (warned
once). If chunked prefill could leave a 1-token final chunk (dispatched as a decode graph)
the fork warns at hook installation — set `max_num_batched_tokens` above your longest
prompt × concurrency. Without the variable, behaviour is exactly 1.1.0's (eager forced).

**known vLLM bug for hybrid GatedDeltaNet models such as Qwen3.5/3.6, vLLM 0.19:** keep `max_num_seqs <= 1024` when CUDA graphs are on. vLLM's packed GDN decode kernel launches a `batch x value_heads` grid (48 heads on Qwen3.6-27B) and the CUDA grid-dimension limit is 65,535; with `max_num_seqs=2048` the engine died at start-up in the graph warm-up (`Triton Error [CUDA]: invalid argument`) with or without vllm-lens, while `max_num_seqs=1024` with vLLM's default capture ladder (`compilation_config={"max_cudagraph_capture_size": 1024}`) works (`bench/diag_engine.py`; LoRA on/off and the packed kernel on/off make no difference). Batches larger than `max_num_seqs` simply run as several scheduler waves.

