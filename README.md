# vllm-lens-port

**A [vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens) fork that is much faster for RL-style workloads: one steering vector per prompt over large batches — indexed hook, prefill-only vectorised injection, CUDA graphs.** Drop-in replacement for vllm-lens 1.1.0 (same package name, same public API).

```bash
pip install git+https://github.com/ceselder/vllm-lens-port
```

<!-- RESULTS:BEGIN -->
_Benchmark numbers are filled in from `bench/results/` — see below._
<!-- RESULTS:END -->

## Why this fork: N× faster per-request steering

The workload this fork targets is the activation-oracle / "meta-model" rollout used in RL:
**every request in a batch carries its own steering vector**, applied at one layer on that
request's marker token, with batches of hundreds to thousands of requests and a few dozen
generated tokens each.

Stock vllm-lens 1.1.0 resolves a request's steering vectors inside its forward hook with
`_find_steering_configs`: **for every decoder layer, for every request in the batch, a
`str.startswith` scan over every registered steering key**, plus two `Tensor.item()`
device syncs per request per layer and one small GPU kernel per steered row. With one key
per request that is O(layers × requests × keys) Python and O(layers × requests) GPU syncs
on *every decode step* — about 1.5 s per step for 64 layers × 1024 requests — which is why
batching looked sub-linear. The plugin also unconditionally forced `enforce_eager=True`, so
vLLM never captured CUDA graphs.

The fork keeps the exact 1.1.0 semantics (same matching rules, unchanged `norm_match` and
`_apply_steering` arithmetic, identical outputs) and changes how the hook gets there:

| | stock 1.1.0 | vllm-lens-port |
|---|---|---|
| resolve a request's vectors | `startswith` over all keys, every layer, every step | dict lookups, once per request, cached |
| per-step bookkeeping | 2 device syncs per request per layer | one plan per forward pass from host-side buffers |
| decode steps with nothing to do | full scan + clone of every layer output | pre-hook flags the pass idle; hooks return on one check (or the decode runs inside a CUDA graph, no Python at all) |
| applying the vectors | one Python iteration + one kernel per row | one `index_add_` for all rows of a layer (norm-match batched) |
| shipping vectors to the worker | one RPC per request | one `[n, d]` block RPC per `generate()` call |
| CUDA graphs | impossible (`enforce_eager` forced) | opt-in: `VLLM_LENS_CUDA_GRAPHS=1` → decode graphs, prompt-position steering |

### What changed (small, upstreamable diff)

Only two library files change against upstream `v1.1.0`; every upstream function stays
verbatim (`_get_layers`, `_find_steering_configs`, `norm_match`, `_apply_steering`, the
capture / state methods) and only the body of `_hook_inner` is replaced. `git diff v1.1.0 --stat`:

<!-- DIFFSTAT:BEGIN -->
```
 pyproject.toml                   |   7 +-
 vllm_lens/_activations_plugin.py | 226 +++++++++++-
 vllm_lens/_worker_ext.py         | 779 ++++++++++++++++++++++++++++++++++-----
 3 files changed, 901 insertions(+), 111 deletions(-)
```
<!-- DIFFSTAT:END -->

`vllm_lens/_worker_ext.py`
- `_SteerEntry` / `_index_configs` / `_prefix_keys` / `_resolve_entries` — per-key summary built at `set_steering_data` time; a request's keys are found by dict lookups on its `_steering_id` and on the `-`-boundary prefixes of its internal id (exactly the set the old scan matched: vLLM ids are `"{external}-{8 hex}"`; multi-key matches keep insertion order).
- `_ReqPlan` (per-request, cached, invalidated on any set/clear) and `_StepPlan` (one per forward pass, built by the first layer hook from `runner.query_start_loc.np` / `input_batch.num_computed_tokens_cpu`, no device syncs). A row is scheduled for a layer only when one of its vectors can touch a position computed in this pass — chunked prefill steers the marker in exactly the chunk that computes it.
- `_make_pre_hook` / `_step_is_idle` — pre-hook on the first decoder layer; idle passes (uniform decode, no broadcast vectors, all positional vectors behind every row, no capture in flight) cost one flag check per layer.
- `_apply_layer_vectorized` — stack all (row, vector) pairs of a layer/pass, one `index_add_`; falls back to upstream's `_apply_steering` when a row would receive several vectors or a broadcast vector covers a multi-token chunk. Bit-identical for `norm_match=False`.
- New RPCs: `set_steering_block`, `set_steering_data_many`, `clear_steering_data_many`, `set_vectorized`, `steering_stats`. Existing `set_steering_data` / `clear_steering_data` also maintain the index.

`vllm_lens/_activations_plugin.py`
- `_patched_create_engine_config` forces `enforce_eager` only unless `VLLM_LENS_CUDA_GRAPHS=1` (`_configure_cuda_graphs` then sets compilation mode `NONE` + `cudagraph_mode=FULL_DECODE_ONLY` unless you passed a compatible config).
- Offline `LLM.generate`: one `set_steering_block` RPC for the call's single-position vectors (+ one `set_steering_data_many` for anything else) instead of one RPC per request; clears in a `finally`.
- `VLLM_LENS_DISABLE=1` no-op switch (as in upstream 1.2.0); `_check_graph_mode_request` fails fast on 2-D vectors under CUDA graphs.

### CUDA graphs: what you get and what you give up

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

Environment variables: `VLLM_LENS_CUDA_GRAPHS` (opt-in graphs), `VLLM_LENS_VECTORIZED=0`
(sequential apply), `VLLM_LENS_BLOCK_RPC=0` (per-key RPC), `VLLM_LENS_DISABLE=1` (plugin off).

### Benchmark

`bench/bench_steering.py` measures generation throughput vs batch size for stock 1.1.0, the
fork (indexed hook), the fork with the vectorised apply, the fork with CUDA graphs, and
no-steering ceilings, with a fixed correctness probe in every configuration (injected
delta vs vector: cos and magnitude ratio; steered hidden state and next-token logprobs vs
stock). `bench/modal_bench.py` runs it on one B200 on [Modal](https://modal.com):

```bash
MODAL_PROFILE=<your workspace> modal run bench/modal_bench.py --small-model Qwen/Qwen3-1.7B
python bench/compare.py bench/results/<timestamp>     # speedup table + correctness assertions
python bench/plot_bench.py bench/results/<timestamp>  # PNG + PDF + data JSON
```

CPU-only unit tests for the indexed path: `pytest vllm_lens/tests/test_steering_index.py --noconftest`.

### Relation to upstream

Branch `main` = upstream `v1.1.0` + the fork commits (this is a GitHub fork, MIT license
preserved; original author Alan Cooney, UK AI Security Institute). Upstream 1.2.x adds
generic hooks and `LLM.chat` support but still has the per-layer key scan and forced eager
mode; two upstream behaviour changes are deliberately not included (see `CHANGELOG.md`).

---

## Upstream README (vllm-lens v1.1.0)


vLLM-Lens enables top-down interpretability (e.g., probes, steering, activation oracles). It offers high performance, supporting tensor parallelism & pipeline parallelism (across GPUs and nodes) out of the box. You can also apply all these techniques concurrently (in the same dynamic batch) - removing the need to switch between model instances.

Note this performance comes at the expense of flexibility - for example, you would need to edit the source to add additional custom hooks (though it should be easy enough for coding agents to do that). For more flexibility out of the box, consider [nnsight](https://nnsight.net/) or [TransformerLens](https://transformerlensorg.github.io/TransformerLens/).

The module auto-registers as a [vLLM general plugin](https://docs.vllm.ai/en/latest/design/plugin_system.html) and an [Inspect](https://inspect.aisi.org.uk/) model provider on install. Interact with model internals per-call via `SamplingParams.extra_args` (vLLM) or `GenerateConfig.extra_body` (Inspect).

### Install

```bash
uv add vllm-lens
```

### Examples

These examples use the Inspect integration. See the [`examples/`](examples/) folder for offline and online direct vLLM usage.

#### Inspect AI provider

An [Inspect AI](https://inspect.ai-safety-institute.org.uk/) model provider is auto-registered as `vllm-lens`, when you install this package. This model provider extends the built-in vLLM provider to serialize `torch.Tensor` steering vectors for HTTP transport and decode base64-encoded activations from responses into `ModelOutput.metadata["activations"]`. It also supports LoRA adaptors.

Usage is the same as the [default vLLM provider](https://inspect.aisi.org.uk/providers.html#vllm) but with the `vllm-lens` prefix (e.g. `vllm-lens/meta-llama/Llama-3.1-1B`).

##### Extracting activations

```python
capture_config = GenerateConfig(
    temperature=0.0,
    max_tokens=1,
    extra_body={
        "extra_args": {"output_residual_stream": extraction_activation_layers},
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
output = await model.generate(state.messages, config=capture_config)
residual_stream = output.metadata["activations"]["residual_stream"]
```

##### Steering with an Activation Oracle

```python
from vllm_lens import SteeringVector

messages = [ChatMessageUser(content=oracle_content)]
oracle_config = GenerateConfig(
    temperature=0.0,
    max_tokens=50,
    extra_body={
        "extra_args": {
            "apply_steering_vectors": [
                SteeringVector(
                    activations=act_vector,
                    layer_indices=[injection_layer],
                    scale=steering_coefficient,
                    norm_match=True,
                    position_indices=[special_pos],
                )
            ],
        },
        "lora_request": {
            "lora_name": "oracle",
            "lora_int_id": 1,
            "lora_path": lora_path,
        },
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
response = await model.generate(messages, config=oracle_config)
```

### Theory

vllm-lens registers as a [vLLM plugin](https://docs.vllm.ai/en/stable/design/plugin_system) and injects itself into vLLM's processing pipeline at broadly 3 stages:

1. **Intercepting generate calls.** To utilise the plugin, you can pass [extra args](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams.extra_args) such as `output_residual_stream` or `apply_steering_vectors` in the sampling parameters. The plugin extracts these, initialises relevant [PyTorch hooks](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html) if they're not already setup (by adding a [worker extension](https://docs.vllm.ai/en/stable/cli/run-batch/?h=worker+extension#-worker-extension-cls)) and sends steering vectors directly to workers (vLLM typically has one worker per GPU).
2. **Per-sample hook operations**. vLLM dynamically batches tokens from multiple concurrent requests into a single forward pass, so a core challenge is "book-keeping" - working out which operations (e.g., activation extraction) should be applied to which parts of the request. To do this we read the `forward_context` metadata, utilising the `query_start_loc` (a tensor of token boundaries per request) and `req_ids` (mapping batch index to request ID). We then, for example, apply hooks to just the slices that correspond to the request. Any extracted activations are moved to CPU ram and compressed (lossless), ready to be requested by the vLLM scheduler process. Steering runs on all tensor-parallel ranks (since it modifies the forward pass), but capture only runs on TP rank 0 (residual streams are identical across TP replicas after all-reduce).
3. **Response collation.** The plugin intercepts the response before it is sent to the client, at which point it queries the relevant vLLM processes for any requested activations. If trims surplus activations, as vLLM does under the hook with tokens (the scheduler often gets ahead of the number of tokens it needs to generate, before stopping). Activations are then returned to the client.

### Credits

Developed by Alan Cooney, with credit going to Sid Black for the original vLLM worker extension idea.
