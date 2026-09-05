# PR draft: indexed hook dispatch, idle fast path, opt-in CUDA graphs

Branch `upstream-pr` (on `ceselder/vllm-metamodels`, based on `UKGovernmentBEIS/vllm-lens`
`main` @ 7d252f7). **Prepared, not opened.** Two library files change
(`vllm_lens/_worker_ext.py`, `vllm_lens/_activations_plugin.py`) plus one CPU test file;
the public API is unchanged and every existing RPC / `extra_args` key keeps its meaning.

## Why

Per-request steering (one `SteeringVector` per prompt in a batch of hundreds -- activation
oracles, meta-model RL rollouts) is dominated by hook overhead in 1.2.x:

* `_hook_inner` re-resolves every request's steering vectors AND hooks on **every layer of
  every forward pass** with a `startswith` scan over all registered keys
  (`_find_steering_configs`, `_find_hook_configs_no_persistent`): O(layers x requests x keys)
  Python per decode step;
* it reads `query_start_loc[i].item()` twice per request per layer: O(layers x requests)
  device syncs per step;
* the plugin forces `enforce_eager`, so decode never uses CUDA graphs.

Measured on 1x B200, Qwen3.6-27B bf16, 96-token prompt, 40 new tokens, one distinct vector
per request at layer 1 (`bench/bench_steering.py` in the fork), wall time of one
`generate()` call:

| requests per call | vllm-lens 1.1.0 (eager, forced) | this branch's algorithm, eager | + CUDA graphs (`VLLM_LENS_CUDA_GRAPHS=1`) | vLLM, no steering, same engine config |
|---:|---:|---:|---:|---:|
| 8 | 2.0 s | 0.7 s | **0.6 s (3.3x)** | 0.6 s |
| 32 | 3.6 s | 1.0 s | **0.8 s (4.5x)** | 0.8 s |
| 128 | 12.3 s | 2.2 s | **1.6 s (7.9x)** | 1.6 s |
| 512 | 76.3 s | 5.8 s | **5.5 s (13.9x)** | 5.5 s |
| 1,024 | 230 s | 12.0 s | **10.5 s (21.9x)** | 10.5 s |
| 2,048 | 778 s | 21.1 s | **20.5 s (37.8x)** | 20.4 s |

Qwen3-1.7B at 1,024 requests: 96.8 s -> 1.6 s (59x). vllm-lens 1.2.1 on vLLM 0.27.1 measured
100-150x slower than the indexed path at B = 512 (same scan, eager forced). Numbers, plots and
the 110-check exactness suite (delta cos 1.0000, magnitude ratio 1.0000, unsteered rows
bit-identical, greedy continuations equal) are in the fork's README / `bench/`.

## What changes

1. **Indexed resolution** (`_prefix_keys`, `_lookup_keyed`). vLLM's internal request id is
   `"{external_id}-{8 hex}"`, so the `startswith(f"{key}-")` scan matched exactly the keys
   that end before a `-` in the internal id. Enumerating those prefixes is a handful of dict
   lookups, independent of the number of registered keys; insertion order is preserved
   (`_steering_seq` / `_hook_seq`), the `_steering_id` / `_hook_id` sentinel is appended last
   as before. `_find_steering_configs` / `_find_hook_configs_no_persistent` keep their
   signatures and results (tested against the old scan).
2. **Per-request plan, cached** (`_ReqPlan`, `_resolve_request`): steering configs, per-request
   hooks, parsed `output_residual_stream`, prompt length, and a summary (layers touched,
   min/max position, broadcast) resolved once per request; invalidated by a generation
   counter bumped on every `set_/clear_*` (steering, hooks, persistent hooks).
3. **One plan per forward pass** (`_StepPlan`, `_get_step_plan`): `query_start_loc` and the
   rows' absolute start positions come from the model runner's HOST buffers
   (`runner.query_start_loc.np`, `input_batch.num_computed_tokens_cpu`; one `tolist()` as the
   fallback) -- no `.item()` per request per layer. Shared by all layer hooks of the pass
   (keyed on the forward-context object). Also makes the hooks independent of which
   attention-metadata entries carry `query_start_loc` (they moved off several backends in
   vLLM >= 0.27).
4. **Idle fast path** (`_begin_pass`, called from the first decoder layer's pre-hook): when no
   row has hooks or capture, there are no persistent hooks, and every steering vector lies
   behind every row's current position (the typical decode step of prompt-position steering),
   every hook of the pass returns on one flag check. Per-layer work is also skipped when no
   hook targets that layer (`Hook.has_layer`) and when no vector touches it.
5. **Opt-in CUDA graphs** (`VLLM_LENS_CUDA_GRAPHS=1`): the plugin keeps
   `compilation_config.mode = NONE` (hooks must stay eager) but sets
   `cudagraph_mode = FULL_DECODE_ONLY` instead of `enforce_eager`. Uniform-decode batches
   replay graphs (no Python), every batch containing prompt tokens runs eagerly with the hooks
   live. Semantics in this mode: steering / hooks / capture are **prompt-position only** -- the
   worker never touches generated positions even on mixed batches (results do not depend on
   batch composition), 2-D broadcast vectors are refused (`ValueError` client-side and in
   `set_steering_data`), `output_residual_stream` returns prompt positions (warned once). Hooks
   also return immediately during graph capture (`is_current_stream_capturing`). Without the
   variable the behaviour is exactly 1.2.x's (eager forced).
6. **V1 model runner pin**: already in 1.2.1 (#12); unchanged.

Not included (kept for a follow-up so this diff stays reviewable): the vectorised
`index_add_` apply across rows of a layer-step, the block RPC that ships all of a
`generate()`'s vectors in one tensor, gather-capture / `capture_positions`, in-engine readout,
early exit, prefix-cache salting, torch.compile-compatible custom-op hooks.

## Tests

* `tests/test_indexed_dispatch.py` (CPU, no engine): indexed resolution == legacy scan on
  prefix / sentinel / duplicate cases; plan caching + generation invalidation; host-buffer plan;
  idle detection (decode past the markers = idle; broadcast vector / capture / persistent hook
  = live); steering applied once at the marker with generated rows untouched; 2-D refusal under
  graphs; capture and per-request hooks follow the plan. Run:
  `pytest tests/test_indexed_dispatch.py --noconftest -p no:cacheprovider`
  (passes with and without a real vLLM install).
* `tests/test_server.py`, `tests/test_client.py`, `tests/test_persistent_hooks.py`: the existing
  server integration tests need a GPU + a running `vllm serve`; **not run yet on this branch**
  (TODO before opening: `VLLM_TEST_MODEL=... pytest tests/`, once eager and once with
  `VLLM_LENS_CUDA_GRAPHS=1` where 2-D vector tests must expect the `ValueError`).
* Exactness on GPU: the fork's `bench/compare.py` suite (110 checks) ran on the same algorithm
  in its 1.1.0-based form; re-run against this branch before opening.

## Caveats for reviewers

* `_begin_pass` relies on the first decoder layer's pre-hook running before any post-hook of
  the pass; PP ranks use their own first layer (`_first_layer_idx`).
* The idle decision treats any per-request hook or capture request as "live" for the whole
  pass (conservative); persistent hooks make every pass live.
* Chunked prefill under `FULL_DECODE_ONLY`: a 1-token final prompt chunk is dispatched as a
  decode graph where hooks do not run; keep `max_num_batched_tokens` above the longest prompt
  x concurrency (the fork warns at hook installation; not ported here).
* Hybrid GatedDeltaNet models on vLLM 0.19 with CUDA graphs: keep `max_num_seqs <= 1024`
  (vLLM's packed GDN decode kernel grid limit; vanilla vLLM fails the same way).
