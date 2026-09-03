# Changelog

This is **vllm-lens-metamodel**, a maintained fork of
[UKGovernmentBEIS/vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens)
(MIT, UK AI Security Institute; original author Alan Cooney).  The fork
branches from upstream **v1.1.0** and keeps the distribution name
`vllm-lens`, so it installs as a drop-in replacement:

```bash
pip install git+https://github.com/ceselder/vllm-lens-metamodel
```

## v1.1.0.post1 (3 September 2026) — vllm-lens-metamodel

Target workload: **RL-style rollouts with one steering vector per prompt**
(activation-oracle / "meta-model" injection: a different vector for every
request in the batch, applied at one layer on that request's marker token,
batches of hundreds to thousands).  Only `vllm_lens/_worker_ext.py` and
`vllm_lens/_activations_plugin.py` change; the public API
(`SamplingParams.extra_args["apply_steering_vectors" | "output_residual_stream"]`,
`SteeringVector`, the Inspect provider) is unchanged.

### Performance — the steering hook

Upstream's forward hook resolved a request's steering vectors with
`_find_steering_configs`: for every decoder layer, for every request in the
batch, a `str.startswith` scan over every registered steering key, plus two
`Tensor.item()` device syncs per request per layer, then one small GPU kernel
per steered row.  With per-request steering (one key per request) that is
O(layers × requests × keys) Python and O(layers × requests) GPU syncs on
*every* decode step — ~1.5 s per step for 64 layers × 1024 requests.

`_worker_ext.py` keeps the exact 1.1.0 matching semantics and the unchanged
`norm_match` / `_apply_steering` arithmetic, and adds:

- **Indexed resolution** (`_SteerEntry`, `_resolve_entries`): each key is
  summarised at `set_steering_data` time (layers, broadcast?, position
  range); a request's keys are found with dict lookups on its `_steering_id`
  and on the `"-"`-boundary prefixes of its internal id — exactly the set the
  `startswith` scan matched (vLLM internal ids are `"{external}-{8 hex}"`).
  Multi-key matches keep insertion order.
- **Per-request cache** (`_ReqPlan`): resolution happens once per request,
  invalidated whenever steering data changes.
- **One plan per forward pass** (`_StepPlan`), built lazily by the first
  layer hook and shared by all layers, from the model runner's host-side
  `query_start_loc` / `num_computed_tokens` buffers (no device syncs).  A row
  is scheduled for a layer only when one of its vectors can touch a position
  being computed in this pass, so chunked prefill steers the marker row in
  exactly the chunk that computes it, and a pure-decode pass for
  prompt-position steering does no GPU work at all (no clone, no add).
- **Idle fast path** (`_make_pre_hook`, `_step_is_idle`): a pre-hook on the
  first decoder layer classifies the pass in O(1); on an idle pass (uniform
  decode, no broadcast vectors, all positional vectors behind every row, no
  capture in flight) every layer hook returns on a single flag check.
- **Vectorised apply** (`_apply_layer_vectorized`): all (row, vector) pairs
  of a layer/pass are stacked into one `[n, hidden]` tensor and added with a
  single `index_add_`, norm-matching in the same batched op.  Falls back to
  the sequential `_apply_steering` when a row would receive several vectors
  or a broadcast vector covers a multi-token chunk, so semantics never
  change.  Bit-identical to the loop for `norm_match=False`; float32
  reduction-order differences possible for `norm_match=True`.
  Toggle: `VLLM_LENS_VECTORIZED=0` or the `set_vectorized` RPC.
- **Batched RPCs**: `set_steering_block` ships a whole call's
  single-position vectors as ONE `[n, hidden]` tensor (one host-to-device
  copy; entries are views into it); `set_steering_data_many` /
  `clear_steering_data_many` handle the rest.  The offline `LLM.generate`
  path uses them (1.1.0 did one RPC per request, ~2–5 ms each) and clears in
  a `finally`.  Toggle packing with `VLLM_LENS_BLOCK_RPC=0`.
- `steering_stats()` RPC exposes hook counters (idle passes, planned passes,
  vectorised layer-steps, rows steered, rows skipped as generated, errors).

### Feature — `enforce_eager` is opt-in instead of forced (CUDA graphs)

Upstream's `EngineArgs.create_engine_config` patch unconditionally set
`enforce_eager=True`, so vLLM never captured CUDA graphs.  The fork keeps
that as the **default** and adds an explicit opt-in:

- `VLLM_LENS_CUDA_GRAPHS=1` — the plugin leaves `enforce_eager` alone and
  sets `compilation_config` to mode `NONE` (no `torch.compile`, so hooks
  still fire) with `cudagraph_mode=FULL_DECODE_ONLY`, unless you passed a
  compatible one.  Other torch.compile modes fall back to eager with a
  warning; `cudagraph_mode` values that would graph prefill batches are
  overridden to `FULL_DECODE_ONLY`; an explicit `enforce_eager=True` wins.
- Semantics in this mode are **prompt-position only**: uniform-decode
  batches are graph replays in which Python hooks do not run (decode is
  completely hook-free), while every batch that contains prompt tokens runs
  eagerly with the hooks live.  The worker never touches generated positions
  (even on mixed batches that happen to run eagerly), so results do not
  depend on batch composition.  2-D (broadcast) steering vectors are
  rejected with a clear `ValueError`; capture warns once that only prompt
  positions are returned.  Chunked prefill that could leave a 1-token final
  chunk (dispatched as a decode graph) is warned about at hook installation
  — raise `max_num_batched_tokens` above your longest prompt.
- `VLLM_LENS_DISABLE=1` makes the plugin a no-op (same switch as upstream
  1.2.0), handy for plain-vLLM baselines.
- Note for hybrid GatedDeltaNet models (Qwen3.5/3.6) on vLLM 0.19: keep
  `max_num_seqs <= 1024` with CUDA graphs on. vLLM's packed GDN decode kernel
  uses a `batch x value_heads` launch grid (48 heads on Qwen3.6-27B; CUDA
  limit 65,535), so `max_num_seqs=2048` fails at engine start-up in the graph
  warm-up (`Triton Error [CUDA]: invalid argument`) -- a vLLM issue, reproducible
  with the plugin disabled (`bench/diag_engine.py`). Use vLLM's default capture
  ladder (`max_cudagraph_capture_size`).

### Measured

See the README section "Why this fork" for the benchmark table and plot
(`bench/bench_steering.py`, run on Modal with `bench/modal_bench.py`).
Steering output is numerically identical to 1.1.0 for every variant:
the same captured hidden states and next-token logprobs, injected delta
cos = 1.000 and magnitude ratio = 1.000.

### Relation to upstream 1.2.x

Upstream has since released 1.2.0 / 1.2.1 (generic hooks, `LLM.chat`
support, examples).  Those releases still contain the per-layer
`startswith` scan and still force `enforce_eager`.  Two upstream changes are
**not** in this fork on purpose, to stay a drop-in for 1.1.0 users:

- upstream #7 changed `norm_match=True` on fused-residual models to scale by
  the full residual stream instead of the MLP-delta component (a behaviour
  change that makes existing `norm_match` steering stronger).  This fork
  keeps the 1.1.0 behaviour.
- the generic `apply_hooks` / `register_hooks` system.

## v1.1.0 (upstream)

- Prior releases — see
  [upstream](https://github.com/UKGovernmentBEIS/vllm-lens/blob/main/CHANGELOG.md).
