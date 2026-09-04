# Changelog

This is **vllm-metamodels**, a maintained fork of
[UKGovernmentBEIS/vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens)
(MIT, UK AI Security Institute; original author Alan Cooney).  The fork
branches from upstream **v1.1.0** and keeps the distribution name
`vllm-lens`, so it installs as a drop-in replacement:

```bash
pip install git+https://github.com/ceselder/vllm-metamodels
```

## v1.1.0.post5 (4 September 2026) — docs + one-call scoring helpers

- `vllm_lens.metamodel`: `readout_scores(llm, token_ids, directions, layer, positions, metric, bias, early_exit, lora_request)`
  → `(values [n, n_layers, n_pos], positions)`, `readout_max(...)` → one reward per text, `capabilities(llm)`.
  One prefill-only `generate()` per call; early exit is used when the engine supports it, otherwise
  dropped with a warning. Exported from `vllm_lens`. 4 CPU tests (`test_metamodel_helpers.py`).
- README: a **Features** table up top, a dedicated **Early exit** section (rules, capability check,
  mixed batches, numbers), **One-call scoring**, and an **API reference** for every `extra_args` key,
  `ReadoutVector`, and the environment switches.

## v1.1.0.post4 (4 September 2026) — fast hidden-state readout: gather-capture, in-engine projection, early exit

Reading the layer-L residual stream out of vLLM (the "re-encode N texts through the
clean base model and score layer 42" pass of an RL reward / eval loop) is now a
first-class, cheap operation.  Measured on 1× B200 (Qwen3.6-27B bf16, 1,024 texts of
96–136 tokens, layer 42 of 64; numbers in the README section "Fast hidden-state readout"
and `bench/results/readout_*`).

- **Gather capture** (`_capture_gather`, default; `VLLM_LENS_FAST_CAPTURE=0` restores the
  1.1.0 path).  Per layer-step the hook now does ONE `index_select` over every capturing
  row's requested positions and ONE asynchronous pinned device→host copy (a CUDA event
  is waited on at retrieval), instead of a blocking `.cpu()` slice per request.  On
  fused-residual layers `hidden_states + residual` is formed on the selected rows only.
  The offline `LLM.generate` retrieves every request of the call in ONE
  `get_captured_states_many` RPC (uncompressed pickle — activations do not compress)
  instead of one zstd-compressed `get_captured_states` RPC per request.
- **Position specs**: `extra_args["capture_positions"] = "all" | {"last": k} | [positions]`
  (`CAPTURE_POSITIONS_KEY`).  `{"last": k}` returns the last `k` prompt positions (plus
  every generated position when running eagerly); explicit lists are absolute, negative
  values count back from the end of the prompt.  `output.activations` gains a
  `"positions"` list; `_trim_activations` / `serialize_activations` / PP merge understand it.
- **`ReadoutVector`** (`extra_args["apply_readout_vectors"]`): an in-engine projection.
  At each requested layer / position the worker computes `metric(h, v) + bias`
  (`metric="cos"|"dot"`, float32, chunked so temporaries stay ≤ 168 MB) with a
  per-request direction and returns only the scalars — `output.readout` is a list (one
  entry per vector) of `{"values": Tensor[n_layers, n_pos], "positions": [...], "layers":
  [...]}`.  Directions travel in one `[n, hidden]` block RPC (`set_readout_block`; general
  multi-layer / multi-vector requests via `set_readout_data_many`), results come back in
  one `get_readouts_many` RPC.  The async `AsyncLLM.generate` path uses
  `set_readout_data` / `get_readouts` per request; `vllm serve` responses carry `readout`
  (values base64-serialised).  `bias` with `metric="dot"` gives SAE-feature
  pre-activations (`bias = b_enc[f] - b_dec·w_f`).
- **Early exit** (`extra_args["lens_early_exit"] = True`, `EARLY_EXIT_KEY`): for a
  `max_tokens=1` capture / readout request.  When EVERY request in a forward pass is such
  a request, the hook of the deepest requested layer raises `_EarlyExit` and the wrapped
  `model_runner._model_forward` returns a zero `[tokens, hidden]` placeholder — the
  remaining layers never run (layer 42 of 64 skips ~34 % of the prefill FLOPs).  The
  sampled token of an early-exit request is meaningless.  Guarded: PP == 1, no aux
  hidden-state outputs, generative model, and **`enable_prefix_caching=False`** (skipped
  layers would leave stale KV blocks a later request could reuse); the engine reports
  `lens_capabilities()["early_exit"]` / `["early_exit_reason"]` and the plugin rejects
  early-exit requests client-side (clear `ValueError`, engine stays alive) when
  unsupported, when `max_tokens != 1`, or with `output_residual_stream=True`.  Mixed
  batches (a generating request in the same pass) simply run to the end.
- New RPCs: `set_readout_block`, `set_readout_data`, `set_readout_data_many`,
  `clear_readout_data`, `clear_readout_data_many`, `get_captured_states_many`,
  `get_readouts`, `get_readouts_many`, `clear_readouts`, `set_fast_capture`;
  `lens_capabilities` gains `fast_capture`, `readout`, `early_exit`, `early_exit_reason`;
  `steering_stats` gains `capture_layer_steps`, `capture_rows`, `hook_capture_s`,
  `readout_layer_steps`, `readout_rows`, `hook_readout_s`, `early_exits`, `retrieval_s`.
- CUDA-graph rule, unchanged but now documented with numbers: prompt-position capture /
  readout is graph-compatible (prefill batches run eagerly); generated-position capture
  needs `enforce_eager` (hooks do not run inside replayed decode graphs) or a re-encode
  pass.  Exports: `ReadoutVector`, `CAPTURE_POSITIONS_KEY`, `EARLY_EXIT_KEY`.
- Bench: `bench/bench_readout.py` (+ `modal run bench/modal_bench.py::readout`) — stock
  1.1.0 vs fork capture (legacy / gather / last-k) vs readout vs early exit vs an HF
  `read_resid`-style early-exit forward (batch 128), with HF-reference correctness
  checks.  33 new CPU tests (`vllm_lens/tests/test_readout.py`; 76 total).

## v1.1.0.post3 (3 September 2026) — hyper-connection architectures (DeepSeek-V4), vLLM 0.27 compatibility

Tested on **DeepSeek-V4-Flash-0731** (284B MoE, mHC `hc_mult=4`, fp8 + fp4 experts) with
**vLLM 0.27.1 at TP4 on 4× B200** (`bench/test_injection_dsv4.py`,
`bench/modal_bench_dsv4.py`; results in the README section "Hyper-connection
architectures"). Fell back from the requested 4× H200: the checkpoint's experts are
`expert_dtype="fp4"`, served by DeepGEMM's SM100 fp8×fp4 kernels (`moe_backend=deep_gemm`,
the configuration the DSv4 NLA session validated); Hopper has no FP4 tensor cores and would
need vLLM's Marlin-mxfp4 fallback, untested for this model.

- **Multi-stream guard.** DeepSeek-V4's decoder layers return
  `(x, residual[T, 4, D], post_mix, res_mix)` — a deferred fold, not a residual
  stream. post2 would have broadcast `output[0] + output[1]` on capture and added
  steering vectors into `x` (mis-injection into the fold) without complaint.
  post3 detects hyper-connection architectures at hook install (`hf_config.hc_mult
  > 1`, override `VLLM_LENS_MULTI_STREAM`), exposes it through a new
  `lens_capabilities` RPC, and refuses layer-output steering / capture with a
  `ValueError` at three levels: client-side in `LLM.generate` / `AsyncLLM.generate`
  before the request is submitted (engine stays alive), worker-side in
  `_prepare_vectors` / `set_steering_block`, and a runtime
  `UnsupportedLayerOutputError` from the layer hook (counted in
  `unsupported_layer_output`, re-raised — never swallowed into a warning). The
  embedding stream (`EMBED_LAYER_INDEX`: replace / add, ± `norm_match`, capture
  as layer -1) is fully supported on these models.
- **vLLM ≥ 0.27.** `_build_step_plan` no longer requires an attention-metadata
  entry with `query_start_loc` (0.27 moved it off several backends' metadata — the
  reason vllm-lens 1.2.1's per-request hooks silently no-op'd there); the model
  runner's host buffers are the primary source. The plugin sets
  `VLLM_ALLOW_INSECURE_SERIALIZATION=1` (0.27 refuses pickled `collective_rpc`
  payloads otherwise).
- **Measured on DeepSeek-V4-Flash-0731 (TP4, 4× B200, vLLM 0.27.1; `bench/results/dsv4_final`):**
  embedding replacement with a distinct vector per request at B ∈ {64, 512}, ± `norm_match`,
  at `scale = ‖e‖`, `scale = 95.5` (the NLA session's alpha) and `scale = 1.0` with `norm_match`:
  marker rows within bf16 rounding of the target (rel ≤ 6.5e-3, cos ≥ 0.999997), every other
  embedding row bit-identical; the prescaled form (`activations = alpha·v/‖v‖`, `scale = 1`) is
  **bit-identical** (max|Δ| = 0) to the NLA session's `nla.utils.dsv4.scale_vector_to_alpha` and to
  the row written by the session's own worker-side pre-hook (`nla.utils.dsv4_fast_hooks`) on the
  same engine; chunked prefill (`max_num_batched_tokens = 4096`, 13 prefill passes at B = 512,
  marker in a non-first chunk) lands every row exactly once; all four layer-output steering /
  capture attempts are refused with a `ValueError` and the engine stays alive; with CUDA graphs
  (`FULL_DECODE_ONLY`, 83 capture sizes) the hooks run only in the prefill passes (14 and 29 per
  call at B = 512 / 1024 vs ~52 / 64 eager) and the stall-free decode-step time is at parity:
  24.7 / 24.3 / 24.5 ms (B = 512) and 38.0 / 37.2 / 33.9 ms (B = 1024) for no-steer / embed-replace /
  embed-add.  Engine load is ~10–16 min per container (167 GB checkpoint from a Modal volume).
- Finding (engine, not fork): on the CUDA-graph engine, UNSTEERED requests co-batched with
  embed-replaced requests show next-token top-20 log-probs that differ from a clean batch by up
  to 1.016 (clean-vs-clean is bit-exact, prefix caching off, `num_cached_tokens = 0`, their
  embedding stream bit-identical to clean, no leakage across `generate()` calls, greedy argmax
  unchanged); the same experiment on the eager engine gives 0.000.  A hook-free control (even
  rows carry a different marker TOKEN, no vllm-lens involvement) reproduces the identical 1.016,
  so this is batch-composition sensitivity of the vLLM 0.27.1 DeepSeek-V4 kernels under the
  CUDA-graph configuration, not the fork (`bench/results/dsv4_final`, case `batch_composition`).
- 5 new CPU tests (43 total).

## v1.1.0.post2 (3 September 2026) — injection modes: embedding replacement, norm_match on the full residual stream

Two things land together: the new **embedding-replacement** injection
(`mode="replace"` + `EMBED_LAYER_INDEX`, PR #1) and the port of upstream's
**norm_match fix for fused-residual models** (upstream #7).  Both are GPU-tested
on Qwen3.6-27B and Qwen3-1.7B, eager and with CUDA graphs, with a distinct
vector per request (`bench/test_injection_modes.py`; results in the README).

### ⚠️ Behaviour change — `norm_match=True` on fused-residual models

Every vLLM decoder layer of the Qwen / Llama / Gemma / Mistral families returns
`(hidden_states, residual)`; the residual stream is their **sum**.  vllm-lens
1.1.0 (and this fork's post1) measured `‖·‖` for `norm_match` on `hidden_states`
alone — the MLP-delta half — so `norm_match=True, scale=c` injected
`c · ‖hidden_states‖ · v/‖v‖` instead of `c · ‖h‖ · v/‖v‖`.  On Qwen3.6-27B at
layer 1 that is a magnitude ratio of **0.123**: about 8× too weak, and a
different factor on every model and layer.  Upstream fixed this in 1.2.0 (#7,
commit c044d31; TP/PP tests in #15).  This release ports the fix:

- `norm_match=True` now scales to the norm of the **full residual stream** at
  that position (`hidden_states + residual`, or the tensor itself on non-fused
  layers and on the embedding stream).  `SteeringVector(norm_match=True,
  scale=coeff)` is therefore exactly the activation-oracle / Karvonen-style
  injection `h' = h + coeff · ‖h‖ · v/‖v‖` — the same arithmetic as an HF
  forward hook on the decoder layer output (verified against `mxf/inject.py`
  on Qwen3.6-27B and Qwen3-1.7B: cos ≥ 0.99998, magnitude ratio within
  5e-4 of `coeff`, next-token log-probs within the vLLM-vs-HF noise floor,
  greedy continuations identical on the 1.7B and 3/4 on the 27B).
- Applies identically in the sequential `_apply_steering` (new required
  `residual` argument, like upstream's `norm_ref`), in the vectorised
  `_apply_layer_vectorized` (batched norm of the full stream, one
  `index_add_`), for 2-D broadcast vectors, and under CUDA graphs (prefill is
  eager, so the hook sees both halves).
- **If you relied on 1.1.0 / post1 `norm_match` on a fused-residual model, your
  effective injection strength changes.**  Trainers that worked around the old
  behaviour by passing an absolute vector with `norm_match=False` are
  unaffected and can now switch to `norm_match=True, scale=coeff`.
- The post1 CHANGELOG note that the fork "keeps the 1.1.0 behaviour on purpose"
  is withdrawn — the 1.1.0 behaviour was a bug for this workload.

### Feature — embedding replacement (`mode="replace"`, `EMBED_LAYER_INDEX`)

New injection mode for NLA-style meta-models and for architectures whose
decoder layers do not emit a single residual tensor (hyper-connection /
multi-stream models such as DeepSeek-V4).  Additive `mode="add"` behaviour is
unchanged.

- `SteeringVector.mode`: `"add"` (default) or `"replace"`.  Replace overwrites
  the target row with `scale * v` (with `norm_match=True`: `scale * ‖h_orig‖ ·
  v/‖v‖`).  Requires 3-D (position-specific) activations; broadcast replacement
  is rejected by the validator.
- `EMBED_LAYER_INDEX` (= -1) as a `layer_indices` value: targets the hidden
  states *entering* decoder layer 0 (the embedding output) instead of a layer
  output.  Applied in the layer-0 pre-hook with one `index_copy_` (replace) /
  `index_add_` (add), using the same step plan and host-side offsets as the
  indexed steering hook, so chunked prefill lands the write in the chunk that
  contains the marker.  Prefill-only by construction, so decode CUDA graphs
  stay legal (measured: with one embed-replace vector per request the Python
  hooks run in 2–6 of 41 forward passes at B=512–1,024 under graphs, and the
  decode-step time on Qwen3.6-27B is within ±1.7% of no steering).
- `mode="replace"` on an ordinary layer index of a fused-residual model
  rewrites **both halves** (`hidden_states[row] = scale·v`, `residual[row] =
  0`), so the full stream equals `scale·v` exactly; the residual half is
  cloned only on layer-steps that contain a replace row.  Vectorised
  (`index_copy_` + `index_fill_`), mixed add/replace batches included.
- The layer-0 pre-hook is registered with `with_kwargs=True` and
  `_find_hidden_states_arg` searches positional **and keyword** inputs
  (`Qwen3NextModel` — Qwen3.5 / Qwen3.6 — calls `layer(positions=...,
  hidden_states=..., residual=...)` by keyword; `Qwen2Model` / `LlamaModel`
  positionally).  Exactly one `[≥ total_tokens, hidden]` floating candidate is
  required (a kwarg literally named `hidden_states` breaks ties); anything else
  raises `EmbedInjectionError` out of the forward pass — counted in
  `embed_errors` — instead of logging a warning and silently skipping the
  injection.  (The PR #1 branch had a positional-only scan that would have
  skipped Qwen3.6 with a warning.)
- `EMBED_LAYER_INDEX` is accepted by the RPC-side layer validation
  (`_prepare_vectors`, `set_steering_block`; the PR branch rejected -1 as out
  of range) and the block RPC carries a per-entry `modes` list, so one
  `set_steering_block` RPC per `generate()` call still covers replace / embed
  vectors (older blocks without `modes` default to add).
- `output_residual_stream=[EMBED_LAYER_INDEX, ...]` captures the embedding
  stream (post-injection) as layer -1; `output_residual_stream=True` is
  unchanged (n_layers rows, no embedding row).
- New stats counters: `rows_replaced`, `embed_apply_steps`, `embed_errors`.
  10 new CPU tests (38 total, `pytest vllm_lens/tests/test_steering_index.py
  --noconftest`).

### GPU test matrix (`bench/test_injection_modes.py`, `modal run bench/modal_bench.py::test_injection`)

Qwen3.6-27B and Qwen3-1.7B, eager and `VLLM_LENS_CUDA_GRAPHS=1`, one distinct
unit vector per request, B ∈ {64, 512}: Karvonen add at layer 1 (`norm_match=True`,
coeff ∈ {1, 4}) vs an HF reference built with the trainer's exact hook;
embedding replacement with and without `norm_match`; replacement at a
fused-residual layer output; a mixed batch (half embed-replace, half add);
chunked prefill with the marker in a non-first chunk (`max_num_batched_tokens=64`);
and throughput at B ∈ {512, 1024} vs `bench/results_summary.json`.  See the
README section "Injection modes: test matrix" for the numbers.

## v1.1.0.post1 (3 September 2026) — vllm-metamodels

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

- ~~upstream #7 changed `norm_match=True` on fused-residual models to scale by
  the full residual stream instead of the MLP-delta component (a behaviour
  change that makes existing `norm_match` steering stronger).  This fork
  keeps the 1.1.0 behaviour.~~ **Superseded: ported in 1.1.0.post2** (see above).
- the generic `apply_hooks` / `register_hooks` system.

## v1.1.0 (upstream)

- Prior releases — see
  [upstream](https://github.com/UKGovernmentBEIS/vllm-lens/blob/main/CHANGELOG.md).
