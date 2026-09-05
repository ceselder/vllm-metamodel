## Unreleased — indexed hook dispatch, idle fast path, opt-in CUDA graphs

Per-request steering / hooks at batch sizes in the hundreds were dominated by hook overhead:
every layer hook re-resolved every request with a `startswith` scan over all registered keys
and paid two device syncs per request, on every forward pass, and the plugin forced
`enforce_eager`. See `UPSTREAM_PR.md` for the numbers (Qwen3.6-27B, 1× B200: one `generate()`
of 1,024 steered requests 230 s → 12 s eager → 10.5 s with CUDA graphs, = the no-steering
time; unsteered rows bit-identical, steering deltas cos 1.000 / ratio 1.000).

- `_find_steering_configs` / `_find_hook_configs_no_persistent`: dict lookups on the internal
  id's `-`-boundary prefixes (+ the `_steering_id` / `_hook_id` sentinel) instead of a scan;
  same results, same order.
- Each request's steering / hooks / capture spec is resolved once and cached (`_ReqPlan`),
  invalidated by a generation counter on every `set_*` / `clear_*`; each forward pass is
  planned once from the model runner's host buffers (`_StepPlan`; no `.item()` syncs; works
  when the attention metadata carries no `query_start_loc`, as on several vLLM ≥ 0.27 backends).
- Idle fast path: the first decoder layer's pre-hook flags passes on which no hook can have
  work (no hooks / capture, every steering vector behind every row's position); layer hooks
  then return on one flag check. Per-layer work is skipped when nothing targets the layer.
- `VLLM_LENS_CUDA_GRAPHS=1`: decode batches replay CUDA graphs (`cudagraph_mode=FULL_DECODE_ONLY`,
  compilation mode NONE) instead of forcing eager; steering / hooks / capture are then
  prompt-position only, 2-D broadcast vectors are refused with a `ValueError`, generated
  positions are never touched (batch-composition independent). Default behaviour unchanged.
- New CPU test `tests/test_indexed_dispatch.py` (no engine needed).

## v1.2.1 (22 July 2026)

- Steering: Fixed offline steering via `LLM.chat` — the plugin now patches `LLM.chat` (which submits requests to the engine directly rather than routing through `LLM.generate`). Previously, live `SteeringVector` objects raised a msgpack `TypeError`, and JSON-serialized vectors ran **silently unsteered**. Activation capture (`output_residual_stream`) and per-request hooks (`apply_hooks`) now also work through `LLM.chat`. (#28)
- Steering: The offline `LLM.generate` / `LLM.chat` path now decodes the JSON-string wire format for `apply_steering_vectors` and `apply_hooks` (the form the HTTP API accepts via `vllm_xargs`), instead of failing inside `collective_rpc`. Both entry points accept both wire forms. (#28)

## v1.2.0 (17 July 2026)

- Hooks: Added a generic, Garçon-style hook system — run arbitrary Python functions per-request and per-layer to capture data (via `ctx.saved`) and/or modify hidden states (by returning a tensor). Supports per-request hooks (`apply_hooks` in `extra_args`), persistent register-once hooks (`register_hooks` / `collect_hook_results` / `clear_hooks`), and pre-hooks (run before the layer forward pass). Exposed over HTTP at `/v1/hooks/*` and through a `VLLMLensClient`; `ctx.get_parameter()` gathers parameters across tensor- and pipeline-parallel ranks. (#3)
- Hooks: Added `POST /v1/hooks/clear_results` (`client.clear_hook_results()`) — drain accumulated persistent-hook results while keeping the hooks registered, so long-lived clients can bound accumulation without re-uploading hook state. (#25)
- Steering: Fixed `norm_match` on fused-residual architectures (Qwen, Gemma, Llama, …). It now scales the steering vector to the full residual stream rather than the MLP-delta half, so the injected magnitude equals `‖residual‖ · scale` as intended. **Behavior change:** existing `norm_match=True` steering becomes correspondingly stronger. (#7, #15)
- Performance: On the offline `LLM.generate` path, a batch's captured activations are now fetched in a single RPC instead of one per request. (#6)
- Plugin: Added the `VLLM_LENS_DISABLE=1` environment variable to make the plugin a no-op, so vllm-lens can be installed alongside another inference server without perturbing it. (#9, #14)
- Plugin: Default to the V1 model runner and raise a clear error if the V2 runner is active (the capture/steering hooks read V1 model-runner internals). (#12)
- Examples: Moved examples to a top-level `examples/` directory and added causal-tracing, logit-lens, deception-probe, and emotion-concepts examples. (#4, #11)
- Examples: Added a Jacobian-lens / J-space example — fit a per-model average Jacobian `J_l = E[∂h_final/∂h_l]` offline (HuggingFace backward pass), then read it out live on vLLM-captured residuals to see what the model is "disposed to say" at each layer. (#19)
- Examples: Hardened the Jacobian-lens run path for tensor / pipeline / expert parallelism, so it works on larger multi-GPU models. (#21)
- Examples: Added a live J-space chat visualizer — chat with a served model in a generated HTML page where every token is hoverable, showing the streaming top-k Jacobian-lens readout across the captured layers. (#25)
- Examples: The Jacobian-lens fitter stamps provenance (model, corpus, sample count, estimator settings) into the saved lens `.pt`. (#27)
- Docs: Documented the `Hook`/`HookContext` interface and added first-class README sections for activation capture and steering. (#13, #16)

## v1.1.0

- Prior releases.
