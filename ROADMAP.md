# vllm-lens Roadmap

Grounded in a review of the codebase from the perspective of an autonomous agent
running interp experiments against a vllm-lens server (July 2026). The recurring
theme: the capability surface is strong; the biggest wins are eliminating
**silent failure modes**, adding **discoverability**, and returning data with
enough **metadata to interpret it** without guesswork.

Items reference open issues/PRs where they exist. Ordering within a tier is
roughly by value.

## At a glance

**Tier 1 — Correctness and agent-critical QoL**
- [1.1](#11-surface-hook-errors-to-the-client) Surface hook errors to the client (+ `/v1/hooks/validate`)
- [1.2](#12-introspection-endpoint) Introspection endpoint (`/v1/lens/info`, provenance, `vllm_lens.introspect`)
- [1.3](#13-token-alignment-metadata-on-captured-activations) Token-alignment metadata on captured activations
- [1.4](#14-prefix-caching-close-the-one-gap-document-add-a-regression-test) Prefix caching: fix the offline `sampling_params=None` gap, document, test
- [1.5](#15-expose-capture_layersall) Expose `capture_layers="all"` + negative indices (server-side)
- [1.6](#16-auth-coverage-on-v1hooks) Auth regression test for `/v1/hooks/*`
- [1.7](#17-fix-the-tp-list-duplication-footgun) TP list-duplication: per-hook reduction policy
- [1.8](#18-hook-execution-time-metadata) Hook execution-time metadata (`abs_start`, phase, token IDs)

**Tier 2 — Research capabilities**
- [2.1](#21-vllm_lensrecipes-promote-example-code-into-the-library) `vllm_lens.recipes`: ablation, projections, probes, logit lens
- [2.2](#22-server-side-tensor-store-handles) Server-side tensor store (handles)
- [2.3](#23-activation-patching-as-a-first-class-primitive) Activation patching as a first-class primitive
- [2.4](#24-sae-support) SAE support (recipe + tensor store for large SAEs)
- [2.5](#25-more-capture-points) More capture points (PR #34 Q/K → named-module capture)

**Tier 3 — Scale and performance**
- [3.1](#31-async-client--sweep-ergonomics) Async client + `generate_batch` (+ multi-choice semantics)
- [3.2](#32-binary-transport-for-activations-31) Binary transport for activations (#31, shape B; RDMA as stretch)
- [3.3](#33-quantify-and-chip-at-the-enforce_eager-tax) Quantify the `enforce_eager` tax
- [3.4](#34-v2-model-runner-support-existential) V2 model runner support + vLLM CI matrix (existential)

**Tier 4 — Docs and agent ergonomics (#32)**
- CLAUDE.md, per-technique skills, fix-stating error messages,
  capture-site semantics contract

**Deferred:** declarative server ops / no-code mode · resource guards ·
multi-tenancy/session isolation · client request IDs · causal-scrubbing
frameworks · #24 fp8 fit path

---

## Tier 1 — Correctness and agent-critical QoL

### 1.1 Surface hook errors to the client
Hook exceptions are currently caught, logged server-side, and swallowed
(`vllm_lens/_worker_ext.py`, `_make_hook` / `_make_pre_hook`). From the client,
a buggy hook is indistinguishable from a hook that ran and saved nothing.

- [ ] Capture per-request, per-hook tracebacks and return them in a
      `hook_errors` field on the response (and via `/v1/hooks/collect`).
- [ ] Add `POST /v1/hooks/validate`, two levels: (a) static checks —
      deserializability, signature, layer bounds; (b) an optional real-model
      canary request (1-token generate with the hook attached, errors
      returned). A dummy-tensor dry-run alone gives false confidence: it
      can't exercise `ctx.model`, `ctx.get_parameter` TP collectives, or
      real device/dtype/fused-layer semantics.

### 1.2 Introspection endpoint
There is no way to ask a server what it's serving. Clients currently guess
layer counts from HF configs and learn topology from `layer_index out of range`
errors.

- [ ] `GET /v1/lens/info` + `client.info()`: architecture, `num_hidden_layers`,
      `hidden_size`, norm / `lm_head` module names, tied embeddings, TP/PP
      topology, currently registered hooks + prefetched params, plugin version.
- [ ] Include server provenance in `info()` for reproducibility: model
      revision/commit, dtype, quantization config, loaded LoRAs, vLLM +
      vllm-lens versions. Optionally (opt-in flag) echo a compact per-response
      run manifest (seed, sampling params, lens args used) so a saved
      activations file can be traced to exactly what produced it — the
      Jacobian-lens fitter already stamps provenance (#27); this generalizes
      that habit.
- [ ] Build on registry-based layer discovery (PR #33); promote the examples'
      `find_norm` / `get_num_layers` / decoder-layout helpers into a public
      `vllm_lens.introspect` module (closes #22 — examples then import it
      instead of carrying per-architecture maps).

### 1.3 Token-alignment metadata on captured activations
`GenerateOutput.activations` is a bare `(n_layers, total_pos, hidden)` tensor.
Users must separately reconstruct token IDs per position, the prompt/generation
boundary, and chat-template effects — the classic off-by-one trap; every
example re-derives it by hand.

- [ ] Return `token_ids`, `prompt_len`, and the captured layer indices
      alongside the tensor (offline, HTTP, and Inspect paths).
- [ ] `client.tokenize(messages)` helper (vLLM already exposes `/tokenize`).
- [ ] Span-finder utility: substring / message-span → position indices, for
      `SteeringVector.position_indices` and position-targeted hooks.
- [ ] Document that the final sampled token has no forward activation —
      capture is trimmed to `prompt + generated − 1` positions
      (`_activations_plugin.py:351`) — as part of the alignment metadata.

### 1.4 Prefix caching: close the one gap, document, add a regression test
Audited (July 2026): the plugin already handles this correctly on the main
paths — any request that needs hooks gets `SamplingParams.
skip_reading_prefix_cache = True` (`_activations_plugin.py:324-327` async,
`:449-451` offline), forcing a fresh prefill while still *writing* to the
cache so other traffic keeps its speedup. Covers HTTP (completions + chat),
offline `LLM.generate`/`LLM.chat` with explicit params, persistent hooks over
HTTP, and the Inspect provider.

- [ ] **Fix the real gap**: offline `LLM.generate(prompts)` /
      `LLM.chat(msgs)` with `sampling_params=None` + persistent hooks — 
      `_prepare_offline_params` iterates an empty `params_list`
      (`:401-406`, `:450-451`), so vLLM builds default `SamplingParams` with
      cache reads enabled and hook firings can silently go missing for
      cached prefixes (e.g. shared system prompts). Construct default params
      and set the flag.
- [ ] Add an explicit repeated-prompt regression test (current coverage is
      only implicit via prompt reuse across module-scoped engines; the
      offline `sampling_params=None` case is untested).
- [ ] Document the behavior + the undocumented per-request
      `extra_args={"skip_reading_prefix_cache": True}` escape hatch in the
      README.

### 1.5 Expose `capture_layers="all"`
The server already supports capture-all — `extra_args=
{"output_residual_stream": True}` (documented only in a docstring,
`_activations_plugin.py`; the worker treats any non-list truthy value as
"all layers"). The client and README only admit lists.

- [ ] Accept `capture_layers="all"` / `True` in `VLLMLensClient`.
- [ ] Support negative layer indices (`-1` = last layer), normalized
      **server-side** — workers currently reject them
      (`_worker_ext.py:729,868`), and client-side normalization would leave
      offline and raw-HTTP users inconsistent.
- [ ] Echo back which layers were actually captured (folds into 1.3).

### 1.6 Auth coverage on `/v1/hooks/*`
The hooks router is mounted into vLLM's FastAPI app under `/v1/hooks`, so
vLLM's `--api-key` middleware (which guards `/v1` paths) should cover it — but
this endpoint executes cloudpickle payloads, so "should" isn't good enough.

- [ ] Add an explicit test: with `--api-key` set, every `/v1/hooks/*` route
      (and `/v1/hooks/ui/`) rejects unauthenticated requests.
- [ ] Document the trust model prominently (hooks = remote code execution).

### 1.7 Fix the TP list-duplication footgun
`ctx.saved` is merged across ranks at collection and **list values are
concatenated across ranks**, so with TP=4 a hook that appends to a plain list
sees every entry 4×. Currently documented (README) rather than fixed; users
will hit it before reading that paragraph.

- [ ] Replace type-based merge guessing with an explicit per-hook reduction
      policy: `rank0` (default for plain lists — fixes the duplication),
      `concat`, `stack`, or custom. A blanket rank-0 gate alone would
      silently discard legitimately rank-local data (MoE expert-local stats,
      PP-stage results — PP ranks genuinely hold different layers), so the
      policy must stay overridable per hook.

### 1.8 Hook execution-time metadata
`HookContext` exposes only `layer_idx` and `seq_len` — a hook cannot tell
what its slice *is*. Under chunked prefill and decode, `h[0]` might be prompt
position 0, the start of a later prefill chunk, or the newest generated
token. The steering path already computes the absolute start position
internally (`_worker_ext.py:314-330`, via `seq_lens` − `n_query`); it just
isn't exposed to hooks. This blocks position-conditioned probes, per-token
monitors, and faithful recipe implementations of anything positional.

- [ ] Expose on `ctx`: absolute position offset (`abs_start`), phase
      (prefill / decode), the request's token IDs, and the request ID.
- [ ] Prerequisite for recipe parity (2.1) and the SAE recipe's per-token
      feature indexing (2.4).

---

## Tier 2 — Research capabilities

### 2.1 `vllm_lens.recipes`: promote example code into the library
Keep the server simple — the Garçon hook interface stays the only server-side
mechanism — and ship the standard techniques as importable, tested,
client-side functions that build hooks under the hood:

- [ ] Ablation (zero / mean) of layers or neuron indices.
- [ ] Direction projection: per-token dot products against supplied vectors
      (the `emotion_tracker` / `deception_probe` pattern) — tiny results
      instead of shipping full residuals.
- [ ] Linear probe inference: per-token probe scores from a registered probe.
- [ ] Logit-lens readout: top-k tokens per (layer, position), computed
      server-side inside the hook via `ctx.get_parameter` / `compute_logits`.
- [ ] Migrate `examples/` to consume these (examples become thin demos).
- [ ] Evaluate migrating the existing prebuilt ops (activation capture,
      steering) onto the same recipes/hook path, so the server carries one
      mechanism instead of three. **Gate on benchmarks AND behavior parity**:
      only do this if efficiency isn't materially affected — the bespoke
      capture path has optimizations a generic hook must match first
      (TP-rank-0 gating, batched RPC collection, zstd compression,
      surplus-token trimming) — and position-specific steering can't be
      reproduced through hooks until 1.8 lands (hooks currently lack the
      absolute-position info the steering path computes internally). Parity
      tests across chunked prefill, decode, TP/PP, and fused-residual
      architectures before switching anything over.

Deferred (revisit on demand): declarative JSON server-side ops and a
`VLLM_LENS_NO_CODE` mode for untrusted clients. Not needed while all known
deployments trust their clients; the recipes API is designed so the same
client calls could later target declarative ops instead of cloudpickle.

### 2.2 Server-side tensor store (handles)
One mechanism, three payoffs: register named tensors on the workers once,
reference them by handle afterwards.

- [ ] `POST /v1/tensors/register` (name → tensor, upload once, lives on
      workers like prefetched params) + list/clear.
- [ ] Hooks and steering can reference stored tensors by name instead of
      embedding them in every request / closure.
- Unlocks: activation patching (2.3), two-stage activation fetch (3.2),
  upload-once SAE / probe weights (2.4).
- Design note: one handle *namespace*, but distinct storage classes —
  registered weights (long-lived, replicated to ranks that need them),
  patch donors (short-lived, TTL'd, stage-local), and fetch buffers
  (host memory, drained on read) have different placement and lifetime
  requirements; a naive replicate-everywhere store multiplies memory the
  way full-parameter prefetch already does.

### 2.3 Activation patching as a first-class primitive
Today patching means capture → download GB → re-upload inside a hook closure.

- [ ] `capture_layers=[...], store_as="clean-1"` keeps captured activations
      server-side under a handle (TTL'd).
- [ ] `patch_from={"handle": "clean-1", "layers": [...], "positions": [...]}`
      on a later request patches them in — no tensors cross the wire.
- [ ] Rewrite `causal_tracing.py` on top. Handles remove the transfer cost;
      it's still one forward pass per grid cell, so the wall-clock win also
      needs concurrent submission (3.1) to exploit dynamic batching.
- [ ] Define alignment rules: donor/recipient position mapping when
      tokenizations differ, dtype/shape checks, and an explicit
      missing-position policy (error vs skip).

### 2.4 SAE support
Minimal viable version is client-side (no server changes): a recipe that
loads an SAELens-format SAE from the Hub and builds a hook that runs the
encoder server-side, saving top-k `(feature_id, activation)` per token —
much cheaper on the wire than shipping residuals.

- [ ] `recipes.sae`: Hub loading + feature-capture hook (weights ride in the
      hook closure for small SAEs; use the tensor store (2.2) for large ones).
- [ ] Feature steering = existing `SteeringVector` with decoder rows; ship a
      helper + example.
- [ ] Worked example / notebook (feature dashboard over a chat, in the style
      of `jacobian_lens_chat`).

### 2.5 More capture points
Hooks fire only at decoder-layer boundaries (`register_forward_hook` on the
layer modules) and see the residual stream. Sub-layer sites — attention out,
MLP out, per-head, Q/K — are outside the per-request slicing bookkeeping that
is vllm-lens's core value.

- [ ] Land per-request Q/K capture + attention-pattern reconstruction (PR #34).
- [ ] Longer term: named-module capture (`capture_modules=
      ["model.layers.15.mlp"]`) — generalize the request-slicing bookkeeping
      to arbitrary submodule outputs rather than adding one flag per site.
      Note the real limitation: only modules whose outputs keep the
      `(total_tokens, ...)` layout are sliceable per request, and anything
      inside fused kernels (FlashAttention internals, fused QKV/MoE kernels)
      is not hookable at all — the introspection endpoint (1.2) should
      report which modules are capturable so users don't discover this by
      trial and error.

---

## Tier 3 — Scale and performance

### 3.1 Async client + sweep ergonomics
`VLLMLensClient` is sync, single-prompt, no retries — painful for
hundreds-of-prompts probe/patching sweeps.

- [ ] `AsyncVLLMLensClient` (httpx), same surface.
- [ ] `generate_batch(prompts, *, concurrency=N)` with bounded parallelism,
      retries with backoff, and per-request error capture (don't fail the
      sweep on one 500).
- [ ] Define multi-choice semantics **before** shipping this: activations and
      hook results are currently attached response-level and the client reads
      `choices[0]` (`client.py:116`, `_activations_plugin.py:569`) — ambiguous
      as soon as `n > 1`, best-of, or beam search is in play. Attach lens data
      per-choice (or explicitly reject `n > 1` with a clear error).

### 3.2 Binary transport for activations (#31)
Activations currently return as zstd → base64 inside the completion JSON:
~33% inflation, full-blob materialization on both ends, and per-request server
memory that caps concurrent capture throughput. ACS Research has offered to
implement.

- [ ] Prefer shape **B** (two-stage fetch): response carries an activation
      handle; `GET /v1/activations/{id}` streams raw binary (TTL'd buffer,
      per-layer range requests). Shares infrastructure with 2.2/2.3 — one
      handle store serves fetch *and* patching.
- [ ] Keep JSON+base64 as the default/fallback; `VLLMLensClient` handles the
      negotiation transparently.
- [ ] Design the fetch side transport-pluggable, then investigate fast-path
      backends for colocated clients: CUDA IPC for a client process on the
      same host (activations never leave the GPU / touch the network stack),
      and GPU RDMA (UCX / NIXL, the stack vLLM already uses for disaggregated
      prefill KV transfer) across nodes. Stretch goal — HTTP streaming ships
      first and is the 90% win; the handle abstraction is what keeps the
      door open.
- [ ] Bulk-harvest mode: `capture_to="dir/shard-{i}.safetensors"` spills
      server-side, response returns a manifest — for dataset-scale probe/SAE
      training data.

### 3.3 Quantify (and chip at) the `enforce_eager` tax
The plugin forces `enforce_eager=True` globally so hooks fire.

- [ ] Benchmark and document the throughput cost across model sizes (extend
      `_benchmarks/`), so users can make informed serving decisions.
- [ ] Investigate piecewise CUDA-graph compatibility for the capture-only
      path (steering/hooks likely genuinely need eager; capture at fixed
      layer boundaries might not). Outcome may well be "not feasible" — write
      that down too.

### 3.4 V2 model runner support (existential)
Capture/steering read V1 model-runner internals; the plugin pins
`VLLM_USE_V2_MODEL_RUNNER=0` and errors on V2. When upstream deprecates V1,
everything breaks at once.

- [ ] Track V2 runner internals; prototype the hook path against it early.
- [ ] CI compatibility matrix against pinned + latest vLLM releases, so
      upstream breakage is caught the week it ships, not by users.

---

## Tier 4 — Docs and agent ergonomics (#32)

Sequenced after Tier 1 — skills can't paper over silent hook failures or
missing introspection, and 1.2's `info()` endpoint is the best "documentation"
of all (queryable at runtime).

- [ ] CLAUDE.md: layer-indexing conventions, TP/PP + rank caveats, wire-format
      rules (JSON-string vs live-object extra_args), known footguns.
- [ ] Skills per technique: capture→probe, steering, patching, logit/J-lens,
      attention (post PR #34).
- [ ] Error messages that state the fix ("layer 40 out of range — this model
      has 32 layers; call client.info()").
- [ ] Capture-site semantics contract: a documented, tested definition of
      what "residual stream after layer N" means per architecture family —
      fused-residual tuple handling (`output[0] + output[1]`,
      `_worker_ext.py:422`), pre-hook input semantics (layer-0 pre-hook =
      embedding output?), norm placement, MoE/hybrid behavior, and the
      generated-token convention (final sampled token has no forward
      activation). State explicitly that gradients / backward-pass workflows
      are out of scope for this plugin.

---

## Explicitly deferred

- **Declarative server-side ops / no-code mode** — see 2.1; revisit if
  untrusted-client deployments materialize.
- **Resource guards on hooks** (result-size caps, wall-clock warnings) —
  judged unnecessary for the current trusted-user model.
- **fp8 / GLM-5.2 (DSA) large-model Jacobian fit path** — tracked in #24,
  orthogonal to the core library roadmap.
- **Multi-tenancy / session isolation** — persistent hooks, prefetched
  params, and results are global mutable worker state: every persistent
  hook fires on every request, and `/v1/hooks/clear` removes everyone's
  hooks. Real (two concurrent users can steer each other's generations and
  read each other's results), but ~99% of users self-host on their own GPUs
  solo, and doing isolation properly (scoped hook-sets, ownership, per-request
  selection) is substantial work. Long-horizon: revisit when shared-server
  or parallel-eval deployments become common; the tensor store (2.2) should
  at least not make the problem worse.
- **Client-supplied stable request IDs** for hook-result correlation
  (persistent collection returns vLLM-internal IDs; `deception_probe.py`
  recovers order by sorting them) — judged too minor for now.
- **Causal scrubbing / path patching / synchronized counterfactual
  frameworks** — research-framework ambitions beyond this plugin's scope;
  recipes (2.1) can grow toward them organically.

---

## Appendix: investigation findings and design rationale (July 2026)

Record of the review discussion behind the items above, so the reasoning
survives outside chat context.

### Prefix caching — handled, with one real gap (basis for 1.4)

Audit result: the plugin never touches `enable_prefix_caching` (APC stays on
globally). Instead, any request that needs hooks gets vLLM's per-request
opt-out `SamplingParams.skip_reading_prefix_cache = True`
(`_activations_plugin.py:324-327` async path, `:449-451` offline path).
Semantics: the request skips *reading* the cache — full prefill runs, hooks
fire for every prompt position — but still *writes* to it, so unrelated
traffic keeps its speedup. The field exists and is consumed scheduler-side in
all vLLM versions ≥ the pyproject floor (`vllm>=0.16.0`), so there is no
version window where it silently no-ops.

Coverage confirmed on: HTTP completions + chat (streaming and not), offline
`LLM.generate`/`LLM.chat` with explicit params, persistent hooks over HTTP
(`_hooks_router.py:41` sets `engine._has_persistent_hooks`, read at
`_activations_plugin.py:317`), and the Inspect provider (transitively, via
HTTP).

**The gap**: offline `LLM.generate(prompts)` / `LLM.chat(msgs)` with
`sampling_params=None` **plus persistent hooks**. `_prepare_offline_params`
builds an empty `params_list` (`:401-406`), so the flag-setting loop at
`:450-451` iterates over nothing; vLLM then constructs default
`SamplingParams` with cache reads enabled, and persistent-hook firings can
silently go missing for cached prefixes (shared system prompts, repeated
prompts across calls). Not reachable over HTTP. Untested — persistent-hook
tests are HTTP-only; existing offline tests only cover prefix caching
*implicitly* via prompt reuse on module-scoped engines.

Also undocumented: the whole mechanism, and the per-request escape hatch
`extra_args={"skip_reading_prefix_cache": True}` (`:314-315`).

Secondary observations from the audit (minor): `_has_persistent_hooks` is a
per-engine-object attribute, so with multiple front-end API server processes
only the process that served `/v1/hooks/register` flags requests as needing
hooks; and only `generate`/`chat` are patched (pooling/`encode` requests never
get the flag — they also never get activations).

### `capture_layers="all"` — half exists (basis for 1.5)

The server accepts `extra_args={"output_residual_stream": True}` to capture
all layers — documented only in a docstring (`_activations_plugin.py:736`);
the worker treats any non-list truthy value as "all layers"
(`_worker_ext.py`, JSON-decode fallback). But `VLLMLensClient.capture_layers`
only accepts a list, and the README only mentions lists — so the shorthand is
invisible in practice. 1.5 is an expose-it item, not a build-it item.

### Auth on `/v1/hooks/*` — probably fine, needs a test (basis for 1.6)

The hooks router is mounted into vLLM's own FastAPI app under `/v1/hooks`
(`_activations_plugin.py:670`), so vLLM's `--api-key` middleware (which
guards `/v1` paths) should cover it. Not verified against vLLM source during
the review (vLLM not installed in the review environment) — and since these
endpoints execute cloudpickle payloads, "should" warrants an explicit
regression test rather than an assumption.

### TP list-duplication footgun, explained (basis for 1.7)

With tensor parallelism, hook `fn` runs on **every** TP rank (steering must
modify hidden states on all ranks). At collection, `ctx.saved` dicts are
merged across ranks and **list** values are concatenated
(`_hooks_router.py:63-72`) — so with TP=4, a hook appending per-token entries
to a plain list gets every entry 4×. Currently documented in the README
rather than fixed; users hit it before reading that paragraph. Fix: rank-gate
list collection to TP rank 0 by default, keeping cross-**PP** merging intact
(PP ranks genuinely hold different layers).

### Why recipes are client-side, not declarative server ops (basis for 2.1)

Maintainer preference is to keep the server simple, with the Garçon hook
interface as the only server-side mechanism. `vllm_lens.recipes` achieves the
same user-facing simplicity by building hooks client-side — mostly moving
already-written example code into the tested package. What this gives up vs
server-side declarative ops: (a) any untrusted-client story (declarative JSON
ops could be validated instead of executed; a `VLLM_LENS_NO_CODE` mode falls
out for free), and (b) freedom from cloudpickle's client↔server Python/library
version coupling. Neither bites current users — all known deployments trust
their clients — so declarative ops are deferred, with the recipes API shaped
so the same calls could later target declarative ops if demand materializes.

### SAE support is mostly a recipe, not a server feature (basis for 2.4)

Minimal version needs zero server changes: `recipes.sae` loads an
SAELens-format SAE from the Hub and builds a hook whose closure carries the
encoder weights — cloudpickle ships them to the workers once via
persistent-hook registration. The hook encodes server-side and saves top-k
`(feature_id, activation)` per token, far cheaper on the wire than shipping
residuals. Feature *steering* already works: `SteeringVector` with decoder
rows. The only real infrastructure want is large SAEs, where a multi-GB hook
closure gets ugly — which is exactly the upload-once tensor store (2.2).

### Why hooks can't already capture sub-layer sites (basis for 2.5)

Hooks are installed only via `register_forward_hook` on the decoder-layer
modules (`_worker_ext.py:704-709`), so `fn` receives the residual stream at
layer boundaries. A hook could technically attach its own torch hooks to
submodules through `ctx.model`, but those fire per *batch*, outside the
`query_start_loc` / `req_ids` bookkeeping that slices tensors per request —
which is the core value of vllm-lens. So attention-out / MLP-out / per-head /
Q-K capture genuinely needs library support: PR #34 first, then generalized
named-module capture.

### Binary transport, explained (basis for 3.2)

Today an activation tensor returns as: zstd-compress → **base64** → string
field inside the completion JSON. Three structural costs (issue #31, from ACS
Research, who run vllm-lens in production on Modal):

1. base64 inflates the already-compressed payload by ~33%;
2. both server and client must materialize and parse the entire response in
   one piece — a multi-hundred-MB JSON string for large captures (8B model,
   all layers, 4k-token prompt ≈ 1.1 GB raw bf16);
3. the server holds each request's full blob in memory until the response
   flushes, capping concurrent capture throughput well below what the forward
   pass sustains (ACS measured ~384 tok/s inline vs multi-k tok/s offline).

Their proposal A: content-negotiated multipart response (normal completion
JSON part + one raw binary part per tensor); less invasive, default behavior
unchanged. Proposal B: two-stage fetch — the completion returns a small
*handle*, and the client separately `GET`s `/v1/activations/{id}` as a binary
stream from a TTL'd server-side buffer. **Recommendation: B**, because the
handle/TTL store it requires is the same infrastructure as the patching donor
store (2.3) and SAE/probe weight upload (2.2/2.4), and it additionally
enables per-layer range requests and decouples capture from response latency.
ACS offered to implement with tests on their GPU infra — likely a design
review rather than an implementation cost.

### Maintainer review outcomes (July 2026)

- Recipes direction confirmed, with the addition that existing prebuilt ops
  (activation capture, steering) should also migrate onto the recipes/hook
  setup **if and only if** benchmarks show no material efficiency loss
  (now in 2.1).
- SAE recipe approach confirmed.
- Named-module capture confirmed, with the caveat that capturable modules
  will be limited (fused kernels, non-token-major layouts) — introspection
  should advertise what's capturable (now in 2.5).
- Binary transport: shape B confirmed. Maintainer asked about NVLink /
  GPU-RDMA transport; assessment: right as an opt-in fast path for
  colocated clients, not the first milestone — HTTP streaming is the 90%
  win, and making the handle-fetch transport pluggable keeps CUDA IPC
  (same host) and UCX/NIXL RDMA (cross-node, same stack vLLM uses for
  disaggregated-prefill KV transfer) open as follow-ups (now in 3.2).
- TP list-duplication fix confirmed (1.7).

### External review outcomes (Codex, July 2026)

A second-model review (framed as an interp researcher) was run over the
roadmap + codebase. Accepted and merged: hook execution-time metadata (new
1.8 — its strongest catch), multi-choice/`n>1` semantics (3.1), capture-site
semantics contract + gradients-out-of-scope statement (Tier 4), concrete
provenance in `info()` (1.2), two-level hook validation replacing the
dummy-tensor dry-run (1.1), server-side negative-index normalization (1.5),
per-hook reduction policy replacing blanket rank-0 gating (1.7), tensor-store
storage classes (2.2), and honest patching-sweep framing + alignment rules
(2.3). Considered and rejected: promoting resource guards / lifecycle /
circuit breakers to Tier 1 (production-SaaS engineering for a trusted
single-team tool), stable client request IDs (too minor), causal-scrubbing/
path-patching frameworks (out of scope), and isolation-first reordering.
Multi-tenancy acknowledged as real but long-horizon (see deferred section).

### Sequencing note

The offline prefix-cache gap fix (1.4), hook-error surfacing (1.1), and the
auth test (1.6) sit in Tier 1 as small, correctness-flavored items. The async
client (3.1) landed in Tier 3 with the other scale work despite being an
acknowledged embarrassment — promote it if the next release should lead with
sweep ergonomics.
