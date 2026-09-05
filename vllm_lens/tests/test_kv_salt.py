"""CPU-only tests for prefix caching with steering (vllm-metamodels 1.1.0.post7, ``_kv_salt``).

Three layers are covered without a GPU:

* the client-side policy (which requests skip reading the cache, where the salt starts,
  which tag) -- pure Python;
* the engine-core patch on vLLM's REAL request block hasher (``vllm.v1.request.Request`` +
  ``get_request_block_hasher``): blocks before the salted token keep vLLM's hash, blocks from
  it on differ per tag, payload tags share, the last-token fallback -- skipped when vLLM's
  V1 core is not importable (the CPU-only CI without vLLM);
* the worker: early exit under prefix caching needs a salt from token 0; the backstop
  counters when a steered request is planned past its marker.

Run without upstream's GPU conftest:  pytest vllm_lens/tests/test_kv_salt.py --noconftest
"""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import pytest
import torch

from vllm_lens import SteeringVector
from vllm_lens import _kv_salt as K

# the shared fake runner / extension harness (also stubs vLLM's two import-time modules)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from test_steering_index import FakeRunner, W, build_plan, make_ext, store, sv2d, sv3d  # noqa: E402

D = 8


# ---------------------------------------------------------------------------
# client-side policy
# ---------------------------------------------------------------------------


def test_min_steered_position_and_salt_from():
    assert K.min_steered_position([sv3d(1, [10])]) == 10
    assert K.min_steered_position([sv3d(1, [10]), sv3d(2, [4, 7])]) == 4
    assert K.min_steered_position([sv3d(1, None, n_pos=3)]) == 0  # sequential 0..2
    assert K.min_steered_position([sv2d(1)]) == 0  # broadcast: every position
    assert K.min_steered_position([]) is None
    # salt starts one token BEFORE the marker (>= 2 recomputed tokens), never below 0
    assert K.steering_salt([sv3d(1, [90])], "nonce", "abc") == [89, "n:abc"]
    assert K.steering_salt([sv3d(1, [0])], "nonce", "abc") == [0, "n:abc"]
    assert K.steering_salt([sv2d(1)], "nonce", "abc") == [0, "n:abc"]
    assert K.steering_salt([sv3d(1, [5])], "mygroup", "abc") == [4, "u:mygroup"]
    assert K.steering_salt([], "nonce", "abc") is None


def test_payload_hash_covers_everything_that_changes_the_kv():
    torch.manual_seed(0)
    v = torch.randn(1, 1, D)
    a = SteeringVector(activations=v, layer_indices=[1], position_indices=[10])
    same = SteeringVector(activations=v.clone(), layer_indices=[1], position_indices=[10])
    assert K.payload_hash([a]) == K.payload_hash([same])
    variants = [
        SteeringVector(activations=v * 1.0001, layer_indices=[1], position_indices=[10]),
        SteeringVector(activations=v, layer_indices=[2], position_indices=[10]),
        SteeringVector(activations=v, layer_indices=[1], position_indices=[11]),
        SteeringVector(activations=v, layer_indices=[1], position_indices=[10], scale=2.0),
        SteeringVector(activations=v, layer_indices=[1], position_indices=[10], norm_match=True),
        SteeringVector(activations=v, layer_indices=[1], position_indices=[10], mode="replace"),
        SteeringVector(activations=v.to(torch.bfloat16), layer_indices=[1], position_indices=[10]),
    ]
    hashes = {K.payload_hash([x]) for x in variants}
    assert len(hashes) == len(variants) and K.payload_hash([a]) not in hashes
    assert K.steering_salt([a], "payload", "n")[1] == "p:" + K.payload_hash([a])


def test_plan_request_kv_policy():
    vec = [sv3d(1, [10])]
    # steering only: keep reading the cache, salt from the marker's predecessor
    assert K.plan_request_kv(vec, False, False, False, None, "x") == (False, [9, "n:x"])
    assert K.plan_request_kv(vec, False, False, False, "payload", "x")[1][1].startswith("p:")
    assert K.plan_request_kv(vec, False, False, False, "grp7", "x") == (False, [9, "u:grp7"])
    # capture / readout: the hooks must see every requested position -> skip reading
    assert K.plan_request_kv(None, True, False, False, None, "x") == (True, None)
    assert K.plan_request_kv(None, False, True, False, None, "x") == (True, None)
    # steering + capture: skip reading AND salt (the written blocks are steered)
    assert K.plan_request_kv(vec, True, False, False, None, "x") == (True, [9, "n:x"])
    # early exit: garbage KV above the exit layer -> salt everything (from token 0)
    assert K.plan_request_kv(None, False, True, True, None, "x") == (True, [0, "n:x"])
    assert K.plan_request_kv(vec, False, True, True, "payload", "x") == (True, [0, "n:x"])
    # a plain request needs nothing
    assert K.plan_request_kv(None, False, False, False, None, "x") == (False, None)


def test_plugin_apply_kv_policy_sets_params():
    from vllm_lens import _activations_plugin as P

    sp = SimpleNamespace(extra_args={"apply_steering_vectors": "popped"}, skip_reading_prefix_cache=None)
    P._apply_kv_policy(sp, [sv3d(1, [20])], False, None, None, None, "nonce1", (True, True), False)
    assert sp.skip_reading_prefix_cache is None and sp.extra_args[K.KV_SALT_KEY] == [19, "n:nonce1"]
    # prefix caching enabled but the scheduler lacks the patch -> pre-post7 behaviour (skip, no salt)
    sp = SimpleNamespace(extra_args={}, skip_reading_prefix_cache=None)
    P._apply_kv_policy(sp, [sv3d(1, [20])], False, None, None, None, "n", (True, False), False)
    assert sp.skip_reading_prefix_cache is True and K.KV_SALT_KEY not in sp.extra_args
    # prefix caching disabled: skip (a no-op for vLLM) and no salt
    sp = SimpleNamespace(extra_args={}, skip_reading_prefix_cache=None)
    P._apply_kv_policy(sp, [sv3d(1, [20])], False, None, None, None, "n", (False, False), False)
    assert sp.skip_reading_prefix_cache is True and K.KV_SALT_KEY not in sp.extra_args
    # VLLM_LENS_PREFIX_CACHE=0: salt for safety, but never read
    sp = SimpleNamespace(extra_args={}, skip_reading_prefix_cache=None)
    import os

    os.environ["VLLM_LENS_PREFIX_CACHE"] = "0"
    try:
        P._apply_kv_policy(sp, [sv3d(1, [20])], False, None, None, None, "n", (True, True), False)
    finally:
        del os.environ["VLLM_LENS_PREFIX_CACHE"]
    assert sp.skip_reading_prefix_cache is True and sp.extra_args[K.KV_SALT_KEY] == [19, "n:n"]
    # an unhooked request in the same call is left alone (unless the caller asked to skip)
    sp = SimpleNamespace(extra_args=None, skip_reading_prefix_cache=None)
    P._apply_kv_policy(sp, None, False, None, None, None, "n", (True, True), False)
    assert sp.skip_reading_prefix_cache is None and sp.extra_args is None
    P._apply_kv_policy(sp, None, False, None, None, None, "n", (True, True), True)
    assert sp.skip_reading_prefix_cache is True
    # effective capabilities: early exit needs the salt under prefix caching
    caps = P._effective_caps({"early_exit": True, "early_exit_reason": "ok"}, True, False)
    assert caps["early_exit"] is False and "salt" in caps["early_exit_reason"]
    assert P._effective_caps({"early_exit": True}, True, True)["early_exit"] is True
    assert P._effective_caps({"early_exit": True}, False, False)["early_exit"] is True


# ---------------------------------------------------------------------------
# engine-core patch on vLLM's real block hasher
# ---------------------------------------------------------------------------


def _real_vllm_hasher():
    ku = pytest.importorskip("vllm.v1.core.kv_cache_utils")
    req_mod = pytest.importorskip("vllm.v1.request")
    from vllm import SamplingParams

    assert K.install(), "patch did not install"
    assert K.is_installed()

    def sha(x):
        import pickle

        return hashlib.sha256(pickle.dumps(x)).digest()

    if hasattr(ku, "init_none_hash"):
        ku.init_none_hash(sha)  # module-level NONE_HASH the hasher expects the engine to have set
    hasher = ku.get_request_block_hasher(16, sha)

    def make(req_id: str, tokens: list[int], extra: dict | None) -> object:
        sp = SamplingParams(max_tokens=4, extra_args=extra)
        return req_mod.Request(
            request_id=req_id,
            prompt_token_ids=tokens,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )

    return make


def test_real_hasher_salts_from_the_marker_block_and_shares_before_it():
    make = _real_vllm_hasher()
    tokens = list(range(1000, 1096))  # 96 tokens = 6 full blocks of 16
    plain = make("plain", tokens, None)
    s1 = make("s1", tokens, {K.KV_SALT_KEY: [89, "n:one"]})
    s2 = make("s2", tokens, {K.KV_SALT_KEY: [89, "n:two"]})
    p1 = make("p1", tokens, {K.KV_SALT_KEY: [89, "p:deadbeef"]})
    p2 = make("p2", tokens, {K.KV_SALT_KEY: [89, "p:deadbeef"]})
    assert len(plain.block_hashes) == 6
    # blocks 0-4 (tokens 0..79) shared by everyone; block 5 (80..95, contains token 89) salted
    for i in range(5):
        assert plain.block_hashes[i] == s1.block_hashes[i] == s2.block_hashes[i] == p1.block_hashes[i]
    assert plain.block_hashes[5] != s1.block_hashes[5] != s2.block_hashes[5]
    assert s1.block_hashes[5] != plain.block_hashes[5] and s2.block_hashes[5] != plain.block_hashes[5]
    assert p1.block_hashes[5] == p2.block_hashes[5] != plain.block_hashes[5]  # payload tags share
    # generated tokens extend the salted chain (block 6 differs from a plain continuation)
    for r in (plain, s1):
        r.append_output_token_ids(list(range(16)))
    assert len(plain.block_hashes) == 7 and plain.block_hashes[6] != s1.block_hashes[6]
    # salt_from on a block boundary: token 80 -> block 5 salted, block 4 (64..79) shared
    b = make("b", tokens, {K.KV_SALT_KEY: [80, "n:b"]})
    assert b.block_hashes[4] == plain.block_hashes[4] and b.block_hashes[5] != plain.block_hashes[5]
    # salt_from 79 -> block 4 salted too
    c = make("c", tokens, {K.KV_SALT_KEY: [79, "n:c"]})
    assert c.block_hashes[3] == plain.block_hashes[3] and c.block_hashes[4] != plain.block_hashes[4]
    # from 0: nothing shared
    z = make("z", tokens, {K.KV_SALT_KEY: [0, "n:z"]})
    assert all(z.block_hashes[i] != plain.block_hashes[i] for i in range(6))


def test_real_hasher_payload_tag_falls_back_to_nonce_when_marker_is_the_last_prompt_token():
    make = _real_vllm_hasher()
    tokens = list(range(2000, 2097))  # 97 tokens: 6 full blocks, marker at 96 (last)
    a = make("a", tokens, {K.KV_SALT_KEY: [95, "p:same"]})
    b = make("b", tokens, {K.KV_SALT_KEY: [95, "p:same"]})
    plain = make("plain", tokens, None)
    # block 5 (80..95) contains token 95 -> salted; identical payload tags must NOT share here
    assert a.block_hashes[5] != plain.block_hashes[5]
    assert a.block_hashes[5] != b.block_hashes[5]
    # marker not last (position 90 of 97): payload tags do share
    c = make("c", tokens, {K.KV_SALT_KEY: [89, "p:same"]})
    d = make("d", tokens, {K.KV_SALT_KEY: [89, "p:same"]})
    assert c.block_hashes[5] == d.block_hashes[5] != plain.block_hashes[5]


def test_install_is_idempotent_and_registers_the_engine_core_utility():
    pytest.importorskip("vllm.v1.core.kv_cache_utils")
    from vllm.v1.core import kv_cache_utils as ku

    assert K.install() and K.install()
    fn = ku.generate_block_hash_extra_keys
    assert getattr(fn, "__wrapped__", None) is not None
    assert getattr(fn.__wrapped__, "__wrapped__", None) is None  # patched once
    from vllm.v1.engine.core import EngineCore

    assert EngineCore.lens_kv_salt_active(None) is True  # type: ignore[attr-defined]
    # client-side probe against a fake engine client
    llm = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(call_utility=lambda m: m == "lens_kv_salt_active")))
    assert K.scheduler_active_sync(llm) is True

    class _NoUtility:
        def call_utility(self, m):
            raise AttributeError(m)

    assert K.scheduler_active_sync(SimpleNamespace(llm_engine=SimpleNamespace(engine_core=_NoUtility()))) is False


def test_salt_key_for_block_rules():
    req = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={K.KV_SALT_KEY: [89, "n:x"]}), num_prompt_tokens=96, request_id="r")
    assert K._salt_key_for_block(req, 80) is None  # block ending at 80 <= salt_from 89
    assert K._salt_key_for_block(req, 96) == "vllm_lens:n:x"
    assert K._salt_key_for_block(SimpleNamespace(sampling_params=None), 96) is None
    assert K._salt_key_for_block(SimpleNamespace(sampling_params=SimpleNamespace(extra_args=None)), 96) is None


# ---------------------------------------------------------------------------
# worker side
# ---------------------------------------------------------------------------


def test_early_exit_under_prefix_caching_requires_salt_from_zero():
    import pickle

    ext = make_ext()
    ext._early_exit_ok = True
    ext._prefix_caching = True
    ext.set_readout_block(pickle.dumps({"keys": ["_read_0"], "vecs": torch.randn(1, D), "layers": [1]}))
    # unsalted early-exit request: refused (runs the full model), counted
    runner = FakeRunner([("0-aaaa1111", {"_readout_id": "_read_0", "lens_early_exit": True}, 4, 0, 4)])
    runner.requests["0-aaaa1111"].sampling_params.max_tokens = 1
    ext.model_runner = runner
    assert build_plan(ext, runner).exit_layer is None
    assert ext._stats["early_exit_refused_unsalted"] == 1
    # salted from token 0: allowed
    ext._req_plan_cache.clear()
    runner = FakeRunner([("1-bbbb2222", {"_readout_id": "_read_0", "lens_early_exit": True, K.KV_SALT_KEY: [0, "n:q"]}, 4, 0, 4)])
    runner.requests["1-bbbb2222"].sampling_params.max_tokens = 1
    ext.model_runner = runner
    assert build_plan(ext, runner).exit_layer == 1
    # salted from a later token is not enough
    ext._req_plan_cache.clear()
    runner = FakeRunner([("2-cccc3333", {"_readout_id": "_read_0", "lens_early_exit": True, K.KV_SALT_KEY: [3, "n:q"]}, 4, 0, 4)])
    runner.requests["2-cccc3333"].sampling_params.max_tokens = 1
    ext.model_runner = runner
    assert build_plan(ext, runner).exit_layer is None
    # without prefix caching no salt is needed (post4 behaviour)
    ext._prefix_caching = False
    ext._req_plan_cache.clear()
    runner = FakeRunner([("3-dddd4444", {"_readout_id": "_read_0", "lens_early_exit": True}, 4, 0, 4)])
    runner.requests["3-dddd4444"].sampling_params.max_tokens = 1
    ext.model_runner = runner
    assert build_plan(ext, runner).exit_layer == 1


def test_backstop_counts_unsalted_and_cache_served_markers():
    ext = make_ext()
    ext._prefix_caching = True
    store(ext, "_steer_0", [sv3d(1, [10])])
    store(ext, "_steer_1", [sv3d(1, [10])])
    store(ext, "_steer_2", [sv3d(1, [10])])
    runner = FakeRunner([
        # first chunk starts at token 0 (nothing cached): fine
        ("a-00000000", {"_steering_id": "_steer_0", K.KV_SALT_KEY: [9, "n:a"]}, 32, 0, 32),
        # starts at 16 (block 0 cached), marker 10 already "computed": the nonce salt did not
        # bite -> counted as a miss
        ("b-00000000", {"_steering_id": "_steer_1", K.KV_SALT_KEY: [9, "n:b"]}, 32, 16, 16),
        # no salt at all while prefix caching is on
        ("c-00000000", {"_steering_id": "_steer_2"}, 32, 0, 32),
    ])
    ext.model_runner = runner
    plan = build_plan(ext, runner)
    assert ext._stats["kv_salt_miss"] == 1 and ext._stats["kv_unsalted_steered"] == 1
    # payload tags may legitimately start past the marker (shared steered blocks)
    ext2 = make_ext()
    ext2._prefix_caching = True
    store(ext2, "_steer_0", [sv3d(1, [10])])
    runner = FakeRunner([("p-00000000", {"_steering_id": "_steer_0", K.KV_SALT_KEY: [9, "p:x"]}, 32, 16, 16)])
    ext2.model_runner = runner
    build_plan(ext2, runner)
    assert ext2._stats["kv_salt_miss"] == 0
    # the check runs once per request (re-resolutions after steering changes do not re-count)
    ext._steering_gen += 1
    build_plan(ext, runner if False else ext.model_runner)
    assert ext._stats["kv_salt_miss"] == 1
    assert plan is not None


def test_capabilities_report_prefix_caching_fields():
    ext = make_ext()
    ext._prefix_caching = True
    ext.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    ext._multi_stream = False
    caps = ext.lens_capabilities()
    assert caps["prefix_caching"] is True and "kv_salt_worker" in caps


# ---------------------------------------------------------------------------
# plugin: the offline generate never mutates the caller's SamplingParams (post7)
# ---------------------------------------------------------------------------


class _FakeWorkerLLM:
    """Records collective_rpc calls; pretends to be an LLM with prefix caching + salt patch."""

    def __init__(self):
        self.calls: list[tuple[str, Any]] = []
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=True)),
            engine_core=SimpleNamespace(call_utility=lambda m: m == "lens_kv_salt_active"),
        )

    def collective_rpc(self, name, args=(), kwargs=None):
        self.calls.append((name, args))
        if name == "lens_capabilities":
            return [{"early_exit": True, "hooks_installed": True, "readout": True}]
        return [None]


from typing import Any  # noqa: E402


def test_offline_generate_does_not_mutate_user_params_and_registers_every_call():
    import pickle

    from vllm_lens import _activations_plugin as P

    vec = sv3d(1, [20])
    user_sp = SimpleNamespace(extra_args={"apply_steering_vectors": [vec], "lens_cache_salt": "payload"},
                              max_tokens=4, skip_reading_prefix_cache=None)
    seen: list = []

    def fake_generate(self, prompts, sampling_params, **kw):
        seen.append(sampling_params)
        return [SimpleNamespace(request_id=str(i), prompt_token_ids=[0] * 8, outputs=[SimpleNamespace(token_ids=[1])])
                for i in range(len(sampling_params))]

    old = P._original_llm_generate
    P._original_llm_generate = fake_generate
    try:
        llm = _FakeWorkerLLM()
        for call in range(2):
            outs = P._patched_llm_generate(llm, ["p", "p"], [user_sp, user_sp])
            assert len(outs) == 2
            # the caller's object is untouched
            assert "apply_steering_vectors" in user_sp.extra_args and "lens_cache_salt" in user_sp.extra_args
            assert "_steering_id" not in user_sp.extra_args and K.KV_SALT_KEY not in user_sp.extra_args
            assert user_sp.skip_reading_prefix_cache is None
            # the copies vLLM received carry the plugin's keys and the payload salt (from token 19)
            sent = seen[-1]
            assert sent[0] is not user_sp and sent[0] is not sent[1]
            assert sent[0].extra_args["_steering_id"] == "_steer_0" and sent[1].extra_args["_steering_id"] == "_steer_1"
            assert sent[0].extra_args[K.KV_SALT_KEY][0] == 19 and sent[0].extra_args[K.KV_SALT_KEY][1].startswith("p:")
            assert "apply_steering_vectors" not in sent[0].extra_args
            assert sent[0].skip_reading_prefix_cache is None  # steering-only: keeps reading the cache
        blocks = [pickle.loads(a[0]) for n, a in llm.calls if n == "set_steering_block"]
        assert len(blocks) == 2 and all(b["keys"] == ["_steer_0", "_steer_1"] for b in blocks)  # both calls steered
        assert ("install_hooks", ()) in llm.calls
    finally:
        P._original_llm_generate = old
