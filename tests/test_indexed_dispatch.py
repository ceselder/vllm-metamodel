"""CPU-only tests for the indexed hook dispatch (no GPU, no vLLM engine).

Exercises the pure-Python pieces of ``_worker_ext`` against a tiny fake model runner:
prefix-key resolution must reproduce the previous ``startswith`` scan exactly, request
plans are cached until steering / hook data changes, a pass is flagged idle only when no
hook can have work, and under decode-only CUDA graphs generated positions are never
touched.

    pytest tests/test_indexed_dispatch.py --noconftest -p no:cacheprovider
"""

from __future__ import annotations

import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

try:  # ``_worker_ext`` imports two vLLM modules at import time; stub them without vLLM.
    import vllm.forward_context
    import vllm.model_executor.models.utils
except Exception:  # noqa: BLE001  # pragma: no cover
    _fc = types.ModuleType("vllm.forward_context")
    _fc._ctx = None
    _fc.get_forward_context = lambda: _fc._ctx
    _fc.is_forward_context_available = lambda: _fc._ctx is not None
    _utils = types.ModuleType("vllm.model_executor.models.utils")

    class PPMissingLayer(torch.nn.Module):
        pass

    _utils.PPMissingLayer = PPMissingLayer
    for name, mod in {
        "vllm": types.ModuleType("vllm"),
        "vllm.forward_context": _fc,
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.models": types.ModuleType("vllm.model_executor.models"),
        "vllm.model_executor.models.utils": _utils,
    }.items():
        sys.modules.setdefault(name, mod)

import vllm.forward_context as _fc_mod

from vllm_lens import _worker_ext as W
from vllm_lens._helpers.types import Hook, SteeringVector

if not hasattr(_fc_mod, "_ctx"):  # real vLLM installed: route the accessors through one attribute
    _fc_mod._ctx = None
    W.get_forward_context = lambda: _fc_mod._ctx
    W.is_forward_context_available = lambda: _fc_mod._ctx is not None

D = 8


def sv2d(layer: int) -> SteeringVector:
    return SteeringVector(activations=torch.randn(1, D), layer_indices=[layer])


def sv3d(layer: int, positions: list[int]) -> SteeringVector:
    return SteeringVector(activations=torch.randn(1, len(positions), D), layer_indices=[layer], position_indices=positions)


def legacy_scan(store: dict, internal_id: str, extra: dict | None, sentinel_key: str) -> list:
    results = []
    for k, cfgs in store.items():
        if internal_id.startswith(f"{k}-"):
            results.extend(cfgs)
    if extra:
        sid = extra.get(sentinel_key)
        if sid and sid in store:
            results.extend(store[sid])
    return results


class FakeModel(torch.nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.model = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Linear(D, D) for _ in range(n_layers)]))
        self.lin = torch.nn.Linear(D, D)


class FakeRunner:
    def __init__(self, rows):
        """rows: (req_id, extra_args, num_prompt, num_computed, n_query)."""
        self.requests = {}
        req_ids, qsl, ncts = [], [0], []
        for rid, extra, num_prompt, computed, n_query in rows:
            self.requests[rid] = SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra), num_prompt_tokens=num_prompt,
                                                 prompt_token_ids=[0] * num_prompt)
            req_ids.append(rid)
            qsl.append(qsl[-1] + n_query)
            ncts.append(computed)
        self.input_batch = SimpleNamespace(num_reqs=len(rows), req_ids=req_ids, num_computed_tokens_cpu=np.array(ncts, dtype=np.int32))
        self.query_start_loc = SimpleNamespace(np=np.array(qsl, dtype=np.int32))
        self.model = FakeModel()


def make_ext(prompt_only: bool = False) -> W.HiddenStatesExtension:
    ext = W.HiddenStatesExtension()
    state = {"_captured_states": {}, "_steering_data": {}, "_hook_data": {}, "_persistent_hooks": [], "_hook_contexts": {},
             "_persistent_hook_contexts": {}, "_steering_seq": {}, "_hook_seq": {}, "_seq_counter": 0, "_gen": 0,
             "_req_plan_cache": {}, "_step_plan": None, "_step_idle": False, "_idle_passes": 0, "_first_layer_idx": 0,
             "_prompt_only": prompt_only, "_should_capture": True, "_prefetched_params": {}}
    for attr, val in state.items():
        setattr(ext, attr, val)
    ext.model_runner = SimpleNamespace(model=FakeModel())
    return ext


def set_ctx(runner) -> None:
    meta = SimpleNamespace(query_start_loc=torch.tensor(runner.query_start_loc.np))
    _fc_mod._ctx = SimpleNamespace(attn_metadata={"layer0": meta})


@pytest.mark.parametrize(
    "internal_id, extra",
    [
        ("req", None),
        ("req-abcd1234", None),
        ("req-0-abcd1234", None),
        ("other-1", {"_steering_id": "_steer_3"}),
        ("12", {"_steering_id": "_steer_3"}),
        ("req-abcd1234", {"_steering_id": "req"}),
    ],
)
def test_indexed_resolution_matches_legacy_scan(internal_id, extra):
    ext = make_ext()
    for key in ("req", "req-0", "req-0-abcd1234", "other", "_steer_3", "zzz"):
        ext._steering_data[key] = [sv3d(1, [key.__len__()])]
        ext._bump(key, ext._steering_seq)
    assert W._find_steering_configs(ext, internal_id, extra) == legacy_scan(ext._steering_data, internal_id, extra, "_steering_id")
    for key in ("req", "other", "_hook_3"):
        ext._hook_data[key] = [Hook(fn=lambda ctx, h: None, layer_indices=[1])]
        ext._bump(key, ext._hook_seq)
    hextra = {"_hook_id": "_hook_3"} if extra else None
    assert W._find_hook_configs_no_persistent(ext, internal_id, hextra) == legacy_scan(ext._hook_data, internal_id, hextra, "_hook_id")


def test_request_plan_cached_and_invalidated_by_generation():
    ext = make_ext()
    runner = FakeRunner([("a-00000000", {"_steering_id": "_steer_0"}, 8, 0, 8)])
    ext.model_runner = runner
    ext._steering_data["_steer_0"] = [sv3d(1, [3])]
    ext._bump("_steer_0", ext._steering_seq)
    p1 = W._resolve_request(ext, runner, "a-00000000")
    assert p1 is not None and p1.steer_layers == frozenset({1}) and (p1.min_pos, p1.max_pos) == (3, 3) and not p1.broadcast
    assert W._resolve_request(ext, runner, "a-00000000") is p1  # cached
    ext._steering_data["_steer_0"] = [sv2d(2)]
    ext._bump("_steer_0", ext._steering_seq)
    p2 = W._resolve_request(ext, runner, "a-00000000")
    assert p2 is not p1 and p2.broadcast and p2.steer_layers == frozenset({2})


def test_step_plan_uses_host_buffers_and_idle_detection():
    ext = make_ext()
    ext._steering_data["_steer_0"] = [sv3d(1, [10])]
    ext._bump("_steer_0", ext._steering_seq)
    # prefill: row 0 covers positions 0..15 (marker 10 inside), row 1 is a plain request
    runner = FakeRunner([("a-00000000", {"_steering_id": "_steer_0"}, 16, 0, 16), ("b-00000000", None, 16, 0, 16)])
    ext.model_runner = runner
    set_ctx(runner)
    W._begin_pass(ext)
    assert not ext._step_idle
    plan = ext._step_plan
    assert plan is not None and plan.qsl == [0, 16, 32] and plan.abs_start == [0, 0]
    # decode step: every row is past the marker -> idle
    runner = FakeRunner([("a-00000000", {"_steering_id": "_steer_0"}, 16, 20, 1), ("b-00000000", None, 16, 20, 1)])
    ext.model_runner = runner
    set_ctx(runner)
    W._begin_pass(ext)
    assert ext._step_idle and ext._idle_passes == 1
    assert W._hook_inner(ext, 1, torch.zeros(2, D)) is None  # returns on the flag
    # a broadcast vector, a capture request or a persistent hook keep the pass live
    ext._steering_data["_steer_0"] = [sv2d(1)]
    ext._bump("_steer_0", ext._steering_seq)
    W._begin_pass(ext)
    assert not ext._step_idle
    ext._steering_data["_steer_0"] = [sv3d(1, [10])]
    ext._bump("_steer_0", ext._steering_seq)
    runner = FakeRunner([("c-00000000", {"output_residual_stream": [2]}, 16, 20, 1)])
    ext.model_runner = runner
    set_ctx(runner)
    W._begin_pass(ext)
    assert not ext._step_idle
    ext._persistent_hooks = [Hook(fn=lambda ctx, h: None, layer_indices=[0])]
    ext._bump()
    runner = FakeRunner([("a-00000000", {"_steering_id": "_steer_0"}, 16, 20, 1)])
    ext.model_runner = runner
    set_ctx(runner)
    W._begin_pass(ext)
    assert not ext._step_idle


def test_steering_applied_once_at_marker_and_prompt_only_skips_generated_rows():
    torch.manual_seed(0)
    for prompt_only in (False, True):
        ext = make_ext(prompt_only=prompt_only)
        vec = sv3d(1, [10])
        ext._steering_data["_steer_0"] = [vec]
        ext._bump("_steer_0", ext._steering_seq)
        # row 0: prefill chunk covering the marker; row 1: same key but already generating (a0 = 20)
        runner = FakeRunner([("a-00000000", {"_steering_id": "_steer_0"}, 16, 0, 16), ("b-00000000", {"_steering_id": "_steer_0"}, 16, 20, 1)])
        ext.model_runner = runner
        set_ctx(runner)
        W._begin_pass(ext)
        h = torch.zeros(17, D)
        out = W._hook_inner(ext, 1, h)
        assert out is not None
        delta = out - h
        assert torch.allclose(delta[10], vec.activations[0, 0]) and delta[:10].abs().sum() == 0 and delta[11:16].abs().sum() == 0
        assert delta[16].abs().sum() == 0  # row 1's single generated token is never a marker position
        assert W._hook_inner(ext, 2, torch.zeros(17, D)) is None  # layer without vectors: untouched
    # a 2-D vector for a decode row is applied eagerly, but refused under graphs
    ext = make_ext(prompt_only=True)
    runner = FakeRunner([("a-00000000", {"_steering_id": "_steer_0"}, 16, 20, 1)])
    ext.model_runner = runner
    set_ctx(runner)
    ext._steering_data["_steer_0"] = [sv2d(1)]
    ext._bump("_steer_0", ext._steering_seq)
    W._begin_pass(ext)
    assert W._hook_inner(ext, 1, torch.zeros(1, D)) is None  # generated row skipped under graphs
    ext.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    with pytest.raises(ValueError, match="2-D"):
        ext.set_steering_data("k", pickle.dumps([sv2d(1)]))


def test_capture_and_hooks_follow_the_plan():
    ext = make_ext()
    seen: list[tuple[int, int]] = []

    def fn(ctx, h):
        seen.append((ctx.layer_idx, int(h.shape[0])))

    ext._hook_data["_hook_0"] = [Hook(fn=fn, layer_indices=[2])]
    ext._bump("_hook_0", ext._hook_seq)
    runner = FakeRunner([("a-00000000", {"output_residual_stream": [1], "_hook_id": "_hook_0"}, 4, 0, 4), ("b-00000000", None, 3, 0, 3)])
    ext.model_runner = runner
    set_ctx(runner)
    W._begin_pass(ext)
    assert not ext._step_idle
    h = torch.randn(7, D)
    assert W._hook_inner(ext, 1, h) is None
    assert torch.equal(ext._captured_states["a-00000000"][1][0], h[:4]) and "b-00000000" not in ext._captured_states
    W._hook_inner(ext, 2, h)
    assert seen == [(2, 4)]  # per-request hook ran once, at its layer, on its rows only
    assert 2 not in ext._captured_states["a-00000000"]
