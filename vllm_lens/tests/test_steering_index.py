"""CPU-only tests for the indexed steering path in ``_worker_ext`` (vllm-lens-metamodel).

No GPU or vLLM engine needed: these exercise the pure-Python / CPU-tensor
pieces the forward hook relies on (prefix-key enumeration, per-key indexing,
per-request resolution order, per-pass planning, the idle fast path, the
vectorised apply, the block RPC) against a tiny fake model runner, and check
they reproduce the 1.1.0 ``startswith``-scan / ``_apply_steering`` semantics.

Run without upstream's GPU conftest:  pytest vllm_lens/tests/test_steering_index.py --noconftest
"""

from __future__ import annotations

import pickle
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# ``_worker_ext`` imports two vLLM modules at import time; stub them when vLLM
# is not installed (CI / laptop) so the pure-Python logic is still testable.
try:  # pragma: no cover - depends on environment
    import vllm.forward_context  # noqa: F401
    import vllm.model_executor.models.utils  # noqa: F401
except Exception:  # pragma: no cover
    _fc = types.ModuleType("vllm.forward_context")
    _fc._ctx = None

    def _get_forward_context():
        return _fc._ctx

    def _is_forward_context_available():
        return _fc._ctx is not None

    _fc.get_forward_context = _get_forward_context
    _fc.is_forward_context_available = _is_forward_context_available
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

from vllm_lens import SteeringVector  # noqa: E402
from vllm_lens import _worker_ext as W  # noqa: E402

D = 8


def sv2d(layer: int, norm_match: bool = False, scale: float = 1.0) -> SteeringVector:
    return SteeringVector(
        activations=torch.randn(1, D),
        layer_indices=[layer],
        norm_match=norm_match,
        scale=scale,
    )


def sv3d(
    layer: int, positions: list[int] | None, n_pos: int | None = None, **kw
) -> SteeringVector:
    n = n_pos if n_pos is not None else len(positions or [])
    return SteeringVector(
        activations=torch.randn(1, n, D),
        layer_indices=[layer],
        position_indices=positions,
        **kw,
    )


def legacy_find(steering_data: dict, internal_req_id: str, extra_args: dict | None):
    """Verbatim 1.1.0 ``_find_steering_configs`` (reference implementation)."""
    results = []
    for external_id, configs in steering_data.items():
        if internal_req_id.startswith(f"{external_id}-"):
            results.extend(configs)
    if extra_args:
        steering_id = extra_args.get("_steering_id")
        if steering_id and steering_id in steering_data:
            results.extend(steering_data[steering_id])
    return results


class FakeModel(torch.nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList([torch.nn.Linear(D, D) for _ in range(n_layers)])
        )
        self.lin = torch.nn.Linear(D, D)  # parameters() -> device/dtype


def make_ext(
    prompt_only: bool = False, vectorized: bool = True
) -> W.HiddenStatesExtension:
    ext = W.HiddenStatesExtension()
    ext._steering_data = {}
    ext._steering_index = {}
    ext._steering_gen = 0
    ext._steering_seq = 0
    ext._req_plan_cache = {}
    ext._capture_live = set()
    ext._step_plan = None
    ext._step_idle = False
    ext._agg_gen = -1
    ext._stats = W._new_stats()
    ext._should_capture = True
    ext._prompt_only = prompt_only
    ext._vectorized = vectorized
    ext.model_runner = SimpleNamespace(model=FakeModel())
    return ext


def store(
    ext: W.HiddenStatesExtension, key: str, configs: list[SteeringVector]
) -> None:
    ext._store(key, configs)
    ext._steering_gen += 1


class FakeRunner:
    def __init__(self, rows):
        """rows: list of (req_id, extra_args, num_prompt, num_computed, n_query)."""
        self.requests = {}
        req_ids, qsl, ncts = [], [0], []
        for rid, extra, num_prompt, computed, n_query in rows:
            sp = SimpleNamespace(extra_args=extra)
            self.requests[rid] = SimpleNamespace(
                sampling_params=sp,
                num_prompt_tokens=num_prompt,
                prompt_token_ids=[0] * num_prompt,
            )
            req_ids.append(rid)
            qsl.append(qsl[-1] + n_query)
            ncts.append(computed)
        self.input_batch = SimpleNamespace(
            num_reqs=len(rows),
            req_ids=req_ids,
            num_computed_tokens_cpu=np.array(ncts, dtype=np.int32),
        )
        self.query_start_loc = SimpleNamespace(np=np.array(qsl, dtype=np.int32))
        self.model = FakeModel()


def build_plan(ext, runner):
    import vllm.forward_context as fc

    meta = SimpleNamespace(query_start_loc=torch.tensor(runner.query_start_loc.np))
    ctx = SimpleNamespace(attn_metadata={"layer0": meta})
    if hasattr(fc, "_ctx"):
        fc._ctx = ctx
        return W._build_step_plan(ext, runner, runner.input_batch.num_reqs)
    orig = W.get_forward_context  # real vLLM installed: patch the accessor
    W.get_forward_context = lambda: ctx
    try:
        return W._build_step_plan(ext, runner, runner.input_batch.num_reqs)
    finally:
        W.get_forward_context = orig


# ---------------------------------------------------------------------------
# resolution == 1.1.0 scan
# ---------------------------------------------------------------------------


def test_prefix_keys_enumerates_dash_boundaries():
    assert list(W._prefix_keys("req-0-a1b2c3d4")) == ["req", "req-0"]
    assert list(W._prefix_keys("cmpl-abc-3-deadbeef")) == [
        "cmpl",
        "cmpl-abc",
        "cmpl-abc-3",
    ]
    assert list(W._prefix_keys("nodash")) == []


@pytest.mark.parametrize(
    "internal_id,extra",
    [
        ("req-0-a1b2c3d4", None),
        ("req-00-b5c6d7e8", None),  # must NOT match key "req-0"
        ("req-0-a1b2c3d4", {"_steering_id": "_steer_3"}),
        ("7-deadbeef", {"_steering_id": "_steer_3"}),
        ("7-deadbeef", {"_steering_id": "missing"}),
        (
            "cmpl-x-1-ffffffff",
            None,
        ),  # two matching prefix keys -> both, insertion order
        ("plain", {"_steering_id": "_steer_3"}),
    ],
)
def test_resolution_matches_legacy_scan(internal_id, extra):
    ext = make_ext()
    keys = {
        "req-0": [sv2d(1)],
        "req-00": [sv2d(2)],
        "_steer_3": [sv3d(4, [5])],
        "cmpl-x-1": [sv2d(6)],
        "cmpl-x": [sv2d(7)],
    }
    for k, v in keys.items():
        store(ext, k, v)
    got = [c for e in W._resolve_entries(ext, internal_id, extra) for c in e.configs]
    want = legacy_find(ext._steering_data, internal_id, extra)
    assert [id(c) for c in got] == [id(c) for c in want]
    # the verbatim 1.1.0 function still works on the same data
    assert [id(c) for c in W._find_steering_configs(ext, internal_id, extra)] == [
        id(c) for c in want
    ]


def test_index_summarises_layers_broadcast_and_positions():
    e = W._index_configs([sv2d(3), sv3d(5, [10, 12]), sv3d(9, None, n_pos=4)], seq=1)
    assert e.layers == frozenset({3, 5, 9})
    assert e.broadcast is True
    assert (e.min_pos, e.max_pos) == (0, 12)
    e2 = W._index_configs([sv3d(9, None, n_pos=4)], seq=2)
    assert e2.broadcast is False and (e2.min_pos, e2.max_pos) == (0, 3)
    # position_indices longer than n_pos: only the first n_pos entries can apply
    e3 = W._index_configs([sv3d(1, [2, 99], n_pos=1)], seq=3)
    assert (e3.min_pos, e3.max_pos) == (2, 2)


def test_clear_and_many_rpcs_keep_index_in_sync():
    ext = make_ext()
    store(ext, "a", [sv2d(1)])
    store(ext, "b", [sv2d(1)])
    assert set(ext._steering_index) == {"a", "b"}
    gen = ext._steering_gen
    ext.clear_steering_data_many(["a", "b", "nope"])
    assert ext._steering_index == {} and ext._steering_data == {}
    assert ext._steering_gen == gen + 1


# ---------------------------------------------------------------------------
# per-pass planning
# ---------------------------------------------------------------------------


def test_step_plan_schedules_only_rows_a_vector_can_touch():
    ext = make_ext()
    store(ext, "_steer_0", [sv3d(1, [10])])  # prompt position 10 only
    store(ext, "_steer_1", [sv2d(5)])  # broadcast at layer 5
    runner = FakeRunner(
        [
            (
                "0-aaaaaaaa",
                {"_steering_id": "_steer_0"},
                96,
                0,
                96,
            ),  # prefill: position 10 computed
            (
                "1-bbbbbbbb",
                {"_steering_id": "_steer_0"},
                96,
                96,
                1,
            ),  # decode of the 3-D request: nothing
            (
                "2-cccccccc",
                {"_steering_id": "_steer_1"},
                96,
                96,
                1,
            ),  # decode of the broadcast request
            ("3-dddddddd", {"output_residual_stream": [5, 7]}, 96, 96, 1),
            (
                "4-eeeeeeee",
                {"_steering_id": "_steer_0"},
                96,
                0,
                8,
            ),  # first prefill CHUNK 0..7: marker not inside
            (
                "5-ffffffff",
                {"_steering_id": "_steer_0"},
                96,
                8,
                8,
            ),  # second chunk 8..15: marker inside
        ]
    )
    plan = build_plan(ext, runner)
    assert plan is not None
    assert plan.qsl == [0, 96, 97, 98, 99, 107, 115]
    assert plan.abs_start == [0, 96, 96, 96, 0, 8]
    assert set(plan.steer) == {1, 5}
    assert [i for i, _ in plan.steer[1]] == [0, 5]
    assert [i for i, _ in plan.steer[5]] == [2]
    assert plan.capture_rows(5) == [3] and plan.capture_rows(7) == [3]
    assert plan.capture_rows(1) == [] and plan.capture_rows(6) == []
    plan2 = build_plan(ext, runner)  # per-request cache reused
    assert plan2.steer.keys() == plan.steer.keys()
    assert ext._stats["steps_planned"] == 2


def test_pure_decode_pass_with_prompt_only_vectors_is_idle_and_fast_path_agrees():
    ext = make_ext()
    store(ext, "_steer_0", [sv3d(1, [10])])
    runner = FakeRunner(
        [
            (f"{i}-aaaaaaaa", {"_steering_id": "_steer_0"}, 96, 96 + i, 1)
            for i in range(4)
        ]
    )
    plan = build_plan(ext, runner)
    assert plan.steer == {} and plan.cap_all == [] and plan.cap_by_layer == {}
    assert ext._stats["steps_idle"] == 1
    assert W._step_is_idle(ext, runner, 4) is True
    # a broadcast key anywhere disables the fast path
    store(ext, "_steer_9", [sv2d(2)])
    assert W._step_is_idle(ext, runner, 4) is False
    ext.clear_steering_data("_steer_9")
    # a positional key at/after some row's position disables it
    store(ext, "_steer_9", [sv3d(2, [97])])
    assert W._step_is_idle(ext, runner, 4) is False
    ext.clear_steering_data("_steer_9")
    # a prefill row (more than one token) disables it
    runner2 = FakeRunner([("0-aaaaaaaa", {"_steering_id": "_steer_0"}, 96, 0, 96)])
    assert W._step_is_idle(ext, runner2, 1) is False
    # a brand-new 1-token request (num_computed == 0) must be planned, not skipped
    runner3 = FakeRunner([("0-aaaaaaaa", {"output_residual_stream": True}, 1, 0, 1)])
    assert W._step_is_idle(ext, runner3, 1) is False
    # capture in flight disables it until the request is gone
    build_plan(ext, runner3)
    assert "0-aaaaaaaa" in ext._capture_live
    runner4 = FakeRunner([("1-bbbbbbbb", {"_steering_id": "_steer_0"}, 96, 96, 1)])
    assert W._step_is_idle(ext, runner4, 1) is True  # stale capture entry pruned
    assert ext._capture_live == set()


def test_prompt_only_mode_never_touches_generated_rows():
    ext = make_ext(prompt_only=True)
    store(ext, "_steer_0", [sv3d(1, [10, 200])])  # 200 would be a generated position
    runner = FakeRunner(
        [
            ("0-aaaaaaaa", {"_steering_id": "_steer_0"}, 96, 0, 96),  # prefill
            ("1-bbbbbbbb", {"_steering_id": "_steer_0"}, 96, 199, 1),  # generated row
            (
                "2-cccccccc",
                {"output_residual_stream": True},
                96,
                100,
                1,
            ),  # generated row
        ]
    )
    plan = build_plan(ext, runner)
    assert [i for i, _ in plan.steer[1]] == [0]
    assert plan.cap_all == []
    assert ext._stats["rows_skipped_generated"] == 2


def test_steering_gen_invalidates_cached_request_plans():
    ext = make_ext()
    store(ext, "_steer_0", [sv2d(1)])
    runner = FakeRunner([("0-aaaaaaaa", {"_steering_id": "_steer_0"}, 4, 4, 1)])
    assert set(build_plan(ext, runner).steer) == {1}
    ext.clear_steering_data("_steer_0")
    assert build_plan(ext, runner).steer == {}
    store(ext, "_steer_0", [sv2d(2)])
    assert set(build_plan(ext, runner).steer) == {2}


def test_capture_all_layers_merges_with_per_layer_rows():
    ext = make_ext()
    runner = FakeRunner(
        [
            ("0-aaaaaaaa", {"output_residual_stream": True}, 4, 4, 1),
            ("1-bbbbbbbb", {"output_residual_stream": [3]}, 4, 4, 1),
        ]
    )
    plan = build_plan(ext, runner)
    assert plan.capture_rows(3) == [0, 1]
    assert plan.capture_rows(2) == [0]


# ---------------------------------------------------------------------------
# vectorised apply == sequential _apply_steering
# ---------------------------------------------------------------------------


def _apply_both(todo, layer_idx, target, qsl, abs_start):
    plan = W._StepPlan(qsl, abs_start, {layer_idx: todo})
    seq = target.clone()
    for i, configs in todo:
        W._apply_steering(configs, layer_idx, seq, qsl[i], qsl[i + 1], abs_start[i])
    vec = target.clone()
    ok = W._apply_layer_vectorized(todo, layer_idx, vec, plan)
    return ok, seq, vec


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vectorized_matches_sequential_bitwise_without_norm_match(dtype):
    torch.manual_seed(0)
    n_reqs, P = 6, 12
    configs = [
        [sv3d(1, [3], scale=1.0)] for _ in range(n_reqs)
    ]  # one marker vector per request
    for cs in configs:
        cs[0].activations = cs[0].activations.to(dtype)
    qsl = [P * i for i in range(n_reqs + 1)]
    abs_start = [0] * n_reqs
    target = (torch.randn(P * n_reqs, D) * 5).to(dtype)
    todo = list(zip(range(n_reqs), configs))
    ok, seq, vec = _apply_both(todo, 1, target, qsl, abs_start)
    assert ok
    assert torch.equal(seq, vec)
    # rows other than the markers untouched
    marks = {qsl[i] + 3 for i in range(n_reqs)}
    keep = [r for r in range(P * n_reqs) if r not in marks]
    assert torch.equal(vec[keep], target[keep])
    # a decode pass: one token per row, mixed scales, broadcast + positional
    configs2 = [[sv2d(1, scale=0.5)], [sv3d(1, [P], scale=2.0)], [sv2d(1, scale=0.5)]]
    for cs in configs2:
        cs[0].activations = cs[0].activations.to(dtype)
    ok, seq, vec = _apply_both(
        list(zip(range(3), configs2)),
        1,
        (torch.randn(3, D) * 5).to(dtype),
        [0, 1, 2, 3],
        [P, P, P],
    )
    assert ok and torch.equal(seq, vec)


def test_vectorized_norm_match_close_to_sequential():
    torch.manual_seed(1)
    configs = [[sv3d(2, [1], norm_match=True, scale=0.7)] for _ in range(5)]
    qsl = [4 * i for i in range(6)]
    target = torch.randn(20, D) * 3
    ok, seq, vec = _apply_both(list(zip(range(5), configs)), 2, target, qsl, [0] * 5)
    assert ok
    assert torch.allclose(seq, vec, rtol=1e-5, atol=1e-5)


def test_vectorized_falls_back_when_semantics_would_differ():
    torch.manual_seed(2)
    # two vectors on the same row -> sequential order/rounding must be kept
    configs = [[sv3d(1, [2]), sv3d(1, [2])]]
    target = torch.randn(8, D)
    plan = W._StepPlan([0, 8], [0], {1: [(0, configs[0])]})
    assert (
        W._apply_layer_vectorized([(0, configs[0])], 1, target.clone(), plan) is False
    )
    # broadcast over a multi-token chunk -> fallback
    plan = W._StepPlan([0, 8], [0], {1: [(0, [sv2d(1)])]})
    assert W._apply_layer_vectorized([(0, [sv2d(1)])], 1, target.clone(), plan) is False
    # nothing in range (later chunk) -> handled (True) and target untouched
    t = target.clone()
    plan = W._StepPlan([0, 8], [50], {1: [(0, [sv3d(1, [2])])]})
    assert W._apply_layer_vectorized([(0, [sv3d(1, [2])])], 1, t, plan) is True
    assert torch.equal(t, target)


def test_apply_steering_unchanged_semantics():
    """The steering arithmetic is untouched: check 2-D + norm_match and 3-D positions."""
    target = torch.zeros(6, D) + 2.0  # rows 0..5, each norm sqrt(D)*2
    cfg2 = SteeringVector(
        activations=torch.ones(1, D), layer_indices=[0], norm_match=True, scale=0.5
    )
    W._apply_steering([cfg2], 0, target, 0, 2, abs_start=0)
    # norm_match: v scaled to ||row|| then * 0.5 -> each element 2 + 0.5*(2*sqrt(D))/sqrt(D) = 3
    assert torch.allclose(target[0:2], torch.full((2, D), 3.0))
    assert torch.allclose(target[2:], torch.full((4, D), 2.0))
    cfg3 = SteeringVector(
        activations=torch.ones(1, 2, D) * 10, layer_indices=[0], position_indices=[7, 9]
    )
    tgt = torch.zeros(6, D)
    W._apply_steering(
        [cfg3], 0, tgt, 1, 5, abs_start=6
    )  # rows 1..4 hold positions 6..9
    assert tgt[2].sum() == 10 * D and tgt[4].sum() == 10 * D
    assert tgt[[0, 1, 3, 5]].abs().sum() == 0


# ---------------------------------------------------------------------------
# block RPC == per-key set_steering_data
# ---------------------------------------------------------------------------


def test_set_steering_block_equals_per_key_vectors():
    torch.manual_seed(3)
    ext_a, ext_b = make_ext(), make_ext()
    n = 5
    vecs = torch.randn(n, D)
    layers = [1, 2, 1, 3, 0]
    positions = [4, 7, 4, 9, 0]
    for i in range(n):
        ext_a.set_steering_data(
            f"_steer_{i}",
            pickle.dumps(
                [
                    SteeringVector(
                        activations=vecs[i].view(1, 1, D),
                        layer_indices=[layers[i]],
                        scale=1.5,
                        norm_match=False,
                        position_indices=[positions[i]],
                    )
                ]
            ),
        )
    ext_b.set_steering_block(
        pickle.dumps(
            {
                "keys": [f"_steer_{i}" for i in range(n)],
                "vecs": vecs,
                "layers": layers,
                "positions": positions,
                "scales": [1.5] * n,
                "norm_match": [False] * n,
            }
        )
    )
    assert set(ext_a._steering_index) == set(ext_b._steering_index)
    for k in ext_a._steering_index:
        ea, eb = ext_a._steering_index[k], ext_b._steering_index[k]
        assert (ea.layers, ea.broadcast, ea.min_pos, ea.max_pos) == (
            eb.layers,
            eb.broadcast,
            eb.min_pos,
            eb.max_pos,
        )
        a, b = ea.configs[0], eb.configs[0]
        assert torch.equal(a.activations, b.activations) and a.activations.shape == (
            1,
            1,
            D,
        )
        assert (a.layer_indices, a.scale, a.norm_match, a.position_indices) == (
            b.layer_indices,
            b.scale,
            b.norm_match,
            b.position_indices,
        )
    # the same forward pass steers identically
    rows = [
        (f"{i}-aaaaaaaa", {"_steering_id": f"_steer_{i}"}, 12, 0, 12) for i in range(n)
    ]
    ra, rb = FakeRunner(rows), FakeRunner(rows)
    pa, pb = build_plan(ext_a, ra), build_plan(ext_b, rb)
    target = torch.randn(12 * n, D)
    ta, tb = target.clone(), target.clone()
    for L in (0, 1, 2, 3):
        if L in pa.steer:
            assert W._apply_layer_vectorized(pa.steer[L], L, ta, pa)
            assert W._apply_layer_vectorized(pb.steer[L], L, tb, pb)
    assert torch.equal(ta, tb)
    assert not torch.equal(ta, target)
    with pytest.raises(ValueError):
        ext_b.set_steering_block(
            pickle.dumps(
                {
                    "keys": ["x"],
                    "vecs": vecs[:1],
                    "layers": [99],
                    "positions": [0],
                    "scales": [1.0],
                    "norm_match": [False],
                }
            )
        )


def test_prompt_only_rejects_broadcast_vectors():
    ext = make_ext(prompt_only=True)
    with pytest.raises(ValueError):
        ext.set_steering_data("k", pickle.dumps([sv2d(1)]))
    ext.set_steering_data("k", pickle.dumps([sv3d(1, [3])]))  # positional is fine
    assert "k" in ext._steering_index


# ---------------------------------------------------------------------------
# embedding replacement (EMBED_LAYER_INDEX + mode="replace")
# ---------------------------------------------------------------------------


def test_replace_mode_requires_positional_activations():
    with pytest.raises(ValueError, match="replace.*3D"):
        SteeringVector(
            activations=torch.randn(1, D), layer_indices=[0], mode="replace"
        )


def test_step_plan_schedules_embed_layer_rows():
    ext = make_ext()
    store(ext, "nla", [sv3d(W.EMBED_LAYER_INDEX, [5], mode="replace", scale=3.0)])
    runner = FakeRunner(
        [
            ("nla-aaaa1111", None, 8, 0, 8),  # prefill covering the marker
            ("other-bbbb2222", None, 8, 8, 1),  # decode row, no configs
        ]
    )
    plan = build_plan(ext, runner)
    assert plan is not None
    todo = plan.steer.get(W.EMBED_LAYER_INDEX)
    assert todo is not None and [i for i, _ in todo] == [0]


def test_apply_embed_replaces_marker_row_and_leaves_others():
    ext = make_ext()
    v = torch.randn(D)
    sv = SteeringVector(
        activations=v.reshape(1, 1, D),
        layer_indices=[W.EMBED_LAYER_INDEX],
        position_indices=[5],
        mode="replace",
        scale=2.5,
    )
    store(ext, "nla", [sv])
    runner = FakeRunner([("nla-aaaa1111", None, 8, 0, 8)])
    plan = build_plan(ext, runner)
    todo = plan.steer[W.EMBED_LAYER_INDEX]
    target = torch.randn(8, D)
    orig = target.clone()
    W._apply_embed(todo, target, plan, ext._stats)
    assert torch.allclose(target[5], v * 2.5)
    keep = [r for r in range(8) if r != 5]
    assert torch.equal(target[keep], orig[keep])
    assert ext._stats["rows_replaced"] == 1


def test_apply_embed_norm_match_scales_to_original_row_norm():
    ext = make_ext()
    v = torch.randn(D)
    sv = SteeringVector(
        activations=v.reshape(1, 1, D),
        layer_indices=[W.EMBED_LAYER_INDEX],
        position_indices=[2],
        mode="replace",
        norm_match=True,
    )
    store(ext, "k", [sv])
    runner = FakeRunner([("k-aaaa1111", None, 4, 0, 4)])
    plan = build_plan(ext, runner)
    target = torch.randn(4, D)
    orig_norm = target[2].norm()
    W._apply_embed(plan.steer[W.EMBED_LAYER_INDEX], target, plan, ext._stats)
    assert torch.allclose(target[2].norm(), orig_norm, rtol=1e-3)
    assert torch.allclose(
        torch.nn.functional.cosine_similarity(target[2], v, dim=0),
        torch.tensor(1.0),
        atol=1e-5,
    )


def test_apply_embed_chunked_prefill_offsets():
    """Marker in the SECOND prefill chunk: only that chunk's pass writes it."""
    ext = make_ext()
    v = torch.randn(D)
    sv = SteeringVector(
        activations=v.reshape(1, 1, D),
        layer_indices=[W.EMBED_LAYER_INDEX],
        position_indices=[10],
        mode="replace",
    )
    store(ext, "k", [sv])
    # chunk 1: rows 0-7 (abs 0..7) — marker abs 10 NOT in range
    r1 = FakeRunner([("k-aaaa1111", None, 16, 0, 8)])
    p1 = build_plan(ext, r1)
    t1 = torch.randn(8, D)
    o1 = t1.clone()
    if p1.steer.get(W.EMBED_LAYER_INDEX):
        W._apply_embed(p1.steer[W.EMBED_LAYER_INDEX], t1, p1, ext._stats)
    assert torch.equal(t1, o1)
    # chunk 2: rows abs 8..15 — marker abs 10 = local row 2
    ext._req_plan_cache = {}
    r2 = FakeRunner([("k-aaaa1111", None, 16, 8, 8)])
    p2 = build_plan(ext, r2)
    t2 = torch.randn(8, D)
    W._apply_embed(p2.steer[W.EMBED_LAYER_INDEX], t2, p2, ext._stats)
    assert torch.allclose(t2[2], v)


def test_vectorized_layer_apply_defers_replace_to_sequential():
    ext = make_ext()
    sv = sv3d(1, [0], mode="replace")
    store(ext, "k", [sv])
    runner = FakeRunner([("k-aaaa1111", None, 4, 0, 4)])
    plan = build_plan(ext, runner)
    target = torch.randn(4, D)
    ok = W._apply_layer_vectorized(plan.steer[1], 1, target, plan)
    assert ok is False  # sequential path must handle replace semantics
