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
    ext._captured_states = {}
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
    ext._fast_capture = True
    ext._cap_blocks = []
    ext._read_blocks = []
    ext._captured_positions = {}
    ext._readout_index = {}
    ext._readouts = {}
    ext._early_exit_ok = False
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


def _apply_both(todo, layer_idx, target, qsl, abs_start, residual=None):
    """Run the sequential and the vectorised apply on copies; returns
    (ok, seq_target, vec_target[, seq_residual, vec_residual])."""
    plan = W._StepPlan(qsl, abs_start, {layer_idx: todo})
    seq = target.clone()
    res_seq = residual.clone() if residual is not None else None
    for i, configs in todo:
        W._apply_steering(
            configs, layer_idx, seq, qsl[i], qsl[i + 1], abs_start[i], res_seq
        )
    vec = target.clone()
    res_vec = residual.clone() if residual is not None else None
    ok = W._apply_layer_vectorized(todo, layer_idx, vec, plan, res_vec)
    if residual is None:
        return ok, seq, vec
    return ok, seq, vec, res_seq, res_vec


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
        W._apply_layer_vectorized([(0, configs[0])], 1, target.clone(), plan, None)
        is False
    )
    # broadcast over a multi-token chunk -> fallback
    plan = W._StepPlan([0, 8], [0], {1: [(0, [sv2d(1)])]})
    assert (
        W._apply_layer_vectorized([(0, [sv2d(1)])], 1, target.clone(), plan, None)
        is False
    )
    # nothing in range (later chunk) -> handled (True) and target untouched
    t = target.clone()
    plan = W._StepPlan([0, 8], [50], {1: [(0, [sv3d(1, [2])])]})
    assert W._apply_layer_vectorized([(0, [sv3d(1, [2])])], 1, t, plan, None) is True
    assert torch.equal(t, target)


def test_apply_steering_unchanged_semantics():
    """The steering arithmetic is untouched: check 2-D + norm_match and 3-D positions."""
    target = torch.zeros(6, D) + 2.0  # rows 0..5, each norm sqrt(D)*2
    cfg2 = SteeringVector(
        activations=torch.ones(1, D), layer_indices=[0], norm_match=True, scale=0.5
    )
    W._apply_steering([cfg2], 0, target, 0, 2, abs_start=0, residual=None)
    # norm_match: v scaled to ||row|| then * 0.5 -> each element 2 + 0.5*(2*sqrt(D))/sqrt(D) = 3
    assert torch.allclose(target[0:2], torch.full((2, D), 3.0))
    assert torch.allclose(target[2:], torch.full((4, D), 2.0))
    cfg3 = SteeringVector(
        activations=torch.ones(1, 2, D) * 10, layer_indices=[0], position_indices=[7, 9]
    )
    tgt = torch.zeros(6, D)
    W._apply_steering(
        [cfg3], 0, tgt, 1, 5, abs_start=6, residual=None
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
            assert W._apply_layer_vectorized(pa.steer[L], L, ta, pa, None)
            assert W._apply_layer_vectorized(pb.steer[L], L, tb, pb, None)
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


def test_vectorized_replace_matches_sequential_and_zeroes_fused_residual():
    """mode="replace" is vectorised (index_copy_); on a fused-residual layer the
    residual half is zeroed so the FULL stream equals scale*v exactly."""
    torch.manual_seed(4)
    n, P = 5, 6
    configs = [[sv3d(1, [2], mode="replace", scale=1.5)] for _ in range(n)]
    configs[3] = [sv3d(1, [2], mode="add", scale=0.5)]  # mixed add + replace batch
    qsl = [P * i for i in range(n + 1)]
    target = torch.randn(P * n, D)
    residual = torch.randn(P * n, D)
    todo = list(zip(range(n), configs))
    ok, seq, vec, rs, rv = _apply_both(todo, 1, target, qsl, [0] * n, residual)
    assert ok
    assert torch.equal(seq, vec) and torch.equal(rs, rv)
    for i in range(n):
        r = qsl[i] + 2
        v = configs[i][0].activations[0, 0]
        if i == 3:
            assert torch.allclose(vec[r], target[r] + 0.5 * v)
            assert torch.equal(rv[r], residual[r])
        else:
            assert torch.allclose(vec[r], 1.5 * v)
            assert torch.equal(rv[r], torch.zeros(D))
            assert torch.allclose(vec[r] + rv[r], 1.5 * v)  # full stream replaced
    keep = [r for r in range(P * n) if r % P != 2]
    assert torch.equal(vec[keep], target[keep]) and torch.equal(rv[keep], residual[keep])
    # non-fused layer (residual None): plain overwrite
    ok2, seq2, vec2 = _apply_both(todo[:1], 1, target.clone(), qsl, [0] * n)
    assert ok2 and torch.equal(seq2, vec2) and torch.allclose(vec2[2], 1.5 * configs[0][0].activations[0, 0])


# ---------------------------------------------------------------------------
# norm_match references the FULL residual stream on fused-residual layers
# (upstream #7 port): h' = h + scale * ||h|| * v/||v|| with h = hidden + residual
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("vectorized", [False, True])
def test_norm_match_scales_to_full_residual_stream_on_fused_layers(vectorized):
    torch.manual_seed(5)
    scale = 4.0
    cfgs = [[sv3d(1, [1], norm_match=True, scale=scale)] for _ in range(3)]
    qsl = [0, 4, 8, 12]
    hidden = torch.randn(12, D) * 0.1  # small MLP-delta half (as in a real model)
    residual = torch.randn(12, D) * 10  # large residual half
    todo = list(zip(range(3), cfgs))
    plan = W._StepPlan(qsl, [0] * 3, {1: todo})
    t, r = hidden.clone(), residual.clone()
    if vectorized:
        assert W._apply_layer_vectorized(todo, 1, t, plan, r)
    else:
        for i, c in todo:
            W._apply_steering(c, 1, t, qsl[i], qsl[i + 1], 0, r)
    for i, c in todo:
        row = qsl[i] + 1
        full = hidden[row] + residual[row]
        delta = (t[row] + r[row]) - full
        v = c[0].activations[0, 0]
        assert torch.nn.functional.cosine_similarity(delta, v, dim=0) > 0.9999
        ratio = delta.norm() / (scale * full.norm())
        assert abs(ratio - 1.0) < 1e-4, ratio  # 1.1.0 would give ~||hidden||/||full||
    assert torch.equal(r, residual)  # add never touches the residual half
    # 2-D broadcast + norm_match on a fused layer: same reference
    t2, r2 = hidden.clone(), residual.clone()
    W._apply_steering([sv2d(1, norm_match=True, scale=2.0)], 1, t2, 0, 4, 0, r2)
    d2 = (t2[:4] + r2[:4]) - (hidden[:4] + residual[:4])
    assert torch.allclose(d2.norm(dim=-1), 2.0 * (hidden[:4] + residual[:4]).norm(dim=-1), rtol=1e-4)


def test_hook_inner_clones_residual_only_for_replace_layers():
    """_hook_inner passes the fused residual half through: read-only for add
    (same tensor object returned), cloned + zeroed for replace."""
    ext = make_ext()
    store(ext, "add", [sv3d(1, [2], norm_match=True, scale=1.0)])
    store(ext, "rep", [sv3d(2, [2], mode="replace", scale=3.0)])
    runner = FakeRunner(
        [
            ("0-aaaaaaaa", {"_steering_id": "add"}, 4, 0, 4),
            ("1-bbbbbbbb", {"_steering_id": "rep"}, 4, 0, 4),
        ]
    )
    ext.model_runner = runner
    import vllm.forward_context as fc

    meta = SimpleNamespace(query_start_loc=torch.tensor(runner.query_start_loc.np))
    ctx = SimpleNamespace(attn_metadata={"layer0": meta})
    fc._ctx = ctx
    try:
        plan = W._build_step_plan(ext, runner, 2)
        assert plan.replace_layers == {2}
        ext._step_plan = plan
        hs, res = torch.randn(8, D), torch.randn(8, D)
        out1 = W._hook_inner(ext, 1, (hs, res))
        assert out1[1] is res  # add-only layer: residual passed through untouched
        full1 = out1[0][2] + out1[1][2]
        assert torch.allclose((full1 - (hs[2] + res[2])).norm(), (hs[2] + res[2]).norm(), rtol=1e-4)
        out2 = W._hook_inner(ext, 2, (hs, res))
        assert out2[1] is not res and torch.equal(res, res)  # cloned, original intact
        v = ext._steering_data["rep"][0].activations[0, 0]
        assert torch.allclose(out2[0][6], 3.0 * v) and torch.equal(out2[1][6], torch.zeros(D))
        assert torch.equal(out2[0][:6], hs[:6]) and torch.equal(out2[1][:6], res[:6])
    finally:
        fc._ctx = None


# ---------------------------------------------------------------------------
# layer-0 pre-hook: keyword-passing architectures, hard error on a miss
# ---------------------------------------------------------------------------


def test_find_hidden_states_arg_searches_kwargs_and_prefers_named():
    h = torch.randn(10, D)
    pos = torch.arange(10)
    # positional (Qwen2 / Llama style)
    assert W._find_hidden_states_arg((pos, h, None), {}, 10) is h
    # keyword (Qwen3Next / Qwen3.5 / Qwen3.6 style)
    assert W._find_hidden_states_arg((), {"positions": pos, "hidden_states": h, "residual": None}, 10) is h
    # padded token dim (sequence parallel) still matches
    hp = torch.randn(12, D)
    assert W._find_hidden_states_arg((pos, hp, None), None, 10) is hp
    # two candidates but one is literally named hidden_states -> that one
    other = torch.randn(10, D)
    assert W._find_hidden_states_arg((other,), {"hidden_states": h}, 10) is h
    # ambiguity without a name -> hard error
    with pytest.raises(W.EmbedInjectionError, match="found 2 candidate"):
        W._find_hidden_states_arg((pos, h, other), {}, 10)
    # nothing that covers the tokens -> hard error (not a warning)
    with pytest.raises(W.EmbedInjectionError, match="found 0 candidate"):
        W._find_hidden_states_arg((pos, torch.randn(4, D)), {"residual": None}, 10)
    with pytest.raises(W.EmbedInjectionError):
        W._find_hidden_states_arg((pos, h.to(torch.int64)), {}, 10)


class _KwLayer(torch.nn.Module):
    """Decoder layer stand-in whose model calls it by KEYWORD (Qwen3Next style)."""

    def forward(self, positions=None, hidden_states=None, residual=None):
        return hidden_states * 1.0, residual


def _run_pre_hook(ext, runner, layer, call):
    import vllm.forward_context as fc

    meta = SimpleNamespace(query_start_loc=torch.tensor(runner.query_start_loc.np))
    fc._ctx = SimpleNamespace(attn_metadata={"layer0": meta})
    try:
        return call()
    finally:
        fc._ctx = None


def test_pre_hook_with_kwargs_applies_embed_on_keyword_calling_layer():
    ext = make_ext()
    v = torch.randn(D)
    sv = SteeringVector(
        activations=v.reshape(1, 1, D),
        layer_indices=[W.EMBED_LAYER_INDEX],
        position_indices=[3],
        mode="replace",
        scale=2.0,
    )
    store(ext, "nla", [sv])
    runner = FakeRunner(
        [
            ("nla-aaaa1111", None, 6, 0, 6),
            ("x-bbbb2222", {"output_residual_stream": [W.EMBED_LAYER_INDEX, 0]}, 4, 0, 4),
        ]
    )
    ext.model_runner = runner
    layer = _KwLayer()
    layer.register_forward_pre_hook(W._make_pre_hook(ext, 0), with_kwargs=True)
    h = torch.randn(10, D)
    orig = h.clone()
    out, _ = _run_pre_hook(
        ext, runner, layer,
        lambda: layer(positions=torch.arange(10), hidden_states=h, residual=None),
    )
    assert torch.allclose(h[3], 2.0 * v)  # injected in place, by keyword
    assert torch.equal(out[3], 2.0 * v)  # ... so the layer saw the replaced row
    keep = [r for r in range(10) if r != 3]
    assert torch.equal(h[keep], orig[keep])
    assert ext._stats["rows_replaced"] == 1 and ext._stats["embed_apply_steps"] == 1
    assert ext._stats["errors"] == 0 and ext._stats["embed_errors"] == 0
    assert ext._step_plan is not None  # plan built once here, reused by layer hooks
    # embedding-stream capture (explicit layer -1) for the second request, post-injection
    ext._flush_host_blocks()  # fast path: one host block per layer-step, split at retrieval
    cap = ext._captured_states["x-bbbb2222"][W.EMBED_LAYER_INDEX]
    assert len(cap) == 1 and torch.equal(cap[0], orig[6:10])
    assert "nla-aaaa1111" not in ext._captured_states


def test_pre_hook_raises_on_missing_hidden_states_instead_of_warning():
    ext = make_ext()
    store(ext, "nla", [sv3d(W.EMBED_LAYER_INDEX, [1], mode="replace")])
    runner = FakeRunner([("nla-aaaa1111", None, 4, 0, 4)])
    ext.model_runner = runner

    class _BadLayer(torch.nn.Module):
        def forward(self, positions, residual=None):  # no hidden_states at all
            return positions

    layer = _BadLayer()
    layer.register_forward_pre_hook(W._make_pre_hook(ext, 0), with_kwargs=True)
    with pytest.raises(W.EmbedInjectionError):
        _run_pre_hook(ext, runner, layer, lambda: layer(torch.arange(4), residual=None))
    assert ext._stats["embed_errors"] == 1 and ext._stats["errors"] == 1


def test_pre_hook_on_non_first_global_layer_never_injects():
    """PP rank > 0: its first local layer is not global layer 0, so the
    embedding stream is not here -- build the plan, but do not touch inputs."""
    ext = make_ext()
    store(ext, "nla", [sv3d(W.EMBED_LAYER_INDEX, [1], mode="replace")])
    runner = FakeRunner([("nla-aaaa1111", None, 4, 0, 4)])
    ext.model_runner = runner
    layer = _KwLayer()
    layer.register_forward_pre_hook(W._make_pre_hook(ext, 8), with_kwargs=True)
    h = torch.randn(4, D)
    orig = h.clone()
    _run_pre_hook(ext, runner, layer, lambda: layer(hidden_states=h, residual=torch.zeros(4, D)))
    assert torch.equal(h, orig) and ext._stats["rows_replaced"] == 0
    assert ext._step_plan is not None


def test_capture_all_layers_does_not_include_embedding_stream():
    """output_residual_stream=True keeps its (n_layers, T, D) shape: the
    embedding stream is captured only when layer -1 is listed explicitly."""
    ext = make_ext()
    runner = FakeRunner([("0-aaaaaaaa", {"output_residual_stream": True}, 4, 0, 4)])
    plan = build_plan(ext, runner)
    assert plan.cap_all == [0]
    assert plan.cap_by_layer.get(W.EMBED_LAYER_INDEX) is None


# ---------------------------------------------------------------------------
# EMBED_LAYER_INDEX through the RPC surface (validation + block packing)
# ---------------------------------------------------------------------------


def test_rpcs_accept_embed_layer_index_and_block_carries_mode():
    ext = make_ext()
    ext.set_steering_data(
        "k", pickle.dumps([sv3d(W.EMBED_LAYER_INDEX, [3], mode="replace", scale=2.0)])
    )
    assert W.EMBED_LAYER_INDEX in ext._steering_index["k"].layers
    assert ext._steering_index["k"].replace_layers == frozenset({W.EMBED_LAYER_INDEX})
    with pytest.raises(ValueError, match="out of range"):
        ext.set_steering_data("bad", pickle.dumps([sv3d(-2, [3])]))
    vecs = torch.randn(3, D)
    ext.set_steering_block(
        pickle.dumps(
            {
                "keys": ["a", "b", "c"],
                "vecs": vecs,
                "layers": [W.EMBED_LAYER_INDEX, 1, W.EMBED_LAYER_INDEX],
                "positions": [5, 5, 7],
                "scales": [1.0, 2.0, 3.0],
                "norm_match": [False, True, True],
                "modes": ["replace", "add", "replace"],
            }
        )
    )
    a, b, c = (ext._steering_data[k][0] for k in "abc")
    assert (a.mode, b.mode, c.mode) == ("replace", "add", "replace")
    assert a.layer_indices == [W.EMBED_LAYER_INDEX] and b.layer_indices == [1]
    assert ext._steering_index["a"].replace_layers == frozenset({W.EMBED_LAYER_INDEX})
    assert ext._steering_index["b"].replace_layers == frozenset()
    # a block without "modes" (older client) defaults to add
    ext.set_steering_block(
        pickle.dumps(
            {"keys": ["d"], "vecs": vecs[:1], "layers": [0], "positions": [0],
             "scales": [1.0], "norm_match": [False]}
        )
    )
    assert ext._steering_data["d"][0].mode == "add"
    with pytest.raises(ValueError, match="modes"):
        ext.set_steering_block(
            pickle.dumps(
                {"keys": ["e"], "vecs": vecs[:1], "layers": [0], "positions": [0],
                 "scales": [1.0], "norm_match": [False], "modes": ["nope"]}
            )
        )
    # the whole thing schedules + applies through the embed path
    ext._refresh_aggregates()
    assert ext._agg_embed is True
    runner = FakeRunner([("a-aaaa1111", None, 8, 0, 8), ("c-cccc3333", None, 8, 0, 8)])
    plan = build_plan(ext, runner)
    target = torch.randn(16, D)
    orig = target.clone()
    W._apply_embed(plan.steer[W.EMBED_LAYER_INDEX], target, plan, ext._stats)
    assert torch.allclose(target[5], vecs[0])  # a: pos 5 -> row 5, scale 1, no norm_match
    # c: second request (rows 8..15), pos 7 -> row 15, norm_match, scale 3
    assert torch.allclose(target[15].norm(), 3.0 * orig[15].norm(), rtol=1e-4)
    assert torch.nn.functional.cosine_similarity(target[15], vecs[2], dim=0) > 0.9999
    keep = [r for r in range(16) if r not in (5, 15)]
    assert torch.equal(target[keep], orig[keep])


def test_pack_steering_carries_mode_and_embed_layer():
    from vllm_lens._activations_plugin import _pack_steering

    payload = {
        "_steer_0": [sv3d(W.EMBED_LAYER_INDEX, [4], mode="replace", scale=2.0)],
        "_steer_1": [sv3d(1, [4], norm_match=True, scale=1.0)],
        "_steer_2": [sv3d(1, [4, 5])],  # two positions: not block-packable
    }
    block, rest = _pack_steering(payload)
    assert block is not None and list(rest) == ["_steer_2"]
    assert block["keys"] == ["_steer_0", "_steer_1"]
    assert block["layers"] == [W.EMBED_LAYER_INDEX, 1]
    assert block["modes"] == ["replace", "add"]
    assert block["norm_match"] == [False, True] and block["scales"] == [2.0, 1.0]


# ---------------------------------------------------------------------------
# hyper-connection / multi-stream architectures (DeepSeek-V4 mHC): embed-only,
# layer-output steering/capture must fail LOUDLY; vLLM 0.27 query_start_loc
# ---------------------------------------------------------------------------


def test_split_layer_output_classifies_outputs():
    h = torch.randn(6, D)
    assert W._split_layer_output(h, 0)[0] is h
    r = torch.randn(6, D)
    s_, res, rest = W._split_layer_output((h, r), 0)
    assert s_ is h and res is r and rest == (r,)
    s_, res, rest = W._split_layer_output((h, None), 0)
    assert res is None and rest == (None,)
    with pytest.raises(W.UnsupportedLayerOutputError, match="multi-stream"):
        W._split_layer_output((h, torch.randn(6, 4, D), torch.randn(6, 1), torch.randn(6, 4)), 3)
    with pytest.raises(W.UnsupportedLayerOutputError):
        W._split_layer_output((h, torch.randn(6, 4, D)), 3)  # 2-tuple but stacked residual


def _fake_ctx(runner, with_qsl=True):
    import vllm.forward_context as fc

    if with_qsl:
        meta = SimpleNamespace(query_start_loc=torch.tensor(runner.query_start_loc.np))
    else:  # vLLM 0.27 style: the attention metadata does not carry query_start_loc
        meta = SimpleNamespace(prefill=SimpleNamespace(cum_seq_lens_q=None))
    fc._ctx = SimpleNamespace(attn_metadata={"layer0": meta})
    return fc


def test_hook_inner_raises_on_multi_stream_output_for_steer_and_capture():
    ext = make_ext()
    store(ext, "add", [sv3d(1, [2])])
    runner = FakeRunner(
        [("0-aaaaaaaa", {"_steering_id": "add"}, 4, 0, 4), ("1-bbbbbbbb", {"output_residual_stream": [1]}, 4, 0, 4)]
    )
    ext.model_runner = runner
    fc = _fake_ctx(runner)
    try:
        ext._step_plan = W._build_step_plan(ext, runner, 2)
        hs, res4 = torch.randn(8, D), torch.randn(8, 4, D)
        hook = W._make_hook(ext, 1)
        with pytest.raises(W.UnsupportedLayerOutputError):  # re-raised, not swallowed
            hook(None, (), (hs, res4, torch.randn(8, 1), torch.randn(8, 4)))
        assert ext._stats["unsupported_layer_output"] == 1 and ext._stats["errors"] == 1
        # ordinary errors are still swallowed into a warning
        assert hook(None, (), "not-a-tensor") is None
        assert ext._stats["errors"] == 2
        # a fused 2-tuple on the same plan still works
        out = hook(None, (), (hs, torch.randn(8, D)))
        assert isinstance(out, tuple) and out[0].shape == hs.shape
    finally:
        fc._ctx = None


def test_multi_stream_detection_and_rpc_rejection():
    ext = make_ext()
    ext.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hc_mult=4)))
    assert W._detect_multi_stream(ext) is True
    ext.vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(hidden_size=8)))
    assert W._detect_multi_stream(ext) is False
    ext.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(text_config=SimpleNamespace(hc_mult=1)))
    )
    assert W._detect_multi_stream(ext) is False
    import os

    os.environ["VLLM_LENS_MULTI_STREAM"] = "1"
    try:
        assert W._detect_multi_stream(ext) is True
    finally:
        del os.environ["VLLM_LENS_MULTI_STREAM"]
    ext._multi_stream = True
    with pytest.raises(ValueError, match="hyper-connection"):
        ext.set_steering_data("k", pickle.dumps([sv3d(1, [3])]))
    with pytest.raises(ValueError, match="hyper-connection"):
        ext.set_steering_block(
            pickle.dumps({"keys": ["a"], "vecs": torch.randn(1, D), "layers": [2], "positions": [0],
                          "scales": [1.0], "norm_match": [False], "modes": ["add"]})
        )
    ext.set_steering_data("k", pickle.dumps([sv3d(W.EMBED_LAYER_INDEX, [3], mode="replace")]))  # embed is fine
    ext.set_steering_block(
        pickle.dumps({"keys": ["a"], "vecs": torch.randn(1, D), "layers": [W.EMBED_LAYER_INDEX], "positions": [0],
                      "scales": [1.0], "norm_match": [False], "modes": ["replace"]})
    )
    ext.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    caps = ext.lens_capabilities()
    assert caps["multi_stream"] is True and caps["num_layers"] == 4


def test_plugin_check_layer_support():
    from vllm_lens._activations_plugin import _check_layer_support

    caps = {"multi_stream": True}
    _check_layer_support(caps, [sv3d(W.EMBED_LAYER_INDEX, [3], mode="replace")], [W.EMBED_LAYER_INDEX])
    _check_layer_support({"multi_stream": False}, [sv3d(1, [3])], True)
    _check_layer_support(None, [sv3d(1, [3])], True)
    with pytest.raises(ValueError, match="unsupported on this hyper-connection"):
        _check_layer_support(caps, [sv3d(1, [3])], None)
    with pytest.raises(ValueError, match="output_residual_stream"):
        _check_layer_support(caps, None, True)
    with pytest.raises(ValueError, match="output_residual_stream"):
        _check_layer_support(caps, None, [W.EMBED_LAYER_INDEX, 5])


def test_step_plan_uses_runner_host_buffers_when_metadata_lacks_query_start_loc():
    """vLLM >= 0.27: several attention backends no longer expose query_start_loc
    on their metadata; the plan must still be built from the runner buffers."""
    ext = make_ext()
    store(ext, "_steer_0", [sv3d(1, [10])])
    runner = FakeRunner([("0-aaaaaaaa", {"_steering_id": "_steer_0"}, 96, 0, 96), ("1-bbbbbbbb", None, 96, 96, 1)])
    fc = _fake_ctx(runner, with_qsl=False)
    try:
        plan = W._build_step_plan(ext, runner, 2)
        assert plan is not None and plan.qsl == [0, 96, 97] and plan.abs_start == [0, 96]
        assert [i for i, _ in plan.steer[1]] == [0]
    finally:
        fc._ctx = None
