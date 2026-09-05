"""CPU-only tests for the fast hidden-state readout (vllm-metamodels 1.1.0.post4):
position specs, gather-capture + host blocks, in-engine projection (ReadoutVector),
early exit, bulk retrieval, and the plugin-side packing / validation.

Run without upstream's GPU conftest:  pytest vllm_lens/tests/test_readout.py --noconftest
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_lens.tests.test_steering_index import (  # noqa: F401  (stubs vLLM when absent)
    D,
    FakeRunner,
    build_plan,
    make_ext,
    store,
    sv3d,
)
from vllm_lens import ReadoutVector, SteeringVector  # noqa: E402
from vllm_lens import _worker_ext as W  # noqa: E402

import vllm.forward_context as _fc_mod  # noqa: E402

if not hasattr(_fc_mod, "_ctx"):
    # Real vLLM installed (CI image / GPU box): route _worker_ext's accessors through the same
    # ``_ctx`` attribute the stub uses so every test can set ``fc._ctx`` regardless of environment.
    _fc_mod._ctx = None
    W.get_forward_context = lambda: _fc_mod._ctx
    W.is_forward_context_available = lambda: _fc_mod._ctx is not None
from vllm_lens._helpers import types as T  # noqa: E402


def _runner(rows):
    """rows: (req_id, extra_args, num_prompt, num_computed, n_query[, max_tokens])."""
    r = FakeRunner([row[:5] for row in rows])
    for row in rows:
        mt = row[5] if len(row) > 5 else None
        r.requests[row[0]].sampling_params.max_tokens = mt
    return r


def _load_stock(blob: bytes) -> dict:
    """Decode the 1.1.0-format ``get_captured_states`` payload (zstd + pickle)."""
    import zstandard as zstd

    return pickle.loads(zstd.ZstdDecompressor().decompress(blob))["activations"]


def _hidden(runner):
    n = int(runner.query_start_loc.np[-1])
    return torch.randn(n, D)


def _run_hook(ext, runner, layer, output):
    """Build the pass plan (as the pre-hook would) and run one layer hook."""
    import vllm.forward_context as fc

    ext.model_runner = runner
    ext._step_plan = None
    ext._step_idle = False
    plan = build_plan(ext, runner)
    ext._step_plan = plan
    meta = SimpleNamespace(query_start_loc=torch.tensor(runner.query_start_loc.np))
    ctx = SimpleNamespace(attn_metadata={"layer0": meta})
    plan.ctx_id = id(ctx)
    if hasattr(fc, "_ctx"):
        fc._ctx = ctx
        return W._hook_inner(ext, layer, output), plan
    orig_get, orig_avail = W.get_forward_context, W.is_forward_context_available
    W.get_forward_context = lambda: ctx
    W.is_forward_context_available = lambda: True
    try:
        return W._hook_inner(ext, layer, output), plan
    finally:
        W.get_forward_context, W.is_forward_context_available = orig_get, orig_avail


# ---------------------------------------------------------------------------
# position specs
# ---------------------------------------------------------------------------


def test_normalize_positions():
    assert T.normalize_positions(None) == ("all", None)
    assert T.normalize_positions("all") == ("all", None)
    assert T.normalize_positions({"last": 5}) == ("last", 5)
    assert T.normalize_positions([3, -1, 7]) == ("list", (3, -1, 7))
    for bad in ({"first": 2}, {"last": 0}, [], "last5", 3.5):
        with pytest.raises(ValueError):
            T.normalize_positions(bad)


def test_select_positions_prefill_chunk_and_generated():
    # one request, prompt 10 tokens, whole prompt in this pass at flat rows 4..14
    idx, pos = W._select_positions(("all", None), 4, 14, 0, 10)
    assert idx.tolist() == list(range(4, 14)) and pos.tolist() == list(range(10))
    idx, pos = W._select_positions(("last", 3), 4, 14, 0, 10)
    assert idx.tolist() == [11, 12, 13] and pos.tolist() == [7, 8, 9]
    idx, pos = W._select_positions(("list", (0, -1, 5, 99)), 4, 14, 0, 10)
    assert idx.tolist() == [4, 9, 13] and pos.tolist() == [0, 5, 9]
    # chunked prefill: second chunk covers absolute 6..10 -> last-3 = 7,8,9
    idx, pos = W._select_positions(("last", 3), 0, 4, 6, 10)
    assert idx.tolist() == [1, 2, 3] and pos.tolist() == [7, 8, 9]
    # first chunk 0..6 contains none of the last 3
    idx, pos = W._select_positions(("last", 3), 0, 6, 0, 10)
    assert len(idx) == 0 and len(pos) == 0
    # decode row (generated position 12) is always selected under "last"
    idx, pos = W._select_positions(("last", 3), 5, 6, 12, 10)
    assert idx.tolist() == [5] and pos.tolist() == [12]
    idx, pos = W._select_positions(("all", None), 5, 6, 12, 10)
    assert idx.tolist() == [5] and pos.tolist() == [12]


# ---------------------------------------------------------------------------
# fast capture == legacy capture, with position specs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fused", [True, False])
def test_fast_capture_matches_legacy_and_honours_positions(fused):
    rows = [
        ("a-aaaa1111", {"output_residual_stream": [2]}, 6, 0, 6, 1),
        ("b-bbbb2222", {"output_residual_stream": [2], "capture_positions": {"last": 2}}, 5, 0, 5, 1),
        ("c-cccc3333", None, 4, 0, 4, 8),
        ("d-dddd4444", {"output_residual_stream": [2], "capture_positions": [0, -1]}, 7, 0, 7, 1),
    ]
    runner = _runner(rows)
    h, r = _hidden(runner), _hidden(runner)
    out = (h, r) if fused else h
    full = h + r if fused else h

    fast = make_ext()
    _run_hook(fast, runner, 2, out)
    assert fast._step_plan.exit_layer is None  # row c generates: no early exit
    assert fast._stats["capture_layer_steps"] == 1 and fast._stats["capture_rows"] == 6 + 2 + 2
    assert len(fast._cap_blocks) == 1 and not fast._captured_states  # lazy split
    blob = fast.get_captured_states_many(["a", "b", "d", "zzz"])
    res = pickle.loads(blob)
    assert set(res) == {"a", "b", "d"}
    assert torch.equal(res["a"]["residual_stream"][0], full[0:6]) and res["a"]["positions"] == list(range(6))
    assert torch.equal(res["b"]["residual_stream"][0], full[6 + 3 : 11]) and res["b"]["positions"] == [3, 4]
    assert torch.equal(res["d"]["residual_stream"][0], full[[15, 21]]) and res["d"]["positions"] == [0, 6]
    assert not fast._captured_states and not fast._captured_positions

    legacy = make_ext()
    legacy._fast_capture = False
    _run_hook(legacy, runner, 2, out)
    st = _load_stock(legacy.get_captured_states("a"))
    assert torch.equal(st["residual_stream"][0], full[0:6]) and "positions" not in st
    # legacy path ignores capture_positions (whole chunk), as documented
    st_b = _load_stock(legacy.get_captured_states("b"))
    assert st_b["residual_stream"].shape[1] == 5


def test_stock_get_captured_states_works_on_fast_path_and_chunked_prefill_concats():
    ext = make_ext()
    # chunk 1: absolute 0..4 of an 8-token prompt; chunk 2: 4..8
    r1 = _runner([("k-aaaa1111", {"output_residual_stream": [1], "capture_positions": {"last": 6}}, 8, 0, 4, 1)])
    h1 = _hidden(r1)
    _run_hook(ext, r1, 1, h1)
    r2 = _runner([("k-aaaa1111", {"output_residual_stream": [1], "capture_positions": {"last": 6}}, 8, 4, 4, 1)])
    h2 = _hidden(r2)
    ext._req_plan_cache.clear()
    _run_hook(ext, r2, 1, h2)
    acts = _load_stock(ext.get_captured_states("k"))
    assert acts["positions"] == [2, 3, 4, 5, 6, 7]
    assert torch.equal(acts["residual_stream"][0], torch.cat([h1[2:4], h2[0:4]]))
    assert ext._debug_captured_states_count() == 0


# ---------------------------------------------------------------------------
# in-engine readout
# ---------------------------------------------------------------------------


def _ref_metric(h_rows, v, cos, bias):
    h = h_rows.float()
    v = v.float()
    d = (h * v).sum(-1)
    if cos:
        d = d / (h.norm(dim=-1) * v.norm()).clamp_min(1e-6)
    return d + bias


@pytest.mark.parametrize("fused", [True, False])
def test_readout_block_values_match_reference(fused):
    ext = make_ext()
    n = 3
    vecs = torch.randn(n, D)
    keys = ["_read_0", "_read_1", "_read_2"]
    ext.set_readout_block(
        pickle.dumps(
            {
                "keys": keys,
                "vecs": vecs,
                "layers": [2, 2, 2],
                "positions": [{"last": 2}, "all", [1, -1]],
                "metric": ["cos", "dot", "cos"],
                "bias": [0.0, 0.5, 0.0],
            }
        )
    )
    rows = [
        ("0-aaaa1111", {"_readout_id": "_read_0"}, 6, 0, 6, 1),
        ("1-bbbb2222", {"_readout_id": "_read_1"}, 4, 0, 4, 1),
        ("2-cccc3333", {"_readout_id": "_read_2"}, 5, 0, 5, 1),
    ]
    runner = _runner(rows)
    h, r = _hidden(runner), _hidden(runner)
    out = (h, r) if fused else h
    full = h + r if fused else h
    _run_hook(ext, runner, 2, out)
    assert ext._stats["readout_layer_steps"] == 1 and ext._stats["readout_rows"] == 2 + 4 + 2
    assert not ext._cap_blocks  # nothing captured, only scalars
    res = pickle.loads(ext.get_readouts_many(["0", "1", "2"]))
    r0, r1, r2 = res["0"][0], res["1"][0], res["2"][0]
    assert r0["positions"] == [4, 5] and r0["layers"] == [2]
    assert torch.allclose(r0["values"][0], _ref_metric(full[4:6], vecs[0], True, 0.0), atol=1e-5)
    assert r1["positions"] == [0, 1, 2, 3]
    assert torch.allclose(r1["values"][0], _ref_metric(full[6:10], vecs[1], False, 0.5), atol=1e-4)
    assert r2["positions"] == [1, 4]
    assert torch.allclose(r2["values"][0], _ref_metric(full[[11, 14]], vecs[2], True, 0.0), atol=1e-5)
    assert r0["values"].dtype == torch.float32
    assert not ext._readouts


def test_readout_data_many_multi_layer_and_multi_vector_equals_block_semantics():
    ext = make_ext()
    v = torch.randn(2, D)  # one vector per layer (layers 1 and 3)
    w = torch.randn(1, D)
    rv_a = ReadoutVector(activations=v, layer_indices=[1, 3], positions={"last": 1}, metric="dot")
    rv_b = ReadoutVector(activations=w, layer_indices=[3], positions="all")
    ext.set_readout_data_many(pickle.dumps({"_read_0": [rv_a, rv_b]}))
    runner = _runner([("0-aaaa1111", {"_readout_id": "_read_0"}, 4, 0, 4, 1)])
    h1, h3 = _hidden(runner), _hidden(runner)
    _run_hook(ext, runner, 1, h1)
    _run_hook(ext, runner, 3, h3)
    res = pickle.loads(ext.get_readouts_many(["0"]))["0"]
    assert len(res) == 2
    a, b = res
    assert a["layers"] == [1, 3] and a["positions"] == [3]
    assert torch.allclose(a["values"][0, 0], _ref_metric(h1[3:4], v[0], False, 0.0)[0], atol=1e-4)
    assert torch.allclose(a["values"][1, 0], _ref_metric(h3[3:4], v[1], False, 0.0)[0], atol=1e-4)
    assert b["layers"] == [3] and b["positions"] == [0, 1, 2, 3]
    assert torch.allclose(b["values"][0], _ref_metric(h3[0:4], w[0], True, 0.0), atol=1e-5)
    ext.clear_readout_data_many(["_read_0"])
    assert not ext._readout_index


def test_readout_and_capture_share_pass_and_steering_is_applied_first():
    ext = make_ext()
    v = torch.randn(D)
    store(ext, "_steer_0", [sv3d(2, [1], activations=None) if False else SteeringVector(
        activations=v.view(1, 1, D), layer_indices=[2], position_indices=[1], mode="replace"
    )])
    ext.set_readout_block(pickle.dumps({"keys": ["_read_0"], "vecs": v.view(1, D), "layers": [2], "positions": [[1]]}))
    runner = _runner([
        ("0-aaaa1111", {"_steering_id": "_steer_0", "_readout_id": "_read_0", "output_residual_stream": [2],
                        "capture_positions": [1]}, 3, 0, 3, 1),
    ])
    h, r = _hidden(runner), _hidden(runner)
    out, _plan = _run_hook(ext, runner, 2, (h, r))
    assert out is not None  # steering modified the output
    res = pickle.loads(ext.get_readouts_many(["0"]))["0"][0]
    assert res["values"][0, 0].item() == pytest.approx(1.0, abs=1e-4)  # replaced row == v -> cos 1
    cap = pickle.loads(ext.get_captured_states_many(["0"]))["0"]
    assert torch.allclose(cap["residual_stream"][0, 0].float(), v, atol=1e-5)


def test_readout_rejects_bad_layers_and_multi_stream():
    ext = make_ext()
    with pytest.raises(ValueError):
        ext.set_readout_block(pickle.dumps({"keys": ["k"], "vecs": torch.randn(1, D), "layers": [99]}))
    with pytest.raises(ValueError):
        ext.set_readout_block(pickle.dumps({"keys": ["k"], "vecs": torch.randn(1, D), "layers": [1], "metric": ["l2"]}))
    ext._multi_stream = True
    with pytest.raises(ValueError):
        ext.set_readout_data("k2", pickle.dumps([ReadoutVector(activations=torch.randn(1, D), layer_indices=[1])]))
    ext.set_readout_data("k3", pickle.dumps([ReadoutVector(activations=torch.randn(1, D), layer_indices=[W.EMBED_LAYER_INDEX])]))
    assert "k" not in ext._readout_index and len(ext._readout_index) == 1


# ---------------------------------------------------------------------------
# early exit
# ---------------------------------------------------------------------------


def test_early_exit_plan_rules():
    ext = make_ext()
    ext._early_exit_ok = True
    ext.set_readout_block(pickle.dumps({"keys": ["_read_0"], "vecs": torch.randn(1, D), "layers": [1]}))
    # all rows readout-only, max_tokens=1 -> exit after the deepest requested layer (3)
    runner = _runner([
        ("0-aaaa1111", {"_readout_id": "_read_0", "lens_early_exit": True}, 4, 0, 4, 1),
        ("1-bbbb2222", {"output_residual_stream": [3, 0], "lens_early_exit": True}, 4, 0, 4, 1),
    ])
    ext.model_runner = runner
    assert build_plan(ext, runner).exit_layer == 3
    # a row that generates (max_tokens > 1) disables the exit for the whole pass
    runner = _runner([
        ("0-aaaa1111", {"_readout_id": "_read_0", "lens_early_exit": True}, 4, 0, 4, 1),
        ("1-bbbb2222", {"output_residual_stream": [3], "lens_early_exit": True}, 4, 0, 4, 4),
    ])
    ext._req_plan_cache.clear()
    assert build_plan(ext, runner).exit_layer is None
    # a plain generation row too
    runner = _runner([
        ("0-aaaa1111", {"_readout_id": "_read_0", "lens_early_exit": True}, 4, 0, 4, 1),
        ("1-bbbb2222", None, 4, 0, 4, 1),
    ])
    ext._req_plan_cache.clear()
    assert build_plan(ext, runner).exit_layer is None
    # output_residual_stream=True (all layers) is never eligible
    runner = _runner([("0-aaaa1111", {"output_residual_stream": True, "lens_early_exit": True}, 4, 0, 4, 1)])
    ext._req_plan_cache.clear()
    assert build_plan(ext, runner).exit_layer is None
    # engine without early-exit support: never
    ext._early_exit_ok = False
    runner = _runner([("0-aaaa1111", {"_readout_id": "_read_0", "lens_early_exit": True}, 4, 0, 4, 1)])
    ext._req_plan_cache.clear()
    assert build_plan(ext, runner).exit_layer is None
    # embedding-only capture exits after layer 0
    ext._early_exit_ok = True
    runner = _runner([("0-aaaa1111", {"output_residual_stream": [W.EMBED_LAYER_INDEX], "lens_early_exit": True}, 4, 0, 4, 1)])
    ext._req_plan_cache.clear()
    assert build_plan(ext, runner).exit_layer == 0


def test_early_exit_raises_after_readout_and_wrapper_returns_placeholder():
    ext = make_ext()
    ext._early_exit_ok = True
    ext.set_readout_block(pickle.dumps({"keys": ["_read_0"], "vecs": torch.randn(1, D), "layers": [2], "positions": [{"last": 1}]}))
    runner = _runner([
        ("0-aaaa1111", {"_readout_id": "_read_0", "lens_early_exit": True}, 4, 0, 4, 1),
        ("1-bbbb2222", {"output_residual_stream": [1], "capture_positions": {"last": 1}, "lens_early_exit": True}, 3, 0, 3, 1),
    ])
    h, r = _hidden(runner), _hidden(runner)
    # layer 1: capture for row 1, no exit yet (deepest requested layer is 2)
    out1, plan = _run_hook(ext, runner, 1, (h, r))
    assert out1 is None and plan.exit_layer == 2
    # layer 2: readout for row 0, then exit
    with pytest.raises(W._EarlyExit) as ei:
        _run_hook(ext, runner, 2, (h, r))
    assert ei.value.placeholder.shape == h.shape and torch.count_nonzero(ei.value.placeholder) == 0
    res = pickle.loads(ext.get_readouts_many(["0"]))["0"][0]
    assert res["positions"] == [3]
    cap = pickle.loads(ext.get_captured_states_many(["1"]))["1"]
    assert cap["positions"] == [2] and torch.equal(cap["residual_stream"][0, 0], (h + r)[6])
    # a layer past the exit with nothing to do still exits (robustness)
    with pytest.raises(W._EarlyExit):
        _run_hook(ext, runner, 3, (h, r))
    # the hook wrapper must not swallow the exit
    hook = W._make_hook(ext, 2)
    ext._step_plan = plan
    with pytest.raises(W._EarlyExit):
        import vllm.forward_context as fc

        if hasattr(fc, "_ctx"):
            hook(None, None, (h, r))
        else:
            _run_hook(ext, runner, 2, (h, r))

    # wrapped _model_forward turns the exception into the placeholder
    calls = []

    def orig(*a, **k):
        calls.append(1)
        raise W._EarlyExit(torch.zeros(5, D))

    fake_runner = SimpleNamespace(_model_forward=orig)
    ext.model_runner = fake_runner
    ext._wrap_model_forward()
    out = fake_runner._model_forward(input_ids=None)
    assert out.shape == (5, D) and ext._stats["early_exits"] == 1 and calls == [1]
    ext.steering_stats(reset=True)  # replaces the stats dict; the wrapper must follow it
    fake_runner._model_forward(input_ids=None)
    assert ext._stats["early_exits"] == 1
    ext._wrap_model_forward()  # idempotent
    assert fake_runner._lens_early_exit_wrapped


def test_early_exit_supported_rules():
    ext = make_ext()
    cfg = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    ext.vllm_config = cfg
    ext.model_runner = SimpleNamespace(_model_forward=lambda **k: None, use_aux_hidden_state_outputs=False, is_pooling_model=False)
    assert ext._early_exit_supported() == (True, "ok")
    # post7: prefix caching no longer disables early exit at the engine level -- each
    # early-exit request must carry a KV salt from token 0 (see test_kv_salt.py)
    cfg.cache_config.enable_prefix_caching = True
    assert ext._early_exit_supported() == (True, "ok")
    cfg.cache_config.enable_prefix_caching = False
    cfg.parallel_config.pipeline_parallel_size = 2
    assert not ext._early_exit_supported()[0]
    cfg.parallel_config.pipeline_parallel_size = 1
    ext.model_runner.use_aux_hidden_state_outputs = True
    assert not ext._early_exit_supported()[0]
    ext.model_runner = SimpleNamespace()  # no _model_forward
    assert not ext._early_exit_supported()[0]


# ---------------------------------------------------------------------------
# retrieval helpers / plugin side
# ---------------------------------------------------------------------------


def test_by_external_prefix_semantics():
    ids = ["req-0-a1b2c3d4", "req-00-b5c6d7e8", "x-1-deadbeef", "odd_suffix-ff-ee"]
    by = W.HiddenStatesExtension._by_external(ids, ["req-0", "req-00", "x-1", "odd_suffix"])
    assert by == {"req-0": ["req-0-a1b2c3d4"], "req-00": ["req-00-b5c6d7e8"], "x-1": ["x-1-deadbeef"], "odd_suffix": ["odd_suffix-ff-ee"]}


def test_plugin_pack_readouts_and_checks():
    from vllm_lens import _activations_plugin as P

    a = ReadoutVector(activations=torch.randn(1, D), layer_indices=[4], positions={"last": 5}, metric="dot", bias=0.25)
    b = ReadoutVector(activations=torch.randn(2, D), layer_indices=[1, 4])
    block, rest = P._pack_readouts({"_read_0": [a], "_read_1": [b], "_read_2": [a, a]})
    assert block["keys"] == ["_read_0"] and block["layers"] == [4] and block["positions"] == [{"last": 5}]
    assert block["metric"] == ["dot"] and block["bias"] == [0.25] and block["vecs"].shape == (1, D)
    assert set(rest) == {"_read_1", "_read_2"}
    caps = {"multi_stream": False, "readout": True, "early_exit": True}
    P._check_readout_request(caps, [a], True, 1, None)
    with pytest.raises(ValueError):
        P._check_readout_request(caps, [a], True, 8, None)  # early exit needs max_tokens=1
    with pytest.raises(ValueError):
        P._check_readout_request(caps, None, True, 1, True)  # all-layer capture never exits
    with pytest.raises(ValueError):
        P._check_readout_request(caps, None, True, 1, None)  # nothing to read
    with pytest.raises(ValueError):
        P._check_readout_request({**caps, "early_exit": False, "early_exit_reason": "prefix caching"}, [a], True, 1, None)
    with pytest.raises(ValueError):
        P._check_readout_request({**caps, "multi_stream": True}, [a], False, 1, None)
    P._check_readout_request({**caps, "multi_stream": True}, [ReadoutVector(activations=torch.randn(1, D), layer_indices=[W.EMBED_LAYER_INDEX])], False, 1, None)
    # JSON (vllm_xargs) form
    rvs = P._coerce_readouts(__import__("json").dumps([a.model_dump()]))
    assert len(rvs) == 1 and torch.allclose(rvs[0].activations, a.activations) and rvs[0].bias == 0.25


def test_trim_activations_and_readout_with_positions():
    from vllm_lens import _activations_plugin as P

    acts = {"residual_stream": torch.randn(1, 4, D), "positions": [7, 8, 9, 10]}
    P._trim_activations(acts, 10)  # one surplus pass past EOS
    assert acts["positions"] == [7, 8, 9] and acts["residual_stream"].shape == (1, 3, D)
    ro = [{"values": torch.randn(1, 4), "positions": [7, 8, 9, 10], "layers": [42]}]
    P._trim_readout(ro, 10)
    assert ro[0]["positions"] == [7, 8, 9] and ro[0]["values"].shape == (1, 3)
    legacy = {"residual_stream": torch.randn(1, 12, D)}
    P._trim_activations(legacy, 10)
    assert legacy["residual_stream"].shape[1] == 10


def test_serialize_passes_positions_through():
    from vllm_lens._helpers._serialize import decode_activations, serialize_activations

    acts = {"residual_stream": torch.randn(1, 3, D).bfloat16(), "positions": [4, 5, 6]}
    enc = serialize_activations(acts)
    assert enc["positions"] == [4, 5, 6]
    dec = decode_activations({"activations": enc})
    assert dec["positions"] == [4, 5, 6] and torch.equal(dec["residual_stream"], acts["residual_stream"])


def test_capabilities_report_readout_features():
    ext = make_ext()
    ext._hooks_installed = True
    ext._early_exit_ok, ext._early_exit_reason = True, "ok"
    caps = ext.lens_capabilities()
    assert caps["readout"] and caps["fast_capture"] and caps["early_exit"] and caps["early_exit_reason"] == "ok"


def test_idle_fast_path_stays_off_while_readout_in_flight():
    ext = make_ext()
    ext.set_readout_block(pickle.dumps({"keys": ["_read_0"], "vecs": torch.randn(1, D), "layers": [1]}))
    runner = _runner([("0-aaaaaaaa", {"_readout_id": "_read_0"}, 4, 0, 4, 1)])
    ext.model_runner = runner
    build_plan(ext, runner)
    assert "0-aaaaaaaa" in ext._capture_live
    dec = _runner([("0-aaaaaaaa", {"_readout_id": "_read_0"}, 4, 4, 1, 1)])
    assert not W._step_is_idle(ext, dec, 1)
