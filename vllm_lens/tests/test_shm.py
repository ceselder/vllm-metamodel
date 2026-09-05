"""CPU-only tests for the shared-memory transport (vllm-metamodels 1.1.0.post7, ``_shm``).

Run without upstream's GPU conftest:  pytest vllm_lens/tests/test_shm.py --noconftest
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from test_steering_index import FakeRunner, W, build_plan, make_ext  # noqa: E402

from vllm_lens import _shm  # noqa: E402

D = 8


def test_put_get_roundtrip_views_and_copies():
    t = {"a": torch.randn(3, 4, D), "b": torch.arange(10, dtype=torch.int64), "c": torch.randn(2, D).to(torch.bfloat16),
         "e": torch.empty(0, D)}
    desc = _shm.put(t)
    assert desc["nbytes"] >= sum(x.numel() * x.element_size() for x in t.values())
    got, handle = _shm.get(desc, copy=False)
    assert handle is not None
    for k in t:
        assert got[k].dtype == t[k].dtype and got[k].shape == t[k].shape and torch.equal(got[k], t[k])
    # views alias the mapping: writing through one is visible in another attach... the name is
    # unlinked on first get, so a second get must fail (single consumer protocol)
    with pytest.raises(FileNotFoundError):
        _shm.get(desc)
    _shm.release(handle)
    desc2 = _shm.put(t)
    got2, handle2 = _shm.get(desc2, copy=True)
    assert handle2 is None and all(torch.equal(got2[k], t[k]) for k in t)
    assert not any(v.is_shared() for v in got2.values())


def test_refuses_foreign_host_descriptor():
    desc = _shm.put({"x": torch.zeros(2)})
    bad = dict(desc, host="elsewhere:0")
    with pytest.raises(RuntimeError, match="another host"):
        _shm.get(bad)
    _shm.get(desc)  # clean up


def _child_put(q):
    torch.manual_seed(7)
    t = {"r0": torch.randn(2, 5, D).to(torch.bfloat16), "r1": torch.randn(2, 3, D).to(torch.bfloat16)}
    q.put((_shm.put(t, tag="test"), {k: v.float().sum().item() for k, v in t.items()}))


def test_cross_process_transfer():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_child_put, args=(q,))
    p.start()
    desc, sums = q.get(timeout=60)
    p.join(timeout=60)
    got, handle = _shm.get(desc, copy=True)
    assert set(got) == {"r0", "r1"} and got["r0"].dtype == torch.bfloat16 and got["r0"].shape == (2, 5, D)
    for k, s in sums.items():
        assert abs(got[k].float().sum().item() - s) < 1e-3
    assert not os.path.exists(f"/dev/shm/{desc['shm']}")  # unlinked by the consumer


def test_shm_mode_env(monkeypatch):
    monkeypatch.delenv("VLLM_LENS_SHM", raising=False)
    assert _shm.shm_mode() == ""
    monkeypatch.setenv("VLLM_LENS_SHM", "1")
    assert _shm.shm_mode() == "copy"
    monkeypatch.setenv("VLLM_LENS_SHM", "view")
    assert _shm.shm_mode() == "view"
    monkeypatch.setenv("VLLM_LENS_SHM", "0")
    assert _shm.shm_mode() == ""


def test_worker_block_rpcs_accept_shm_descriptor():
    ext = make_ext()
    vecs = torch.randn(3, D)
    ext.set_steering_block(pickle.dumps({"keys": ["k0", "k1", "k2"], "vecs": vecs, "layers": [1, 1, 2], "positions": [3, 4, 5],
                                         "scales": [1.0, 2.0, 1.0], "norm_match": [False, True, False]}))
    ref = {k: ext._steering_data[k][0].activations.clone() for k in ("k0", "k1", "k2")}
    ext2 = make_ext()
    desc = _shm.put({"vecs": vecs}, tag="steer")
    ext2.set_steering_block(pickle.dumps({"keys": ["k0", "k1", "k2"], "vecs": None, "shm": desc, "layers": [1, 1, 2],
                                          "positions": [3, 4, 5], "scales": [1.0, 2.0, 1.0], "norm_match": [False, True, False]}))
    for k in ref:
        assert torch.equal(ext2._steering_data[k][0].activations, ref[k])
        assert ext2._steering_data[k][0].scale == ext._steering_data[k][0].scale
    desc = _shm.put({"vecs": torch.randn(2, D)}, tag="read")
    n = ext2.set_readout_block(pickle.dumps({"keys": ["r0", "r1"], "vecs": None, "shm": desc, "layers": [1, 2]}))
    assert n == 2 and set(ext2._readout_index) == {"r0", "r1"}


def test_get_captured_states_shm_equals_pickled_path():
    ext = make_ext()
    runner = FakeRunner([("0-aaaa1111", {"output_residual_stream": [1, 2]}, 4, 0, 4), ("1-bbbb2222", {"output_residual_stream": [1]}, 3, 0, 3)])
    ext.model_runner = runner
    plan = build_plan(ext, runner)
    ext._step_plan = plan
    torch.manual_seed(0)
    h1, h2 = torch.randn(7, D), torch.randn(7, D)
    W._capture_gather(ext, runner, 1, h1, None, plan, plan.capture_rows(1))
    W._capture_gather(ext, runner, 2, h2, None, plan, plan.capture_rows(2))
    # pickled path on a twin
    ext_b = make_ext()
    ext_b.model_runner = runner
    ext_b._step_plan = plan
    W._capture_gather(ext_b, runner, 1, h1, None, plan, plan.capture_rows(1))
    W._capture_gather(ext_b, runner, 2, h2, None, plan, plan.capture_rows(2))
    ref = pickle.loads(ext_b.get_captured_states_many(["0", "1"]))
    blob = pickle.loads(ext.get_captured_states_shm(["0", "1"], "view"))
    assert "shm" in blob and blob["positions"]["0"] == ref["0"]["positions"]
    from vllm_lens._activations_plugin import _unpack_shm_capture

    outs = [SimpleNamespace(request_id="0"), SimpleNamespace(request_id="1")]
    got = _unpack_shm_capture(blob, "view", outs)
    assert torch.equal(got["0"]["residual_stream"], ref["0"]["residual_stream"])
    assert torch.equal(got["1"]["residual_stream"], ref["1"]["residual_stream"])
    assert got["0"]["positions"] == ref["0"]["positions"]
    assert all(hasattr(o, "lens_shm") for o in outs)  # views: the mapping handle rides along
    assert not ext._captured_states  # consumed


def test_plugin_ships_blocks_through_shm_when_enabled(monkeypatch):
    from vllm_lens import _activations_plugin as P

    block = {"keys": ["a"], "vecs": torch.randn(1, D), "layers": [1], "positions": [0], "scales": [1.0], "norm_match": [False], "modes": ["add"]}
    monkeypatch.delenv("VLLM_LENS_SHM", raising=False)
    assert P._ship_block(block, "steer") is block
    monkeypatch.setenv("VLLM_LENS_SHM", "1")
    shipped = P._ship_block(block, "steer")
    assert shipped["vecs"] is None and "shm" in shipped
    got, _ = _shm.get(shipped["shm"], copy=True)
    assert torch.equal(got["vecs"], block["vecs"])


def test_arena_reuse_and_growth():
    _shm.release_arena()
    t1 = {"a": torch.randn(4, D).to(torch.bfloat16)}
    d1 = _shm.put_arena(t1, tag="t")
    assert d1["arena"] and d1["gen"] >= 1
    got = _shm.get_arena(d1)
    assert torch.equal(got["a"], t1["a"])
    t2 = {"a": torch.randn(2, D).to(torch.bfloat16)}  # smaller: same segment reused
    d2 = _shm.put_arena(t2, tag="t")
    assert d2["shm"] == d1["shm"] and torch.equal(_shm.get_arena(d2)["a"], t2["a"])
    t3 = {"a": torch.randn(64, 64, D)}  # larger: grown, old name unlinked
    d3 = _shm.put_arena(t3, tag="t")
    assert d3["shm"] != d1["shm"] and torch.equal(_shm.get_arena(d3)["a"], t3["a"])
    assert not os.path.exists("/dev/shm" + d1["shm"])
    _shm.release_arena()
    assert not os.path.exists("/dev/shm" + d3["shm"])


def test_worker_shm_copy_mode_uses_arena():
    ext = make_ext()
    runner = FakeRunner([("0-aaaa1111", {"output_residual_stream": [1]}, 4, 0, 4)])
    ext.model_runner = runner
    plan = build_plan(ext, runner); ext._step_plan = plan
    h1 = torch.randn(4, D)
    W._capture_gather(ext, runner, 1, h1, None, plan, plan.capture_rows(1))
    blob = pickle.loads(ext.get_captured_states_shm(["0"], "copy"))
    assert blob["shm"]["arena"] is True
    from vllm_lens._activations_plugin import _unpack_shm_capture
    got = _unpack_shm_capture(blob, "copy", [SimpleNamespace(request_id="0")])
    assert torch.equal(got["0"]["residual_stream"][0], h1)
    _shm.release_arena()
