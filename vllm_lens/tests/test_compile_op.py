"""CPU-only tests for the torch.compile-compatible hooks (vllm-metamodels 1.1.0.post7, ``_compile_op``).

A toy fused-residual decoder stack (keyword-calling layers, like Qwen3-Next) is wired to the
fork's worker extension through the custom-op hooks and compiled with
``torch.compile(fullgraph=True)`` -- ``aot_eager`` (functionalization + autograd tracing, no
codegen) and ``inductor`` when a C++ toolchain is available.  The compiled model must apply the
same per-request steering / embedding replacement / capture / readout as the eager hooks, with
zero graph breaks, and hooks registered AFTER compilation must be detected as dead.

Run without upstream's GPU conftest:  pytest vllm_lens/tests/test_compile_op.py --noconftest
"""

from __future__ import annotations

import pickle
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from test_steering_index import FakeRunner, W, make_ext, sv3d  # noqa: E402

from vllm_lens import ReadoutVector, SteeringVector  # noqa: E402
from vllm_lens import _compile_op as C  # noqa: E402
from vllm_lens._helpers.types import EMBED_LAYER_INDEX  # noqa: E402

D = 8


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(D, D)
        self.b = torch.nn.Linear(D, D)

    def forward(self, positions, hidden_states, residual=None):
        if residual is None:
            residual = hidden_states
            h = torch.nn.functional.rms_norm(hidden_states, (D,))
        else:
            residual = hidden_states + residual
            h = torch.nn.functional.rms_norm(residual, (D,))
        return self.b(torch.relu(self.a(h))), residual


class _Stack(torch.nn.Module):
    """``model.model.layers`` like a vLLM causal LM; layers called by KEYWORD."""

    def __init__(self, n: int):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer() for _ in range(n)])
        self.embed = torch.nn.Embedding(32, D)

    def forward(self, input_ids, positions):
        h = self.embed(input_ids)
        residual = None
        for layer in self.model.layers:
            h, residual = layer(positions=positions, hidden_states=h, residual=residual)
        return h + residual


def _ctx(runner):
    import vllm.forward_context as fc

    meta = SimpleNamespace(query_start_loc=torch.tensor(runner.query_start_loc.np))
    ctx = SimpleNamespace(attn_metadata={"layer0": meta})
    if hasattr(fc, "_ctx"):
        fc._ctx = ctx
    else:  # real vLLM: patch the accessors the worker uses
        W.get_forward_context = lambda: ctx
        W.is_forward_context_available = lambda: True
    return ctx


def _wire(ext, model, compile_mode: bool):
    """Register the extension's hooks on ``model`` the way install_hooks does."""
    layers = model.model.layers
    ext._compile_mode = compile_mode
    ext._first_layer_idx = 0
    for i, layer in enumerate(layers):
        if i == 0:
            layer.register_forward_pre_hook(
                C.make_pre_hook() if compile_mode else W._make_pre_hook(ext, 0), with_kwargs=True
            )
        layer.register_forward_hook(C.make_post_hook(i) if compile_mode else W._make_hook(ext, i))
    if compile_mode:
        C._ACTIVE = ext


def _setup(n_layers=3, T=6):
    torch.manual_seed(0)
    model = _Stack(n_layers).eval()
    runner = FakeRunner([
        ("a-00000000", {"_steering_id": "_steer_0", "output_residual_stream": [1], "_readout_id": "_read_0"}, T, 0, T),
        ("b-00000000", {"_steering_id": "_steer_1"}, T, 0, T),
    ])
    runner.model = model  # parameters() for device / dtype
    ids = torch.randint(0, 32, (2 * T,))
    pos = torch.cat([torch.arange(T), torch.arange(T)])
    return model, runner, ids, pos


def _register(ext, T):
    # request a: add at layer 1 pos 2 + embedding replace at pos 0; request b: norm-matched add at layer 2 pos 4
    ext.set_steering_data_many(pickle.dumps({
        "_steer_0": [sv3d(1, [2]), SteeringVector(activations=torch.randn(1, 1, D), layer_indices=[EMBED_LAYER_INDEX],
                                                  mode="replace", position_indices=[0])],
        "_steer_1": [sv3d(2, [4], norm_match=True, scale=0.5)],
    }))
    ext.set_readout_block(pickle.dumps({"keys": ["_read_0"], "vecs": torch.randn(1, D), "layers": [2], "positions": [{"last": 2}]}))


def _run(ext, model, runner, ids, pos):
    ext.model_runner = runner
    _ctx(runner)
    with torch.no_grad():
        out = model(ids, pos).clone()
    ext._flush_host_blocks()
    cap = ext._pop_activations("a-00000000")["residual_stream"].clone() if "a-00000000" in ext._captured_states else None
    read = ext._pop_readouts("a-00000000")[0]["values"].clone() if "a-00000000" in ext._readouts else None
    return out, cap, read


def _eager_reference():
    model, runner, ids, pos = _setup()
    ext = make_ext()
    _wire(ext, model, compile_mode=False)
    with torch.no_grad():
        clean = model(ids, pos).clone()  # hooks idle (nothing registered)
    _register(ext, 6)
    out, cap, read = _run(ext, model, runner, ids, pos)
    return model.state_dict(), ids, pos, clean, out, cap, read


BACKENDS = ["aot_eager"] + (["inductor"] if shutil.which("g++") or shutil.which("clang++") else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_compiled_op_hooks_match_eager_hooks(backend):
    sd, ids, pos, clean_ref, out_ref, cap_ref, read_ref = _eager_reference()
    assert not torch.allclose(clean_ref, out_ref), "steering must change the output"

    model, runner, _, _ = _setup()
    model.load_state_dict(sd)
    ext = make_ext()
    _wire(ext, model, compile_mode=True)
    torch._dynamo.reset()
    cm = torch.compile(model, fullgraph=True, backend=backend, dynamic=False)
    ext.model_runner = runner
    _ctx(runner)
    with torch.no_grad():
        clean = cm(ids, pos).clone()  # compiles here, hooks present but nothing registered
    assert torch.allclose(clean, clean_ref, atol=1e-5), "compiled clean forward differs"
    n0 = ext._stats["op_calls"]
    assert n0 == 4, f"expected 3 post-hook + 1 pre-hook op calls per forward, got {n0}"  # traced hooks fire
    # request a asks for capture / readout in its extra_args: drop what the clean pass collected
    ext._flush_host_blocks()
    ext._captured_states.clear(); ext._captured_positions.clear(); ext._readouts.clear()
    _register(ext, 6)
    out, cap, read = _run(ext, cm, runner, ids, pos)
    assert torch.allclose(out, out_ref, atol=1e-5), f"compiled steering differs: {(out - out_ref).abs().max()}"
    assert cap is not None and torch.allclose(cap, cap_ref, atol=1e-5), "compiled capture differs"
    assert read is not None and torch.allclose(read, read_ref, atol=1e-5), "compiled readout differs"
    assert ext._stats["rows_steered"] >= 2 and ext._stats["rows_replaced"] == 1 and ext._stats["errors"] == 0
    # data-dependent plans change between passes without recompiling (the op is opaque)
    ext.clear_steering_data_many(["_steer_0", "_steer_1"])
    out2, _, _ = _run(ext, cm, runner, ids, pos)
    assert torch.allclose(out2, clean_ref, atol=1e-5)


def test_zero_graph_breaks_and_one_op_per_layer():
    model, runner, ids, pos = _setup()
    ext = make_ext()
    _wire(ext, model, compile_mode=True)
    ext.model_runner = runner
    _ctx(runner)
    torch._dynamo.reset()
    ex = torch._dynamo.explain(model)(ids, pos)
    assert ex.graph_break_count == 0 and ex.graph_count == 1, (ex.graph_break_count, ex.graph_count)
    n_ops = sum("lens_layer_" in str(n.target) for g in ex.graphs for n in g.graph.nodes)
    assert n_ops == 4  # 3 layers + the layer-0 pre-hook


def test_hooks_added_after_compile_are_dead_and_install_hooks_refuses():
    model, runner, ids, pos = _setup()
    ext = make_ext()
    torch._dynamo.reset()
    cm = torch.compile(model, fullgraph=True, backend="aot_eager", dynamic=False)
    with torch.no_grad():
        _ = cm(ids, pos)
    _wire(ext, model, compile_mode=True)  # too late
    ext.model_runner = runner
    _ctx(runner)
    with torch.no_grad():
        _ = cm(ids, pos)
    assert ext._stats["op_calls"] == 0, "hooks registered after compilation must not run"
    # install_hooks detects the situation through the compiled-module marker
    ext2 = make_ext()
    ext2._hooks_installed = False
    ext2.parallel_config = SimpleNamespace(tensor_parallel_size=1)
    marker = torch.nn.Module()
    marker.do_not_compile = False
    marker.compiled = True
    model.compiled_marker = marker
    ext2.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(mode=SimpleNamespace(name="VLLM_COMPILE"), cudagraph_mode=SimpleNamespace(name="FULL_DECODE_ONLY")),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(pipeline_parallel_size=1),
    )
    ext2.model_runner = SimpleNamespace(model=model, input_batch=runner.input_batch, requests=runner.requests,
                                        _model_forward=lambda *a, **k: None, vllm_config=ext2.vllm_config)
    with pytest.raises(RuntimeError, match="compiled before the hooks"):
        ext2.install_hooks()


def test_model_is_compiled_and_config_rules():
    mk = lambda eager, mode: SimpleNamespace(model_config=SimpleNamespace(enforce_eager=eager),  # noqa: E731
                                              compilation_config=SimpleNamespace(mode=mode))
    assert C.model_is_compiled(mk(False, SimpleNamespace(name="VLLM_COMPILE")))
    assert not C.model_is_compiled(mk(False, SimpleNamespace(name="NONE")))
    assert not C.model_is_compiled(mk(True, SimpleNamespace(name="VLLM_COMPILE")))
    assert not C.model_is_compiled(mk(False, None))
    assert C.model_is_compiled(mk(False, 3)) and not C.model_is_compiled(mk(False, 0))


def test_embed_target_validation():
    assert W._embed_target(torch.zeros(6, D), 6) is not None
    with pytest.raises(W.EmbedInjectionError):
        W._embed_target(torch.zeros(4, D), 6)
    with pytest.raises(W.EmbedInjectionError):
        W._embed_target(torch.zeros(6, D, dtype=torch.long), 6)


def test_plugin_compile_env_keeps_compile_mode_and_forces_decode_only_graphs(monkeypatch):
    pytest.importorskip("vllm.config")
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CompilationMode, CUDAGraphMode

    from vllm_lens import _activations_plugin as P

    monkeypatch.setenv("VLLM_LENS_COMPILE", "1")
    ea = SimpleNamespace(compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE), enforce_eager=False)
    assert P._configure_cuda_graphs(ea) is True
    assert ea.compilation_config.mode == CompilationMode.VLLM_COMPILE and not ea.enforce_eager
    assert ea.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    ea = SimpleNamespace(compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE), enforce_eager=False)
    P._configure_cuda_graphs(ea)
    assert ea.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY  # prefill never inside a graph
    monkeypatch.delenv("VLLM_LENS_COMPILE")
    ea = SimpleNamespace(compilation_config=CompilationConfig(mode=CompilationMode.VLLM_COMPILE), enforce_eager=False)
    assert P._configure_cuda_graphs(ea) is False and ea.enforce_eager  # pre-post7: compile -> forced eager
