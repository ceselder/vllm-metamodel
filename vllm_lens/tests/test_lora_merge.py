"""CPU tests for vllm_lens._lora_merge (LoRA merge-on-publish) against fake vLLM linear layers.

The fakes reproduce the attribute / class-name surface the merger duck-types on
(``QKVParallelLinear``, ``MergedColumnParallelLinear``, ``RowParallelLinear``,
``ColumnParallelLinear`` with ``tp_size`` / ``tp_rank`` / ``output_sizes`` / head counts).

Run:  pytest vllm_lens/tests/test_lora_merge.py --noconftest
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from vllm_lens import _lora_merge as LM

H, R = 32, 4  # hidden, LoRA rank
HEADS, KV, HD = 4, 2, 8  # attention heads / kv heads / head size
INTER = 48


class LinearBase(torch.nn.Module):
    def __init__(self, in_size: int, out_size: int, tp_size: int = 1, tp_rank: int = 0, dtype=torch.bfloat16):
        super().__init__()
        self.input_size, self.output_size, self.tp_size, self.tp_rank = in_size, out_size, tp_size, tp_rank
        self.weight = torch.nn.Parameter(torch.randn(out_size, in_size).to(dtype), requires_grad=False)


class ColumnParallelLinear(LinearBase):
    def __init__(self, in_size, out_size, tp_size=1, tp_rank=0):
        super().__init__(in_size, out_size // tp_size, tp_size, tp_rank)
        self.output_size = out_size
        self.output_size_per_partition = out_size // tp_size


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self, in_size, output_sizes, tp_size=1, tp_rank=0):
        super().__init__(in_size, sum(output_sizes), tp_size, tp_rank)
        self.output_sizes = list(output_sizes)


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden, head_size, total_num_heads, total_num_kv_heads, tp_size=1, tp_rank=0):
        self.head_size, self.total_num_heads, self.total_num_kv_heads = head_size, total_num_heads, total_num_kv_heads
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)
        self.num_kv_head_replicas = max(1, tp_size // total_num_kv_heads)
        out = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden, out, tp_size, tp_rank)
        # per-partition rows: q shard + 2 kv shards
        rows = (self.num_heads + 2 * self.num_kv_heads) * head_size
        self.weight = torch.nn.Parameter(torch.randn(rows, hidden).to(torch.bfloat16), requires_grad=False)


class RowParallelLinear(LinearBase):
    def __init__(self, in_size, out_size, tp_size=1, tp_rank=0):
        super().__init__(in_size // tp_size, out_size, tp_size, tp_rank)
        self.input_size = in_size
        self.input_size_per_partition = in_size // tp_size


class Attn(torch.nn.Module):
    def __init__(self, tp_size=1, tp_rank=0):
        super().__init__()
        self.qkv_proj = QKVParallelLinear(H, HD, HEADS, KV, tp_size, tp_rank)
        self.o_proj = RowParallelLinear(HEADS * HD, H, tp_size, tp_rank)


class MLP(torch.nn.Module):
    def __init__(self, tp_size=1, tp_rank=0):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(H, [INTER, INTER], tp_size, tp_rank)
        self.down_proj = RowParallelLinear(INTER, H, tp_size, tp_rank)


class Layer(torch.nn.Module):
    def __init__(self, tp_size=1, tp_rank=0):
        super().__init__()
        self.self_attn = Attn(tp_size, tp_rank)
        self.mlp = MLP(tp_size, tp_rank)


class Inner(torch.nn.Module):
    def __init__(self, n, tp_size=1, tp_rank=0):
        super().__init__()
        self.layers = torch.nn.ModuleList([Layer(tp_size, tp_rank) for _ in range(n)])


class FakeModel(torch.nn.Module):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}

    def __init__(self, n=2, tp_size=1, tp_rank=0):
        super().__init__()
        self.model = Inner(n, tp_size, tp_rank)


def hf_shapes(model: FakeModel) -> dict[str, tuple[int, int]]:
    """HF module -> (out, in) of the FULL (unsharded) weight."""
    out = {}
    for i in range(len(model.model.layers)):
        p = f"model.layers.{i}."
        out[p + "self_attn.q_proj"] = (HEADS * HD, H)
        out[p + "self_attn.k_proj"] = (KV * HD, H)
        out[p + "self_attn.v_proj"] = (KV * HD, H)
        out[p + "self_attn.o_proj"] = (H, HEADS * HD)
        out[p + "mlp.gate_proj"] = (INTER, H)
        out[p + "mlp.up_proj"] = (INTER, H)
        out[p + "mlp.down_proj"] = (H, INTER)
    return out


def random_adapter(model: FakeModel, seed: int, scale: float = 0.05) -> dict[str, dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    mods = {}
    for name, (o, i) in hf_shapes(model).items():
        mods[name] = {"A": torch.randn(R, i, generator=g) / math.sqrt(i), "B": torch.randn(o, R, generator=g) * scale}
    return mods


def hf_full_weights(model: FakeModel) -> dict[str, torch.Tensor]:
    """Reassemble the FULL HF weights from a TP=1 fake model (qkv / gate_up split)."""
    full = {}
    for i, layer in enumerate(model.model.layers):
        p = f"model.layers.{i}."
        w = layer.self_attn.qkv_proj.weight.data
        q, k, v = HEADS * HD, KV * HD, KV * HD
        full[p + "self_attn.q_proj"], full[p + "self_attn.k_proj"], full[p + "self_attn.v_proj"] = w[:q], w[q : q + k], w[q + k : q + k + v]
        full[p + "self_attn.o_proj"] = layer.self_attn.o_proj.weight.data
        gu = layer.mlp.gate_up_proj.weight.data
        full[p + "mlp.gate_proj"], full[p + "mlp.up_proj"] = gu[:INTER], gu[INTER:]
        full[p + "mlp.down_proj"] = layer.mlp.down_proj.weight.data
    return {k: v.clone() for k, v in full.items()}


def reference_merge(w0: dict[str, torch.Tensor], mods, scaling: float) -> dict[str, torch.Tensor]:
    """HF-style merge: round(W0 + s B A) once, per module."""
    return {k: (w0[k].float() + scaling * (mods[k]["B"].float() @ mods[k]["A"].float())).to(w0[k].dtype) for k in w0}


# ---------------------------------------------------------------------------


def test_discover_layout_fused_and_row_parallel():
    torch.manual_seed(0)
    model = FakeModel(n=1)
    targets = LM.discover_targets(model)
    by = {t.vllm_name: t for t in targets}
    assert set(by) == {"model.layers.0.self_attn.qkv_proj", "model.layers.0.self_attn.o_proj", "model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj"}
    qkv = by["model.layers.0.self_attn.qkv_proj"]
    assert qkv.kind == "qkv" and [s.hf_name.rsplit(".", 1)[1] for s in qkv.subs] == ["q_proj", "k_proj", "v_proj"]
    assert [s.rows for s in qkv.subs] == [(0, 32), (32, 48), (48, 64)]
    gu = by["model.layers.0.mlp.gate_up_proj"]
    assert gu.kind == "merged_col" and [s.rows for s in gu.subs] == [(0, INTER), (INTER, 2 * INTER)]
    down = by["model.layers.0.mlp.down_proj"]
    assert down.kind == "row" and down.subs[0].col_src == (0, INTER) and down.subs[0].out_full == H
    summary = LM.layout_summary(targets, with_norms=True)
    assert all("frob" in s for t in summary for s in t["subs"])
    json.dumps(summary)  # RPC payload must be JSON-able


def test_merge_matches_hf_reference_and_unmerge_is_exact_gpu_mode():
    torch.manual_seed(1)
    model = FakeModel(n=2)
    w0 = hf_full_weights(model)
    mods = random_adapter(model, seed=1)
    scaling = 16 / math.sqrt(R)
    merger = LM.LoRAMerger(model)
    fp0 = merger.fingerprint()
    out = merger.merge(mods, scaling, keep_base="gpu")
    assert out["mode"] == "gpu" and out["n_modules"] == len(mods) and out["n_params"] == 8
    ref = reference_merge(w0, mods, scaling)
    got = hf_full_weights(model)
    for k in ref:
        assert torch.equal(got[k], ref[k]), k  # bit-identical: same single rounding
    assert merger.status()["merged"] and merger.base_bytes() == sum(t.numel() * 2 for t in merger.base.values())
    # replacing the adapter is computed from W0, not from the merged weights
    mods2 = random_adapter(model, seed=2)
    merger.merge(mods2, scaling, keep_base="gpu")
    got2 = hf_full_weights(model)
    ref2 = reference_merge(w0, mods2, scaling)
    for k in ref2:
        assert torch.equal(got2[k], ref2[k]), k
    u = merger.unmerge()
    assert u["how"] == "copy"
    assert merger.fingerprint() == fp0
    for k, v in hf_full_weights(model).items():
        assert torch.equal(v, w0[k])


def test_cpu_mode_is_exact_and_auto_picks_a_mode():
    torch.manual_seed(2)
    model = FakeModel(n=1)
    w0 = hf_full_weights(model)
    mods = random_adapter(model, seed=3)
    merger = LM.LoRAMerger(model)
    out = merger.merge(mods, 2.0, keep_base="cpu")
    assert out["mode"] == "cpu" and all(t.device.type == "cpu" for t in merger.base.values())
    ref = reference_merge(w0, mods, 2.0)
    for k, v in hf_full_weights(model).items():
        assert torch.equal(v, ref[k])
    merger.unmerge(release=True)
    assert not merger.base
    for k, v in hf_full_weights(model).items():
        assert torch.equal(v, w0[k])
    out = merger.merge(mods, 2.0, keep_base="auto")
    assert out["mode"] in ("gpu", "cpu")
    merger.unmerge(release=True)


def test_none_mode_replaces_adapter_by_subtraction_with_bounded_drift():
    torch.manual_seed(3)
    model = FakeModel(n=1)
    w0 = hf_full_weights(model)
    merger = LM.LoRAMerger(model)
    scaling = 2.0
    mods = [random_adapter(model, seed=10 + k) for k in range(6)]
    for m in mods:
        out = merger.merge(m, scaling, keep_base="none")
        assert out["mode"] == "none" and out["base_bytes"] == 0
    # currently merged: mods[-1]; compare with an exact merge of it
    ref = reference_merge(w0, mods[-1], scaling)
    got = hf_full_weights(model)
    def ulps(x: torch.Tensor, ref_t: torch.Tensor) -> float:  # max |diff| in bf16 spacings at the RMS magnitude
        return float((x.float() - ref_t.float()).abs().max()) / (float(ref_t.float().pow(2).mean().sqrt()) * 2.0**-7)

    worst = max(ulps(got[k], ref[k]) for k in ref)
    # random walk of half-spacing roundings over 6 publishes: a few spacings at most, never wildly off
    assert 0.0 < worst <= 8.0, worst
    u = merger.unmerge()
    assert u["how"] == "subtract"
    assert max(ulps(v, w0[k]) for k, v in hf_full_weights(model).items()) <= 8.0


def test_compare_to_exact_reports_zero_for_exact_merge_and_nonzero_for_drift():
    torch.manual_seed(4)
    model = FakeModel(n=1)
    merger = LM.LoRAMerger(model)
    a0, a1 = random_adapter(model, seed=20), random_adapter(model, seed=21)
    merger.merge(a0, 2.0, keep_base="gpu")
    z = merger.compare_to_exact(a0, 2.0)
    assert z["max_diff_ulps"] == 0.0 and z["frac_changed"] == 0.0
    merger.unmerge()  # copies retained
    merger.merge(a1, 2.0, keep_base="none")
    merger.merge(a0, 2.0, keep_base="none")
    d = merger.compare_to_exact(a0, 2.0)
    assert d["n_params"] == 4 and 0.0 < d["max_diff_ulps"] <= 8.0 and 0.0 < d["frac_changed"] < 1.0, d
    merger.unmerge(how="subtract")
    b = merger.compare_to_exact(None)
    assert 0.0 < b["max_diff_ulps"] <= 8.0, b
    # exact restore with the retained copies
    merger.merge(a0, 2.0, keep_base="gpu")
    merger.unmerge(how="copy", release=True)
    assert merger.compare_to_exact.__doc__  # still importable; copies are gone
    with pytest.raises(ValueError):
        merger.compare_to_exact(a0, 2.0)


def test_tp_sharding_slices_match_weight_loader_rules():
    """tp_size=2, rank 1: q rows [q_shard:2q_shard) of the full q_proj, kv shard id = rank // replicas,
    gate/up rows [half:] of each, o_proj / down_proj input columns [half:]."""
    torch.manual_seed(5)
    model = FakeModel(n=1, tp_size=2, tp_rank=1)
    targets = {t.vllm_name: t for t in LM.discover_targets(model)}
    qkv = targets["model.layers.0.self_attn.qkv_proj"]
    q, k, v = qkv.subs
    assert q.row_src == (HEADS * HD // 2, HEADS * HD) and q.rows == (0, HEADS * HD // 2)
    assert k.row_src == (KV * HD // 2, KV * HD) and v.row_src == (KV * HD // 2, KV * HD)  # 2 kv heads / tp 2 -> 1 head each
    gu = targets["model.layers.0.mlp.gate_up_proj"]
    assert [s.row_src for s in gu.subs] == [(INTER // 2, INTER), (INTER // 2, INTER)] and [s.rows for s in gu.subs] == [(0, INTER // 2), (INTER // 2, INTER)]
    o = targets["model.layers.0.self_attn.o_proj"]
    assert o.subs[0].col_src == (HEADS * HD // 2, HEADS * HD) and o.subs[0].rows == (0, H)
    # merging on the shard equals slicing the full merged weight
    mods = random_adapter(model, seed=30)
    merger = LM.LoRAMerger(model)
    w_before = {n: t.param.detach().clone() for n, t in targets.items()}
    merger.merge(mods, 1.5, keep_base="gpu")
    for name, t in targets.items():
        for s in t.subs:
            delta_full = 1.5 * (mods[s.hf_name]["B"].float() @ mods[s.hf_name]["A"].float())
            shard = delta_full[s.row_src[0] : s.row_src[1], s.col_src[0] : s.col_src[1]]
            expect = (w_before[name][s.rows[0] : s.rows[1]].float() + shard).to(torch.bfloat16)
            assert torch.equal(t.param[s.rows[0] : s.rows[1]], expect), (name, s.hf_name)


def test_adapter_parsing_config_scaling_and_errors():
    model = FakeModel(n=1)
    mods = random_adapter(model, seed=40)
    tensors = {}
    for m, ab in mods.items():
        tensors[f"base_model.model.{m}.lora_A.weight"] = ab["A"]
        tensors[f"base_model.model.{m}.lora_B.default.weight"] = ab["B"]  # adapter-named variant
    parsed = LM.parse_adapter_tensors(tensors)
    assert set(parsed) == set(mods) and torch.equal(parsed[next(iter(mods))]["A"], mods[next(iter(mods))]["A"])
    assert LM.scaling_from_config({"r": 64, "lora_alpha": 16, "use_rslora": True}) == pytest.approx(2.0)
    assert LM.scaling_from_config({"r": 64, "lora_alpha": 16}) == pytest.approx(0.25)
    merger = LM.LoRAMerger(model)
    with pytest.raises(ValueError, match="do not match any linear layer"):
        merger.merge({"model.layers.0.self_attn.nonexistent": mods["model.layers.0.self_attn.q_proj"]}, 1.0)
    bad = {"model.layers.0.self_attn.q_proj": {"A": torch.randn(R, H + 1), "B": torch.randn(HEADS * HD, R)}}
    with pytest.raises(ValueError, match="do not match the served weight"):
        merger.merge(bad, 1.0)
    with pytest.raises(ValueError, match="missing lora_A or lora_B"):
        LM.parse_adapter_tensors({"base_model.model.x.lora_A.weight": torch.randn(R, H)})
    assert merger.unmerge()["how"] == "nothing"


def test_synth_adapter_hits_requested_relative_norm_and_roundtrips_through_dir(tmp_path):
    torch.manual_seed(6)
    model = FakeModel(n=1)
    merger = LM.LoRAMerger(model)
    layout = LM.layout_summary(merger.targets, with_norms=True)
    tensors, cfg = LM.synth_adapter(layout, rank=R, rel_norm=0.01, seed=7, dtype=torch.float32)
    scaling = LM.scaling_from_config(cfg)
    mods = LM.parse_adapter_tensors(tensors)
    for t in layout:
        for s in t["subs"]:
            d = scaling * (mods[s["hf_name"]]["B"] @ mods[s["hf_name"]]["A"])
            assert float(d.norm()) / s["frob"] == pytest.approx(0.01, rel=1e-3)
    LM.save_adapter(str(tmp_path / "ad"), tensors, cfg)
    mods2, scaling2, cfg2 = LM.load_adapter_dir(str(tmp_path / "ad"))
    assert scaling2 == pytest.approx(scaling) and set(mods2) == set(mods) and set(cfg2["target_modules"]) == {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    w0 = hf_full_weights(model)
    merger.merge(mods2, scaling2, keep_base="gpu")
    ref = reference_merge(w0, mods2, scaling2)
    for k, v in hf_full_weights(model).items():
        assert torch.equal(v, ref[k])


def test_name_map_and_language_model_prefix_fallback():
    model = FakeModel(n=1)
    merger = LM.LoRAMerger(model)
    # adapter saved from a wrapper checkpoint naming: model.language_model.layers.N -> fall back to model.layers.N
    mods = {("model.language_model." + k[len("model."):]): v for k, v in random_adapter(model, seed=50).items()}

    class Mapper:
        def _map_name(self, n):
            return n.replace("model.language_model.", "model.")

    merger.name_map = Mapper()._map_name
    out = merger.merge(mods, 1.0, keep_base="gpu")
    assert out["n_modules"] == len(mods)
    merger.unmerge(release=True)


def test_lora_wrapped_linears_are_discovered_once_under_the_wrapper_name():
    """enable_lora=True: vLLM replaces each linear by a ``*WithLoRA`` wrapper holding ``base_layer``;
    the weight must be targeted exactly once (the 1.7B run without this dedup merged every delta twice)."""

    class Wrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base_layer = base
            self.output_slices = (base.weight.shape[0],)

    model = FakeModel(n=1)
    layer = model.model.layers[0]
    layer.self_attn.qkv_proj = Wrapper(layer.self_attn.qkv_proj)
    layer.mlp.down_proj = Wrapper(layer.mlp.down_proj)
    targets = LM.discover_targets(model)
    names = [t.vllm_name for t in targets]
    assert names.count("model.layers.0.self_attn.qkv_proj") == 1 and names.count("model.layers.0.mlp.down_proj") == 1
    assert not any(n.endswith(".base_layer") for n in names) and len(names) == 4
    w0 = {n: t.param.detach().clone() for n, t in zip(names, targets)}
    merger = LM.LoRAMerger(model)
    mods = random_adapter(model, seed=60)
    merger.merge(mods, 1.0, keep_base="gpu")
    qkv = next(t for t in merger.targets if t.vllm_name == "model.layers.0.self_attn.qkv_proj")
    q = qkv.subs[0]
    expect = (w0["model.layers.0.self_attn.qkv_proj"][q.rows[0]:q.rows[1]].float() + (mods[q.hf_name]["B"].float() @ mods[q.hf_name]["A"].float())).to(torch.bfloat16)
    assert torch.equal(layer.self_attn.qkv_proj.base_layer.weight[q.rows[0]:q.rows[1]], expect)  # applied exactly once


def test_single_hf_module_spanning_several_output_slices_is_merged_over_all_rows():
    """Qwen3-Next GDN: HF `in_proj_qkvz` [q+k+v+z, hidden] is ONE LoRA module but vLLM's layer has
    output_sizes [q, k, v, z]; the delta must land on every row, not just the first slice."""

    class Inner2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkvz = MergedColumnParallelLinear(H, [8, 8, 16, 16])
            self.in_proj_ba = MergedColumnParallelLinear(H, [4, 4])

    class M(torch.nn.Module):
        packed_modules_mapping = {"in_proj_qkvz": ["in_proj_qkvz"], "in_proj_ba": ["in_proj_ba"]}

        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Inner2()])

    torch.manual_seed(8)
    model = M()
    targets = {t.vllm_name: t for t in LM.discover_targets(model)}
    q = targets["model.layers.0.in_proj_qkvz"]
    assert q.kind == "merged_col_single" and len(q.subs) == 1 and q.subs[0].rows == (0, 48) and q.subs[0].out_full == 48
    w0 = {n: t.param.detach().clone() for n, t in targets.items()}
    mods = {"model.layers.0.in_proj_qkvz": {"A": torch.randn(R, H), "B": torch.randn(48, R) * 0.1},
            "model.layers.0.in_proj_ba": {"A": torch.randn(R, H), "B": torch.randn(8, R) * 0.1}}
    merger = LM.LoRAMerger(model)
    merger.merge(mods, 1.0, keep_base="gpu")
    for name, t in targets.items():
        ab = mods[name]
        expect = (w0[name].float() + ab["B"].float() @ ab["A"].float()).to(torch.bfloat16)
        assert torch.equal(t.param, expect), name
        assert not torch.equal(t.param[-1], w0[name][-1])  # the LAST slice moved too
    merger.unmerge()
    for name, t in targets.items():
        assert torch.equal(t.param, w0[name])
