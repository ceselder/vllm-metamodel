"""CPU tests for vllm_lens.metamodel (fake engine, no vLLM needed)."""
import math

import torch

from vllm_lens.metamodel import capabilities, readout_max, readout_scores

D = 16


class _Out:
    def __init__(self, values, positions):
        self.readout = [{"values": values, "positions": positions, "layers": [42]}]


class FakeSP:
    def __init__(self, max_tokens, temperature, extra_args):
        self.max_tokens, self.temperature, self.extra_args = max_tokens, temperature, extra_args


class FakeLLM:
    """Computes the readout exactly like the worker would: cos/dot of a fake hidden row with the direction."""

    def __init__(self, early_exit=True, hidden=None):
        self.early_exit, self.calls = early_exit, []
        self.hidden = hidden

    def collective_rpc(self, name, args=(), kwargs=None):
        # post7: capabilities() installs the hooks first (several capabilities are only known then)
        assert name in ("install_hooks", "lens_capabilities"), name
        if name == "install_hooks":
            return [None]
        return [{"early_exit": self.early_exit, "early_exit_reason": "" if self.early_exit else "prefix caching on"}]

    def generate(self, prompts, params, lora_request=None, use_tqdm=False):
        self.calls.append((prompts, params, lora_request))
        outs = []
        for p, sp in zip(prompts, params):
            rv = sp.extra_args["apply_readout_vectors"][0]
            n_tok = len(p["prompt_token_ids"])
            k = rv.positions["last"] if isinstance(rv.positions, dict) else n_tok
            pos = list(range(max(0, n_tok - k), n_tok))
            h = self.hidden[: n_tok] if self.hidden is not None else torch.arange(n_tok, dtype=torch.float32)[:, None].repeat(1, D)
            d = rv.activations[0]
            rows = h[pos]
            if rv.metric == "cos":
                v = torch.nn.functional.cosine_similarity(rows, d[None], dim=-1)
            else:
                v = rows @ d
            outs.append(_Out((v + rv.bias)[None], pos))
        return outs


def test_readout_scores_shapes_positions_and_padding():
    llm = FakeLLM()
    ids = [list(range(10)), list(range(3))]           # second text shorter than the window
    dirs = torch.randn(2, D)
    values, positions = readout_scores(llm, ids, dirs, layer=42, positions={"last": 5}, sampling_params_cls=FakeSP)
    assert values.shape == (2, 1, 5)
    assert positions[0] == [5, 6, 7, 8, 9] and positions[1] == [0, 1, 2]
    assert not torch.isnan(values[0]).any() and torch.isnan(values[1, 0, 3:]).all()
    prompts, params, lora = llm.calls[0]
    assert lora is None and all(sp.max_tokens == 1 for sp in params)
    assert all(sp.extra_args.get("lens_early_exit") is True for sp in params)


def test_early_exit_dropped_when_unsupported():
    llm = FakeLLM(early_exit=False)
    readout_scores(llm, [list(range(6))], torch.randn(1, D), layer=42, sampling_params_cls=FakeSP)
    _, params, _ = llm.calls[0]
    assert "lens_early_exit" not in params[0].extra_args
    assert capabilities(llm)["early_exit"] is False


def test_readout_max_matches_direct_computation():
    torch.manual_seed(0)
    hidden = torch.randn(12, D)
    llm = FakeLLM(hidden=hidden)
    ids = [list(range(12))]
    d = torch.randn(1, D)
    r = readout_max(llm, ids, d, layer=42, positions={"last": 4}, sampling_params_cls=FakeSP)
    expect = torch.nn.functional.cosine_similarity(hidden[8:12], d, dim=-1).max()
    assert math.isclose(float(r[0]), float(expect), rel_tol=1e-6)


def test_dot_metric_with_bias_and_direction_shape_check():
    llm = FakeLLM(hidden=torch.ones(5, D))
    values, _ = readout_scores(llm, [list(range(5))], torch.ones(1, D), layer=42, positions={"last": 1},
                               metric="dot", bias=-1.0, sampling_params_cls=FakeSP)
    assert math.isclose(float(values[0, 0, 0]), D - 1.0)
    try:
        readout_scores(llm, [list(range(5))], torch.ones(3, D), layer=42, sampling_params_cls=FakeSP)
        assert False, "shape mismatch must raise"
    except ValueError:
        pass
