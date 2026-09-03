"""Fill the README's injection-mode test-matrix block from a results directory.

    python bench/render_injection_readme.py bench/results/injection_<timestamp>

Rewrites everything between ``<!-- INJECTION:BEGIN -->`` and ``<!-- INJECTION:END -->``
in README.md (run metadata, the correctness matrix, the throughput table, how to
re-run) and copies the throughput figure (PNG + PDF + data JSON) into ``bench/``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from test_injection_modes import markdown_table  # noqa: E402


def main(results_dir: str) -> None:
    d = Path(results_dir)
    s = json.loads((d / "summary.json").read_text())
    subprocess.run([sys.executable, str(HERE / "plot_injection.py"), str(d), "--out-dir", str(HERE)], check=True)
    runs = s["runs"]
    gpu = next((r["gpu"] for r in runs if r.get("gpu")), "GPU").replace("NVIDIA ", "")
    ver = next((r["versions"] for r in runs if r.get("versions")), {})
    models = sorted({r["model"] for r in runs}, key=lambda m: ("27B" not in m, m))
    engines = sorted({r["engine"] + (" +chunked" if r.get("chunked") else "") for r in runs})
    block = f"""<!-- INJECTION:BEGIN -->
`bench/test_injection_modes.py` on 1× {gpu} (vLLM {ver.get('vllm')}, torch {ver.get('torch')}, vllm-lens {ver.get('vllm_lens')}),
models {', '.join(models)}; engines {', '.join(engines)}; **{s['n_pass']}/{s['n_gated']} gated checks pass**
({'all' if s['all_pass'] else 'NOT all'}{f"; {s['n_info']} throughput gates on the 1.7B — decode-step time and the eager wall ratio — are below the measurement resolution of a 0.5–2 s call (control repeat spread 13–28%) and are reported as informational, not as passes" if s.get('n_info') else ''}). Every request in a batch carries its own unit vector; B is the number of requests
per `generate()` call; the marker is prompt position 10 (chunked engines also test position 70, i.e. a non-first 64-token
chunk). "cos(Δ, v)" / "‖Δ‖/(c·‖h‖) − 1" are the worst request of the batch; "other rows" is the max absolute change of any
non-marker row of the captured layer (0 = bit-identical). The HF reference is the same model in transformers with the
trainer's exact hook (`mxf/inject.py`, `h + coeff·‖h‖·v/‖v‖` on the decoder-layer output); the log-prob noise floor is
the vLLM-vs-HF difference on the clean prompt.

{markdown_table(s)}

![steering throughput by injection mode](bench/injection_throughput.png)

Re-run: `MODAL_PROFILE=safety-sahan modal run bench/modal_bench.py::test_injection` (both models, eager + graphs, chunked
engines, HF reference; ~25 min of B200), then `python bench/test_injection_modes.py bench/results/injection_<ts>` for the
summary and `python bench/render_injection_readme.py bench/results/injection_<ts>` for this block. Single engine by hand:
`python bench/test_injection_modes.py --model Qwen/Qwen3-1.7B --stage hf-ref --out ref.pt` then
`VLLM_LENS_CUDA_GRAPHS=1 python bench/test_injection_modes.py --model Qwen/Qwen3-1.7B --stage vllm --engine graphs --ref ref.pt --out graphs.json`.
<!-- INJECTION:END -->"""
    readme = REPO / "README.md"
    txt = readme.read_text()
    i, j = txt.index("<!-- INJECTION:BEGIN -->"), txt.index("<!-- INJECTION:END -->") + len("<!-- INJECTION:END -->")
    readme.write_text(txt[:i] + block + txt[j:])
    print("README injection block updated from", d)


if __name__ == "__main__":
    main(sys.argv[1])
