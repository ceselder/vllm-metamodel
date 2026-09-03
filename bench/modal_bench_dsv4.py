"""Modal launcher for bench/test_injection_dsv4.py: vllm-metamodel on DeepSeek-V4-Flash-0731
(hyper-connection / multi-stream architecture), vLLM 0.27.1, TP4.

App ``vllm-metamodel-bench-dsv4`` (our own). The image REPLICATES the NLA session's
training image layer-by-layer (nla-deepseek-v4/scripts_local/modal_common.py: CUDA 13.0.1
devel base, vllm==0.27.1, DeepGEMM from source, kernels pin) so the expensive layers are
cache hits, then installs THIS checkout over vllm-lens 1.2.1 (same distribution name).
The ``nla-dsv4`` volume (weights in /vol/hf, JIT caches in /vol/cache) is mounted
READ-ONLY; caches are copied into the container at start so DeepGEMM/flashinfer do not
recompile. The NLA session's easyNLA repo is mounted at /repo (reference implementation,
read-only local mount) -- nothing of theirs is written to.

    modal run bench/modal_bench_dsv4.py::run1       # eager correctness matrix
    modal run bench/modal_bench_dsv4.py::run2       # VLLM_LENS_CUDA_GRAPHS=1 correctness + throughput
    modal run bench/modal_bench_dsv4.py::run3       # throughput repeat (interleaved, 3 repeats), graphs
    modal run bench/modal_bench_dsv4.py::main --engine eager --extra "--batches 64"

Results: bench/results/dsv4_<timestamp>/<engine>[_throughput].json (+ summary via
bench/test_injection_modes.py <dir>).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
NLA = Path(os.environ.get("NLA_DSV4_DIR", "/home/celeste/nla-deepseek-v4"))
EASYNLA = NLA / "easyNLA"
GPU = os.environ.get("DSV4_GPU", "B200:4")
TP = int(GPU.split(":")[1]) if ":" in GPU else 4

app = modal.App("vllm-metamodel-bench-dsv4")
vol = modal.Volume.from_name("nla-dsv4")  # mounted read-only below

# --- layers identical to nla-deepseek-v4/scripts_local/modal_common.py (cache hits) ---
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "clang", "gcc-12", "g++-12")
    .pip_install(
        "ninja", "cmake", "vllm==0.27.1", "vllm-lens==1.2.1", "transformers>=5.12", "peft>=0.17", "wandb",
        "safetensors", "pyyaml", "pyarrow", "datasets", "anthropic", "orjson", "httpx", "tqdm", "numpy",
        "huggingface_hub",
    )
    .run_commands(
        "git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /build/DeepGEMM",
        "cd /build/DeepGEMM && CUDA_HOME=/usr/local/cuda python -m pip install --no-build-isolation .",
    )
    .pip_install("kernels>=0.16.0,<0.17.0")
    # --- ours: the fork replaces vllm-lens 1.2.1 (same dist name) ---------------------
    .add_local_dir(
        REPO, "/opt/vllm-metamodel", copy=True,
        ignore=[".git", "bench/results", "**/__pycache__", "uv.lock", ".venv", "*.png", "*.pdf"],
    )
    .run_commands(
        "pip install --no-deps --force-reinstall /opt/vllm-metamodel",
        "python -c 'import vllm_lens; from vllm_lens import _worker_ext as W; "
        'assert hasattr(W.HiddenStatesExtension, "lens_capabilities"); print(vllm_lens.__version__)\'',
    )
    .env({
        "PYTHONPATH": "/repo",                       # NLA session's easyNLA (reference impl)
        "HF_HOME": "/vol/hf",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUDA_HOME": "/usr/local/cuda",              # DeepGEMM JIT needs nvcc at runtime
        "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR": "1",
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",    # pickled steering RPC payloads (vLLM >= 0.27)
        "NVCC_PREPEND_FLAGS": "-ccbin g++-12",
        "MAX_JOBS": "4",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_CACHE_ROOT": "/root/.cache/vllm",      # seeded from the (read-only) volume at start
        "DG_CACHE_DIR": "/root/.cache/deepgemm",
    })
    .add_local_dir(EASYNLA, remote_path="/repo", ignore=["**/.git/**", "**/__pycache__/**", "**/.pytest_cache/**"])
    .add_local_file(HERE / "test_injection_dsv4.py", "/bench/test_injection_dsv4.py")
    .add_local_file(HERE / "test_injection_modes.py", "/bench/test_injection_modes.py")
    .add_local_file(HERE / "bench_steering.py", "/bench/bench_steering.py")
)


def _seed_caches() -> str:
    """Copy the NLA session's JIT caches (DeepGEMM under vLLM's cache root, flashinfer) from the
    read-only volume into the container so nothing recompiles; report sizes."""
    import shutil

    notes = []
    for src, dst in (("/vol/cache/vllm", "/root/.cache/vllm"), ("/vol/cache/flashinfer", "/root/.cache/flashinfer")):
        if os.path.isdir(src) and not os.path.exists(dst):
            t0 = time.time()
            shutil.copytree(src, dst, symlinks=True)
            size = subprocess.run(["du", "-sh", dst], capture_output=True, text=True).stdout.split()[0]
            notes.append(f"{src} -> {dst} ({size}, {time.time() - t0:.0f}s)")
    return "; ".join(notes) or "no caches copied"


@app.function(image=image, gpu=GPU, volumes={"/vol": vol.read_only()}, timeout=75 * 60, cpu=32, memory=512 * 1024)
def run_dsv4(engine: str, extra: str, tag: str) -> dict:
    print(f"[modal] caches: {_seed_caches()}", flush=True)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout, flush=True)
    out = f"/tmp/{tag}.json"
    cmd = [sys.executable, "/bench/test_injection_dsv4.py", "--engine", engine, "--tp", str(TP), "--out", out, *extra.split()]
    print(f"[modal] >>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:  # stream live so progress is visible in `modal run` output
        print(line, end="", flush=True)
        lines.append(line)
        if len(lines) > 4000:
            del lines[:1000]
    proc.wait()
    tail = "".join(lines)[-30000:]
    rec = {"returncode": proc.returncode, "elapsed_s": time.time() - t0, "log_tail": tail, "gpu": GPU}
    if os.path.exists(out):
        with open(out) as f:
            rec["result"] = json.load(f)
    print(f"[modal] <<< {tag} rc={proc.returncode} in {rec['elapsed_s']:.0f}s", flush=True)
    return rec


def _save(dest: Path, name: str, rec: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(f"[local] saved {dest / (name + '.json')} rc={rec['returncode']} ({rec['elapsed_s']:.0f}s)", flush=True)
    if rec["returncode"] != 0:
        print(rec["log_tail"][-6000:], flush=True)


def _summarize(dest: Path) -> None:
    from test_injection_modes import markdown_table, summarize

    s = summarize(dest)
    (dest / "summary.json").write_text(json.dumps(s, indent=1))
    (dest / "summary.md").write_text(markdown_table(s))
    print(markdown_table(s))
    for c in s["checks"]:
        print(f"[{'PASS' if c['ok'] else ('n/a ' if c['ok'] is None else 'FAIL')}] {c['engine']} {c['case']}: {c['check']}  {c['detail'][:200]}")
    print(f"{s['n_pass']}/{s['n_gated']} gated checks pass, {s['n_info']} informational" + (" -- ALL PASS" if s["all_pass"] else " -- SOME FAILED"))
    print(f"[local] results in {dest}")


def _dest(label: str) -> Path:
    return HERE / "results" / f"dsv4_{label}_{time.strftime('%Y%m%d_%H%M%S')}"


@app.local_entrypoint()
def run1(extra: str = "--skip-throughput"):
    """(1) eager correctness matrix."""
    dest = _dest("run1_eager")
    _save(dest, "eager", run_dsv4.remote("eager", extra, "eager"))
    _summarize(dest)


@app.local_entrypoint()
def run2(extra: str = ""):
    """(2) CUDA graphs: correctness matrix + throughput (2 repeats)."""
    dest = _dest("run2_graphs")
    _save(dest, "graphs", run_dsv4.remote("graphs", extra, "graphs"))
    _summarize(dest)


@app.local_entrypoint()
def run3(extra: str = "--only-throughput --tp-repeats 3"):
    """(3) throughput repeat with interleaved repeats (error bars), graphs."""
    dest = _dest("run3_graphs_tp")
    _save(dest, "graphs_throughput", run_dsv4.remote("graphs", extra, "graphs_throughput"))
    _summarize(dest)


@app.local_entrypoint()
def run1b(extra: str = "--only-mixed"):
    """(1b) eager: mixed batch + effect probe only (with the clean-vs-clean noise control)."""
    dest = _dest("run1b_eager_mixed")
    _save(dest, "eager_mixed", run_dsv4.remote("eager", extra, "eager_mixed"))
    _summarize(dest)


@app.local_entrypoint()
def main(engine: str = "eager", extra: str = "", tag: str = ""):
    dest = _dest(tag or engine)
    _save(dest, engine, run_dsv4.remote(engine, extra, engine))
    _summarize(dest)
