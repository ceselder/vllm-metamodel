"""Modal launcher for the vllm-metamodels benchmarks on one B200, parametrised by vLLM version.

    BENCH_VLLM=0.19.0   (default; the RL trainer's pin: torch 2.10 cu128, transformers 5.15)
    BENCH_VLLM=0.27.1   (DeepSeek-V4 session's version: torch 2.13)
    BENCH_VLLM=0.16.0 / 0.28.0 / ...  any PyPI release: ``pip install vllm==<ver>`` picks its torch

Two images per version share the vLLM install:
  * ``image_stock``  pip installs upstream vllm-lens (1.1.0 for vLLM <= 0.19, 1.2.1 for >= 0.27;
                     ``BENCH_STOCK_LENS`` overrides)
  * ``image_fork``   installs THIS checkout (``pip install --no-deps /opt/vllm-metamodels``)
Each engine mode runs in its own subprocess inside the container (the plugin reads its
environment variables at import time), all in one container so the model stays in the
page cache between engines.

Entrypoints (from the repo root, ``MODAL_PROFILE`` selecting your workspace):

    modal run bench/modal_bench.py::main                 # steering throughput: Qwen3.6-27B (maemm-data volume)
    modal run bench/modal_bench.py::main --small-model Qwen/Qwen3-1.7B
    modal run bench/modal_bench.py::test_injection       # injection-mode GPU test matrix
    modal run bench/modal_bench.py::readout              # hidden-state readout benchmark
    BENCH_VLLM=0.27.1 modal run bench/modal_bench.py::matrix --profile full --skip-big
                                                         # version-compatibility matrix (steering + injection + readout
                                                         #   + CPU tests + stock comparison) -> bench/results/matrix_<ver>_<ts>/
    modal run bench/modal_bench.py::lora                 # LoRA decode overhead + merge-on-publish (bench/bench_lora.py)

Results land in ``bench/results/<...>/`` as JSON plus summaries (``bench/compare.py``,
``bench/test_injection_modes.py``, ``bench/summarize_readout.py``, ``bench/summarize_matrix.py``).
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
# BENCH_REPO: a frozen copy of the checkout to build the image from (Modal aborts the build if a
# file changes while it is being hashed; ``rsync -a --exclude .git repo/ /tmp/snap/`` then point here)
REPO = Path(os.environ.get("BENCH_REPO") or HERE.parent)
sys.path.insert(0, str(HERE))

VLLM = os.environ.get("BENCH_VLLM", "0.19.0")
VLLM_TUPLE = tuple(int(x) for x in VLLM.split(".")[:3])
STOCK_LENS = os.environ.get("BENCH_STOCK_LENS") or ("1.1.0" if VLLM_TUPLE < (0, 27, 0) else "1.2.1")
GPU = os.environ.get("BENCH_GPU", "B200")
app = modal.App("vllm-metamodels-bench")

COMMON_FILES = ["bench_steering.py", "bench_readout.py", "test_injection_modes.py", "diag_engine.py", "bench_lora.py"]


def _base() -> modal.Image:
    """vLLM ``VLLM`` + a python 3.12; every other pin follows from vLLM's own requirements."""
    if VLLM == "0.19.0":
        # the RL trainer's exact environment (also what the 1.1.0.post1-post5 numbers were taken on)
        return (
            modal.Image.debian_slim(python_version="3.12")
            .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128")
            .pip_install("vllm==0.19.0")
            .pip_install(
                "transformers==5.15.0",
                "tokenizers==0.22.2",
                "huggingface_hub==1.27.0",
                "hf_xet",
                "safetensors==0.8.0",
                "numpy==2.4.6",
            )
        )
    if VLLM_TUPLE >= (0, 27, 0):
        # torch 2.13 / CUDA 13 wheels; the devel base gives Triton and FlashInfer JIT a working nvcc
        img = modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12").apt_install(
            "git", "build-essential"
        )
    else:
        img = modal.Image.debian_slim(python_version="3.12").apt_install("git")
    return img.pip_install(f"vllm=={VLLM}").pip_install("hf_xet", "accelerate>=1.0")


def _stock_image(lens_version: str) -> modal.Image:
    img = _base().pip_install(f"vllm-lens=={lens_version}", "pytest")
    for f in ("bench_steering.py", "bench_readout.py", "test_injection_modes.py"):
        img = img.add_local_file(HERE / f, f"/bench/{f}")
    return img


image_stock = _stock_image(STOCK_LENS)
image_stock_110 = _stock_image("1.1.0")  # upstream 1.1.0 on a newer vLLM: what a pinned user sees

image_fork = (
    _base()
    .pip_install("datasets>=4.0.0", "pydantic>=2.0", "zstandard>=0.23.0", "accelerate>=1.0", "pytest", "peft>=0.15", "safetensors")
    # HF reference for hybrid GatedDeltaNet models (Qwen3.5/3.6): the trainer runs the fla kernels, not the torch fallback
    .pip_install("flash-linear-attention")
    .add_local_dir(
        REPO,
        "/opt/vllm-metamodels",
        copy=True,
        ignore=[".git", "bench/results", "**/__pycache__", "uv.lock", ".venv", "*.png", "*.pdf"],
    )
    .run_commands(
        "pip install --no-deps --force-reinstall /opt/vllm-metamodels",
        "python -c 'import vllm_lens; from vllm_lens import _worker_ext as W; "
        'assert hasattr(W.HiddenStatesExtension, "set_steering_data_many"); print(vllm_lens.__version__)\'',
    )
)
for _f in COMMON_FILES:
    image_fork = image_fork.add_local_file(HERE / _f, f"/bench/{_f}")

vol = modal.Volume.from_name("maemm-data")
hf_secret = modal.Secret.from_name("maemm-hf")
_FN = dict(gpu=GPU, volumes={"/data": vol}, secrets=[hf_secret], timeout=3 * 3600)


def _env(offline: bool) -> dict:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    # vLLM >= 0.27 refuses pickled collective_rpc payloads unless opted in; the fork's plugin sets this
    # itself, stock 1.1.0/1.2.1 do not -- set it for every run so version comparisons measure the hooks.
    env["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
    if offline:
        env["HF_HOME"] = "/data/hf_cache"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env["HF_HOME"] = "/root/hf_small"
        env.pop("HF_HUB_OFFLINE", None)
    return env


def _in_volume(model: str) -> bool:
    return os.path.isdir(f"/data/hf_cache/hub/models--{model.replace('/', '--')}")


def _versions() -> dict:
    out = {}
    for pkg in ("vllm", "torch", "transformers", "vllm-lens", "flashinfer-python", "triton"):
        try:
            import importlib.metadata as md

            out[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = None
    return out


def _run(cmd: list[str], env: dict, out_path: str | None, tag: str, tail_chars: int = 12000) -> dict:
    print(f"[modal] >>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    tail = proc.stdout[-tail_chars:] + "\n--- stderr ---\n" + proc.stderr[-8000:]
    print(tail[-6000:], flush=True)
    rec = {"returncode": proc.returncode, "elapsed_s": time.time() - t0, "log_tail": tail, "versions": _versions()}
    if out_path and os.path.exists(out_path):
        with open(out_path) as f:
            rec["result"] = json.load(f)
    print(f"[modal] <<< {tag} rc={proc.returncode} in {rec['elapsed_s']:.0f}s", flush=True)
    return rec


def _run_engines(model: str, engines: list[str], extra: str, offline: bool) -> dict:
    env = _env(offline)
    out: dict = {}
    for eng in engines:
        path = f"/tmp/{eng}.json"
        cmd = [sys.executable, "/bench/bench_steering.py", "--model", model, "--engine", eng, "--out", path, *extra.split()]
        out[eng] = _run(cmd, env, path, eng, tail_chars=6000)
    return out


@app.function(image=image_stock, **_FN)
def run_stock(model: str, engines: list[str], extra: str, offline: bool) -> dict:
    return _run_engines(model, engines, extra, offline)


@app.function(image=image_fork, **_FN)
def run_fork(model: str, engines: list[str], extra: str, offline: bool) -> dict:
    return _run_engines(model, engines, extra, offline)


@app.function(image=image_fork, **_FN)
def run_injection_tests(model: str, extra: str, engines: list[str], chunked: bool, hf_ref: bool) -> dict:
    """bench/test_injection_modes.py: HF reference, then one subprocess per engine
    (eager / graphs, each optionally repeated with the chunked-prefill engine), all in
    one container so the weights stay in the page cache."""
    offline = _in_volume(model)
    env = _env(offline)
    if offline:
        print(f"[modal] {model}: found in /data/hf_cache -> offline", flush=True)
    return _injection_stages(model, extra, engines, chunked, hf_ref, env)


def _injection_stages(model: str, extra: str, engines: list[str], chunked: bool, hf_ref: bool, env: dict) -> dict:
    out: dict = {}
    ref_path = f"/tmp/hf_ref_{model.replace('/', '__')}.pt"
    baseline = "/opt/vllm-metamodels/bench/results_summary.json"

    def run(tag: str, args: list[str]) -> dict:
        cmd = [sys.executable, "/bench/test_injection_modes.py", "--model", model, *args, *extra.split()]
        return _run(cmd, env, f"/tmp/{tag}.json", tag)

    if hf_ref:
        out["hf_ref"] = run("hf_ref", ["--stage", "hf-ref", "--out", ref_path])
        if out["hf_ref"]["returncode"] != 0:
            ref_path = ""  # engines still run; HF comparisons are skipped
    for eng in engines:
        common = ["--stage", "vllm", "--engine", eng, "--ref", ref_path, "--baseline", baseline]
        out[eng] = run(eng, [*common, "--out", f"/tmp/{eng}.json"])
        if chunked:
            out[f"{eng}_chunked"] = run(f"{eng}_chunked", [*common, "--chunked", "--out", f"/tmp/{eng}_chunked.json"])
    return out


@app.local_entrypoint()
def test_injection(
    model: str = "Qwen/Qwen3.6-27B",
    small_model: str = "Qwen/Qwen3-1.7B",
    engines: str = "eager,graphs",
    coeffs: str = "1.0,4.0",
    batches: str = "64,512",
    tp_batches: str = "512,1024",
    prompt_tokens: int = 96,
    marker: int = 10,
    inject_layer: int = 1,
    attention_backend: str = "TRITON_ATTN",
    skip_chunked: bool = False,
    skip_hf_ref: bool = False,
    skip_throughput: bool = False,
    skip_big: bool = False,
    extra_args: str = "",
    out_dir: str = str(HERE / "results"),
):
    """GPU test matrix for the injection modes (see bench/test_injection_modes.py).

        modal run bench/modal_bench.py::test_injection
    """
    from test_injection_modes import markdown_table, summarize

    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = Path(out_dir) / f"injection_{ts}"
    dest.mkdir(parents=True, exist_ok=True)
    common = (
        f"--prompt-tokens {prompt_tokens} --marker {marker} --inject-layer {inject_layer} "
        f"--coeffs {coeffs} --batches {batches} --tp-batches {tp_batches}"
    )
    if attention_backend:
        common += f" --attention-backend {attention_backend}"
    if skip_throughput:
        common += " --skip-throughput"
    if extra_args:
        common += " " + extra_args
    eng_list = [e for e in engines.split(",") if e]
    jobs = []
    if not skip_big:
        jobs.append((model, run_injection_tests.spawn(model, common + " --language-model-only", eng_list, not skip_chunked, not skip_hf_ref)))
    if small_model:
        jobs.append((small_model, run_injection_tests.spawn(small_model, common, eng_list, not skip_chunked, not skip_hf_ref)))
    print(f"[local] {len(jobs)} container jobs spawned; results -> {dest}", flush=True)
    for mtag, fut in jobs:
        res = fut.get()
        for tag, rec in res.items():
            name = f"{mtag.replace('/', '__')}__{tag}"
            (dest / f"{name}.json").write_text(json.dumps(rec, indent=1))
            print(f"[local] saved {name}.json rc={rec['returncode']} ({rec['elapsed_s']:.0f}s)", flush=True)
            if rec["returncode"] != 0:
                print(rec["log_tail"][-4000:], flush=True)
    summary = summarize(dest)
    (dest / "summary.json").write_text(json.dumps(summary, indent=1))
    (dest / "summary.md").write_text(markdown_table(summary))
    print(markdown_table(summary))
    for c in summary["checks"]:
        print(f"[{'PASS' if c['ok'] else 'FAIL'}] {c['model']} {c['engine']} {c['case']}: {c['check']}  {c['detail'][:200]}")
    print(f"{summary['n_pass']}/{summary['n_checks']} checks pass" + (" -- ALL PASS" if summary["all_pass"] else " -- SOME FAILED"))
    print(f"[local] results in {dest}")


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen3.6-27B",
    small_model: str = "",
    sizes: str = "8,32,128,512,1024,2048",
    small_sizes: str = "8,32,128,512,1024",
    sizes_2d: str = "8,32,128,512",
    max_tokens: int = 40,
    prompt_tokens: int = 96,
    inject_layer: int = 1,
    marker: int = 10,
    attention_backend: str = "TRITON_ATTN",
    skip_plain: bool = False,
    skip_stock: bool = False,
    skip_big: bool = False,
    fork_engines: str = "",
    max_capture_size: int = 0,
    max_num_seqs: int = 0,
    extra_args: str = "",
    out_dir: str = str(HERE / "results"),
):
    from compare import summarize

    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = Path(out_dir) / ts
    dest.mkdir(parents=True, exist_ok=True)
    common = (
        f"--max-tokens {max_tokens} --prompt-tokens {prompt_tokens} --sizes-2d {sizes_2d} "
        f"--inject-layer {inject_layer} --marker {marker}"
    )
    if attention_backend:
        common += f" --attention-backend {attention_backend}"
    if max_capture_size:
        common += f" --max-capture-size {max_capture_size}"
    if max_num_seqs:
        common += f" --max-num-seqs {max_num_seqs}"
    if extra_args:
        common += " " + extra_args
    big_extra = common + f" --sizes {sizes} --language-model-only"
    small_extra = common + f" --sizes {small_sizes}"
    fork_engines = (
        [e for e in fork_engines.split(",") if e]
        if fork_engines
        else ["eager", "graphs"] + ([] if skip_plain else ["plain"])
    )

    jobs = []  # (model_tag, variant, future)
    if not skip_big:
        if not skip_stock:
            jobs.append((model, "stock", run_stock.spawn(model, ["eager"], big_extra, True)))
        jobs.append((model, "fork", run_fork.spawn(model, fork_engines, big_extra, True)))
    if small_model:
        if not skip_stock:
            jobs.append((small_model, "stock", run_stock.spawn(small_model, ["eager"], small_extra, False)))
        jobs.append((small_model, "fork", run_fork.spawn(small_model, fork_engines, small_extra, False)))
    print(f"[local] {len(jobs)} container jobs spawned; results -> {dest}", flush=True)

    all_results: dict = {}
    for mtag, variant, fut in jobs:
        res = fut.get()
        for eng, rec in res.items():
            name = f"{mtag.replace('/', '__')}__{variant}__{eng}"
            (dest / f"{name}.json").write_text(json.dumps(rec, indent=1))
            print(f"[local] saved {dest / (name + '.json')} rc={rec['returncode']} ({rec['elapsed_s']:.0f}s)", flush=True)
            if rec["returncode"] != 0:
                print(rec["log_tail"][-3000:], flush=True)
        all_results.setdefault(mtag, {})[variant] = res

    summary = summarize(all_results)
    (dest / "summary.json").write_text(json.dumps(summary, indent=1))
    _print_steering_summary(summary)
    print(f"[local] results in {dest}")


def _print_steering_summary(summary: dict) -> None:
    for m, t in summary["tables"].items():
        print(f"== {m} ==")
        for name, by_b in t["series"].items():
            print(f"  {name:22s} " + "  ".join(f"B={b}: {v['wall_s']:.1f}s/{v['tok_per_s']:.0f}tok/s" for b, v in by_b.items()))
        for name, by_b in t["speedup_vs_stock"].items():
            print(f"  speedup {name:14s} " + "  ".join(f"B={b}: {x:.2f}x" for b, x in by_b.items()))
    print("\n".join(summary["assertions_text"]))
    print("ALL PASS" if summary["all_pass"] else "SOME CHECKS FAILED")


# ---------------------------------------------------------------------------
# Engine-init bisect (bench/diag_engine.py):  modal run bench/modal_bench.py::diag_main --configs a,b,c
# ---------------------------------------------------------------------------


@app.function(image=image_fork, gpu=GPU, volumes={"/data": vol}, secrets=[hf_secret], timeout=3600)
def run_diag(model: str, configs: list[str], max_num_seqs: int, offline: bool) -> dict:
    env = _env(offline)
    out = {}
    for c in configs:
        cmd = [sys.executable, "/bench/diag_engine.py", "--model", model, "--config", c, "--max-num-seqs", str(max_num_seqs)]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        text = proc.stdout + "\n--- stderr ---\n" + proc.stderr
        keep = [ln for ln in text.splitlines() if "[diag]" in ln or "Error" in ln]
        out[c] = {"rc": proc.returncode, "diag": keep[-40:], "stderr_tail": proc.stderr[-12000:]}
        print(f"[modal] {c}: rc={proc.returncode}\n" + "\n".join(keep[-12:]), flush=True)
    return out


@app.local_entrypoint()
def diag_main(
    model: str = "Qwen/Qwen3.6-27B",
    configs: str = "lora_maxcap,nolora_maxcap,nolora_maxcap_nopacked",
    max_num_seqs: int = 1024,
    offline: bool = True,
    out: str = "/tmp/vlp_diag.json",
):
    res = run_diag.remote(model, [c for c in configs.split(",") if c], max_num_seqs, offline)
    Path(out).write_text(json.dumps(res, indent=1))
    for c, r in res.items():
        print(f"== {c}: rc={r['rc']}")
        print("\n".join(r["diag"][-6:]))
    print(f"[local] wrote {out}")


# ---------------------------------------------------------------------------
# Hidden-state readout benchmark (bench/bench_readout.py):
#   modal run bench/modal_bench.py::readout
# ---------------------------------------------------------------------------


def _readout_runs(model: str, stages: list[tuple[str, list[str]]], extra: str, env: dict) -> dict:
    """stages: [(tag, argv)] run sequentially in this container; JSON per stage."""
    out: dict = {}
    for tag, argv in stages:
        path = f"/tmp/readout_{tag}.json"
        cmd = [sys.executable, "/bench/bench_readout.py", "--model", model, "--out", path, *argv, *extra.split()]
        out[tag] = _run(cmd, env, path, tag, tail_chars=14000)
    return out


@app.function(image=image_fork, **_FN)
def run_readout_fork(model: str, engines: list[str], extra: str, hf: bool) -> dict:
    offline = _in_volume(model)
    ref = f"/tmp/readout_ref_{model.replace('/', '__')}.pt"
    stages: list[tuple[str, list[str]]] = []
    if hf:
        stages.append(("hf", ["--stage", "hf", "--ref", ref]))
    for eng in engines:
        stages.append((f"fork_{eng}", ["--stage", "vllm", "--engine", eng, "--ref", ref]))
    return _readout_runs(model, stages, extra, _env(offline))


@app.function(image=image_stock, **_FN)
def run_readout_stock(model: str, extra: str) -> dict:
    offline = _in_volume(model)
    return _readout_runs(model, [("stock_eager", ["--stage", "vllm", "--engine", "eager"])], extra, _env(offline))


@app.local_entrypoint()
def readout(
    model: str = "Qwen/Qwen3.6-27B",
    layer: int = 42,
    small_model: str = "Qwen/Qwen3-1.7B",
    small_layer: int = 18,
    engines: str = "eager,graphs",
    sizes: str = "64,512,1024",
    gen_sizes: str = "64,512",
    gen_tokens: int = 40,
    n_texts: int = 1024,
    repeats: int = 2,
    attention_backend: str = "TRITON_ATTN",
    skip_stock: bool = False,
    skip_fork: bool = False,
    skip_small: bool = False,
    skip_big: bool = False,
    skip_hf: bool = False,
    extra_args: str = "",
    out_dir: str = str(HERE / "results"),
):
    """Hidden-state readout benchmark on one B200 per container (stock + fork containers in parallel).

        modal run bench/modal_bench.py::readout
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = Path(out_dir) / f"readout_{ts}"
    dest.mkdir(parents=True, exist_ok=True)
    common = f"--sizes {sizes} --gen-sizes {gen_sizes} --gen-tokens {gen_tokens} --n-texts {n_texts} --repeats {repeats}"
    if attention_backend:
        common += f" --attention-backend {attention_backend}"
    if extra_args:
        common += " " + extra_args
    eng_list = [e for e in engines.split(",") if e]
    jobs = []
    if not skip_big:
        big = common + f" --layer {layer} --language-model-only"
        if not skip_fork:
            jobs.append((model, "fork", run_readout_fork.spawn(model, eng_list, big, not skip_hf)))
        if not skip_stock:
            jobs.append((model, "stock", run_readout_stock.spawn(model, big)))
    if small_model and not skip_small:
        small = common + f" --layer {small_layer}"
        if not skip_fork:
            jobs.append((small_model, "fork", run_readout_fork.spawn(small_model, eng_list, small, not skip_hf)))
        if not skip_stock:
            jobs.append((small_model, "stock", run_readout_stock.spawn(small_model, small)))
    print(f"[local] {len(jobs)} container jobs spawned; results -> {dest}", flush=True)
    for mtag, variant, fut in jobs:
        res = fut.get()
        for tag, rec in res.items():
            name = f"{mtag.replace('/', '__')}__{tag}"
            (dest / f"{name}.json").write_text(json.dumps(rec, indent=1))
            print(f"[local] saved {name}.json rc={rec['returncode']} ({rec['elapsed_s']:.0f}s)", flush=True)
            if rec["returncode"] != 0:
                print(rec["log_tail"][-5000:], flush=True)
    print(f"[local] results in {dest}")


# ---------------------------------------------------------------------------
# vLLM version-compatibility matrix:  BENCH_VLLM=<ver> modal run bench/modal_bench.py::matrix
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict] = {
    # 1.7B, full: everything the 0.19 numbers were taken with, plus stock rows
    "full": dict(
        steer_eager="--sizes 8,32,128,512,1024", steer_graphs="--sizes 8,32,128,512,1024 --repeats 2",
        steer_plain="--sizes 8,128,512,1024 --repeats 2", stock_sizes="8,32,128,512,1024",
        inj_engines=["eager", "graphs"], inj_chunked=True, inj_hf=True,
        inj_args="--coeffs 1.0,4.0 --batches 64,512 --tp-batches 512,1024",
        ro_engines=["eager", "graphs"], ro_hf=True, ro_args="--sizes 64,512,1024 --gen-sizes 64,512 --repeats 2",
    ),
    # 1.7B, quick: lower/upper bound versions
    "quick": dict(
        steer_eager="--sizes 8,128,512", steer_graphs="--sizes 8,128,512,1024 --repeats 2",
        steer_plain="--sizes 512,1024 --repeats 2", stock_sizes="8,128",
        inj_engines=["eager", "graphs"], inj_chunked=False, inj_hf=True,
        inj_args="--coeffs 1.0 --batches 64 --tp-batches 512 --skip-throughput",
        ro_engines=["graphs"], ro_hf=False, ro_args="--sizes 64,512,1024 --gen-sizes 64 --repeats 1",
    ),
    # 27B headline rows (graphs), one eager steering engine at B=512 for the fork-eager row
    "big": dict(
        steer_eager="--sizes 512 --conditions nosteer,steer3d", steer_graphs="--sizes 512,1024 --repeats 2 --conditions nosteer,steer3d",
        steer_plain="--sizes 512,1024 --repeats 2", stock_sizes="",
        inj_engines=["graphs"], inj_chunked=False, inj_hf=False,
        inj_args="--coeffs 1.0 --batches 64 --tp-batches 512 --skip-throughput",
        ro_engines=["graphs"], ro_hf=False,
        ro_args="--sizes 1024 --gen-sizes 512 --repeats 2 --conditions nocap,cap_all,cap_last5,read_last5,exit_read_last5,gen_nocap,gen_then_exit_read",
    ),
}


@app.function(image=image_fork, **_FN)
def run_matrix_fork(model: str, profile: str, common: str, attention_backend: str, layer: int, big: bool, plain_v1: bool = False) -> dict:
    """One container: CPU test suites, steering (eager / graphs / plain), injection matrix, readout.
    ``plain_v1``: also run plain vLLM on its V1 model runner (vLLM >= 0.23; decided locally -- the
    container does not see BENCH_VLLM)."""
    p = PROFILES[profile]
    offline = _in_volume(model)
    env = _env(offline)
    out: dict = {"versions": _versions()}
    lmo = " --language-model-only" if big else ""
    ab = f" --attention-backend {attention_backend}" if attention_backend else ""
    # CPU test suites against the real vLLM modules of this version
    cmd = [sys.executable, "-m", "pytest", "-q", "--noconftest", "-p", "no:cacheprovider",
           "/opt/vllm-metamodels/vllm_lens/tests/test_steering_index.py",
           "/opt/vllm-metamodels/vllm_lens/tests/test_readout.py",
           "/opt/vllm-metamodels/vllm_lens/tests/test_metamodel_helpers.py"]
    out["cpu_tests"] = _run(cmd, env, None, "cpu_tests", tail_chars=4000)
    # steering throughput + correctness probes (+ on vLLM >= 0.23: plain vLLM on its V1 runner as
    # well, since the plugin forces V1 and "plain" runs vLLM's default = the V2 runner)
    stages = [("eager", p["steer_eager"], ""), ("graphs", p["steer_graphs"], ""), ("plain", p["steer_plain"], "")]
    if plain_v1 and p["steer_plain"]:
        stages.append(("plain_v1", p["steer_plain"] + " --model-runner v1", "plain"))
    for tag, args, eng in stages:
        if not args:
            continue
        eng = eng or tag
        path = f"/tmp/steer_{tag}.json"
        cmd = [sys.executable, "/bench/bench_steering.py", "--model", model, "--engine", eng, "--out", path,
               *(common + ab + lmo + " " + args).split()]
        out[f"steer_{tag}"] = _run(cmd, env, path, f"steer_{tag}", tail_chars=6000)
    # injection modes
    inj_common = common + ab + lmo + " " + p["inj_args"]
    for tag, rec in _injection_stages(model, inj_common, p["inj_engines"], p["inj_chunked"], p["inj_hf"], env).items():
        out[f"inj_{tag}"] = rec
    # readout
    ref = f"/tmp/readout_ref_{model.replace('/', '__')}.pt"
    stages: list[tuple[str, list[str]]] = []
    if p["ro_hf"]:
        stages.append(("hf", ["--stage", "hf", "--ref", ref]))
    for eng in p["ro_engines"]:
        stages.append((f"fork_{eng}", ["--stage", "vllm", "--engine", eng, "--ref", ref]))
    ro_common = f"--layer {layer}" + ab + lmo + " " + p["ro_args"]
    for tag, rec in _readout_runs(model, stages, ro_common, env).items():
        out[f"ro_{tag}"] = rec
    return out


def _stock_matrix(model: str, sizes: str, common: str, big: bool) -> dict:
    offline = _in_volume(model)
    env = _env(offline)
    out: dict = {"versions": _versions()}
    lmo = " --language-model-only" if big else ""
    path = "/tmp/stock_eager.json"
    cmd = [sys.executable, "/bench/bench_steering.py", "--model", model, "--engine", "eager", "--out", path,
           *(common + lmo + f" --sizes {sizes}").split()]
    out["steer_eager"] = _run(cmd, env, path, "stock_steer_eager", tail_chars=8000)
    return out


@app.function(image=image_stock, **_FN)
def run_matrix_stock(model: str, sizes: str, common: str, big: bool) -> dict:
    return _stock_matrix(model, sizes, common, big)


@app.function(image=image_stock_110, **_FN)
def run_matrix_stock_110(model: str, sizes: str, common: str, big: bool) -> dict:
    return _stock_matrix(model, sizes, common, big)


@app.local_entrypoint()
def matrix(
    model: str = "Qwen/Qwen3.6-27B",
    small_model: str = "Qwen/Qwen3-1.7B",
    profile: str = "full",
    big_profile: str = "big",
    layer: int = 42,
    small_layer: int = 18,
    max_tokens: int = 40,
    prompt_tokens: int = 96,
    inject_layer: int = 1,
    marker: int = 10,
    attention_backend: str = "TRITON_ATTN",
    skip_big: bool = False,
    skip_small: bool = False,
    skip_stock: bool = False,
    skip_stock_110: bool = False,
    out_dir: str = str(HERE / "results"),
):
    """Version-compatibility matrix for the vLLM release ``BENCH_VLLM`` (default 0.19.0):

        BENCH_VLLM=0.27.1 modal run bench/modal_bench.py::matrix --profile full --skip-big

    Writes bench/results/matrix_<ver>_<ts>/{steering,injection,readout}/... in the layouts the
    existing summarisers read, plus matrix.json (one record per stage) and versions.
    """
    from compare import summarize as summarize_steering

    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = Path(out_dir) / f"matrix_{VLLM}_{ts}"
    for sub in ("steering", "injection", "readout"):
        (dest / sub).mkdir(parents=True, exist_ok=True)
    common = f"--max-tokens {max_tokens} --prompt-tokens {prompt_tokens} --inject-layer {inject_layer} --marker {marker}"
    stock_common = common + (f" --attention-backend {attention_backend}" if attention_backend else "")
    jobs = []
    plain_v1 = VLLM_TUPLE >= (0, 23, 0)
    if not skip_small and small_model:
        jobs.append((small_model, "fork", run_matrix_fork.spawn(small_model, profile, common, attention_backend, small_layer, False, plain_v1)))
        sizes = PROFILES[profile]["stock_sizes"]
        if sizes and not skip_stock:
            jobs.append((small_model, f"stock{STOCK_LENS}", run_matrix_stock.spawn(small_model, sizes, stock_common, False)))
            if STOCK_LENS != "1.1.0" and not skip_stock_110:
                jobs.append((small_model, "stock1.1.0", run_matrix_stock_110.spawn(small_model, sizes, stock_common, False)))
    if not skip_big:
        jobs.append((model, "fork", run_matrix_fork.spawn(model, big_profile, common, attention_backend, layer, True, plain_v1)))
    print(f"[local] vLLM {VLLM}: {len(jobs)} container jobs spawned; results -> {dest}", flush=True)
    matrix_json: dict = {"vllm": VLLM, "stock_lens": STOCK_LENS, "gpu": GPU, "profile": profile, "big_profile": big_profile, "models": {}}
    steering_results: dict = {}
    for mtag, variant, fut in jobs:
        res = fut.get()
        mkey = mtag.replace("/", "__")
        matrix_json["models"].setdefault(mtag, {})[variant] = {k: v for k, v in res.items() if k == "versions"}
        for tag, rec in res.items():
            if tag == "versions":
                continue
            rc = rec.get("returncode")
            matrix_json["models"][mtag][variant][tag] = {"returncode": rc, "elapsed_s": rec.get("elapsed_s")}
            print(f"[local] {mtag} {variant} {tag}: rc={rc} ({rec.get('elapsed_s', 0):.0f}s)", flush=True)
            if rc != 0:
                print(rec.get("log_tail", "")[-3000:], flush=True)
            if tag.startswith("steer_"):
                eng = tag[len("steer_"):]
                var = "fork" if variant == "fork" else ("stock" if variant == f"stock{STOCK_LENS}" else "stock110")
                (dest / "steering" / f"{mkey}__{var}__{eng}.json").write_text(json.dumps(rec, indent=1))
                steering_results.setdefault(mtag, {}).setdefault(var, {})[eng] = rec
            elif tag.startswith("inj_"):
                (dest / "injection" / f"{mkey}__{tag[4:]}.json").write_text(json.dumps(rec, indent=1))
            elif tag.startswith("ro_"):
                (dest / "readout" / f"{mkey}__{tag[3:]}.json").write_text(json.dumps(rec, indent=1))
            else:
                (dest / f"{mkey}__{variant}__{tag}.json").write_text(json.dumps(rec, indent=1))
    (dest / "matrix.json").write_text(json.dumps(matrix_json, indent=1))
    try:
        s = summarize_steering({m: {v: e for v, e in by.items() if v in ("stock", "fork")} for m, by in steering_results.items()})
        (dest / "steering" / "summary.json").write_text(json.dumps(s, indent=1))
        _print_steering_summary(s)
    except Exception as e:  # noqa: BLE001
        print(f"[local] steering summary failed: {e!r}")
    try:
        from test_injection_modes import markdown_table, summarize as summarize_injection

        si = summarize_injection(dest / "injection")
        (dest / "injection" / "summary.json").write_text(json.dumps(si, indent=1))
        (dest / "injection" / "summary.md").write_text(markdown_table(si))
        print(f"[local] injection: {si['n_pass']}/{si['n_checks']} checks pass")
    except Exception as e:  # noqa: BLE001
        print(f"[local] injection summary failed: {e!r}")
    print(f"[local] results in {dest}; summarise with: python bench/summarize_matrix.py bench/results/matrix_*")


@app.function(image=image_fork, cpu=4, timeout=1800)
def run_cpu_tests() -> dict:
    """The fork's CPU suites against this vLLM version's real modules (no GPU)."""
    cmd = [sys.executable, "-m", "pytest", "-q", "--noconftest", "-p", "no:cacheprovider",
           "/opt/vllm-metamodels/vllm_lens/tests/test_steering_index.py", "/opt/vllm-metamodels/vllm_lens/tests/test_readout.py",
           "/opt/vllm-metamodels/vllm_lens/tests/test_metamodel_helpers.py", "/opt/vllm-metamodels/vllm_lens/tests/test_lora_merge.py"]
    return _run(cmd, _env(False), None, "cpu_tests", tail_chars=4000)


@app.local_entrypoint()
def cpu_tests(out_dir: str = str(HERE / "results")):
    """BENCH_VLLM=<ver> modal run bench/modal_bench.py::cpu_tests -- refresh the CPU-suite cell of a matrix dir."""
    rec = run_cpu_tests.remote()
    dests = sorted(Path(out_dir).glob(f"matrix_{VLLM}_*"))
    for d in dests[-1:]:
        for f in d.glob("*__fork__cpu_tests.json"):
            f.write_text(json.dumps(rec, indent=1))
            print(f"[local] refreshed {f}")
    print(f"[local] vLLM {VLLM} cpu tests rc={rec['returncode']}: {rec['log_tail'][-300:]}")


# ---------------------------------------------------------------------------
# LoRA decode overhead + merge-on-publish:  modal run bench/modal_bench.py::lora
# ---------------------------------------------------------------------------


@app.function(image=image_fork, gpu=GPU, volumes={"/data": vol}, secrets=[hf_secret], timeout=3 * 3600, memory=(96 << 10))
def run_lora(model: str, engine: str, extra: str, big: bool, stages: list[str]) -> dict:
    """bench/bench_lora.py: LoRA-capable engine stage (nolora / lora / merged) then the plain engine
    stage (plain / merged, enable_lora=False), same container (page cache); 96 GB RAM for the pinned
    host copy of the 27B's targeted weights (keep_base="cpu")."""
    offline = _in_volume(model)
    env = _env(offline)
    out: dict = {"versions": _versions()}
    for stage in stages:
        path = f"/tmp/lora_{engine}_{stage}.json"
        cmd = [sys.executable, "/bench/bench_lora.py", "--model", model, "--engine", engine, "--stage", stage, "--out", path,
               *(extra + (" --language-model-only" if big else "")).split()]
        out[f"lora_{engine}_{stage}"] = _run(cmd, env, path, f"lora_{engine}_{stage}", tail_chars=20000)
    return out


@app.local_entrypoint()
def lora(
    model: str = "Qwen/Qwen3.6-27B",
    small_model: str = "Qwen/Qwen3-1.7B",
    engine: str = "graphs",
    sizes: str = "512,1024",
    max_tokens: int = 40,
    prompt_tokens: int = 96,
    n_publishes: int = 30,
    attention_backend: str = "TRITON_ATTN",
    skip_big: bool = False,
    skip_small: bool = False,
    stages: str = "lora_engine,plain_engine",
    big_gpu_mem: float = 0.55,
    extra_args: str = "",
    out_dir: str = str(HERE / "results"),
):
    """LoRA rank-64 decode overhead vs merged weights (bench/bench_lora.py) on vLLM ``BENCH_VLLM``.
    ``big_gpu_mem``: gpu_memory_utilization for the 27B (0.55 leaves room for the 48 GB base copy on
    vLLM 0.19; vLLM >= 0.27 needs >= 0.78 so the GDN state pool has one block per sequence at
    max_num_seqs=1024 -- the base copy then lives in pinned host memory, keep_base auto -> cpu)."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    dest = Path(out_dir) / f"lora_{VLLM}_{ts}"
    dest.mkdir(parents=True, exist_ok=True)
    common = f"--sizes {sizes} --max-tokens {max_tokens} --prompt-tokens {prompt_tokens} --n-publishes {n_publishes}"
    if attention_backend:
        common += f" --attention-backend {attention_backend}"
    if extra_args:
        common += " " + extra_args
    stage_list = [s for s in stages.split(",") if s]
    jobs = []
    if not skip_small and small_model:
        jobs.append((small_model, run_lora.spawn(small_model, engine, common, False, stage_list)))
    if not skip_big:
        jobs.append((model, run_lora.spawn(model, engine, common + f" --gpu-mem {big_gpu_mem}", True, stage_list)))
    print(f"[local] vLLM {VLLM}: {len(jobs)} container jobs spawned; results -> {dest}", flush=True)
    for mtag, fut in jobs:
        res = fut.get()
        for tag, rec in res.items():
            if tag == "versions":
                continue
            name = f"{mtag.replace('/', '__')}__{tag}"
            rec["versions"] = res.get("versions")
            (dest / f"{name}.json").write_text(json.dumps(rec, indent=1))
            print(f"[local] saved {name}.json rc={rec['returncode']} ({rec['elapsed_s']:.0f}s)", flush=True)
            if rec["returncode"] != 0:
                print(rec["log_tail"][-6000:], flush=True)
            else:
                print(rec["log_tail"][-3000:], flush=True)
    print(f"[local] results in {dest}")
