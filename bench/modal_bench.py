"""Modal launcher for bench/bench_steering.py: stock vllm-lens 1.1.0 vs vllm-lens-port on one B200.

Two images share the same pins (python 3.12, torch 2.10 cu128, vllm 0.19.0):
  * ``image_stock``  pip installs ``vllm-lens==1.1.0`` from PyPI
  * ``image_fork``   installs THIS checkout (``pip install /opt/vllm-lens-port``)
Each engine mode runs in its own subprocess inside the container (the plugin reads its
environment variables at import time), all in one container so the model stays in the
page cache between engines.

Usage (from the repo root, ``MODAL_PROFILE`` selecting your workspace):

    modal run bench/modal_bench.py::main                 # Qwen3.6-27B from the maemm-data volume
    modal run bench/modal_bench.py::main --small-model Qwen/Qwen3-1.7B   # + a second, downloaded model
    modal run bench/modal_bench.py::main --skip-plain          # skip the torch.compile'd vLLM ceiling

Results land in ``bench/results/<timestamp>/`` as JSON (one file per model x variant x engine)
plus ``summary.json`` (speedup table + correctness assertions) via ``bench/compare.py``.
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

app = modal.App("vllm-lens-port-bench")
GPU = os.environ.get("BENCH_GPU", "B200")


def _base() -> modal.Image:
    return (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.10.0", index_url="https://download.pytorch.org/whl/cu128"
        )
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


image_stock = (
    _base()
    .pip_install("vllm-lens==1.1.0")
    .add_local_file(HERE / "bench_steering.py", "/bench/bench_steering.py")
)

image_fork = (
    _base()
    .pip_install("datasets>=4.0.0", "pydantic>=2.0", "zstandard>=0.23.0")
    .add_local_dir(
        REPO,
        "/opt/vllm-lens-port",
        copy=True,
        ignore=[
            ".git",
            "bench/results",
            "**/__pycache__",
            "uv.lock",
            ".venv",
            "*.png",
            "*.pdf",
        ],
    )
    .run_commands(
        "pip install --no-deps /opt/vllm-lens-port",
        "python -c 'import vllm_lens; from vllm_lens import _worker_ext as W; "
        'assert hasattr(W.HiddenStatesExtension, "set_steering_data_many"); print(vllm_lens.__version__)\'',
    )
    .add_local_file(HERE / "bench_steering.py", "/bench/bench_steering.py")
    .add_local_file(HERE / "diag_engine.py", "/bench/diag_engine.py")
)

vol = modal.Volume.from_name("maemm-data")
hf_secret = modal.Secret.from_name("maemm-hf")


def _run_engines(model: str, engines: list[str], extra: str, offline: bool) -> dict:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    if offline:
        env["HF_HOME"] = "/data/hf_cache"
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
    else:
        env["HF_HOME"] = "/root/hf_small"
        env.pop("HF_HUB_OFFLINE", None)
    out: dict = {}
    for eng in engines:
        path = f"/tmp/{eng}.json"
        cmd = [
            sys.executable,
            "/bench/bench_steering.py",
            "--model",
            model,
            "--engine",
            eng,
            "--out",
            path,
            *extra.split(),
        ]
        print(f"[modal] >>> {' '.join(cmd)}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        tail = proc.stdout[-6000:] + "\n--- stderr ---\n" + proc.stderr[-6000:]
        print(tail, flush=True)
        rec = {
            "returncode": proc.returncode,
            "elapsed_s": time.time() - t0,
            "log_tail": tail,
        }
        if os.path.exists(path):
            with open(path) as f:
                rec["result"] = json.load(f)
        out[eng] = rec
        print(
            f"[modal] <<< {eng} rc={proc.returncode} in {rec['elapsed_s']:.0f}s",
            flush=True,
        )
    return out


@app.function(
    image=image_stock,
    gpu=GPU,
    volumes={"/data": vol},
    secrets=[hf_secret],
    timeout=3 * 3600,
)
def run_stock(model: str, engines: list[str], extra: str, offline: bool) -> dict:
    return _run_engines(model, engines, extra, offline)


@app.function(
    image=image_fork,
    gpu=GPU,
    volumes={"/data": vol},
    secrets=[hf_secret],
    timeout=3 * 3600,
)
def run_fork(model: str, engines: list[str], extra: str, offline: bool) -> dict:
    return _run_engines(model, engines, extra, offline)


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
    big_extra = common + f" --sizes {sizes} --language-model-only"
    small_extra = common + f" --sizes {small_sizes}"
    if max_capture_size:
        common += f" --max-capture-size {max_capture_size}"
    if max_num_seqs:
        common += f" --max-num-seqs {max_num_seqs}"
    if extra_args:
        common += " " + extra_args
    fork_engines = (
        [e for e in fork_engines.split(",") if e]
        if fork_engines
        else ["eager", "graphs"] + ([] if skip_plain else ["plain"])
    )

    jobs = []  # (model_tag, variant, future)
    if not skip_big:
        if not skip_stock:
            jobs.append(
                (model, "stock", run_stock.spawn(model, ["eager"], big_extra, True))
            )
        jobs.append(
            (model, "fork", run_fork.spawn(model, fork_engines, big_extra, True))
        )
    if small_model:
        if not skip_stock:
            jobs.append(
                (
                    small_model,
                    "stock",
                    run_stock.spawn(small_model, ["eager"], small_extra, False),
                )
            )
        jobs.append(
            (
                small_model,
                "fork",
                run_fork.spawn(small_model, fork_engines, small_extra, False),
            )
        )
    print(f"[local] {len(jobs)} container jobs spawned; results -> {dest}", flush=True)

    all_results: dict = {}
    for mtag, variant, fut in jobs:
        res = fut.get()
        for eng, rec in res.items():
            name = f"{mtag.replace('/', '__')}__{variant}__{eng}"
            (dest / f"{name}.json").write_text(json.dumps(rec, indent=1))
            print(
                f"[local] saved {dest / (name + '.json')} rc={rec['returncode']} ({rec['elapsed_s']:.0f}s)",
                flush=True,
            )
            if rec["returncode"] != 0:
                print(rec["log_tail"][-3000:], flush=True)
        all_results.setdefault(mtag, {})[variant] = res

    summary = summarize(all_results)
    (dest / "summary.json").write_text(json.dumps(summary, indent=1))
    for m, t in summary["tables"].items():
        print(f"== {m} ==")
        for name, by_b in t["series"].items():
            print(
                f"  {name:22s} "
                + "  ".join(
                    f"B={b}: {v['wall_s']:.1f}s/{v['tok_per_s']:.0f}tok/s"
                    for b, v in by_b.items()
                )
            )
        for name, by_b in t["speedup_vs_stock"].items():
            print(
                f"  speedup {name:14s} "
                + "  ".join(f"B={b}: {x:.2f}x" for b, x in by_b.items())
            )
    print("\n".join(summary["assertions_text"]))
    print("ALL PASS" if summary["all_pass"] else "SOME CHECKS FAILED")
    print(f"[local] results in {dest}")


# ---------------------------------------------------------------------------
# Engine-init bisect (bench/diag_engine.py):  modal run bench/modal_bench.py::diag_main --configs a,b,c
# ---------------------------------------------------------------------------


@app.function(
    image=image_fork, gpu=GPU, volumes={"/data": vol}, secrets=[hf_secret], timeout=3600
)
def run_diag(model: str, configs: list[str], max_num_seqs: int, offline: bool) -> dict:
    env = os.environ.copy()
    env["TOKENIZERS_PARALLELISM"] = "false"
    if offline:
        env.update(
            HF_HOME="/data/hf_cache", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
        )
    out = {}
    for c in configs:
        cmd = [
            sys.executable,
            "/bench/diag_engine.py",
            "--model",
            model,
            "--config",
            c,
            "--max-num-seqs",
            str(max_num_seqs),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        text = proc.stdout + "\n--- stderr ---\n" + proc.stderr
        keep = [ln for ln in text.splitlines() if "[diag]" in ln or "Error" in ln]
        out[c] = {
            "rc": proc.returncode,
            "diag": keep[-40:],
            "stderr_tail": proc.stderr[-12000:],
        }
        print(
            f"[modal] {c}: rc={proc.returncode}\n" + "\n".join(keep[-12:]), flush=True
        )
    return out


@app.local_entrypoint()
def diag_main(
    model: str = "Qwen/Qwen3.6-27B",
    configs: str = "lora_maxcap,nolora_maxcap,nolora_maxcap_nopacked",
    max_num_seqs: int = 1024,
    offline: bool = True,
    out: str = "/tmp/vlp_diag.json",
):
    res = run_diag.remote(
        model, [c for c in configs.split(",") if c], max_num_seqs, offline
    )
    Path(out).write_text(json.dumps(res, indent=1))
    for c, r in res.items():
        print(f"== {c}: rc={r['rc']}")
        print("\n".join(r["diag"][-6:]))
    print(f"[local] wrote {out}")
