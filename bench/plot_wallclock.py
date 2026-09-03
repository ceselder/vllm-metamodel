"""Wall-clock time of one LLM.generate() call vs batch size, one line per configuration.
Reads bench/results_summary.json (tables[model].series[cfg][batch].wall_s). No speedup axis: just seconds.
    python bench/plot_wallclock.py [--out-dir bench]"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SERIES = [  # key, legend label, color (validated categorical palette), marker
    ("stock_eager", "stock vllm-lens 1.1.0", "#2a78d6", "o"),
    ("fork_vectorized", "vllm-metamodel (eager)", "#eb6834", "s"),
    ("fork_graphs", "vllm-metamodel + CUDA graphs", "#1baf7a", "D"),
    ("ceiling_plain", "plain vLLM, no steering (torch.compile + graphs)", "#87867F", "^"),
]
CLAIM = {"Qwen/Qwen3.6-27B": "Per-request steering on Qwen3.6-27B (1× B200): vllm-metamodel generates as fast as plain vLLM; stock vllm-lens is 38× slower at batch 2,048",
         "Qwen/Qwen3-1.7B": "Per-request steering on Qwen3-1.7B (1× B200): vllm-metamodel matches plain vLLM; stock vllm-lens is 59× slower at batch 1,024"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=str(HERE)); a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tables = json.load(open(HERE / "results_summary.json"))["tables"]
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e8e5dc", "grid.linewidth": 0.7, "axes.axisbelow": True, "font.size": 10.5, "legend.frameon": False})
    for model, tab in tables.items():
        stem = "wallclock_vs_batch" + ("" if "27B" in model else "_qwen3-1.7b")
        fig, ax = plt.subplots(figsize=(9, 5.6)); data = {"model": model, "series": {}}
        for key, label, col, mk in SERIES:
            s = tab["series"].get(key)
            if not s: continue
            bs = sorted(int(b) for b in s); ws = [s[str(b)]["wall_s"] for b in bs]
            data["series"][label] = {"batch": bs, "wall_s": ws}
            ax.plot(bs, ws, marker=mk, ms=7, lw=2.2, color=col, label=label, markeredgecolor="white", markeredgewidth=1.2)
            ax.annotate(f"{ws[-1]:.1f} s", (bs[-1], ws[-1]), textcoords="offset points", xytext=(7, -3), fontsize=9, color="#52514e")
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks([8, 32, 128, 512, 1024, 2048][: len(bs)]); ax.set_xticklabels([str(b) for b in [8, 32, 128, 512, 1024, 2048][: len(bs)]])
        ax.set_xlabel("batch size (requests per generate() call, one steering vector each)"); ax.set_ylabel("wall-clock seconds per generate() call  (log)")
        ax.set_title(CLAIM.get(model, model), fontsize=10.5, loc="left", pad=12)
        ax.text(0, 1.005, "96-token prompt, 40 new tokens, bf16; lower is better", transform=ax.transAxes, fontsize=9, color="#52514e", va="bottom")
        ax.legend(loc="upper left", fontsize=9.5); fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=160, bbox_inches="tight"); fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
        json.dump(data, open(out / f"{stem}_data.json", "w"), indent=1); plt.close(fig); print("wrote", out / f"{stem}.png")

if __name__ == "__main__":
    main()
