"""Wall-clock time of one LLM.generate() call vs batch size (minimal). Reads bench/results_summary.json.
    python bench/plot_wallclock.py [--out-dir bench]"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
SERIES = [  # key, legend label, color, marker  (CUDA-graph mode is the default vllm-metamodel)
    ("stock_eager", "vllm-lens 1.1.0", "#2a78d6", "o"),
    ("fork_vectorized", "vllm-metamodel (eager)", "#eb6834", "s"),
    ("fork_graphs", "vllm-metamodel", "#1baf7a", "D"),
    ("ceiling_plain", "vLLM, no steering", "#87867F", "^"),
]
TITLE = {"Qwen/Qwen3.6-27B": "Generation performance (Qwen 3.6 27B)", "Qwen/Qwen3-1.7B": "Generation performance (Qwen 3 1.7B)"}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-dir", default=str(HERE)); a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tables = json.load(open(HERE / "results_summary.json"))["tables"]
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e8e5dc", "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 15,
                         "axes.titlesize": 18, "axes.labelsize": 15, "legend.fontsize": 14, "legend.frameon": False})
    for model, tab in tables.items():
        stem = "wallclock_vs_batch" + ("" if "27B" in model else "_qwen3-1.7b")
        fig, ax = plt.subplots(figsize=(9, 6)); data = {"model": model, "series": {}}
        for key, label, col, mk in SERIES:
            s = tab["series"].get(key)
            if not s: continue
            bs = sorted(int(b) for b in s); ws = [s[str(b)]["wall_s"] for b in bs]
            data["series"][label] = {"batch": bs, "wall_s": ws}
            ax.plot(bs, ws, marker=mk, ms=8, lw=2.6, color=col, label=label, markeredgecolor="white", markeredgewidth=1.2)
        ticks = [8, 32, 128, 512, 1024, 2048][: len(bs)]
        ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xticks(ticks); ax.set_xticklabels([str(b) for b in ticks])
        ax.set_xlabel("batch size"); ax.set_ylabel("seconds per generate() call")
        ax.set_title(TITLE.get(model, model), loc="left", pad=14, fontweight="bold")
        leg = ax.legend(loc="upper left")
        for t in leg.get_texts():
            if t.get_text().startswith("vllm-metamodel"):
                t.set_fontweight("bold")
        fig.tight_layout()
        fig.savefig(out / f"{stem}.png", dpi=170, bbox_inches="tight"); fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
        json.dump(data, open(out / f"{stem}_data.json", "w"), indent=1); plt.close(fig); print("wrote", out / f"{stem}.png")

if __name__ == "__main__":
    main()
