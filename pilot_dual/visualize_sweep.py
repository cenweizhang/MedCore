"""Quick visualization: compression vs accuracy for v8 sweep results.

python -m pilot_dual.visualize_sweep \
    --results results/pilot_cascade_v8_sweep/cascade_results_v8.json \
    --out results/pilot_cascade_v8_sweep/sweep_viz.png


"""
import json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/pilot_cascade_v8_sweep/cascade_results_v8.json")
    p.add_argument("--out", default="results/pilot_cascade_v8_sweep/sweep_viz.png")
    args = p.parse_args()

    data = load(args.results)
    cr = data["cascade_results"]
    bl = data["baseline"]
    bl_ds = data["baseline_per_dataset"]

    baseline_bf1  = bl["mean_boundary_f1"]
    baseline_dice = bl["mean_dice"]

    head_only = [e for e in cr if e["phase"] == "head_only"]
    cascade   = [e for e in cr if e["phase"] == "cascade"]

    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("v8 Sweep: Compression vs Accuracy", fontsize=14, fontweight="bold")

    mlp_vals = sorted({round(e.get("mlp_sp_target", 0), 2) for e in cascade})
    colors   = cm.viridis(np.linspace(0.1, 0.9, len(mlp_vals)))
    mlp_color = {m: c for m, c in zip(mlp_vals, colors)}

    def plot_panel(ax, metric_key, metric_label, baseline_val, title):
        # Baseline line
        ax.axhline(baseline_val, color="black", lw=1.5, ls="--", label="Baseline")
        ax.axhline(baseline_val - 0.015, color="gray", lw=1, ls=":", alpha=0.7, label="ΔBF1=−0.015")

        # Head-only (black markers)
        ho_par  = [e["param_reduction_pct"] for e in head_only]
        ho_met  = [e["macro"][metric_key]    for e in head_only]
        ho_h    = [e["head_sp_target"]       for e in head_only]
        ax.plot(ho_par, ho_met, "k-o", lw=1.5, ms=7, zorder=5, label="Head-only")
        for par, met, h in zip(ho_par, ho_met, ho_h):
            ax.annotate(f"h{h:.1f}", (par, met), textcoords="offset points",
                        xytext=(4, 3), fontsize=6.5, color="black")

        # Cascade: one line per mlp_sp
        for m in mlp_vals:
            subset = sorted(
                [e for e in cascade if round(e.get("mlp_sp_target", 0), 2) == m],
                key=lambda e: e["param_reduction_pct"],
            )
            if not subset:
                continue
            xs = [e["param_reduction_pct"]   for e in subset]
            ys = [e["macro"][metric_key]      for e in subset]
            ax.plot(xs, ys, "-o", color=mlp_color[m], lw=1.2, ms=5,
                    label=f"m={m:.2f}", alpha=0.85)

        ax.set_xlabel("Parameter reduction (%)", fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.5, ncol=2, loc="lower left")

    plot_panel(axes[0, 0], "mean_boundary_f1", "Macro BF1",  baseline_bf1,  "Macro BF1 vs Compression")
    plot_panel(axes[0, 1], "mean_dice",        "Macro Dice", baseline_dice, "Macro Dice vs Compression")

    # ------------------------------------------------------------------ #
    # Bottom row: per-dataset BF1 breakdown for cascade configs only
    datasets  = ["Kvasir", "ColonDB", "ClinicDB"]
    ds_colors = ["steelblue", "tomato", "mediumseagreen"]

    for ax, (metric_key, metric_label, title) in zip(
        [axes[1, 0], axes[1, 1]],
        [
            ("mean_boundary_f1", "BF1",  "Per-dataset BF1 (cascade only)"),
            ("mean_dice",        "Dice", "Per-dataset Dice (cascade only)"),
        ],
    ):
        ax.axhline(baseline_bf1 if "BF1" in metric_label else baseline_dice,
                   color="black", lw=1.2, ls="--", alpha=0.5, label="Baseline macro")

        for ds, c in zip(datasets, ds_colors):
            bl_val = bl_ds[ds][metric_key]
            ax.axhline(bl_val, color=c, lw=0.8, ls=":", alpha=0.4)

            subset = sorted(cascade, key=lambda e: e["param_reduction_pct"])
            xs = [e["param_reduction_pct"]        for e in subset]
            ys = [e["per_dataset"][ds][metric_key] for e in subset]
            ax.plot(xs, ys, "o-", color=c, lw=1.2, ms=4, alpha=0.7, label=ds)

        ax.set_xlabel("Parameter reduction (%)", fontsize=9)
        ax.set_ylabel(metric_label, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
