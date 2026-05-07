"""
Quick analysis of v5 scores:
  Method A: Δ_zero vs Δ_reset correlation (head + MLP)
  Method C: per-block weight drift (MedSAM vs SAM) vs Fisher sensitivity

python -m pilot_dual.analyze_scores \
    --scores results/pilot_cascade_v5/scores.npz \
    --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
    --sam_ckpt work_dir/SAM/sam_vit_b_01ec64.pth \
    --output results/pilot_cascade_v5/score_analysis.png
"""
import argparse, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

def load_ckpt_flat(path):
    """Load checkpoint; unwrap 'model' or 'state_dict' wrapper if present."""
    raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("model", "state_dict", "image_encoder"):
            if key in raw:
                return raw[key]
    return raw

def find_keys(state, pattern):
    return [k for k in state if pattern in k]

def block_drift(medsam, sam, block_idx):
    """L2 norm of (medsam - sam) for each sub-module in a block."""
    result = {}
    prefixes = {
        "attn_qkv":  f"image_encoder.blocks.{block_idx}.attn.qkv.weight",
        "attn_proj": f"image_encoder.blocks.{block_idx}.attn.proj.weight",
        "mlp_lin1":  f"image_encoder.blocks.{block_idx}.mlp.lin1.weight",
        "mlp_lin2":  f"image_encoder.blocks.{block_idx}.mlp.lin2.weight",
        "norm1":     f"image_encoder.blocks.{block_idx}.norm1.weight",
        "norm2":     f"image_encoder.blocks.{block_idx}.norm2.weight",
    }
    for name, key in prefixes.items():
        if key in medsam and key in sam:
            diff = (medsam[key].float() - sam[key].float()).norm().item()
            result[name] = diff
        else:
            result[name] = None
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores",     default="results/pilot_cascade_v5/scores.npz")
    parser.add_argument("--medsam_ckpt",default="work_dir/MedSAM/medsam_vit_b.pth")
    parser.add_argument("--sam_ckpt",   default="work_dir/SAM/sam_vit_b_01ec64.pth")
    parser.add_argument("--output",     default="results/pilot_cascade_v5/score_analysis.png")
    args = parser.parse_args()

    sc = np.load(args.scores)
    dz_head = sc["delta_zero_head"]   # (144,)
    dr_head = sc["delta_reset_head"]  # (144,)
    dz_mlp  = sc["delta_zero_mlp"]    # (36864,)
    dr_mlp  = sc["delta_reset_mlp"]
    block_sens = sc["block_sensitivity"]  # (12,)

    # ── Method A ────────────────────────────────────────────────────────────
    corr_head = float(np.corrcoef(dz_head, dr_head)[0, 1])
    corr_mlp  = float(np.corrcoef(dz_mlp,  dr_mlp )[0, 1])

    print("=" * 55)
    print("Method A: Δ_zero vs Δ_reset correlation")
    print(f"  Head scores (144):    r = {corr_head:.4f}")
    print(f"  MLP  scores (36864):  r = {corr_mlp:.4f}")
    if corr_head > 0.95:
        print("  → High correlation: model is deeply specialised;")
        print("    Δ_reset carries little additional info (α≈1 is sufficient)")
    elif corr_head > 0.7:
        print("  → Moderate correlation: some heads have diverged scoring;")
        print("    intermediate α may help")
    else:
        print("  → Low correlation: Δ_reset captures distinct signal;")
        print("    α tuning matters significantly")

    # Per-block head correlation
    print("\nPer-block head correlation (Δ_zero vs Δ_reset):")
    for b in range(12):
        dz_b = dz_head[b*12:(b+1)*12]
        dr_b = dr_head[b*12:(b+1)*12]
        r = float(np.corrcoef(dz_b, dr_b)[0, 1]) if dz_b.std() > 0 and dr_b.std() > 0 else float("nan")
        bar = "#" * int(abs(r) * 20)
        print(f"  Block {b:2d}: r={r:+.3f}  {bar}")

    # ── Method C ────────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("Method C: per-block weight drift (MedSAM vs SAM)")
    medsam = load_ckpt_flat(args.medsam_ckpt)
    sam    = load_ckpt_flat(args.sam_ckpt)

    # Check key presence
    sample_key = "image_encoder.blocks.0.attn.qkv.weight"
    if sample_key not in medsam:
        print(f"  [WARN] key '{sample_key}' not found in MedSAM ckpt.")
        print(f"  Available prefixes: {set(k.split('.')[0] for k in medsam)}")
    if sample_key not in sam:
        print(f"  [WARN] key '{sample_key}' not found in SAM ckpt.")
        print(f"  Available prefixes: {set(k.split('.')[0] for k in sam)}")

    drifts = []
    print(f"\n{'Block':>6}  {'attn_qkv':>10}  {'attn_proj':>10}  {'mlp_lin1':>10}  {'mlp_lin2':>10}  {'Fisher':>10}")
    for b in range(12):
        d = block_drift(medsam, sam, b)
        total = sum(v for v in d.values() if v is not None)
        drifts.append(d)
        qkv  = f"{d['attn_qkv']:.3f}"  if d["attn_qkv"]  is not None else "N/A"
        proj = f"{d['attn_proj']:.3f}" if d["attn_proj"] is not None else "N/A"
        l1   = f"{d['mlp_lin1']:.3f}"  if d["mlp_lin1"]  is not None else "N/A"
        l2   = f"{d['mlp_lin2']:.3f}"  if d["mlp_lin2"]  is not None else "N/A"
        fs   = f"{block_sens[b]:.4f}"
        print(f"  {b:4d}  {qkv:>10}  {proj:>10}  {l1:>10}  {l2:>10}  {fs:>10}")

    # ── Figures ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Score Analysis (v5): Δ_zero vs Δ_reset + Weight Drift", fontsize=13)

    # A1: head scatter
    ax = axes[0, 0]
    ax.scatter(dz_head, dr_head, alpha=0.5, s=20)
    ax.set_xlabel("Δ_zero (head)"); ax.set_ylabel("Δ_reset (head)")
    ax.set_title(f"Head scores  r={corr_head:.3f}")

    # A2: MLP scatter (subsample for speed)
    ax = axes[0, 1]
    idx = np.random.choice(len(dz_mlp), min(3000, len(dz_mlp)), replace=False)
    ax.scatter(dz_mlp[idx], dr_mlp[idx], alpha=0.3, s=5)
    ax.set_xlabel("Δ_zero (MLP)"); ax.set_ylabel("Δ_reset (MLP)")
    ax.set_title(f"MLP neuron scores  r={corr_mlp:.3f}")

    # A3: per-block correlation
    ax = axes[0, 2]
    block_corrs = []
    for b in range(12):
        dz_b = dz_head[b*12:(b+1)*12]; dr_b = dr_head[b*12:(b+1)*12]
        r = float(np.corrcoef(dz_b, dr_b)[0, 1]) if dz_b.std() > 0 and dr_b.std() > 0 else 0.0
        block_corrs.append(r)
    ax.bar(range(12), block_corrs)
    ax.axhline(0.9, color="red", linestyle="--", label="r=0.9")
    ax.set_xlabel("Block"); ax.set_ylabel("Pearson r")
    ax.set_title("Per-block Δ_zero/Δ_reset correlation"); ax.legend()

    # C1: total drift per block
    ax = axes[1, 0]
    total_drifts = [sum(v for v in d.values() if v is not None) for d in drifts]
    ax.bar(range(12), total_drifts, color="steelblue")
    ax.set_xlabel("Block"); ax.set_ylabel("Total L2 drift")
    ax.set_title("Total weight drift per block (MedSAM − SAM)")
    ax2 = ax.twinx()
    ax2.plot(range(12), block_sens, color="orange", marker="o", label="Fisher")
    ax2.set_ylabel("Fisher sensitivity", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange"); ax2.legend(loc="upper left")

    # C2: drift breakdown by layer type
    ax = axes[1, 1]
    layer_names = ["attn_qkv", "attn_proj", "mlp_lin1", "mlp_lin2"]
    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]
    bottoms = np.zeros(12)
    for ln, col in zip(layer_names, colors):
        vals = np.array([d[ln] if d[ln] is not None else 0.0 for d in drifts])
        ax.bar(range(12), vals, bottom=bottoms, label=ln, color=col)
        bottoms += vals
    ax.set_xlabel("Block"); ax.set_ylabel("L2 drift")
    ax.set_title("Drift breakdown by layer type"); ax.legend(fontsize=7)

    # C3: drift vs Fisher scatter per block
    ax = axes[1, 2]
    attn_drifts = [((d["attn_qkv"] or 0) + (d["attn_proj"] or 0)) for d in drifts]
    ax.scatter(attn_drifts, block_sens, s=80, zorder=3)
    for b in range(12):
        ax.annotate(str(b), (attn_drifts[b], block_sens[b]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("Attn weight drift (L2)"); ax.set_ylabel("Fisher sensitivity")
    ax.set_title("Block drift vs Fisher sensitivity")

    plt.tight_layout()
    plt.savefig(args.output, dpi=120)
    print(f"\nFigure saved → {args.output}")

if __name__ == "__main__":
    main()
