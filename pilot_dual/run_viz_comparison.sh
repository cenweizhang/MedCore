#!/bin/bash
# Run all mask generation steps and create final visualization.
# Execute on a GPU-enabled node from repo root:
#   cd /volume/med-train/users/jshan/testSAMpruning
#   bash pilot_dual/run_viz_comparison.sh [cuda:0] [seed]
#
# Examples:
#   bash pilot_dual/run_viz_comparison.sh cuda:0 42    # seed=42 (default)
#   bash pilot_dual/run_viz_comparison.sh cuda:0 123   # try a different seed

set -e
DEVICE=${1:-cuda:0}
SEED=${2:-42}
OUT=results/viz_comparison

echo "========================================================"
echo "  Polyp Comparison Visualization Pipeline"
echo "  Device: $DEVICE   Seed: $SEED   Output: $OUT"
echo "========================================================"

# Step 1: SAM-based models (medsam env)
echo ""
echo "[1/4] medsam env: MedSAM + Ours h70/h80_m95 + EfficientSAM-Ti + SlimSAM + SAMed + QMedSAM"
conda run -n medsam python -m pilot_dual.generate_masks_medsam_env \
    --output_dir $OUT --device $DEVICE --seed $SEED

# Step 2: EMCAD + MK-UNet (emcadenv)
echo ""
echo "[2/4] emcadenv: EMCAD + MK-UNet"
conda run -n emcadenv python pilot_dual/generate_masks_emcad_env.py \
    --output_dir $OUT --device $DEVICE --seed $SEED

# Step 3: Swin-Unet + nnUNet (medsam env; nnUNet predictions already exist)
echo ""
echo "[3/4] medsam env: Swin-Unet + nnUNet (copy existing preds)"
conda run -n medsam python pilot_dual/generate_masks_swinunet_nnunet.py \
    --output_dir $OUT --device $DEVICE --seed $SEED

# Step 4: Visualize
echo ""
echo "[4/4] Creating final visualization..."
conda run -n medsam python pilot_dual/visualize_comparison.py \
    --mask_dir ${OUT}/masks \
    --output   ${OUT}/comparison_seed${SEED}.png \
    --seed $SEED \
    --dpi 200

echo ""
echo "Done! Results in $OUT/"
echo "  comparison_seed${SEED}.png          — full comparison figure"
echo "  comparison_seed${SEED}_1200px.png   — 1200px wide version"
echo "  metrics_medsam_env.json             — per-image metrics (SAM-based models)"
echo "  metrics_emcad_env.json              — per-image metrics (EMCAD/MK-UNet)"
echo "  metrics_swinunet_nnunet.json        — per-image metrics (Swin-Unet/nnUNet)"
