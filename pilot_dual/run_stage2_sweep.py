# -*- coding: utf-8 -*-
"""
Stage-2 sequential pruning sweep: prune blocks 10-11 on top of Stage-1 model.

Stage 1 already pruned blocks 0-9 at h70_m95 (protected=[10,11]).
Stage 2 uses the Stage-1 baked checkpoint and prunes blocks 10-11
(protected=[0..9]), targeting per-block h>=60%, m>=70%.

Global sparsity ↔ per-block translation (blocks 10-11 = 2/12 of params):
  global_h=0.10 → ~60%/blk   global_m=0.117 → ~70%/blk
  global_h=0.12 → ~72%/blk   global_m=0.14  → ~84%/blk
  global_h=0.14 → ~84%/blk   global_m=0.158 → ~95%/blk

Ablation variants (abl1/2/3) are run at the highest compression config (C).

Workflow:
  1. Run bake_stage1_masks.py first to produce stage1_pruned_h70m95.pth
  2. Run this script (serial on one GPU)

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m pilot_dual.run_stage2_sweep \
        --stage1_ckpt  results/stage2/stage1_pruned_h70m95.pth \
        --sam_ckpt     work_dir/SAM/sam_vit_b_01ec64.pth \
        --output_dir   results/stage2_sweep
"""

import argparse, json, os, subprocess, sys, time

PROTECTED_BLOCKS = list(range(10))   # protect 0-9, expose 10-11

# (global_head_sp, global_mlp_sp, name)
BASE_CONFIGS = [
    (0.10, 0.117, "cfg_A_h60m70"),
    (0.12, 0.14,  "cfg_B_h72m84"),
    (0.14, 0.158, "cfg_C_h84m95"),
]

# Ablation variants at highest compression config C
ABL_VARIANTS = [
    # (name, extra_flags)
    ("cfg_C_abl1_no_BF",   ["--boundary_fisher_weight", "0"]),
    ("cfg_C_abl2_zero",    ["--no_adaptive_alpha", "--phase1_alpha", "1.0"]),
    ("cfg_C_abl3_b0",      ["--dist_beta", "0"]),
]


def build_cmd(args, h, m, name, extra=None):
    out_dir = os.path.join(args.output_dir, name)
    cmd = [
        sys.executable, "-m", "pilot_dual.run_cascade_v8",
        "--medsam_ckpt",     args.stage1_ckpt,
        "--sam_ckpt",        args.sam_ckpt,
        "--output_dir",      out_dir,
        "--device",          "cuda:0",        # remapped via CUDA_VISIBLE_DEVICES
        "--phase",           "all",
        "--head_sparsities", str(h),
        "--mlp_sparsities",  str(m),
        "--protected_blocks"] + [str(b) for b in PROTECTED_BLOCKS] + [
        "--data_roots",     ] + args.data_roots + [
        "--dataset_names",  ] + args.dataset_names + [
        "--cal_sizes",      ] + [str(c) for c in args.cal_sizes] + [
        "--recovery_steps",   str(args.recovery_steps),
        "--recovery_lr",      str(args.recovery_lr),
        "--seed",             "42",
        "--num_workers",      "4",
    ]
    if extra:
        cmd += extra
    return cmd, out_dir


def run_one(cmd, out_dir, name, log_dir):
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{name}.log")
    print(f"\n[START] {name}")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = (time.time() - t0) / 60
    status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
    print(f"[{status}] {name}  ({elapsed:.1f} min)  log={log_path}")
    return proc.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", default="results/stage2/stage1_pruned_h70m95.pth")
    p.add_argument("--sam_ckpt",    default="work_dir/SAM/sam_vit_b_01ec64.pth")
    p.add_argument("--output_dir",  default="results/stage2_sweep")
    p.add_argument("--data_roots",  nargs="+",
                   default=["asserts/kvasir-seg/Kvasir-SEG",
                            "asserts/CVC-ColonDB",
                            "asserts/CVC-ClinicDB"])
    p.add_argument("--dataset_names", nargs="+",
                   default=["Kvasir", "ColonDB", "ClinicDB"])
    p.add_argument("--cal_sizes",   nargs="+", type=int, default=[128, 128, 128])
    p.add_argument("--recovery_steps", type=int,   default=100)
    p.add_argument("--recovery_lr",    type=float, default=1e-5)
    p.add_argument("--skip_ablations", action="store_true",
                   help="Run base configs only (skip abl variants)")
    args = p.parse_args()

    if not os.path.exists(args.stage1_ckpt):
        print(f"ERROR: Stage-1 checkpoint not found: {args.stage1_ckpt}")
        print("Run first:  python -m pilot_dual.bake_stage1_masks")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    print("=" * 65)
    print("STAGE-2 SWEEP — prune blocks 10-11 on Stage-1 pruned model")
    print(f"  Stage-1 ckpt : {args.stage1_ckpt}")
    print(f"  Protected    : blocks {PROTECTED_BLOCKS}")
    print(f"  Datasets     : {args.dataset_names}")
    for h, m, name in BASE_CONFIGS:
        print(f"  {name:<22}  h={h:.3f}  m={m:.3f}")
    if not args.skip_ablations:
        for name, _ in ABL_VARIANTS:
            print(f"  {name:<22}  (ablation)")
    print("=" * 65)

    summary = []
    t_total = time.time()

    # Base configs A, B, C
    for h, m, name in BASE_CONFIGS:
        cmd, out_dir = build_cmd(args, h, m, name)
        rc = run_one(cmd, out_dir, name, log_dir)
        summary.append((name, rc, out_dir))

    # Ablation variants at config C
    if not args.skip_ablations:
        h_c, m_c, _ = BASE_CONFIGS[-1]   # highest compression
        for abl_name, extra in ABL_VARIANTS:
            cmd, out_dir = build_cmd(args, h_c, m_c, abl_name, extra)
            rc = run_one(cmd, out_dir, abl_name, log_dir)
            summary.append((abl_name, rc, out_dir))

    # Results table
    print("\n" + "=" * 65)
    print("STAGE-2 RESULTS  (protect blocks 0-9 | cascade results)")
    print(f"{'Config':<25} {'BF1':>7} {'Dice':>7} {'IoU':>7} {'Par%':>7}")
    print("-" * 55)
    for name, rc, out_dir in summary:
        jf = os.path.join(out_dir, "cascade_results_v8.json")
        if rc != 0 or not os.path.exists(jf):
            print(f"{name:<25}  FAILED")
            continue
        data = json.load(open(jf))
        for r in data.get("cascade_results", []):
            if r.get("phase") == "cascade":
                m_res = r["macro"]
                bf1  = m_res.get("mean_boundary_f1", float("nan"))
                dice = m_res.get("mean_dice",         float("nan"))
                iou  = m_res.get("mean_iou",          float("nan"))
                par  = r.get("param_reduction_pct",   float("nan"))
                print(f"{name:<25} {bf1:>7.4f} {dice:>7.4f} {iou:>7.4f} {par:>6.1f}%")

    print(f"\nTotal elapsed: {(time.time()-t_total)/60:.1f} min")

    failed = [n for n, rc, _ in summary if rc != 0]
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
