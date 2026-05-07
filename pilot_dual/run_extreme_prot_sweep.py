# -*- coding: utf-8 -*-
"""
Extreme-protection sweep: protect blocks 0-9, expose blocks 10-11 to pruning.
Runs Full_v8 + abl1 + abl2 + abl3 variants at h70_m95, serial on one GPU.

Reuses Phase B caches — only Phase C is executed (~15-20 min per variant).

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m pilot_dual.run_extreme_prot_sweep \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth
"""

import argparse, json, os, subprocess, sys, time

V8_CACHE  = "results/pilot_cascade_v8_sweep/_phase_b_cache"
ABL1_CACHE = "results/ablation_v8/abl1_no_boundary_fisher/_phase_b_cache"

VARIANTS = [
    {
        "name":  "full_v8",
        "cache": V8_CACHE,
        "extra": [],
    },
    {
        "name":  "abl1_no_BF",
        "cache": ABL1_CACHE,
        "extra": ["--boundary_fisher_weight", "0"],
    },
    {
        "name":  "abl2_zero",
        "cache": V8_CACHE,
        "extra": ["--no_adaptive_alpha", "--phase1_alpha", "1.0"],
    },
    {
        "name":  "abl3_b0",
        "cache": V8_CACHE,
        "extra": ["--dist_beta", "0"],
    },
]


def build_cmd(args, variant):
    out_dir = os.path.join(args.output_dir, variant["name"])
    cmd = [
        sys.executable, "-m", "pilot_dual.run_cascade_v8",
        "--medsam_ckpt",    args.medsam_ckpt,
        "--sam_ckpt",       args.sam_ckpt,
        "--output_dir",     out_dir,
        "--device",         "cuda:0",
        "--phase",          "c_only",
        "--cache_dir",      variant["cache"],
        "--head_sparsities", "0.7",
        "--mlp_sparsities",  "0.95",
        "--protected_blocks"] + [str(b) for b in range(10)] + [
        "--data_roots",     ] + args.data_roots + [
        "--dataset_names",  ] + args.dataset_names + [
        "--cal_sizes",      ] + [str(c) for c in args.cal_sizes] + [
        "--recovery_steps",   str(args.recovery_steps),
        "--recovery_lr",      str(args.recovery_lr),
        "--seed",             "42",
        "--num_workers",      "4",
    ] + variant["extra"]
    return cmd, out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medsam_ckpt",  default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--sam_ckpt",     default="work_dir/SAM/sam_vit_b_01ec64.pth")
    p.add_argument("--output_dir",   default="results/extreme_prot_v8")
    p.add_argument("--data_roots",   nargs="+",
                   default=["asserts/kvasir-seg/Kvasir-SEG",
                            "asserts/CVC-ColonDB",
                            "asserts/CVC-ClinicDB"])
    p.add_argument("--dataset_names", nargs="+",
                   default=["Kvasir", "ColonDB", "ClinicDB"])
    p.add_argument("--cal_sizes",    nargs="+", type=int, default=[128, 128, 128])
    p.add_argument("--recovery_steps", type=int,   default=100)
    p.add_argument("--recovery_lr",    type=float, default=1e-5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print("EXTREME PROTECTION SWEEP — protect blocks 0-9, prune 10-11")
    print(f"  Config   : h70_m95")
    print(f"  Variants : {[v['name'] for v in VARIANTS]}")
    print(f"  Phase    : c_only (reuse Phase B cache)")
    print("=" * 65)

    summary = []
    t_total = time.time()

    for variant in VARIANTS:
        cmd, out_dir = build_cmd(args, variant)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(args.output_dir, f"{variant['name']}.log")

        print(f"\n[START] {variant['name']}  cache={os.path.basename(variant['cache'])}")
        t0 = time.time()
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
        elapsed = (time.time() - t0) / 60
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        print(f"[{status}] {variant['name']}  ({elapsed:.1f} min)")
        summary.append((variant["name"], proc.returncode, out_dir))

    # Results
    print("\n" + "=" * 65)
    print("RESULTS  (protect blocks 0-9 | h70_m95 | Phase C only)")
    print(f"{'Variant':<15} {'BF1':>7} {'Dice':>7} {'IoU':>7} {'Par%':>7}")
    print("-" * 50)
    for name, rc, out_dir in summary:
        jf = os.path.join(out_dir, "cascade_results_v8.json")
        if rc != 0 or not os.path.exists(jf):
            print(f"{name:<15}  FAILED")
            continue
        data = json.load(open(jf))
        for r in data.get("cascade_results", []):
            if r.get("phase") == "cascade":
                m = r["macro"]
                bf1  = m.get("mean_boundary_f1", float("nan"))
                dice = m.get("mean_dice",         float("nan"))
                iou  = m.get("mean_iou",          float("nan"))
                par  = r.get("param_reduction_pct", float("nan"))
                print(f"{name:<15} {bf1:>7.4f} {dice:>7.4f} {iou:>7.4f} {par:>6.1f}%")

    print(f"\nTotal elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"Logs: {args.output_dir}/<variant>.log")

    failed = [n for n, rc, _ in summary if rc != 0]
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
