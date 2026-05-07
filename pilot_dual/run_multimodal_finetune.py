# -*- coding: utf-8 -*-
"""
Post-pruning fine-tuning sweep for multi-modal pruned models (Phase D).

Reads Phase-C results from results/multimodal_v8/ and launches
run_finetune_pruned workers in parallel across 2 GPUs, 2 workers per GPU.

GPU 1: h50_m70  h70_m70
GPU 2: h70_m95  h80_m95

(h40_m30 omitted by default; add --include_h40 to include it serially after.)

Usage:
    python -m pilot_dual.run_multimodal_finetune \
        --medsam_ckpt  work_dir/MedSAM/medsam_vit_b.pth \
        --phase_c_dir  results/multimodal_v8 \
        --output_dir   results/multimodal_finetune \
        --devices 1 2
"""

import argparse, json, os, subprocess, sys, time

# (head_sp, mlp_sp, logical_gpu)   logical 0→devices[0], 1→devices[1]
CONFIGS = [
    (0.5, 0.70, 0),   # GPU 1 slot 1
    (0.7, 0.70, 0),   # GPU 1 slot 2
    (0.7, 0.95, 1),   # GPU 2 slot 1
    (0.8, 0.95, 1),   # GPU 2 slot 2
]


def tag(h, m):
    return f"h{int(h*100)}_m{int(m*100)}"


def build_cmd(args, h, m):
    t = tag(h, m)
    phase_c_json = os.path.join(args.phase_c_dir, t, "cascade_results_v8.json")
    # run_finetune_pruned appends its own config_tag subdir, so pass parent directly
    cmd = [
        sys.executable, "-m", "pilot_dual.run_finetune_pruned",
        "--medsam_ckpt",    args.medsam_ckpt,
        "--phase_c_json",   phase_c_json,
        "--head_sparsity",  str(h),
        "--mlp_sparsity",   str(m),
        "--output_dir",     args.output_dir,
        "--data_roots",     ] + args.data_roots + [
        "--dataset_names",  ] + args.dataset_names + [
        "--num_epochs",     str(args.num_epochs),
        "--batch_size",     str(args.batch_size),
        "--lr",             str(args.lr),
        "--val_frac",       str(args.val_frac),
        "--device",         "cuda:0",      # remapped via CUDA_VISIBLE_DEVICES
        "--seed",           "42",
        "--num_workers",    "4",
    ]
    if args.use_amp:
        cmd.append("--use_amp")
    return cmd, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medsam_ckpt",  default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--phase_c_dir",  default="results/multimodal_v8",
                   help="Directory containing per-config subdirs (h50_m70 etc.)")
    p.add_argument("--output_dir",   default="results/multimodal_finetune")
    p.add_argument("--devices",      nargs="+", type=int, default=[1, 2],
                   help="Physical GPU ids for logical [0, 1]")
    p.add_argument("--data_roots",   nargs="+",
                   default=["asserts/BUSI",
                            "asserts/CVC-ClinicDB",
                            "asserts/ISIC2018"])
    p.add_argument("--dataset_names", nargs="+",
                   default=["BUSI", "ClinicDB", "ISIC2018"])
    p.add_argument("--num_epochs",   type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--val_frac",     type=float, default=0.2)
    p.add_argument("--use_amp",      action="store_true", default=True)
    p.add_argument("--include_h40",  action="store_true",
                   help="Also fine-tune h40_m30 (runs serially after the 4 main configs)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    dev_map = {i: args.devices[i] for i in range(len(args.devices))}

    print("=" * 65)
    print("MULTI-MODAL FINE-TUNING SWEEP  (Phase D)")
    print(f"  Phase-C dir : {args.phase_c_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Datasets    : {args.dataset_names}")
    print(f"  Epochs      : {args.num_epochs}  batch={args.batch_size}  lr={args.lr}")
    for h, m, lg in CONFIGS:
        print(f"  {tag(h,m):<12}  cuda:{dev_map.get(lg, lg)}")
    print("=" * 65)

    # Validate all phase_c JSONs exist before launching
    for h, m, _ in CONFIGS:
        jf = os.path.join(args.phase_c_dir, tag(h, m), "cascade_results_v8.json")
        if not os.path.exists(jf):
            print(f"  ERROR: missing {jf}")
            sys.exit(1)

    # Launch all 4 simultaneously
    procs = []
    for h, m, lg in CONFIGS:
        cmd, t = build_cmd(args, h, m)
        log_path = os.path.join(logs_dir, f"{t}.log")
        physical_gpu = dev_map.get(lg, lg)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        logf = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append({"tag": t, "proc": proc, "logf": logf, "log": log_path,
                      "h": h, "m": m})
        print(f"  STARTED  {t:<12}  cuda:{physical_gpu}  pid={proc.pid}  log={log_path}")

    # Poll until all done
    pending = set(range(len(procs)))
    t0 = time.time()
    while pending:
        time.sleep(60)
        for i in list(pending):
            rc = procs[i]["proc"].poll()
            if rc is not None:
                procs[i]["logf"].close()
                elapsed = (time.time() - t0) / 60
                status = "OK" if rc == 0 else f"FAIL({rc})"
                print(f"  [{elapsed:6.1f}min]  {status}  {procs[i]['tag']}")
                pending.discard(i)

    # Optional h40_m30 serial run (after main 4 complete)
    extra_summary = []
    if args.include_h40:
        print("\n  [h40_m30] starting serially ...")
        h, m = 0.4, 0.3
        cmd, t = build_cmd(args, h, m)
        log_path = os.path.join(logs_dir, f"{t}.log")
        # use whichever GPU just freed up first
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(dev_map.get(0, args.devices[0]))
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        extra_summary.append({"tag": t, "proc": proc, "log": log_path, "h": h, "m": m})
        status = "OK" if proc.returncode == 0 else f"FAIL({proc.returncode})"
        print(f"  {status}  {t}")

    # Results summary
    print("\n" + "=" * 65)
    print("FINE-TUNING RESULTS  (val set — best epoch)")
    print(f"{'Config':<12} {'BF1_ft':>8} {'Dice_ft':>8} {'ΔBDF1':>8} {'Par%':>7}")
    print("-" * 50)

    all_entries = procs + extra_summary
    failed = []
    for entry in all_entries:
        t_ = entry["tag"]
        h, m = entry["h"], entry["m"]
        rc   = entry["proc"].returncode
        # run_finetune_pruned writes to <output_dir>/<config_tag>/finetune_results.json
        jf   = os.path.join(args.output_dir, t_, "finetune_results.json")
        if rc != 0 or not os.path.exists(jf):
            print(f"  {t_:<12}  FAILED")
            failed.append(t_)
            continue
        data    = json.load(open(jf))
        fm      = data["final_eval"]["macro"]
        pc_bf1  = data["phase_c_metrics"]["mean_boundary_f1"]
        bf1_ft  = fm["mean_boundary_f1"]
        dice_ft = fm["mean_dice"]
        par     = data["pruning_stats"].get("param_reduction_pct", float("nan"))
        delta   = bf1_ft - pc_bf1
        print(f"  {t_:<12} {bf1_ft:>8.4f} {dice_ft:>8.4f} {delta:>+8.4f} {par:>6.1f}%")

    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")
    print(f"Logs: {logs_dir}/")

    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
