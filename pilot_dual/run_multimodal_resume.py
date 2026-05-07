# -*- coding: utf-8 -*-
"""
Resume interrupted multimodal fine-tuning runs.

Picks up h50_m70 (stopped epoch 6), h70_m95 (stopped epoch 7),
h80_m95 (stopped epoch 4) from their checkpoint_latest.pth files.

GPU layout:
  GPU devices[0]: h70_m95  h50_m70   (parallel)
  GPU devices[1]: h80_m95             (serial, one process)

Usage:
    python -m pilot_dual.run_multimodal_resume \
        --medsam_ckpt  work_dir/MedSAM/medsam_vit_b.pth \
        --phase_c_dir  results/multimodal_v8 \
        --finetune_dir results/multimodal_finetune \
        --devices 0 1
    python -m pilot_dual.run_multimodal_resume \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --devices 0 1

"""

import argparse, json, os, subprocess, sys, time

CONFIGS = [
    # (h, m, logical_gpu)
    (0.7, 0.95, 0),   # GPU devices[0] — furthest along, kick off first
    (0.5, 0.70, 0),   # GPU devices[0]
    (0.8, 0.95, 1),   # GPU devices[1]
]


def tag(h, m):
    return f"h{int(h*100)}_m{int(m*100)}"


def build_cmd(args, h, m):
    t = tag(h, m)
    phase_c_json  = os.path.join(args.phase_c_dir, t, "cascade_results_v8.json")
    resume_ckpt   = os.path.join(args.finetune_dir, t, "checkpoint_latest.pth")

    if not os.path.exists(phase_c_json):
        raise FileNotFoundError(f"Phase-C JSON not found: {phase_c_json}")
    if not os.path.exists(resume_ckpt):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")

    cmd = [
        sys.executable, "-m", "pilot_dual.run_finetune_pruned",
        "--medsam_ckpt",    args.medsam_ckpt,
        "--phase_c_json",   phase_c_json,
        "--head_sparsity",  str(h),
        "--mlp_sparsity",   str(m),
        "--output_dir",     args.finetune_dir,
        "--data_roots",     ] + args.data_roots + [
        "--dataset_names",  ] + args.dataset_names + [
        "--num_epochs",     str(args.num_epochs),
        "--batch_size",     str(args.batch_size),
        "--lr",             str(args.lr),
        "--val_frac",       str(args.val_frac),
        "--device",         "cuda:0",
        "--seed",           "42",
        "--num_workers",    "4",
        "--resume",         resume_ckpt,
    ]
    if args.use_amp:
        cmd.append("--use_amp")
    return cmd, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medsam_ckpt",  default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--phase_c_dir",  default="results/multimodal_v8")
    p.add_argument("--finetune_dir", default="results/multimodal_finetune",
                   help="Directory that already contains partial checkpoints.")
    p.add_argument("--devices",      nargs="+", type=int, default=[1, 2])
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
    args = p.parse_args()

    dev_map  = {i: args.devices[i] for i in range(len(args.devices))}
    logs_dir = os.path.join(args.finetune_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Confirm resume checkpoints before launching
    print("Checking resume checkpoints ...")
    for h, m, lg in CONFIGS:
        t = tag(h, m)
        ckpt = os.path.join(args.finetune_dir, t, "checkpoint_latest.pth")
        if os.path.exists(ckpt):
            import torch
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            epoch = ck.get("epoch", "?")
            bvl   = ck.get("best_val_loss", float("nan"))
            print(f"  {t:<12}  resumed from epoch {epoch:>2d}/19"
                  f"  best_val_loss={bvl:.4f}  -> cuda:{dev_map[lg]}")
        else:
            print(f"  {t:<12}  WARNING: checkpoint not found at {ckpt}")

    print()

    # Launch all 3 simultaneously
    procs = []
    for h, m, lg in CONFIGS:
        cmd, t = build_cmd(args, h, m)
        log_path = os.path.join(logs_dir, f"{t}_resume.log")
        physical_gpu = dev_map[lg]
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

    # Results summary
    print("\n" + "=" * 60)
    print("RESUME RESULTS")
    print(f"{'Config':<12} {'BF1_ft':>8} {'Dice_ft':>8} {'ΔBDF1':>8} {'Par%':>7}")
    print("-" * 50)

    failed = []
    for entry in procs:
        t_ = entry["tag"]
        h, m = entry["h"], entry["m"]
        rc   = entry["proc"].returncode
        jf   = os.path.join(args.finetune_dir, t_, "finetune_results.json")
        if rc != 0 or not os.path.exists(jf):
            print(f"  {t_:<12}  FAILED (rc={rc})")
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
