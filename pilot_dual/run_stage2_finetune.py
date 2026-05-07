# -*- coding: utf-8 -*-
"""
Full post-pruning fine-tuning for Stage-2 (extreme compression) configs.

Runs run_finetune_pruned on all 6 Stage-2 configs using the baked Stage-1
checkpoint as the base model.  Protected blocks are [0-9] so fine-tuning
hooks are only applied to blocks 10-11.

GPU assignment (2 processes per GPU):
  GPU 0: cfg_A_h60m70   cfg_B_h72m84      (first batch)
  GPU 3: cfg_C_h84m95   cfg_C_abl1_no_BF  (first batch)
  GPU 0: cfg_C_abl2_zero                  (second batch, serial)
  GPU 3: cfg_C_abl3_b0                    (second batch, serial)

Usage:
    python -m pilot_dual.run_stage2_finetune \
        --stage1_ckpt results/stage2/stage1_pruned_h70m95.pth \
        --stage2_dir  results/stage2_sweep \
        --output_dir  results/stage2_finetune \
        --devices 0 3
    python -m pilot_dual.run_stage2_finetune \
        --stage1_ckpt results/stage2/stage1_pruned_h70m95.pth \
        --devices 0 3
"""

import argparse, json, os, subprocess, sys, time

PROTECTED_BLOCKS = list(range(10))

# (name, head_sp, mlp_sp)
CONFIGS = [
    ("cfg_A_h60m70",     0.10,  0.117),
    ("cfg_B_h72m84",     0.12,  0.140),
    ("cfg_C_h84m95",     0.14,  0.158),
    ("cfg_C_abl1_no_BF", 0.14,  0.158),
    ("cfg_C_abl2_zero",  0.14,  0.158),
    ("cfg_C_abl3_b0",    0.14,  0.158),
]

# First batch: 4 simultaneous (2 per GPU)
BATCH1 = [
    ("cfg_A_h60m70",     0),   # GPU devices[0]
    ("cfg_B_h72m84",     0),   # GPU devices[0]
    ("cfg_C_h84m95",     1),   # GPU devices[1]
    ("cfg_C_abl1_no_BF", 1),   # GPU devices[1]
]

# Second batch: 2 serial (one per GPU)
BATCH2 = [
    ("cfg_C_abl2_zero",  0),   # GPU devices[0]
    ("cfg_C_abl3_b0",    1),   # GPU devices[1]
]


def get_cfg(name):
    return next(c for c in CONFIGS if c[0] == name)


def build_cmd(args, name, h, m):
    phase_c_json = os.path.join(args.stage2_dir, name, "cascade_results_v8.json")
    # Give each config its own parent dir to avoid collision when h/m targets are identical
    per_config_out = os.path.join(args.output_dir, name)
    cmd = [
        sys.executable, "-m", "pilot_dual.run_finetune_pruned",
        "--medsam_ckpt",    args.stage1_ckpt,
        "--phase_c_json",   phase_c_json,
        "--head_sparsity",  str(h),
        "--mlp_sparsity",   str(m),
        "--output_dir",     per_config_out,
        "--protected_blocks"] + [str(b) for b in PROTECTED_BLOCKS] + [
        "--data_roots",     ] + args.data_roots + [
        "--dataset_names",  ] + args.dataset_names + [
        "--num_epochs",     str(args.num_epochs),
        "--batch_size",     str(args.batch_size),
        "--lr",             str(args.lr),
        "--val_frac",       str(args.val_frac),
        "--device",         "cuda:0",
        "--seed",           "42",
        "--num_workers",    "4",
    ]
    if args.use_amp:
        cmd.append("--use_amp")
    return cmd


def launch(cmd, name, logs_dir, physical_gpu, env_base):
    log_path = os.path.join(logs_dir, f"{name}.log")
    env = env_base.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    logf = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    print(f"  STARTED  {name:<22}  cuda:{physical_gpu}  pid={proc.pid}")
    return proc, logf, log_path


def wait_all(procs):
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
                print(f"  [{elapsed:6.1f}min]  {status}  {procs[i]['name']}")
                pending.discard(i)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", default="results/stage2/stage1_pruned_h70m95.pth",
                   help="Baked Stage-1 checkpoint (output of bake_stage1_masks.py)")
    p.add_argument("--stage2_dir",  default="results/stage2_sweep",
                   help="Directory with Stage-2 Phase-C results")
    p.add_argument("--output_dir",  default="results/stage2_finetune")
    p.add_argument("--devices",     nargs="+", type=int, default=[0, 3])
    p.add_argument("--data_roots",  nargs="+",
                   default=["asserts/kvasir-seg/Kvasir-SEG",
                            "asserts/CVC-ColonDB",
                            "asserts/CVC-ClinicDB"])
    p.add_argument("--dataset_names", nargs="+",
                   default=["Kvasir", "ColonDB", "ClinicDB"])
    p.add_argument("--num_epochs",  type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--val_frac",    type=float, default=0.2)
    p.add_argument("--use_amp",     action="store_true", default=True)
    args = p.parse_args()

    if not os.path.exists(args.stage1_ckpt):
        print(f"ERROR: baked Stage-1 checkpoint not found: {args.stage1_ckpt}")
        print("Run first:  python -m pilot_dual.bake_stage1_masks")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    dev = {i: args.devices[i] for i in range(len(args.devices))}
    env_base = os.environ.copy()

    print("=" * 65)
    print("STAGE-2 FINE-TUNING SWEEP")
    print(f"  Stage-1 ckpt : {args.stage1_ckpt}")
    print(f"  Stage-2 dir  : {args.stage2_dir}")
    print(f"  Protected    : blocks {PROTECTED_BLOCKS}")
    print(f"  Epochs       : {args.num_epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  Batch 1 (parallel):  GPU{dev[0]}: cfg_A + cfg_B  |  GPU{dev[1]}: cfg_C + abl1")
    print(f"  Batch 2 (parallel):  GPU{dev[0]}: abl2           |  GPU{dev[1]}: abl3")
    print("=" * 65)

    t_total = time.time()
    all_procs = []

    # ------------------------------------------------------------------
    # Batch 1: 4 processes simultaneously
    # ------------------------------------------------------------------
    print("\n[Batch 1] Launching 4 workers ...")
    batch1_procs = []
    for name, lg in BATCH1:
        _, h, m = get_cfg(name)
        cmd = build_cmd(args, name, h, m)
        proc, logf, log_path = launch(cmd, name, logs_dir, dev[lg], env_base)
        entry = {"name": name, "proc": proc, "logf": logf, "h": h, "m": m}
        batch1_procs.append(entry)
        all_procs.append(entry)

    wait_all(batch1_procs)

    # ------------------------------------------------------------------
    # Batch 2: 2 processes simultaneously (reuse freed GPU slots)
    # ------------------------------------------------------------------
    print("\n[Batch 2] Launching 2 workers ...")
    batch2_procs = []
    for name, lg in BATCH2:
        _, h, m = get_cfg(name)
        cmd = build_cmd(args, name, h, m)
        proc, logf, log_path = launch(cmd, name, logs_dir, dev[lg], env_base)
        entry = {"name": name, "proc": proc, "logf": logf, "h": h, "m": m}
        batch2_procs.append(entry)
        all_procs.append(entry)

    wait_all(batch2_procs)

    # ------------------------------------------------------------------
    # Results summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STAGE-2 FINE-TUNING RESULTS")
    print(f"{'Config':<22} {'BF1_ft':>8} {'Dice_ft':>8} {'IoU_ft':>7} {'HD95_ft':>8} {'ΔBDF1':>8} {'Par%':>7}")
    print("-" * 75)

    failed = []
    for entry in all_procs:
        name = entry["name"]
        h, m  = entry["h"], entry["m"]
        rc    = entry["proc"].returncode
        # run_finetune_pruned creates <output_dir>/<name>/h{H}_m{M}/finetune_results.json
        tag   = f"h{int(h*100)}_m{int(m*100)}"
        jf    = os.path.join(args.output_dir, name, tag, "finetune_results.json")
        if rc != 0 or not os.path.exists(jf):
            print(f"  {name:<22}  FAILED")
            failed.append(name)
            continue
        data    = json.load(open(jf))
        fm      = data["final_eval"]["macro"]
        pc_bf1  = data["phase_c_metrics"]["mean_boundary_f1"]
        bf1_ft  = fm["mean_boundary_f1"]
        dice_ft = fm["mean_dice"]
        iou_ft  = fm.get("mean_iou", float("nan"))
        hd95_ft = fm["mean_hd95"]
        par     = data["pruning_stats"].get("param_reduction_pct", float("nan"))
        delta   = bf1_ft - pc_bf1
        print(f"  {name:<22} {bf1_ft:>8.4f} {dice_ft:>8.4f} {iou_ft:>7.4f} {hd95_ft:>8.2f} {delta:>+8.4f} {par:>6.1f}%")

    print(f"\nTotal elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"Logs: {logs_dir}/")

    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
