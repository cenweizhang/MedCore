# -*- coding: utf-8 -*-
"""
Multi-modal pruning sweep (BUSI + ClinicDB + ISIC2018).
5 configs assigned to 3 GPUs; all 5 processes launch simultaneously.

GPU 0: h40_m30, h50_m70  (2 concurrent processes)
GPU 1: h70_m70, h70_m95  (2 concurrent processes)
GPU 2: h80_m95            (1 process)

Usage:
    python -m pilot_dual.run_multimodal_sweep \
        --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
        --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \
        --output_dir  results/multimodal_v8 \
        --devices 0 1 2
"""

import argparse, json, os, subprocess, sys, time

# (head_sparsity, mlp_sparsity, gpu_device)
CONFIGS = [
    (0.4, 0.3,  0),
    (0.5, 0.7,  0),
    (0.7, 0.7,  1),
    (0.7, 0.95, 1),
    (0.8, 0.95, 2),
]


def build_cmd(args, h, m):
    tag = f"h{int(h*100)}_m{int(m*100)}"
    out_dir = os.path.join(args.output_dir, tag)
    cmd = [
        sys.executable, "-m", "pilot_dual.run_cascade_v8",
        "--medsam_ckpt",    args.medsam_ckpt,
        "--sam_ckpt",       args.sam_ckpt,
        "--output_dir",     out_dir,
        "--device",         "cuda:0",          # remapped via CUDA_VISIBLE_DEVICES
        "--phase",          "all",
        "--head_sparsities", str(h),
        "--mlp_sparsities",  str(m),
        "--data_roots",     ] + args.data_roots + [
        "--dataset_names",  ] + args.dataset_names + [
        "--cal_sizes",      ] + [str(c) for c in args.cal_sizes] + [
        "--protected_blocks", "10", "11",
        "--recovery_steps",   str(args.recovery_steps),
        "--recovery_lr",      str(args.recovery_lr),
        "--seed",             "42",
        "--num_workers",      "4",
    ]
    return cmd, out_dir, tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medsam_ckpt",  default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--sam_ckpt",     default="work_dir/SAM/sam_vit_b_01ec64.pth")
    p.add_argument("--output_dir",   default="results/multimodal_v8")
    p.add_argument("--devices",      nargs="+", type=int, default=[0, 1, 2],
                   help="Override GPU ids: positional mapping to CONFIGS order")
    p.add_argument("--data_roots",   nargs="+",
                   default=["asserts/BUSI",
                            "asserts/CVC-ClinicDB",
                            "asserts/ISIC2018"])
    p.add_argument("--dataset_names", nargs="+",
                   default=["BUSI", "ClinicDB", "ISIC2018"])
    p.add_argument("--cal_sizes",    nargs="+", type=int, default=[128, 128, 128])
    p.add_argument("--recovery_steps", type=int,   default=100)
    p.add_argument("--recovery_lr",    type=float, default=1e-5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Allow --devices to remap the 3 logical GPUs [0,1,2] to physical ids
    dev_map = {i: args.devices[i] for i in range(len(args.devices))}

    print("=" * 65)
    print("MULTI-MODAL SWEEP — v8 cascade pruning  (all 5 jobs in parallel)")
    print(f"  Datasets : {args.dataset_names}")
    print(f"  Cal sizes: {args.cal_sizes}")
    for h, m, lg in CONFIGS:
        tag = f"h{int(h*100)}_m{int(m*100)}"
        print(f"  {tag:<12}  cuda:{dev_map.get(lg, lg)}")
    print("=" * 65)

    # Spawn all 5 processes at once
    procs = []
    for h, m, logical_gpu in CONFIGS:
        cmd, out_dir, tag = build_cmd(args, h, m)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"{tag}.log")
        physical_gpu = dev_map.get(logical_gpu, logical_gpu)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        logf = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append({"tag": tag, "proc": proc, "logf": logf, "log": log_path})
        print(f"  STARTED  {tag:<12}  cuda:{physical_gpu}  pid={proc.pid}")

    # Poll until all done
    pending = set(range(len(procs)))
    t0 = time.time()
    while pending:
        time.sleep(60)
        for i in list(pending):
            ret = procs[i]["proc"].poll()
            if ret is not None:
                procs[i]["logf"].close()
                elapsed = (time.time() - t0) / 60
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"  [{elapsed:6.1f}min]  {status}  {procs[i]['tag']}")
                pending.discard(i)

    # Summary
    print("\n" + "=" * 65)
    failed = [p["tag"] for p in procs if p["proc"].returncode != 0]
    if failed:
        print(f"FAILED: {failed}")
    else:
        print("All configs completed.")

    print("\nQuick results (macro BF1 from cascade_results_v8.json):")
    for entry in procs:
        tag = entry["tag"]
        jf = os.path.join(args.output_dir, tag, "cascade_results_v8.json")
        if not os.path.exists(jf):
            print(f"  {tag:<15}  (no JSON)")
            continue
        data = json.load(open(jf))
        for r in data.get("cascade_results", []):
            if r.get("phase") == "cascade":
                bf1  = r["macro"].get("mean_boundary_f1", float("nan"))
                dice = r["macro"].get("mean_dice", float("nan"))
                par  = r.get("param_reduction_pct", float("nan"))
                print(f"  {tag:<15}  BF1={bf1:.4f}  Dice={dice:.4f}  Par↓{par:.1f}%")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
