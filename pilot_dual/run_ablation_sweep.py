# -*- coding: utf-8 -*-
"""
Ablation sweep for v8 cascade pruning.  4 ablations × 4 configs, 4 GPUs in parallel.

Ablations:
  abl1_no_boundary_fisher  --boundary_fisher_weight 0   (Phase B+C, ~full run)
  abl2_zero_only_fisher    --no_adaptive_alpha           (Phase C only, reuse cache)
  abl3_beta0               --dist_beta 0                (Phase C only, reuse cache)
  abl4_uniform             --uniform_allocation          (Phase C only, reuse cache)

Configs: h40_m30, h50_m70, h70_m70, h70_m95

Usage:
  python -m pilot_dual.run_ablation_sweep \
      --medsam_ckpt work_dir/MedSAM/medsam_vit_b.pth \
      --sam_ckpt    work_dir/SAM/sam_vit_b_01ec64.pth \
      --phase_b_cache results/pilot_cascade_v8_sweep/_phase_b_cache \
      --output_dir    results/ablation_v8 \
      --devices 0 1 2 3
"""

import argparse, json, os, subprocess, sys, time

CONFIGS = {
    "head_sparsities": [0.4, 0.5, 0.7, 0.7],
    "mlp_sparsities":  [0.3, 0.7, 0.7, 0.95],
}

ABLATIONS = {
    "abl1_no_boundary_fisher": {
        "phase": "all",          # must redo Fisher
        "extra": ["--boundary_fisher_weight", "0"],
    },
    "abl2_zero_only_fisher": {
        "phase": "c_only",
        "extra": ["--no_adaptive_alpha", "--phase1_alpha", "1.0"],
    },
    "abl3_beta0": {
        "phase": "c_only",
        "extra": ["--dist_beta", "0"],
    },
    "abl4_uniform": {
        "phase": "c_only",
        "extra": ["--uniform_allocation"],
    },
}


def build_cmd(args, abl_name, abl_cfg, device_id):
    out_dir = os.path.join(args.output_dir, abl_name)
    cmd = [
        sys.executable, "-m", "pilot_dual.run_cascade_v8",
        "--medsam_ckpt",   args.medsam_ckpt,
        "--sam_ckpt",      args.sam_ckpt,
        "--output_dir",    out_dir,
        "--device",        "cuda:0",   # remapped via CUDA_VISIBLE_DEVICES
        "--phase",         abl_cfg["phase"],
        "--head_sparsities"] + [str(h) for h in CONFIGS["head_sparsities"]] + [
        "--mlp_sparsities"] + [str(m) for m in CONFIGS["mlp_sparsities"]] + [
        "--data_roots"]    + args.data_roots + [
        "--dataset_names"] + args.dataset_names + [
        "--cal_sizes"]     + [str(c) for c in args.cal_sizes] + [
        "--protected_blocks", "10", "11",
        "--recovery_steps",   str(args.recovery_steps),
        "--recovery_lr",      str(args.recovery_lr),
        "--seed",             "42",
        "--num_workers",      "4",
    ]

    # For c_only: point at the shared Phase B cache
    if abl_cfg["phase"] == "c_only":
        cmd += ["--cache_dir", args.phase_b_cache]

    cmd += abl_cfg["extra"]
    return cmd, out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--medsam_ckpt",  default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--sam_ckpt",     default="work_dir/SAM/sam_vit_b_01ec64.pth")
    p.add_argument("--phase_b_cache",
                   default="results/pilot_cascade_v8_sweep/_phase_b_cache",
                   help="Shared Phase B cache from the original v8 sweep run.")
    p.add_argument("--output_dir",   default="results/ablation_v8")
    p.add_argument("--devices",      nargs="+", type=int, default=[0, 1, 2, 3])
    p.add_argument("--data_roots",   nargs="+",
                   default=["asserts/kvasir-seg/Kvasir-SEG",
                            "asserts/CVC-ColonDB",
                            "asserts/CVC-ClinicDB"])
    p.add_argument("--dataset_names", nargs="+", default=["Kvasir", "ColonDB", "ClinicDB"])
    p.add_argument("--cal_sizes",    nargs="+", type=int, default=[128, 128, 128])
    p.add_argument("--recovery_steps", type=int,   default=100)
    p.add_argument("--recovery_lr",    type=float, default=1e-5)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    abl_names = list(ABLATIONS.keys())
    assert len(abl_names) == len(args.devices), \
        f"Need exactly {len(abl_names)} devices, got {args.devices}"

    print("=" * 70)
    print("ABLATION SWEEP — v8 cascade pruning")
    print(f"  Configs : {list(zip(CONFIGS['head_sparsities'], CONFIGS['mlp_sparsities']))}")
    print(f"  Ablations: {abl_names}")
    print(f"  Devices  : {args.devices}")
    print(f"  Phase B cache: {args.phase_b_cache}")
    print("=" * 70)

    procs = []
    for abl_name, device_id in zip(abl_names, args.devices):
        abl_cfg = ABLATIONS[abl_name]
        cmd, out_dir = build_cmd(args, abl_name, abl_cfg, device_id)
        os.makedirs(out_dir, exist_ok=True)

        log_path = os.path.join(args.output_dir, f"{abl_name}.log")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)

        logf = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append({"name": abl_name, "proc": proc, "logf": logf, "log": log_path})
        print(f"  STARTED  {abl_name:<30}  cuda:{device_id}  pid={proc.pid}")
        print(f"           phase={abl_cfg['phase']}  log={log_path}")

    # Poll
    pending = set(range(len(procs)))
    t0 = time.time()
    while pending:
        time.sleep(60)
        for i in list(pending):
            ret = procs[i]["proc"].poll()
            if ret is not None:
                procs[i]["logf"].close()
                status  = "OK" if ret == 0 else f"FAIL({ret})"
                elapsed = (time.time() - t0) / 60
                print(f"  [{elapsed:6.1f}min]  {status}  {procs[i]['name']}")
                pending.discard(i)

    # Summary
    print("\n" + "=" * 70)
    failed = [procs[i]["name"] for i in range(len(procs))
              if procs[i]["proc"].returncode != 0]
    if failed:
        print(f"FAILED: {failed}")
        print(f"Logs: {args.output_dir}/<name>.log")
    else:
        print("All ablations completed successfully.")

    # Print per-ablation macro BF1 for quick comparison
    print("\nQuick results (macro BF1 from cascade_results_v8.json):")
    for abl_name in abl_names:
        jf = os.path.join(args.output_dir, abl_name, "cascade_results_v8.json")
        if not os.path.exists(jf):
            print(f"  {abl_name:<30}  (no JSON)")
            continue
        data = json.load(open(jf))
        rows = data.get("cascade_results", [])
        print(f"  {abl_name}")
        for r in rows:
            if r.get("phase") == "cascade":
                h = r.get("head_sp_target", 0)
                m = r.get("mlp_sp_target",  0)
                bf1 = r["macro"].get("mean_boundary_f1", float("nan"))
                print(f"    h={h:.2f} m={m:.2f}  BF1={bf1:.4f}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
