# -*- coding: utf-8 -*-
"""
Parallel sweep launcher for post-pruning fine-tuning  (Phase D).

Reads Phase C results (cascade_results_v8.json), distributes configs
across multiple GPUs, spawns one run_finetune_pruned worker per GPU slot,
and processes each slot's configs **sequentially** to avoid OOM.

Usage
-----
  # Hardcoded configs (12 combos, 4 GPUs × 3 sequential per GPU):
  python -m pilot_dual.run_finetune_sweep \
      --phase_c_json  results/pilot_cascade_v8_sweep/cascade_results_v8.json \
      --medsam_ckpt   work_dir/MedSAM/medsam_vit_b.pth \
      --output_dir    results/finetune_pruned_sweep \
      --devices       0 1 2 3 \
      --fixed_configs 0.4,0.3 0.8,0.3 0.6,0.5 0.5,0.7 0.7,0.7 \
                      0.3,0.8 0.3,0.85 0.3,0.9 0.3,0.95 \
                      0.7,0.85 0.7,0.9 0.7,0.95 \
      --num_epochs 50 \
      --batch_size 4 \
      --lr 1e-4 \
      --use_amp
"""

import argparse
import json
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_configs(phase_c_json, fixed_configs=None,
                 phases=("cascade",), min_head_sp=0.0, min_mlp_sp=0.0, top_k=0):
    """
    Build config list.

    When fixed_configs is given (list of (head_sp, mlp_sp) tuples), those
    pairs are used directly; the Phase-C JSON is still opened to read
    baseline_bf1 and to look up Phase-C BF1 for each pair (if present).

    Otherwise, all cascade_results rows are used subject to the filters.
    """
    with open(phase_c_json) as f:
        data = json.load(f)

    baseline_bf1 = data.get("baseline", {}).get("mean_boundary_f1")
    rows_all = data["cascade_results"]

    if fixed_configs:
        # Build minimal row dicts for the requested pairs.
        # Try to look up Phase-C BF1 from the existing rows.
        lookup = {}
        for r in rows_all:
            key = (round(r.get("head_sp_target", 0), 4),
                   round(r.get("mlp_sp_target",  0), 4))
            if key not in lookup or (
                r["macro"]["mean_boundary_f1"] >
                lookup[key]["macro"]["mean_boundary_f1"]
            ):
                lookup[key] = r

        configs = []
        for (h, m) in fixed_configs:
            key = (round(h, 4), round(m, 4))
            if key in lookup:
                configs.append(lookup[key])
            else:
                # Phase-C row not found → create a stub (BF1 = N/A)
                configs.append({
                    "phase":           "cascade",
                    "head_sp_target":  h,
                    "mlp_sp_target":   m,
                    "macro":           {"mean_boundary_f1": float("nan"),
                                        "mean_dice":        float("nan"),
                                        "mean_hd95":        float("nan"),
                                        "mean_iou":         float("nan")},
                })
                print(f"  WARNING: (h={h}, m={m}) not found in Phase-C results; "
                      "will proceed without Phase-C reference BF1.")
    else:
        rows = [r for r in rows_all if r.get("phase") in phases]
        rows = [r for r in rows if r.get("head_sp_target", 0) >= min_head_sp - 1e-6]
        rows = [r for r in rows if r.get("mlp_sp_target",  0) >= min_mlp_sp - 1e-6]

        seen = {}
        for r in rows:
            key = (round(r.get("head_sp_target", 0), 4),
                   round(r.get("mlp_sp_target",  0), 4))
            bf1 = r["macro"]["mean_boundary_f1"]
            if key not in seen or bf1 > seen[key]["macro"]["mean_boundary_f1"]:
                seen[key] = r
        rows = list(seen.values())
        rows.sort(key=lambda r: (r.get("head_sp_target", 0),
                                  r.get("mlp_sp_target",  0)))
        if top_k > 0:
            rows = sorted(rows, key=lambda r: r["macro"]["mean_boundary_f1"],
                          reverse=True)[:top_k]
            rows.sort(key=lambda r: (r.get("head_sp_target", 0),
                                      r.get("mlp_sp_target",  0)))
        configs = rows

    print(f"\n[Launcher] {len(configs)} configs to fine-tune")
    print(f"  baseline macro BF1 = "
          f"{f'{baseline_bf1:.4f}' if baseline_bf1 else 'N/A'}")
    for r in configs:
        bf1 = r["macro"]["mean_boundary_f1"]
        bf1_s = f"{bf1:.4f}" if bf1 == bf1 else "N/A"   # nan check
        print(f"  h={r.get('head_sp_target',0)*100:.0f}%  "
              f"m={r.get('mlp_sp_target',0)*100:.0f}%  "
              f"Phase-C BF1={bf1_s}")

    return configs, baseline_bf1


# ---------------------------------------------------------------------------
# Worker assignment  (round-robin over GPU slots)
# ---------------------------------------------------------------------------

def assign_workers(configs, devices, chains_per_gpu):
    """
    Assign configs to (device, chain) slots round-robin.
    Configs in the same slot are executed **sequentially** on that GPU.
    """
    total_slots = len(devices) * chains_per_gpu
    slots = [[] for _ in range(total_slots)]
    for i, cfg in enumerate(configs):
        slots[i % total_slots].append(cfg)

    workers = []
    for slot_idx, cfgs in enumerate(slots):
        if not cfgs:
            continue
        workers.append({
            "worker_idx": len(workers),
            "device_id":  devices[slot_idx // chains_per_gpu],
            "configs":    cfgs,
        })
    return workers


# ---------------------------------------------------------------------------
# Command builder  (forwards training flags to run_finetune_pruned)
# ---------------------------------------------------------------------------

def build_base_cmd(args):
    cmd = [sys.executable, "-m", "pilot_dual.run_finetune_pruned"]

    scalars = [
        ("--phase_c_json",             args.phase_c_json),
        ("--medsam_ckpt",              args.medsam_ckpt),
        ("--output_dir",               args.output_dir),
        ("--val_frac",                 args.val_frac),
        ("--num_epochs",               args.num_epochs),
        ("--batch_size",               args.batch_size),
        ("--lr",                       args.lr),
        ("--weight_decay",             args.weight_decay),
        ("--grad_clip",                args.grad_clip),
        ("--patience",                 args.patience),
        ("--eval_every",               args.eval_every),
        ("--seed",                     args.seed),
        ("--num_workers",              args.num_workers),
        ("--nonuniform_min_keep_head", args.nonuniform_min_keep_head),
        ("--nonuniform_min_frac_mlp",  args.nonuniform_min_frac_mlp),
    ]
    for flag, val in scalars:
        cmd.extend([flag, str(val)])

    lists = [
        ("--data_roots",       args.data_roots),
        ("--dataset_names",    args.dataset_names),
        ("--protected_blocks", args.protected_blocks),
    ]
    for flag, vals in lists:
        cmd.append(flag)
        cmd.extend(str(v) for v in vals)

    if args.use_amp:
        cmd.append("--use_amp")

    return cmd


# ---------------------------------------------------------------------------
# Phase D: 4 GPU slots run in parallel; each slot's configs run sequentially
# ---------------------------------------------------------------------------

def run_phase_d(args, base_cmd, workers):
    """
    Spawn all configs simultaneously — one subprocess per config.
    Each config gets its own CUDA_VISIBLE_DEVICES so multiple configs
    can share a GPU at the same time.
    Returns list of failed config_tags.
    """
    all_procs = []
    n_total   = sum(len(w["configs"]) for w in workers)
    print(f"\n[Launcher] Phase D — spawning {n_total} workers across "
          f"{len(workers)} slots")

    for w in workers:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(w["device_id"])

        for cfg in w["configs"]:
            h   = cfg.get("head_sp_target", 0.0)
            m   = cfg.get("mlp_sp_target",  0.0)
            tag = f"h{int(h * 100)}_m{int(m * 100)}"

            cmd = base_cmd + [
                "--device",        "cuda:0",
                "--head_sparsity", str(h),
                "--mlp_sparsity",  str(m),
                "--worker_idx",    str(w["worker_idx"]),
            ]
            log_path = os.path.join(args.output_dir, f"finetune_{tag}.log")
            logf     = open(log_path, "w")
            proc     = subprocess.Popen(
                cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
            all_procs.append({
                "config_tag": tag,
                "device_id":  w["device_id"],
                "proc":       proc,
                "logf":       logf,
                "log_path":   log_path,
            })
            print(f"  {tag:<14}  cuda:{w['device_id']}  "
                  f"pid={proc.pid}  log={log_path}")

    # Poll until all processes finish
    pending = set(range(len(all_procs)))
    t_start = time.time()
    while pending:
        time.sleep(30)
        for i in list(pending):
            ret = all_procs[i]["proc"].poll()
            if ret is not None:
                all_procs[i]["logf"].close()
                status  = "OK" if ret == 0 else f"FAIL({ret})"
                elapsed = (time.time() - t_start) / 60.0
                print(f"  [{elapsed:7.1f}min]  {all_procs[i]['config_tag']:<14} {status}")
                pending.discard(i)

    failed = [all_procs[i]["config_tag"]
              for i in range(len(all_procs))
              if all_procs[i]["proc"].returncode != 0]
    return failed


# ---------------------------------------------------------------------------
# Merge per-config JSONs
# ---------------------------------------------------------------------------

def merge_results(output_dir, configs, baseline_bf1):
    merged = []
    for cfg in configs:
        h   = cfg.get("head_sp_target", 0.0)
        m   = cfg.get("mlp_sp_target",  0.0)
        tag = f"h{int(h * 100)}_m{int(m * 100)}"
        p   = os.path.join(output_dir, tag, "finetune_results.json")
        if os.path.exists(p):
            with open(p) as f:
                merged.append(json.load(f))
        else:
            print(f"  WARNING: {p} not found (worker may have failed).")

    out = {"baseline_bf1": baseline_bf1, "n_configs": len(merged), "configs": merged}
    out_path = os.path.join(output_dir, "finetune_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Launcher] Merged {len(merged)} configs → {out_path}")
    return merged


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(merged, baseline_bf1):
    if not merged:
        return

    print("\n" + "=" * 115)
    print(f"FINE-TUNING SWEEP SUMMARY   "
          f"(baseline macro BF1 = "
          f"{f'{baseline_bf1:.4f}' if baseline_bf1 else 'N/A'})")
    print("=" * 115)
    header = (f"  {'Config':<14}  {'Phase-C BF1':>11}  {'FT BF1':>8}  "
              f"{'Δ BF1':>7}  {'FT Dice':>8}  {'FT HD95':>8}  "
              f"{'Par↓%':>6}  {'FL↓%':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in sorted(merged, key=lambda x: (x.get("head_sp_target", 0),
                                            x.get("mlp_sp_target",  0))):
        tag    = r.get("config_tag", "?")
        pc_bf1 = r.get("phase_c_metrics", {}).get("mean_boundary_f1", float("nan"))
        fm     = r.get("final_eval", {}).get("macro", {})
        ft_bf1 = fm.get("mean_boundary_f1", float("nan"))
        delta  = ft_bf1 - pc_bf1 if (ft_bf1 == ft_bf1 and pc_bf1 == pc_bf1) else float("nan")
        st     = r.get("pruning_stats", {})
        pc_s   = f"{pc_bf1:.4f}" if pc_bf1 == pc_bf1 else "   N/A"
        ft_s   = f"{ft_bf1:.4f}" if ft_bf1 == ft_bf1 else "   N/A"
        dl_s   = f"{delta:+.4f}" if delta == delta else "   N/A"
        print(f"  {tag:<14}  {pc_s:>11}  {ft_s:>8}  "
              f"{dl_s:>7}  {fm.get('mean_dice',0):>8.4f}  "
              f"{fm.get('mean_hd95',0):>8.2f}  "
              f"{st.get('param_reduction_pct',0):>6.1f}  "
              f"{st.get('flop_reduction_pct',0):>6.1f}")

    best = max(merged, key=lambda r: r.get("final_eval", {})
               .get("macro", {}).get("mean_boundary_f1", -1))
    bfm  = best.get("final_eval", {}).get("macro", {})
    print(f"\n  Best fine-tuned macro BF1 : {bfm.get('mean_boundary_f1',0):.4f}  "
          f"({best.get('config_tag','?')})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase D — parallel fine-tuning sweep launcher")

    # --- Primary input ---
    p.add_argument("--phase_c_json", required=True,
                   help="Path to cascade_results_v8.json from Phase C.")
    p.add_argument("--output_dir",  default="results/finetune_pruned_sweep",
                   help="Parent output dir; per-config subdirs created inside.")
    p.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")

    # --- Hardcoded config list (bypasses Phase-C filtering) ---
    p.add_argument("--fixed_configs", nargs="+", default=None,
                   metavar="H,M",
                   help="Explicit list of (head_sp,mlp_sp) pairs, e.g. "
                        "'0.4,0.3 0.7,0.7 0.3,0.95'.  "
                        "When provided, --phases/--min_*/--top_k are ignored.")

    # --- Config filtering (used only when --fixed_configs is NOT given) ---
    p.add_argument("--phases", nargs="+", default=["cascade"],
                   choices=["cascade", "head_only"])
    p.add_argument("--min_head_sp",   type=float, default=0.0)
    p.add_argument("--min_mlp_sp",    type=float, default=0.0)
    p.add_argument("--top_k_configs", type=int,   default=0)

    # --- Parallelism ---
    p.add_argument("--devices",        nargs="+", type=int, default=[0, 1, 2, 3],
                   help="Physical GPU IDs to use.")
    p.add_argument("--chains_per_gpu", type=int, default=3,
                   help="Concurrent workers per GPU.")

    # --- Datasets (forwarded) ---
    p.add_argument("--data_roots", nargs="+",
                   default=["asserts/kvasir-seg/Kvasir-SEG",
                            "asserts/CVC-ColonDB",
                            "asserts/CVC-ClinicDB"])
    p.add_argument("--dataset_names", nargs="+",
                   default=["Kvasir", "ColonDB", "ClinicDB"])
    p.add_argument("--val_frac", type=float, default=0.2)

    # --- Pruning mask params (forwarded) ---
    p.add_argument("--protected_blocks",         type=int, nargs="*", default=[10, 11])
    p.add_argument("--nonuniform_min_keep_head", type=int,   default=1)
    p.add_argument("--nonuniform_min_frac_mlp",  type=float, default=0.05)

    # --- Training hyper-parameters (forwarded) ---
    p.add_argument("--num_epochs",   type=int,   default=20)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--use_amp",      action="store_true")
    p.add_argument("--patience",     type=int,   default=5)
    p.add_argument("--eval_every",   type=int,   default=1)

    # --- Misc (forwarded) ---
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Parse --fixed_configs "0.4,0.3" → [(0.4, 0.3), ...]
    fixed = None
    if args.fixed_configs:
        fixed = []
        for s in args.fixed_configs:
            h, m = s.split(",")
            fixed.append((float(h), float(m)))

    print("=" * 80)
    print("FINE-TUNING SWEEP LAUNCHER — Phase D")
    print(f"  phase_c_json   : {args.phase_c_json}")
    print(f"  output_dir     : {args.output_dir}")
    print(f"  devices        : {args.devices}  chains/gpu={args.chains_per_gpu}")
    if fixed:
        print(f"  fixed_configs  : {len(fixed)} pairs  (filter flags ignored)")
    else:
        print(f"  phases         : {args.phases}")
        print(f"  min_head_sp    : {args.min_head_sp}  min_mlp_sp={args.min_mlp_sp}")
        print(f"  top_k_configs  : {args.top_k_configs or 'all'}")
    print(f"  num_epochs     : {args.num_epochs}  batch={args.batch_size}  lr={args.lr}")
    print(f"  AMP            : {args.use_amp}")
    print("=" * 80)

    configs, baseline_bf1 = load_configs(
        args.phase_c_json,
        fixed_configs=fixed,
        phases=tuple(args.phases),
        min_head_sp=args.min_head_sp,
        min_mlp_sp=args.min_mlp_sp,
        top_k=args.top_k_configs,
    )
    if not configs:
        print("[Launcher] No configs matched. Exiting.")
        return

    workers  = assign_workers(configs, args.devices, args.chains_per_gpu)
    base_cmd = build_base_cmd(args)

    t_start = time.time()
    failed  = run_phase_d(args, base_cmd, workers)

    if failed:
        print(f"\n[Launcher] Failed configs: {failed}")
        print(f"  Check logs in: {args.output_dir}/")

    merged = merge_results(args.output_dir, configs, baseline_bf1)

    total_min = (time.time() - t_start) / 60.0
    print(f"\n[Launcher] Total wall time: {total_min:.1f} min")

    print_summary(merged, baseline_bf1)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
