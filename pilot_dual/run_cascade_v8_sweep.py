# -*- coding: utf-8 -*-
"""
Parallel sweep launcher for run_cascade_v8.

Pipeline
--------
  1. Phase B (serial, one GPU)
       Invokes `run_cascade_v8 --phase b_only` once to produce
       scores + baseline_eval + original_state in cache_dir.
  2. Phase C (parallel, multi-GPU × multi-chain)
       Splits head_sparsities across N devices × chains_per_gpu slots;
       each worker runs `run_cascade_v8 --phase c_only` with its subset
       of head_sparsities and CUDA_VISIBLE_DEVICES={device_id}.
       Each worker writes cascade_results_v8_wXX.json.
  3. Merge per-worker JSONs → cascade_results_v8.json.
  4. Best-config summary printed.

Intra-GPU concurrency
---------------------
When `chains_per_gpu > 1`, each GPU hosts multiple subprocesses sharing
the same device via separate CUDA contexts.  With ViT-B ViT-B peak
memory ~20 GB per chain, 140 GB VRAM comfortably holds 4+ concurrent
chains per GPU.

Usage
-----
    python -m pilot_dual.run_cascade_v8_sweep \
        --output_dir       results/pilot_cascade_v8_sweep \
        --head_sparsities  0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
        --mlp_sparsities   0.3 0.5 0.7 0.8 0.85 0.9 0.95 \
        --devices          0 1 2 3 \
        --chains_per_gpu   2 \
        --parallel_fisher \
        --recovery_steps   100 \
        --eval_batch_size 16


The launcher forwards every other CLI flag to run_cascade_v8 unchanged.

    python -m pilot_dual.run_cascade_v8_sweep \                                               
      --output_dir      results/pilot_cascade_v8_sweep \                                    
      --head_sparsities 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \                                       
      --mlp_sparsities  0.3 0.5 0.7 0.8 0.85 0.9 0.95 \                                     
      --devices         0 1 2 3 \                                                           
      --chains_per_gpu  2 \                                                        
      --parallel_fisher \                                                                   
      --recovery_steps  100 \                                                               
      --eval_batch_size 16
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Parallel sweep launcher for run_cascade_v8")

    # --- Paths (forwarded) ---
    p.add_argument("--medsam_ckpt", default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--sam_ckpt",    default="work_dir/SAM/sam_vit_b_01ec64.pth")
    p.add_argument("--output_dir",  default="results/pilot_cascade_v8_sweep")
    p.add_argument("--cache_dir",   default=None,
                   help="Phase B cache location; default is output_dir/_phase_b_cache.")

    # --- Data (forwarded) ---
    p.add_argument("--data_roots", nargs="+",
                   default=["asserts/kvasir-seg/Kvasir-SEG",
                            "asserts/CVC-ColonDB",
                            "asserts/CVC-ClinicDB"])
    p.add_argument("--dataset_names", nargs="+",
                   default=["Kvasir", "ColonDB", "ClinicDB"])
    p.add_argument("--cal_sizes", nargs="+", type=int, default=[64, 38, 26])
    p.add_argument("--pi_r", nargs="+", type=float, default=[0.5, 0.3, 0.2])

    # --- Sweep grid (forwarded) ---
    p.add_argument("--head_sparsities", nargs="+", type=float,
                   default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--mlp_sparsities",  nargs="+", type=float,
                   default=[0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95])
    p.add_argument("--mlp_alpha_values", nargs="+", type=float, default=[1.0])

    # --- Compute options (forwarded) ---
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_workers",     type=int, default=4)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--recovery_steps",  type=int, default=100)
    p.add_argument("--recovery_lr",     type=float, default=1e-5)
    p.add_argument("--boundary_fisher_weight", type=float, default=3.0)
    p.add_argument("--feat_distill_weight",    type=float, default=0.5)
    p.add_argument("--boundary_loss_weight",   type=float, default=1.0)
    p.add_argument("--logit_distill_weight",   type=float, default=2.0)
    p.add_argument("--freq_pred_loss_weight",  type=float, default=0.5)
    p.add_argument("--dist_beta",       type=float, default=0.3)
    p.add_argument("--phase1_alpha",    type=float, default=1.0)
    p.add_argument("--tau",             type=float, default=0.0)
    p.add_argument("--exact_validate_topk",       type=int, default=0)
    p.add_argument("--n_visualize_per_dataset",   type=int, default=3)

    # --- Boolean forwards ---
    p.add_argument("--no_cross_fisher",   action="store_true")
    p.add_argument("--no_adaptive_alpha", action="store_true")
    p.add_argument("--recompute_fisher_between_stages", action="store_true")
    p.add_argument("--oneshot_mlp",       action="store_true")
    p.add_argument("--no_freq_sampling",  action="store_true")

    # --- Protection ---
    p.add_argument("--protected_blocks",         type=int, nargs="*", default=[10, 11])
    p.add_argument("--nonuniform_min_keep_head", type=int,   default=1)
    p.add_argument("--nonuniform_min_frac_mlp",  type=float, default=0.05)

    # --- Launcher-specific ---
    p.add_argument("--devices", nargs="+", type=int, default=[0, 1, 2, 3],
                   help="Physical GPU IDs to use. Each worker sets "
                        "CUDA_VISIBLE_DEVICES to one of these.")
    p.add_argument("--chains_per_gpu", type=int, default=2,
                   help="How many worker processes share one GPU "
                        "(each gets its own CUDA context).")
    p.add_argument("--phase_b_device", type=int, default=0,
                   help="Physical GPU ID used for the one-time Phase B "
                        "(Fisher + scoring + baseline eval).")
    p.add_argument("--skip_phase_b", action="store_true",
                   help="Assume Phase B cache already exists and skip it.")
    p.add_argument("--parallel_fisher", action="store_true",
                   help="Fan out Fisher computation across 4 GPUs. "
                        "Requires --devices to cover at least 4 GPU IDs.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Command assembly  (forwards scalar/list/bool args to run_cascade_v8)
# ---------------------------------------------------------------------------

def build_base_v8_cmd(args):
    """Construct the list of CLI tokens common to every v8 invocation."""
    cmd = [sys.executable, "-m", "pilot_dual.run_cascade_v8"]

    # scalar pass-through
    scalars = [
        ("--medsam_ckpt", args.medsam_ckpt),
        ("--sam_ckpt",    args.sam_ckpt),
        ("--output_dir",  args.output_dir),
        ("--eval_batch_size", args.eval_batch_size),
        ("--num_workers",     args.num_workers),
        ("--seed",            args.seed),
        ("--recovery_steps",  args.recovery_steps),
        ("--recovery_lr",     args.recovery_lr),
        ("--boundary_fisher_weight", args.boundary_fisher_weight),
        ("--feat_distill_weight",    args.feat_distill_weight),
        ("--boundary_loss_weight",   args.boundary_loss_weight),
        ("--logit_distill_weight",   args.logit_distill_weight),
        ("--freq_pred_loss_weight",  args.freq_pred_loss_weight),
        ("--dist_beta",              args.dist_beta),
        ("--phase1_alpha",           args.phase1_alpha),
        ("--tau",                    args.tau),
        ("--exact_validate_topk",    args.exact_validate_topk),
        ("--n_visualize_per_dataset", args.n_visualize_per_dataset),
        ("--nonuniform_min_keep_head", args.nonuniform_min_keep_head),
        ("--nonuniform_min_frac_mlp",  args.nonuniform_min_frac_mlp),
    ]
    if args.cache_dir is not None:
        scalars.append(("--cache_dir", args.cache_dir))
    for flag, val in scalars:
        cmd.extend([flag, str(val)])

    # list pass-through
    lists = [
        ("--data_roots",      args.data_roots),
        ("--dataset_names",   args.dataset_names),
        ("--cal_sizes",       args.cal_sizes),
        ("--pi_r",            args.pi_r),
        ("--mlp_sparsities",  args.mlp_sparsities),
        ("--mlp_alpha_values", args.mlp_alpha_values),
        ("--protected_blocks", args.protected_blocks),
    ]
    for flag, vals in lists:
        cmd.append(flag)
        cmd.extend(str(v) for v in vals)

    # bool pass-through
    bools = [
        ("--no_cross_fisher",   args.no_cross_fisher),
        ("--no_adaptive_alpha", args.no_adaptive_alpha),
        ("--recompute_fisher_between_stages", args.recompute_fisher_between_stages),
        ("--oneshot_mlp",       args.oneshot_mlp),
        ("--no_freq_sampling",  args.no_freq_sampling),
    ]
    for flag, on in bools:
        if on:
            cmd.append(flag)

    return cmd


# ---------------------------------------------------------------------------
# Phase B (serial)
# ---------------------------------------------------------------------------

def run_phase_b(args, base_cmd):
    """Run `v8 --phase b_only` once on args.phase_b_device."""
    os.makedirs(args.output_dir, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.phase_b_device)

    cmd = base_cmd + [
        "--phase", "b_only",
        "--device", "cuda:0",           # post-remap, only one GPU visible
        "--head_sparsities", str(args.head_sparsities[0]),  # placeholder (unused)
    ]

    log_path = os.path.join(args.output_dir, "phase_b.log")
    print(f"\n[Launcher] Phase B on cuda:{args.phase_b_device}  "
          f"(log → {log_path})")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Phase B failed ({proc.returncode}); see {log_path}")
    print(f"[Launcher] Phase B done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Phase B — parallel Fisher fan-out (task 8)
# ---------------------------------------------------------------------------

def run_phase_b_parallel(args, base_cmd):
    """
    Parallel Phase B:
      1. Spawn 4 subprocesses (3 MedSAM F^M subsets + 1 SAM F^S) on 4 GPUs.
      2. Each subprocess runs v8 with --fisher_task and saves fisher dict to cache.
      3. After all done, run v8 --phase b_only --load_precomputed_fisher on one GPU
         to do scoring + baseline eval + cache save.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    assert len(args.devices) >= 4, \
        "--parallel_fisher needs at least 4 devices"
    assert len(args.data_roots) == 3, \
        "Current --parallel_fisher layout assumes 3 datasets + 1 SAM = 4 tasks"

    tasks = [
        ("medsam:0", args.devices[0], "fisher_m_0"),
        ("medsam:1", args.devices[1], "fisher_m_1"),
        ("medsam:2", args.devices[2], "fisher_m_2"),
        ("sam:combined", args.devices[3], "fisher_s"),
    ]

    print(f"\n[Launcher] Phase B parallel Fisher: {len(tasks)} tasks on "
          f"cuda:{[d for _,d,_ in tasks]}")
    procs = []
    for task_spec, dev_id, tag in tasks:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(dev_id)
        cmd = base_cmd + [
            "--fisher_task", task_spec,
            "--device", "cuda:0",   # post-remap
            "--head_sparsities", str(args.head_sparsities[0]),  # placeholder
            "--phase", "all",       # phase arg is irrelevant since --fisher_task exits early
        ]
        log_path = os.path.join(args.output_dir, f"phase_b_fisher_{tag}.log")
        logf = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append((task_spec, dev_id, tag, proc, logf, log_path))
        print(f"    task {task_spec:<20}  cuda:{dev_id}  pid={proc.pid}  log={log_path}")

    t0 = time.time()
    pending = set(range(len(procs)))
    while pending:
        time.sleep(10)
        for i in list(pending):
            task_spec, dev_id, tag, proc, logf, log_path = procs[i]
            ret = proc.poll()
            if ret is not None:
                logf.close()
                elapsed = (time.time() - t0) / 60.0
                status = "OK" if ret == 0 else f"FAIL({ret})"
                print(f"    [{elapsed:5.1f}min] {task_spec:<20} {status}")
                pending.discard(i)
                if ret != 0:
                    # Early kill others so launcher doesn't hang
                    for other in procs:
                        try:
                            other[3].terminate()
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"Fisher task {task_spec} failed; see {log_path}")

    print(f"[Launcher] Parallel Fisher done in {(time.time()-t0)/60:.1f} min")

    # Serial scoring + baseline eval phase (uses precomputed Fisher)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.phase_b_device)
    cmd = base_cmd + [
        "--phase", "b_only",
        "--load_precomputed_fisher",
        "--device", "cuda:0",
        "--head_sparsities", str(args.head_sparsities[0]),
    ]
    log_path = os.path.join(args.output_dir, "phase_b_score.log")
    print(f"[Launcher] Phase B scoring+baseline on cuda:{args.phase_b_device}  "
          f"(log → {log_path})")
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Phase B scoring failed ({proc.returncode}); see {log_path}")
    print(f"[Launcher] Phase B scoring done in {(time.time()-t0)/60:.1f} min")


# ---------------------------------------------------------------------------
# Phase C (parallel)
# ---------------------------------------------------------------------------

def assign_workers(head_sparsities, devices, chains_per_gpu):
    """
    Round-robin assign head_sp values across (len(devices) × chains_per_gpu) slots.
    Each slot becomes a worker; slots on the same GPU run as separate subprocesses
    with their own CUDA contexts.
    """
    total_slots = len(devices) * chains_per_gpu
    slots = [[] for _ in range(total_slots)]
    for i, h in enumerate(sorted(head_sparsities)):
        slots[i % total_slots].append(h)

    workers = []
    for slot_idx, hs in enumerate(slots):
        if not hs:
            continue
        dev = devices[slot_idx // chains_per_gpu]
        workers.append({
            "worker_idx":   len(workers),
            "device_id":    dev,
            "head_sp_list": hs,
        })
    return workers


def run_phase_c(args, base_cmd, workers):
    """Spawn all Phase C workers; poll until all complete."""
    procs = []
    print(f"\n[Launcher] Phase C: {len(workers)} workers")
    for w in workers:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(w["device_id"])

        cmd = base_cmd + [
            "--phase",       "c_only",
            "--device",      "cuda:0",                    # post-remap
            "--worker_idx",  str(w["worker_idx"]),
            "--head_sparsities",
        ] + [str(h) for h in w["head_sp_list"]]

        log_path = os.path.join(args.output_dir,
                                 f"phase_c_w{w['worker_idx']:02d}.log")
        logf = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        procs.append((w, proc, logf, log_path))
        print(f"    worker {w['worker_idx']:02d} | cuda:{w['device_id']} | "
              f"head_sp={w['head_sp_list']} | pid={proc.pid} | log={log_path}")

    # Poll loop so we report progress instead of blocking silently
    pending = set(range(len(procs)))
    t_start = time.time()
    while pending:
        time.sleep(15)
        for i in list(pending):
            w, proc, logf, log_path = procs[i]
            ret = proc.poll()
            if ret is not None:
                logf.close()
                status = "OK" if ret == 0 else f"FAIL({ret})"
                elapsed = (time.time() - t_start) / 60.0
                print(f"    [{elapsed:5.1f}min] worker {w['worker_idx']:02d} "
                      f"finished: {status}")
                pending.discard(i)

    failed = [procs[i][0]["worker_idx"] for i in range(len(procs))
              if procs[i][1].returncode != 0]
    return failed


# ---------------------------------------------------------------------------
# Merge per-worker outputs
# ---------------------------------------------------------------------------

def merge_results(args):
    paths = sorted(glob.glob(os.path.join(
        args.output_dir, "cascade_results_v8_w*.json")))
    if not paths:
        raise RuntimeError(
            f"No worker output files found in {args.output_dir}")

    merged = None
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if merged is None:
            merged = {
                "config":               d.get("config"),
                "baseline":             d.get("baseline"),
                "baseline_per_dataset": d.get("baseline_per_dataset"),
                "baseline_pool":        d.get("baseline_pool"),
                "head_score_summary":   d.get("head_score_summary"),
                "block_sensitivity":    d.get("block_sensitivity"),
                "alpha_per_block":      d.get("alpha_per_block"),
                "pi_r":                 d.get("pi_r"),
                "exact_validation":     d.get("exact_validation"),
                "cascade_results":      list(d.get("cascade_results", [])),
                "merged_from":          [os.path.basename(p)],
            }
        else:
            merged["cascade_results"].extend(d.get("cascade_results", []))
            merged["merged_from"].append(os.path.basename(p))

    # Sort for readability
    merged["cascade_results"].sort(
        key=lambda r: (r.get("head_sp_target", 0.0),
                       r.get("mlp_sp_target",  0.0)))

    out_path = os.path.join(args.output_dir, "cascade_results_v8.json")
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\n[Launcher] Merged {len(paths)} workers → {out_path}")
    return merged


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(merged):
    rows = merged["cascade_results"]
    baseline_macro = merged["baseline"]
    print("\n" + "=" * 130)
    print(f"SWEEP SUMMARY — {len(rows)} configs")
    print("=" * 130)
    print(f"  Baseline : macro Dice={baseline_macro['mean_dice']:.4f}  "
          f"BF1={baseline_macro['mean_boundary_f1']:.4f}  "
          f"HD95={baseline_macro['mean_hd95']:.2f}")
    print()

    header = (f"  {'Config':<35} "
              f"{'macro_D':>8} {'macro_BF1':>10} {'worst_BF1':>10} "
              f"{'Par%':>6} {'FL%':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in rows:
        if r["phase"] == "head_only":
            lbl = f"HEAD_ONLY h={r['head_sp_target']*100:.0f}%"
        else:
            lbl = (f"h={r['head_sp_target']*100:.0f}% "
                   f"m={r['mlp_sp_target']*100:.0f}%")
        ma = r["macro"]; wo = r["worst"]
        print(f"  {lbl:<35} "
              f"{ma['mean_dice']:>8.4f} {ma['mean_boundary_f1']:>10.4f} "
              f"{wo['mean_boundary_f1']:>10.4f} "
              f"{r.get('param_reduction_pct', 0):>6.1f} "
              f"{r.get('flop_reduction_pct', 0):>6.1f}")

    # Best and Pareto knee
    best = max(rows, key=lambda r: r["macro"]["mean_boundary_f1"])
    print(f"\n  Best macro-BF1 : {best['macro']['mean_boundary_f1']:.4f}  "
          f"at  h={best.get('head_sp_target',0)*100:.0f}% "
          f"m={best.get('mlp_sp_target',0)*100:.0f}%  "
          f"(Par↓{best.get('param_reduction_pct',0):.1f}% "
          f"FL↓{best.get('flop_reduction_pct',0):.1f}%)")

    # Most aggressive config with BF1 drop ≤ 0.015
    thresh = baseline_macro["mean_boundary_f1"] - 0.015
    accepted = [r for r in rows if r["macro"]["mean_boundary_f1"] >= thresh]
    if accepted:
        max_comp = max(accepted, key=lambda r: r.get("param_reduction_pct", 0))
        print(f"  Max compression (ΔBF1 ≤ 0.015): "
              f"h={max_comp.get('head_sp_target',0)*100:.0f}% "
              f"m={max_comp.get('mlp_sp_target',0)*100:.0f}%  "
              f"Par↓{max_comp.get('param_reduction_pct',0):.1f}% "
              f"FL↓{max_comp.get('flop_reduction_pct',0):.1f}%  "
              f"BF1={max_comp['macro']['mean_boundary_f1']:.4f}")
    else:
        print(f"  No config meets ΔBF1 ≤ 0.015 threshold (BF1 ≥ {thresh:.4f}).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Assertion: input grid sanity
    assert len(args.head_sparsities) >= 1, "--head_sparsities must be non-empty"
    assert args.chains_per_gpu >= 1

    # Print config
    print("=" * 80)
    print("PARALLEL SWEEP LAUNCHER — run_cascade_v8")
    print(f"  devices            : {args.devices}")
    print(f"  chains_per_gpu     : {args.chains_per_gpu}  "
          f"(total slots = {len(args.devices) * args.chains_per_gpu})")
    print(f"  head_sparsities    : {args.head_sparsities}")
    print(f"  mlp_sparsities     : {args.mlp_sparsities}")
    print(f"  output_dir         : {args.output_dir}")
    print(f"  phase_b_device     : cuda:{args.phase_b_device}")
    print(f"  skip_phase_b       : {args.skip_phase_b}")
    print(f"  parallel_fisher    : {args.parallel_fisher}")
    print("=" * 80)

    base_cmd = build_base_v8_cmd(args)

    t_start = time.time()

    # ----- Phase B -----
    if not args.skip_phase_b:
        if args.parallel_fisher:
            run_phase_b_parallel(args, base_cmd)
        else:
            run_phase_b(args, base_cmd)
    else:
        print("\n[Launcher] --skip_phase_b set; assuming cache already exists.")

    # ----- Phase C -----
    workers = assign_workers(args.head_sparsities, args.devices,
                              args.chains_per_gpu)
    failed = run_phase_c(args, base_cmd, workers)
    if failed:
        print(f"\n[Launcher] Failed workers: {failed}")
        print(f"Check individual logs in {args.output_dir}/phase_c_wXX.log")
        sys.exit(1)

    # ----- Merge -----
    merged = merge_results(args)

    total_min = (time.time() - t_start) / 60.0
    print(f"\n[Launcher] Total wall time: {total_min:.1f} min")

    # ----- Summary -----
    print_summary(merged)
    print(f"\n[Launcher] Done. Merged result: "
          f"{os.path.join(args.output_dir, 'cascade_results_v8.json')}")


if __name__ == "__main__":
    main()
