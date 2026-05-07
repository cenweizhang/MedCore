# -*- coding: utf-8 -*-
"""
Boundary Leverage Validation  (§5.3)

Phase A (--phase logits)
    For each of the 49 (h,m) sweep configs: regenerate pruning masks from
    the Phase-B score cache, apply them to the base MedSAM model, run the
    calibration set through the network, and save the 256×256 logit maps.
    Two workers run in parallel on separate GPUs.

Phase B (--phase leverage)
    Load the saved logit maps.  For every adjacent grid step (head step
    and MLP step), compute the boundary-band logit perturbation normalised
    by Sobel gradient magnitude and parameter compression ΔC.
    Report BLR_95, WinRate_95, BSR (boundary specificity ratio).

Usage (recommended):
    python -m pilot_dual.compute_boundary_leverage \
        --medsam_ckpt  work_dir/MedSAM/medsam_vit_b.pth \
        --phase_b_cache results/pilot_cascade_v8_sweep/_phase_b_cache \
        --output_dir   results/boundary_leverage \
        --devices 2 3

Run phases manually:
    CUDA_VISIBLE_DEVICES=2 python -m pilot_dual.compute_boundary_leverage \\
        --phase logits --worker_id 0 --n_workers 2 --device cuda:0 ...
    CUDA_VISIBLE_DEVICES=3 python -m pilot_dual.compute_boundary_leverage \\
        --phase logits --worker_id 1 --n_workers 2 --device cuda:0 ...
    python -m pilot_dual.compute_boundary_leverage --phase leverage ...
"""

import argparse, json, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# Grid definition  (must match the sweep)
# ---------------------------------------------------------------------------
HEAD_SP     = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MLP_SP      = [0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]
ALL_CONFIGS = [(h, m) for h in HEAD_SP for m in MLP_SP]   # 49 points
TAU_DEFAULT = [3, 5]


# ---------------------------------------------------------------------------
# Mask generation from Phase-B score cache
# ---------------------------------------------------------------------------
def _make_masks(cache_path, h_sp, m_sp, protected_blocks=(10, 11)):
    from pilot_dual.run_cascade_v8 import combine_scores_dist_adaptive
    from pilot_dual.pruning import (
        allocate_nonuniform_head_sparsity,
        allocate_nonuniform_neuron_sparsity,
        generate_head_mask_nonuniform,
        generate_neuron_mask_nonuniform,
    )

    sc     = np.load(cache_path, allow_pickle=True)
    R      = len(sc["pi_r"])
    beta   = float(sc["dist_beta"])
    alpha  = sc["alpha_per_block"]
    sens   = sc["block_sensitivity"]
    pi     = sc["pi_r"]

    dz_h   = [sc["dz_head_per_subset"][i] for i in range(R)]
    dr_h   = [sc["dr_head_per_subset"][i] for i in range(R)]
    dz_m   = [sc["dz_mlp_per_subset"][i]  for i in range(R)]
    dr_m   = [sc["dr_mlp_per_subset"][i]  for i in range(R)]

    head_sc = combine_scores_dist_adaptive(dz_h, dr_h, alpha, items_per_block=12,
                                           pi=pi, beta=beta)
    mlp_sc  = combine_scores_dist_adaptive(dz_m, dr_m, alpha, items_per_block=3072,
                                           pi=pi, beta=beta)

    pb_head = allocate_nonuniform_head_sparsity(
        sens, h_sp, num_heads=12, min_keep=1,
        protected_blocks=list(protected_blocks))
    pb_mlp  = allocate_nonuniform_neuron_sparsity(
        sens, m_sp, mlp_dim=3072,
        protected_blocks=list(protected_blocks))

    head_mask   = generate_head_mask_nonuniform(head_sc, pb_head)
    neuron_mask = generate_neuron_mask_nonuniform(mlp_sc, pb_mlp)
    return head_mask, neuron_mask


# ---------------------------------------------------------------------------
# Phase A — compute and save logit maps
# ---------------------------------------------------------------------------
@torch.no_grad()
def _collect_logits_and_masks(model, loader, device, max_n=None):
    """
    Returns
    -------
    logits : list of (256, 256) float16 tensors  (one per sample)
    gt_masks: list of (256, 256) bool arrays
    """
    model.eval()
    logits_out, masks_out = [], []
    for batch in loader:
        images = batch["image"].to(device)                  # (B,3,1024,1024)
        bboxes = batch["bbox"].to(device).float()           # (B,4)
        gt_256 = batch["mask_256"]                          # (B,1,256,256) CPU

        if bboxes.dim() == 2:
            bboxes = bboxes[:, None, :]

        emb        = model.image_encoder(images)
        sp, dp     = model.prompt_encoder(points=None, boxes=bboxes, masks=None)
        low, _     = model.mask_decoder(
            image_embeddings=emb,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sp,
            dense_prompt_embeddings=dp,
            multimask_output=False)                         # (B,1,256,256)

        for i in range(low.shape[0]):
            logits_out.append(low[i, 0].cpu().to(torch.float16))  # (256,256)
            masks_out.append((gt_256[i, 0].numpy() > 0.5))        # (256,256) bool

        if max_n and len(logits_out) >= max_n:
            break

    return logits_out, masks_out


def phase_logits(args):
    from segment_anything import sam_model_registry
    from pilot_dual.pruning import (
        apply_head_mask_to_model, apply_mlp_mask_to_model, remove_hooks)
    from pilot_phase1.dataset import PolypDataset

    device = torch.device(args.device)
    logit_dir = os.path.join(args.output_dir, "logits")
    os.makedirs(logit_dir, exist_ok=True)

    # --- calibration loader (same split as Phase-B) ---
    print("Building calibration loader ...")
    cal_subsets = []
    for root, name, n_cal in zip(args.data_roots, args.dataset_names, args.cal_sizes):
        ds  = PolypDataset(root, bbox_shift=0)
        gen = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(ds), generator=gen).tolist()[:n_cal]
        cal_subsets.append(Subset(ds, idx))
        print(f"  [{name}] root={root}  cal={n_cal}/{len(ds)}")

    cal_ds     = ConcatDataset(cal_subsets)
    cal_loader = DataLoader(cal_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    N = len(cal_ds)
    print(f"  Total calibration samples: {N}\n")

    # --- load model ---
    print("Loading MedSAM ...")
    model = sam_model_registry["vit_b"](checkpoint=args.medsam_ckpt)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # --- save GT masks once (worker 0 only to avoid race) ---
    gt_path = os.path.join(logit_dir, "gt_masks.npz")
    if args.worker_id == 0 and not os.path.exists(gt_path):
        print("Collecting GT masks (unpruned forward pass) ...")
        _, masks_list = _collect_logits_and_masks(model, cal_loader, device)
        np.savez_compressed(gt_path,
                            masks=np.stack(masks_list, axis=0).astype(np.bool_))
        print(f"  Saved → {gt_path}\n")
    else:
        # Collect without saving (we still need to iterate to populate masks)
        pass

    # --- assign configs to this worker ---
    my_configs = [c for i, c in enumerate(ALL_CONFIGS)
                  if i % args.n_workers == args.worker_id]
    print(f"Worker {args.worker_id}/{args.n_workers}: "
          f"{len(my_configs)} configs  (device={args.device})\n")

    cache_path = os.path.join(args.phase_b_cache, "scores.npz")
    hooks      = []

    for step_i, (h, m) in enumerate(my_configs):
        tag      = f"h{int(round(h*100))}_m{int(round(m*100))}"
        out_path = os.path.join(logit_dir, f"{tag}.npz")
        if os.path.exists(out_path):
            print(f"  [{step_i+1:2d}/{len(my_configs)}] {tag} — skip (exists)")
            continue

        t0 = time.time()

        # swap masks
        if hooks:
            remove_hooks(hooks)
            hooks = []

        head_mask, neuron_mask = _make_masks(cache_path, h, m)
        hooks  = apply_head_mask_to_model(model, head_mask)
        hooks += apply_mlp_mask_to_model(model, neuron_mask)

        logits_list, _ = _collect_logits_and_masks(model, cal_loader, device)
        arr = np.stack([l.numpy() for l in logits_list], axis=0)   # (N,256,256) fp16
        np.savez_compressed(out_path, logits=arr)

        elapsed = time.time() - t0
        print(f"  [{step_i+1:2d}/{len(my_configs)}] {tag}  "
              f"N={len(logits_list)}  {elapsed:.1f}s")

    if hooks:
        remove_hooks(hooks)
    print(f"\nWorker {args.worker_id} done.")


# ---------------------------------------------------------------------------
# Phase B — compute boundary leverage
# ---------------------------------------------------------------------------
def _boundary_band(mask_bool, tau):
    """mask_bool: (H,W) bool → narrow band within τ pixels of boundary."""
    dist_fg = distance_transform_edt(mask_bool)
    dist_bg = distance_transform_edt(~mask_bool)
    return np.minimum(dist_fg, dist_bg) <= tau


def _sobel_mag(logit_np):
    """logit_np: (N,H,W) float32 → (N,H,W) float32 Sobel gradient magnitude."""
    t  = torch.from_numpy(logit_np).unsqueeze(1)          # (N,1,H,W)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                      dtype=torch.float32).view(1, 1, 3, 3) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                      dtype=torch.float32).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(t, kx, padding=1)
    gy = F.conv2d(t, ky, padding=1)
    return (gx**2 + gy**2).sqrt().squeeze(1).numpy()      # (N,H,W)


def _lambda_bd(log_before, log_after, gt_masks, dC, tau, eps=1e-6):
    """Returns (lbd_mean, lbd_95)."""
    if dC <= 0.01:
        return np.nan, np.nan

    delta  = (log_after - log_before)                     # (N,256,256)
    grad_A = _sobel_mag(log_before)                        # (N,256,256)

    # clip denominator at 5th-percentile to avoid explosion in flat regions
    flat_vals = grad_A.ravel()
    clip_low  = float(np.percentile(flat_vals[flat_vals > 0], 5)) if (flat_vals > 0).any() else eps
    denom     = np.maximum(grad_A, max(clip_low, eps))

    r_all = []
    for n in range(len(gt_masks)):
        band = _boundary_band(gt_masks[n], tau)
        if not band.any():
            continue
        r_all.append(np.abs(delta[n][band]) / denom[n][band])

    if not r_all:
        return np.nan, np.nan

    flat = np.concatenate(r_all)
    return float(flat.mean()) / dC, float(np.percentile(flat, 95)) / dC


def _lambda_reg(log_before, log_after, dC):
    """Full-image mean |Δlogit| / ΔC."""
    if dC <= 0.01:
        return np.nan
    return float(np.mean(np.abs(log_after - log_before))) / dC


def phase_leverage(args):
    logit_dir  = os.path.join(args.output_dir, "logits")
    sweep_dir  = args.sweep_dir

    # --- GT masks ---
    gt_path = os.path.join(logit_dir, "gt_masks.npz")
    if not os.path.exists(gt_path):
        sys.exit(f"ERROR: GT masks not found at {gt_path}")
    gt_masks = np.load(gt_path)["masks"].astype(bool)     # (N,256,256)
    N = gt_masks.shape[0]
    print(f"GT masks: {gt_masks.shape}")

    # --- par% table from sweep JSON ---
    par_cache = {}
    for fname in sorted(os.listdir(sweep_dir)):
        if not fname.startswith("cascade_results_v8_w"):
            continue
        d = json.load(open(os.path.join(sweep_dir, fname)))
        for r in d.get("cascade_results", []):
            if r.get("phase") != "cascade":
                continue
            h = round(r.get("head_sp_target", 0), 2)
            m = round(r.get("mlp_sp_target",  0), 2)
            par_cache[(h, m)] = r.get("param_reduction_pct", 0.0)

    # --- load all 49 logit maps ---
    logit_cache, missing = {}, []
    print("Loading logit maps ...")
    for h, m in ALL_CONFIGS:
        tag  = f"h{int(round(h*100))}_m{int(round(m*100))}"
        path = os.path.join(logit_dir, f"{tag}.npz")
        if not os.path.exists(path):
            missing.append(tag)
            continue
        logit_cache[(h, m)] = np.load(path)["logits"].astype(np.float32)

    if missing:
        print(f"  WARNING: missing {len(missing)} logit maps: {missing}")
    print(f"  Loaded {len(logit_cache)}/49 configs\n")

    all_rows = []

    for tau in args.tau_list:
        print(f"--- tau = {tau} ---")

        # Head steps: (h_i, m_j) → (h_{i+1}, m_j)
        for m in MLP_SP:
            for i, h in enumerate(HEAD_SP[:-1]):
                h2 = HEAD_SP[i + 1]
                if (h, m) not in logit_cache or (h2, m) not in logit_cache:
                    continue
                dC = par_cache.get((h2, m), 0) - par_cache.get((h, m), 0)
                lm, l95 = _lambda_bd(logit_cache[(h, m)], logit_cache[(h2, m)],
                                     gt_masks, dC, tau)
                lr = _lambda_reg(logit_cache[(h, m)], logit_cache[(h2, m)], dC)
                all_rows.append(dict(tau=tau, step_type="head",
                                     from_h=h, from_m=m, to_h=h2, to_m=m,
                                     delta_C=dC, lambda_bd_mean=lm,
                                     lambda_bd_95=l95, lambda_reg=lr))

        # MLP steps: (h_i, m_j) → (h_i, m_{j+1})
        for h in HEAD_SP:
            for j, m in enumerate(MLP_SP[:-1]):
                m2 = MLP_SP[j + 1]
                if (h, m) not in logit_cache or (h, m2) not in logit_cache:
                    continue
                dC = par_cache.get((h, m2), 0) - par_cache.get((h, m), 0)
                lm, l95 = _lambda_bd(logit_cache[(h, m)], logit_cache[(h, m2)],
                                     gt_masks, dC, tau)
                lr = _lambda_reg(logit_cache[(h, m)], logit_cache[(h, m2)], dC)
                all_rows.append(dict(tau=tau, step_type="mlp",
                                     from_h=h, from_m=m, to_h=h, to_m=m2,
                                     delta_C=dC, lambda_bd_mean=lm,
                                     lambda_bd_95=l95, lambda_reg=lr))

    # --- save CSV ---
    csv_path = os.path.join(args.output_dir, "boundary_leverage_steps.csv")
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in all_rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
        print(f"Saved step CSV → {csv_path}")

    # --- summary per tau ---
    print("\n" + "=" * 70)
    print("BOUNDARY LEVERAGE SUMMARY")
    print("=" * 70)

    tau_summary = {}
    for tau in args.tau_list:
        rh = [r for r in all_rows
              if r["tau"] == tau and r["step_type"] == "head"
              and not np.isnan(r["lambda_bd_95"]) and r["delta_C"] > 0.01]
        rm = [r for r in all_rows
              if r["tau"] == tau and r["step_type"] == "mlp"
              and not np.isnan(r["lambda_bd_95"]) and r["delta_C"] > 0.01]
        if not rh or not rm:
            print(f"tau={tau}: insufficient data (head={len(rh)}, mlp={len(rm)})")
            continue

        lh95  = np.array([r["lambda_bd_95"]  for r in rh])
        lm95  = np.array([r["lambda_bd_95"]  for r in rm])
        lhreg = np.array([r["lambda_reg"]     for r in rh])
        lmreg = np.array([r["lambda_reg"]     for r in rm])

        blr95 = float(np.median(lh95) / (np.median(lm95) + 1e-9))
        rlr   = float(np.median(lhreg) / (np.median(lmreg) + 1e-9))
        bsr   = float(blr95 / (rlr + 1e-9))

        # Paired win rate: for each grid point (h_i, m_j), compare head vs mlp step
        pairs_win, pairs_diff = [], []
        for r_h in rh:
            for r_m in rm:
                if (r_h["from_h"] == r_m["from_h"] and
                        r_h["from_m"] == r_m["from_m"]):
                    pairs_win.append(r_h["lambda_bd_95"] > r_m["lambda_bd_95"])
                    pairs_diff.append(r_h["lambda_bd_95"] - r_m["lambda_bd_95"])

        winrate   = float(np.mean(pairs_win))  if pairs_win  else np.nan
        diff_med  = float(np.median(pairs_diff)) if pairs_diff else np.nan

        print(f"\ntau = {tau} pixels:")
        print(f"  Head  n={len(lh95):3d}  Median λ_bd95={np.median(lh95):.5f}"
              f"  Mean={np.mean(lh95):.5f}"
              f"  p25={np.percentile(lh95,25):.5f}  p75={np.percentile(lh95,75):.5f}")
        print(f"  MLP   n={len(lm95):3d}  Median λ_bd95={np.median(lm95):.5f}"
              f"  Mean={np.mean(lm95):.5f}"
              f"  p25={np.percentile(lm95,25):.5f}  p75={np.percentile(lm95,75):.5f}")
        print(f"  BLR_95     = {blr95:.3f}  {'✓ (>1)' if blr95>1 else '✗ (≤1)'}")
        print(f"  RLR        = {rlr:.3f}")
        print(f"  BSR        = {bsr:.3f}  {'✓ (>1, boundary-specific)' if bsr>1 else '✗'}")
        print(f"  WinRate_95 = {winrate:.3f}  ({sum(pairs_win)}/{len(pairs_win)})"
              f"  {'✓ (>0.6)' if winrate>0.6 else '✗'}")
        print(f"  Diff_med   = {diff_med:.5f}  {'✓ (>0)' if diff_med>0 else '✗'}")

        tau_summary[tau] = dict(n_head=len(lh95), n_mlp=len(lm95),
                                median_head=float(np.median(lh95)),
                                median_mlp=float(np.median(lm95)),
                                BLR_95=blr95, RLR=rlr, BSR=bsr,
                                WinRate_95=winrate, Diff95=diff_med,
                                n_pairs=len(pairs_win))

    summary_path = os.path.join(args.output_dir, "boundary_leverage_summary.json")
    with open(summary_path, "w") as f:
        json.dump(tau_summary, f, indent=2)
    print(f"\nSaved summary → {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Boundary Leverage Validation §5.3")
    p.add_argument("--phase", choices=["logits", "leverage", "all"], default="all",
                   help="'all' launches two GPU workers then runs leverage.")
    p.add_argument("--medsam_ckpt",   default="work_dir/MedSAM/medsam_vit_b.pth")
    p.add_argument("--phase_b_cache", default="results/pilot_cascade_v8_sweep/_phase_b_cache")
    p.add_argument("--sweep_dir",     default="results/pilot_cascade_v8_sweep")
    p.add_argument("--output_dir",    default="results/boundary_leverage")
    p.add_argument("--data_roots",    nargs="+",
                   default=["asserts/kvasir-seg/Kvasir-SEG",
                            "asserts/CVC-ColonDB",
                            "asserts/CVC-ClinicDB"])
    p.add_argument("--dataset_names", nargs="+",
                   default=["Kvasir", "ColonDB", "ClinicDB"])
    p.add_argument("--cal_sizes",     nargs="+", type=int, default=[128, 128, 128])
    p.add_argument("--batch_size",    type=int, default=32,
                   help="Inference batch size per GPU.")
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--tau_list",      nargs="+", type=int, default=[3, 5])
    p.add_argument("--devices",       nargs="+", type=int, default=[2, 3],
                   help="Physical GPU ids for the two parallel logit workers.")
    # Internal flags used when launched as a subprocess worker
    p.add_argument("--device",        default="cuda:0",
                   help="Device for this worker (overridden by CUDA_VISIBLE_DEVICES).")
    p.add_argument("--worker_id",     type=int, default=0)
    p.add_argument("--n_workers",     type=int, default=2)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.phase == "logits":
        phase_logits(args)

    elif args.phase == "leverage":
        phase_leverage(args)

    elif args.phase == "all":
        logs_dir = os.path.join(args.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)

        # Shared CLI flags for subprocess workers
        base_cmd = [
            sys.executable, "-m", "pilot_dual.compute_boundary_leverage",
            "--phase", "logits",
            "--medsam_ckpt",   args.medsam_ckpt,
            "--phase_b_cache", args.phase_b_cache,
            "--sweep_dir",     args.sweep_dir,
            "--output_dir",    args.output_dir,
            "--data_roots",    ] + args.data_roots + [
            "--dataset_names", ] + args.dataset_names + [
            "--cal_sizes",     ] + [str(c) for c in args.cal_sizes] + [
            "--batch_size",    str(args.batch_size),
            "--num_workers",   str(args.num_workers),
            "--seed",          str(args.seed),
            "--n_workers",     str(len(args.devices)),
            "--device",        "cuda:0",   # each worker sees only its own GPU
        ]

        procs = []
        print("Launching logit workers ...")
        for wid, gpu_id in enumerate(args.devices):
            cmd      = base_cmd + ["--worker_id", str(wid)]
            log_path = os.path.join(logs_dir, f"worker_{wid}.log")
            env      = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logf     = open(log_path, "w")
            proc     = subprocess.Popen(cmd, stdout=logf,
                                        stderr=subprocess.STDOUT, env=env)
            procs.append({"proc": proc, "logf": logf, "wid": wid, "gpu": gpu_id})
            print(f"  Worker {wid}  cuda:{gpu_id}  pid={proc.pid}  log={log_path}")

        t0      = time.time()
        pending = set(range(len(procs)))
        while pending:
            time.sleep(30)
            for i in list(pending):
                rc = procs[i]["proc"].poll()
                if rc is not None:
                    procs[i]["logf"].close()
                    status = "OK" if rc == 0 else f"FAIL({rc})"
                    print(f"  [{(time.time()-t0)/60:.1f}min]  {status}  worker {procs[i]['wid']}")
                    pending.discard(i)

        failed = [p["wid"] for p in procs if p["proc"].returncode != 0]
        if failed:
            print(f"ERROR: workers {failed} failed. Logs: {logs_dir}/")
            sys.exit(1)

        print(f"\nAll logit workers done ({(time.time()-t0)/60:.1f} min). "
              f"Running leverage computation ...\n")
        phase_leverage(args)


if __name__ == "__main__":
    main()
