"""Prune MedSAM attention heads and MLP neurons, then recover the encoder."""

import argparse
import math
from pathlib import Path

from medcore_pruning.utils import add_runtime_args, parse_config, write_json


def build_parser(last_blocks=False):
    description = "Continue pruning only ViT blocks 10 and 11." if last_blocks else __doc__
    parser = argparse.ArgumentParser(description=description)
    if last_blocks:
        parser.add_argument("--checkpoint", help="First-stage dense checkpoint with pruning masks.")
    else:
        parser.add_argument("--medsam_ckpt", default="checkpoints/medsam_vit_b.pth")
    parser.add_argument("--sam_ckpt", default="checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--data_roots", nargs="+")
    parser.add_argument("--dataset_names", nargs="+")
    parser.add_argument("--pi_r", nargs="+", type=float)
    if not last_blocks:
        parser.add_argument("--cal_sizes", nargs="+", type=int)
        parser.add_argument("--split_manifest", help="Reuse a recorded train/validation/test split.")
        parser.add_argument("--val_fraction", type=float, default=0.2)
        parser.add_argument("--test_fraction", type=float, default=0.2)
        parser.add_argument("--protected_blocks", nargs="*", type=int, default=[10, 11])
    else:
        parser.set_defaults(protected_blocks=list(range(10)))
    budget_help = "Pruning budget relative to all 12 original blocks; allocated to eligible blocks."
    parser.add_argument("--head_sparsity", type=float, default=0.10 if last_blocks else 0.5,
                        help=budget_help)
    parser.add_argument("--mlp_sparsity", type=float, default=0.117 if last_blocks else 0.7,
                        help=budget_help)
    parser.add_argument("--min_keep_heads", type=int, default=1)
    parser.add_argument("--min_keep_mlp_fraction", type=float, default=0.05)
    parser.add_argument("--boundary_fisher_weight", type=float, default=3.0)
    parser.add_argument("--dist_beta", type=float, default=0.3)
    parser.add_argument("--recompute_mlp_scores", action="store_true")
    parser.add_argument("--recovery_steps", type=int, default=100)
    parser.add_argument("--recovery_lr", type=float, default=1e-5)
    parser.add_argument("--feat_distill_weight", type=float, default=0.5)
    parser.add_argument("--boundary_loss_weight", type=float, default=1.0)
    parser.add_argument("--logit_distill_weight", type=float, default=2.0)
    parser.add_argument("--freq_pred_loss_weight", type=float, default=0.5)
    parser.add_argument("--output_dir", default="outputs/last_blocks" if last_blocks else "outputs/medcore")
    add_runtime_args(parser)
    return parser


def validate_args(args):
    for name in ("head_sparsity", "mlp_sparsity"):
        if not 0 <= getattr(args, name) < 1:
            raise ValueError(f"{name} must be in [0, 1).")
    if not 1 <= args.min_keep_heads <= 12 or not 0 < args.min_keep_mlp_fraction <= 1:
        raise ValueError("Each block must retain at least one head and a positive MLP fraction.")
    if any(block not in range(12) for block in args.protected_blocks):
        raise ValueError("protected_blocks must contain ViT-B block indices 0 through 11.")
    if args.recovery_steps < 0 or not math.isfinite(args.recovery_lr) or args.recovery_lr <= 0:
        raise ValueError("recovery_steps must be nonnegative and recovery_lr positive.")
    if args.eval_batch_size < 1 or args.num_workers < 0:
        raise ValueError("eval_batch_size must be positive and num_workers nonnegative.")
    for name in ("boundary_fisher_weight", "dist_beta", "feat_distill_weight",
                 "boundary_loss_weight", "logit_distill_weight", "freq_pred_loss_weight"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")


def starting_masks(payload, last_blocks):
    """Validate a dense pruning source and return its existing structured masks."""
    import numpy as np
    from medcore_pruning.utils import numpy_masks

    if payload.get("compact_manifest") is not None:
        raise ValueError("Pruning uses aligned dense SAM parameters. Use the pre-export checkpoint.")
    heads, neurons = numpy_masks(payload)
    if not last_blocks:
        if heads is not None or neurons is not None:
            raise ValueError("Pruning starts from unpruned MedSAM. Use prune_last_blocks.py to continue.")
        return np.ones(144, dtype=np.float32), np.ones(36864, dtype=np.float32)
    for name, mask, width in (("head_mask", heads, 12), ("neuron_mask", neurons, 3072)):
        if mask is None or mask.size != 12 * width:
            raise ValueError(f"The first-stage checkpoint must contain {name} with {12 * width} entries.")
        blocks = mask.reshape(12, width)
        if np.any(blocks.sum(axis=1) == 0):
            raise ValueError(f"{name} must retain at least one unit per block.")
        if not np.all(blocks[10:] == 1):
            raise ValueError("The source must leave blocks 10 and 11 unpruned.")
    if np.all(heads == 1) and np.all(neurons == 1):
        raise ValueError("The source must already contain pruning in blocks 0 through 9.")
    return heads, neurons


def run_pruning(args, last_blocks=False):
    """Run a head-to-MLP cascade while preserving all masks from its input checkpoint."""
    validate_args(args)
    if last_blocks and not args.checkpoint:
        raise ValueError("Provide --checkpoint from the first pruning stage.")

    import copy
    import json
    import numpy as np
    import torch
    from segment_anything import sam_model_registry
    from medcore_pruning.checkpoints import load_checkpoint, save_checkpoint
    from medcore_pruning.evaluation import eval_all
    from medcore_pruning.fisher import (
        build_cross_fisher_list, compute_adaptive_alpha_per_block,
        compute_diagonal_fisher_boundary_aware, compute_sam_fisher_on_medical,
        compute_scores_per_subset, sum_fisher_dicts,
    )
    from medcore_pruning.pruning import (
        allocate_nonuniform_head_sparsity, allocate_nonuniform_neuron_sparsity,
        compute_block_sensitivity, compute_cascade_stats,
        generate_head_mask_nonuniform, generate_neuron_mask_nonuniform, remove_hooks,
    )
    from medcore_pruning.recovery import recovery_finetune
    from medcore_pruning.reproducibility import (
        build_data_splits, build_split_loaders, load_split_manifest,
        save_split_manifest, seed_everything,
    )
    from medcore_pruning.scoring import combine_scores_dist_adaptive, load_sam_encoder_params
    from medcore_pruning.utils import attach_masks, checkpoint_manifest, file_sha256, validate_device

    device = validate_device(args.device)
    seed_everything(args.seed)
    source = args.checkpoint if last_blocks else args.medsam_ckpt
    output = Path(args.output_dir)
    if output.resolve() == Path(source).resolve().parent:
        raise FileExistsError("Choose an output directory separate from the source checkpoint directory.")
    for artifact in ("checkpoint.pth", "splits.json", "scores.npz", "metrics.json", "config.json"):
        if (output / artifact).exists():
            raise FileExistsError(f"{output / artifact} already exists; choose another output directory.")
    output.mkdir(parents=True, exist_ok=True)
    payload = load_checkpoint(source)
    previous_heads, previous_neurons = starting_masks(payload, last_blocks)
    if last_blocks:
        manifest = checkpoint_manifest(payload, data_roots=args.data_roots,
                                       dataset_names=args.dataset_names)
        if args.pi_r is None:
            args.pi_r = payload.get("metadata", {}).get("config", {}).get("pi_r")
    elif args.split_manifest:
        manifest = load_split_manifest(args.split_manifest, args.data_roots, args.dataset_names)
    else:
        if not args.data_roots or not args.dataset_names:
            raise ValueError("Provide data_roots and dataset_names in --config or on the command line.")
        sizes = args.cal_sizes or [128] * len(args.data_roots)
        manifest = build_data_splits(args.data_roots, args.dataset_names, sizes,
                                    args.seed, args.val_fraction, args.test_fraction)
    save_split_manifest(manifest, output / "splits.json")
    names = [dataset["name"] for dataset in manifest["datasets"]]
    pi = np.asarray(args.pi_r if args.pi_r is not None else [1.0] * len(names), dtype=float)
    if pi.shape != (len(names),) or not np.isfinite(pi).all() or (pi < 0).any() or pi.sum() <= 0:
        raise ValueError("pi_r must contain one nonnegative finite weight per dataset and have positive sum.")
    pi /= pi.sum()
    args.pi_r = pi.tolist()
    loaders = build_split_loaders(manifest, eval_batch_size=args.eval_batch_size,
                                 num_workers=args.num_workers)
    model = sam_model_registry["vit_b"]()
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    teacher = None
    if args.recovery_steps and (args.feat_distill_weight or args.logit_distill_weight):
        teacher = copy.deepcopy(model).cpu().eval()
        if last_blocks:
            attach_masks(teacher, previous_heads, previous_neurons)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    handles = attach_masks(model, previous_heads, previous_neurons) if last_blocks else []
    print("Evaluating the starting checkpoint on validation data...")
    baseline = eval_all(model, loaders["val_loaders"], names, device)
    sam_params = load_sam_encoder_params(args.sam_ckpt)
    cal = loaders["combined_cal_loader"]

    def estimate():
        return [compute_diagonal_fisher_boundary_aware(model, loader, device,
                boundary_weight=args.boundary_fisher_weight) for loader in loaders["cal_loaders"]]

    fisher = estimate()
    sam_fisher = compute_sam_fisher_on_medical(
        args.sam_ckpt, cal, device, boundary_weight=args.boundary_fisher_weight)

    def score(fisher_list):
        cross = build_cross_fisher_list(fisher_list, sam_fisher)
        return compute_scores_per_subset(model, sam_params, fisher_list, cross)

    dz_h, dr_h, dz_m, dr_m = score(fisher)
    alpha = compute_adaptive_alpha_per_block(np.average(dz_h, axis=0, weights=pi),
                                            np.average(dr_h, axis=0, weights=pi))
    sensitivity = compute_block_sensitivity(sum_fisher_dicts(fisher, pi), 12)
    head_scores = combine_scores_dist_adaptive(dz_h, dr_h, alpha, 12, pi, args.dist_beta)
    head_quota = allocate_nonuniform_head_sparsity(sensitivity, args.head_sparsity,
        min_keep=args.min_keep_heads, protected_blocks=args.protected_blocks)
    heads = previous_heads * generate_head_mask_nonuniform(head_scores, head_quota)

    def recover():
        recovery_finetune(model, cal, device, n_steps=args.recovery_steps, lr=args.recovery_lr,
            feature_teacher=teacher, feat_distill_weight=args.feat_distill_weight,
            boundary_loss_weight=args.boundary_loss_weight, logit_distill_weight=args.logit_distill_weight,
            freq_pred_loss_weight=args.freq_pred_loss_weight,
            sampling_weights=loaders["sampling_weights"])

    remove_hooks(handles)
    handles = attach_masks(model, heads, previous_neurons)
    print("Recovering after attention-head pruning...")
    recover()
    if args.recompute_mlp_scores:
        fisher = estimate()
        _, _, dz_m, dr_m = score(fisher)
        sensitivity = compute_block_sensitivity(sum_fisher_dicts(fisher, pi), 12)
    mlp_scores = combine_scores_dist_adaptive(dz_m, dr_m, alpha, 3072, pi, args.dist_beta)
    mlp_quota = allocate_nonuniform_neuron_sparsity(sensitivity, args.mlp_sparsity,
        min_frac=args.min_keep_mlp_fraction, protected_blocks=args.protected_blocks)
    neurons = previous_neurons * generate_neuron_mask_nonuniform(mlp_scores, mlp_quota)
    remove_hooks(handles)
    handles = attach_masks(model, heads, neurons)
    print("Recovering after MLP pruning...")
    recover()
    validation = eval_all(model, loaders["val_loaders"], names, device)
    stats = compute_cascade_stats(model, heads, neurons)
    stats["representation"] = "dense_with_structured_masks"
    stats["actual_encoder_parameters"] = sum(p.numel() for p in model.image_encoder.parameters())
    config = json.loads(json.dumps(vars(args), default=str))
    metadata = {"config": config, "split_manifest": manifest, "validation": validation,
                "source_checkpoint": str(Path(source).resolve()), "pruning_statistics": stats,
                "pruning_stage": "last_blocks" if last_blocks else "first_blocks",
                "source_sha256": file_sha256(source), "sam_sha256": file_sha256(args.sam_ckpt),
                "torch_version": str(torch.__version__), "numpy_version": np.__version__}
    save_checkpoint(output / "checkpoint.pth", model, heads, neurons, metadata)
    np.savez(output / "scores.npz", head_scores=head_scores, mlp_scores=mlp_scores,
             dz_head_per_subset=np.stack(dz_h), dr_head_per_subset=np.stack(dr_h),
             dz_mlp_per_subset=np.stack(dz_m), dr_mlp_per_subset=np.stack(dr_m),
             alpha_per_block=alpha, block_sensitivity=sensitivity, pi_r=pi)
    write_json(output / "metrics.json", {"split": "validation", "baseline": baseline,
               "pruned": validation, "statistics": stats})
    write_json(output / "config.json", config)
    remove_hooks(handles)
    print(f"Saved {output / 'checkpoint.pth'}. Run evaluate.py for the held-out test split.")


def main(argv=None):
    run_pruning(parse_config(build_parser(), argv))


if __name__ == "__main__":
    main()
