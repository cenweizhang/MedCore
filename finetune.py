"""Fine-tune the saved recovered weights while retaining all pruning masks."""

import argparse
from pathlib import Path

from medcore_pruning.utils import add_runtime_args, parse_config, write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Dense masked checkpoint from either pruning entry.")
    parser.add_argument("--resume", help="training_state.pt from an interrupted fine-tuning run.")
    parser.add_argument("--split_manifest")
    parser.add_argument("--data_roots", nargs="+", help="Relocate datasets in manifest order.")
    parser.add_argument("--dataset_names", nargs="+")
    parser.add_argument("--output_dir", default="outputs/finetune")
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5, help="Early-stop patience; 0 disables it.")
    parser.add_argument("--use_amp", action="store_true", help="Use CUDA mixed precision.")
    add_runtime_args(parser)
    return parser


def main(argv=None):
    args = parse_config(build_parser(), argv)
    if not (args.checkpoint or args.resume):
        raise ValueError("Provide --checkpoint or --resume.")
    if min(args.num_epochs, args.batch_size, args.eval_batch_size) < 1 or args.lr <= 0:
        raise ValueError("Epochs, batch sizes and learning rate must be positive.")
    if args.patience < 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("Invalid optimizer or early-stopping settings.")

    import json
    import shutil
    import torch
    from segment_anything import sam_model_registry
    from medcore_pruning.checkpoints import load_checkpoint, save_checkpoint
    from medcore_pruning.evaluation import eval_all
    from medcore_pruning.pruning import remove_hooks
    from medcore_pruning.reproducibility import build_split_loaders, save_split_manifest, seed_everything
    from medcore_pruning.training import capture_random_state, restore_random_state, run_epoch
    from medcore_pruning.utils import attach_masks, checkpoint_manifest, numpy_masks, validate_device

    device = validate_device(args.device)
    if args.use_amp and device.type != "cuda":
        raise ValueError("--use_amp requires a CUDA device.")
    seed_everything(args.seed)
    resume = None
    if args.resume:
        resume_path = Path(args.resume)
        resume = torch.load(resume_path, map_location="cpu", weights_only=True)
        source = resume_path.parent / "checkpoint_latest.pth"
        for key in ("num_epochs", "batch_size", "lr", "weight_decay", "use_amp", "seed"):
            if resume["config"][key] != getattr(args, key):
                raise ValueError(f"Resume requires the original {key}={resume['config'][key]}.")
    else:
        source = Path(args.checkpoint)
    payload = load_checkpoint(source)
    if resume and payload.get("metadata", {}).get("epoch") != resume["epoch"] + 1:
        raise ValueError("The latest checkpoint and training state have different epochs; use a matching pair.")
    if payload.get("compact_manifest") is not None:
        raise ValueError("Fine-tune the dense masked checkpoint, then run export.py.")
    manifest = checkpoint_manifest(payload, args.split_manifest, args.data_roots, args.dataset_names)
    names = [dataset["name"] for dataset in manifest["datasets"]]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    latest, best = output / "checkpoint_latest.pth", output / "checkpoint_best.pth"
    if latest.exists() and resume is None:
        raise FileExistsError(f"{latest} exists; use --resume or a new output directory.")
    save_split_manifest(manifest, output / "splits.json")
    loaders = build_split_loaders(manifest, batch_size=args.batch_size,
                                 eval_batch_size=args.eval_batch_size, num_workers=args.num_workers)
    model = sam_model_registry["vit_b"]()
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    heads, neurons = numpy_masks(payload)
    handles = attach_masks(model, heads, neurons)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for parameter in model.prompt_encoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=args.lr * .01)
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None
    start, best_loss, patience_count, history = 0, float("inf"), 0, []
    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        if scaler is not None:
            scaler.load_state_dict(resume["scaler"])
        start, best_loss = resume["epoch"] + 1, resume["best_val_loss"]
        patience_count, history = resume["patience_count"], resume["history"]
        previous_best = Path(args.resume).parent / "checkpoint_best.pth"
        if previous_best.resolve() != best.resolve():
            shutil.copy2(previous_best, best)
        restore_random_state(resume["random_state"], loaders["train_loader"].generator)
    config = json.loads(json.dumps(vars(args), default=str))
    write_json(output / "config.json", config)
    metadata = dict(payload.get("metadata", {}))
    if "validation" in metadata:
        metadata["pruning_validation"] = metadata.pop("validation")
    metadata.update(split_manifest=manifest, finetune_config=config)
    for epoch in range(start, args.num_epochs):
        if args.patience and patience_count >= args.patience:
            break
        train_loss = run_epoch(model, [loaders["train_loader"]], device, optimizer, scaler, args.grad_clip)
        val_loss = run_epoch(model, loaders["val_loaders"], device)
        scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        metadata.update(epoch=epoch + 1, val_loss=val_loss)
        if val_loss < best_loss:
            best_loss, patience_count = val_loss, 0
            save_checkpoint(best, model, heads, neurons, metadata)
        else:
            patience_count += 1
        save_checkpoint(latest, model, heads, neurons, metadata)
        state = {"epoch": epoch, "best_val_loss": best_loss, "patience_count": patience_count,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict() if scaler is not None else None,
                 "random_state": capture_random_state(loaders["train_loader"].generator),
                 "history": history, "config": config}
        temporary = output / "training_state.tmp"
        torch.save(state, temporary)
        temporary.replace(output / "training_state.pt")
        write_json(output / "history.json", history)
        print(f"Epoch {epoch + 1}/{args.num_epochs}: train={train_loss:.5f}, val={val_loss:.5f}")
    if not best.exists():
        raise RuntimeError("No best checkpoint is available; check the resume directory.")
    load_checkpoint(best, model=model)
    validation = eval_all(model, loaders["val_loaders"], names, device)
    write_json(output / "metrics.json", {"split": "validation", "metrics": validation,
                                         "best_val_loss": best_loss})
    remove_hooks(handles)
    print(f"Saved {best}. Run evaluate.py for the held-out test split.")


if __name__ == "__main__":
    main()
