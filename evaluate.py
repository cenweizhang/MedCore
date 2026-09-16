"""Evaluate one checkpoint using deterministic prompts on a recorded split."""

import argparse

from medcore_pruning.utils import add_runtime_args, parse_config, write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split_manifest")
    parser.add_argument("--data_roots", nargs="+")
    parser.add_argument("--dataset_names", nargs="+")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--output", default="outputs/evaluation.json")
    add_runtime_args(parser)
    return parser


def main(argv=None):
    args = parse_config(build_parser(), argv)
    if not args.checkpoint:
        raise ValueError("Provide --checkpoint.")
    from medcore_pruning.evaluation import eval_all
    from medcore_pruning.reproducibility import build_split_loaders, seed_everything
    from medcore_pruning.utils import checkpoint_manifest, load_inference_model, validate_device

    device = validate_device(args.device)
    seed_everything(args.seed)
    model, payload = load_inference_model(args.checkpoint, device)
    manifest = checkpoint_manifest(payload, args.split_manifest, args.data_roots, args.dataset_names)
    names = [dataset["name"] for dataset in manifest["datasets"]]
    loaders = build_split_loaders(manifest, eval_batch_size=args.eval_batch_size, num_workers=args.num_workers)
    selected = loaders["test_loaders" if args.split == "test" else "val_loaders"]
    metrics = eval_all(model, selected, names, device)
    result = {"checkpoint": args.checkpoint, "split": args.split, "metrics": metrics,
              "encoder_parameters": sum(p.numel() for p in model.image_encoder.parameters()),
              "total_parameters": sum(p.numel() for p in model.parameters()),
              "representation": "compact" if payload.get("compact_manifest") else "dense"}
    write_json(args.output, result)
    print(result["metrics"]["macro"])
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
