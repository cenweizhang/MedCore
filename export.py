"""Export masked MedCore weights as a physically smaller SAM model."""

import argparse
from pathlib import Path

import numpy as np

from medcore_pruning.checkpoints import load_checkpoint
from medcore_pruning.compact import export_compact_model, save_compact_checkpoint
from segment_anything import sam_model_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Dense MedCore or MedSAM checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output compact checkpoint (.pth).")
    parser.add_argument("--head-mask", type=Path, help="Optional binary .npy mask overriding the saved head mask.")
    parser.add_argument("--neuron-mask", type=Path, help="Optional binary .npy mask overriding the saved MLP mask.")
    args = parser.parse_args()
    if args.checkpoint.resolve() == args.output.resolve():
        parser.error("--output must differ from --checkpoint to preserve the dense training artifact.")
    payload = load_checkpoint(args.checkpoint)
    if payload.get("compact_manifest") is not None:
        parser.error("The input is already compact; provide a dense checkpoint with pruning masks.")
    head_mask = payload["head_mask"] if args.head_mask is None else np.load(args.head_mask, allow_pickle=False)
    neuron_mask = payload["neuron_mask"] if args.neuron_mask is None else np.load(args.neuron_mask, allow_pickle=False)
    if head_mask is None and neuron_mask is None:
        parser.error("No pruning masks were found. Supply a MedCore checkpoint or explicit .npy masks.")
    model_type = payload["model_type"]
    if model_type not in sam_model_registry:
        parser.error(f"Unknown SAM model type: {model_type}.")
    model = sam_model_registry[model_type](checkpoint=None)
    model.load_state_dict(payload["state_dict"], strict=True)
    before = sum(parameter.numel() for parameter in model.parameters())
    model, manifest = export_compact_model(model, head_mask, neuron_mask, inplace=True)
    after = sum(parameter.numel() for parameter in model.parameters())
    metadata = dict(payload["metadata"])
    metadata["export"] = {
        "source_checkpoint": str(args.checkpoint),
        "parameters_dense": before,
        "parameters_compact": after,
    }
    save_compact_checkpoint(args.output, model, manifest, metadata)
    print(f"Saved {args.output}")
    print(f"Parameters: {before:,} -> {after:,} ({100 * (1 - after / before):.2f}% reduction)")


if __name__ == "__main__":
    main()
