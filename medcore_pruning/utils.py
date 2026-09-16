"""Small shared helpers for the public command-line entry points."""

import hashlib
import json
from pathlib import Path


def parse_config(parser, argv=None):
    """Load JSON defaults; explicit command-line arguments take precedence."""
    parser.add_argument("--config", type=Path, help="JSON file containing CLI defaults.")
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config:
        with preliminary.config.open(encoding="utf-8") as stream:
            defaults = json.load(stream)
        if not isinstance(defaults, dict):
            parser.error("The configuration must be a JSON object.")
        known = {action.dest for action in parser._actions} - {"help", "config"}
        unknown = set(defaults) - known
        if unknown:
            parser.error(f"Unknown configuration fields: {sorted(unknown)}")
        parser.set_defaults(**defaults)
    return parser.parse_args(argv)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, default=str)
        stream.write("\n")


def add_runtime_args(parser):
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=1)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_device(name):
    import torch

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA build of PyTorch or use --device cpu.")
    return device


def attach_masks(model, head_mask, neuron_mask):
    from medcore_pruning.pruning import apply_head_mask_to_model, apply_mlp_mask_to_model

    handles = []
    if head_mask is not None:
        handles += apply_head_mask_to_model(model, head_mask)
    if neuron_mask is not None:
        handles += apply_mlp_mask_to_model(model, neuron_mask)
    return handles


def numpy_masks(payload):
    def convert(value):
        return None if value is None else value.detach().cpu().numpy().astype("float32")
    return convert(payload.get("head_mask")), convert(payload.get("neuron_mask"))


def checkpoint_manifest(payload, path=None, data_roots=None, dataset_names=None):
    from medcore_pruning.reproducibility import load_split_manifest

    source = path or payload.get("metadata", {}).get("split_manifest")
    if source is None:
        raise ValueError("No data split is recorded. Provide --split_manifest from a pruning run.")
    return load_split_manifest(source, data_roots=data_roots, dataset_names=dataset_names)


def load_inference_model(path, device):
    from medcore_pruning.checkpoints import load_checkpoint
    from medcore_pruning.compact import load_compact_checkpoint
    from segment_anything import sam_model_registry

    payload = load_checkpoint(path)
    if payload.get("compact_manifest") is not None:
        model, payload = load_compact_checkpoint(path)
    else:
        model = sam_model_registry["vit_b"]()
        model.load_state_dict(payload["state_dict"])
        attach_masks(model, *numpy_masks(payload))
    return model.to(device).eval(), payload
