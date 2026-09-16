"""Portable model checkpoints with explicit pruning masks and experiment metadata."""

import json
from collections.abc import Mapping
from pathlib import Path

import torch


FORMAT_VERSION = 1


def _json_metadata(metadata):
    """Copy metadata while rejecting values that cannot be written as JSON."""
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("Checkpoint metadata must be a JSON-compatible mapping.")
    return json.loads(json.dumps(dict(metadata), allow_nan=False))


def _mask_tensor(mask, name):
    if mask is None:
        return None
    value = torch.as_tensor(mask).detach().to(device="cpu", dtype=torch.float32)
    value = value.flatten().clone()
    if value.numel() == 0 or not torch.all((value == 0) | (value == 1)):
        raise ValueError(f"{name} must contain binary values (0 = prune, 1 = keep).")
    return value


def _validate_model_masks(model, head_mask, neuron_mask):
    if head_mask is None and neuron_mask is None:
        return
    if not hasattr(model, "image_encoder") or not hasattr(model.image_encoder, "blocks"):
        raise ValueError("Pruning masks require a model with image_encoder.blocks.")
    blocks = model.image_encoder.blocks
    expected = {
        "head_mask": (head_mask, sum(block.attn.num_heads for block in blocks)),
        "neuron_mask": (neuron_mask, sum(block.mlp.lin1.out_features for block in blocks)),
    }
    for name, (mask, size) in expected.items():
        if mask is not None and mask.numel() != size:
            raise ValueError(f"{name} has {mask.numel()} entries; expected {size}.")


def save_checkpoint(path, model, head_mask=None, neuron_mask=None, metadata=None):
    """Save weights and flattened binary masks; optimizer state is not included.

    Checkpoints contain only tensors and primitive values and can be read with
    ``torch.load(..., weights_only=True)``. Masks describe the dense architecture
    and must be reapplied by the caller when resuming masked training.
    """
    head_mask = _mask_tensor(head_mask, "head_mask")
    neuron_mask = _mask_tensor(neuron_mask, "neuron_mask")
    _validate_model_masks(model, head_mask, neuron_mask)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_type": getattr(model, "_medcore_model_type", "vit_b"),
        "state_dict": {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        },
        "head_mask": head_mask,
        "neuron_mask": neuron_mask,
        "metadata": _json_metadata(metadata),
    }
    manifest = getattr(model, "_medcore_compact_manifest", None)
    if manifest is not None:
        if head_mask is not None or neuron_mask is not None:
            raise ValueError("A compact model must not carry dense pruning masks.")
        payload["compact_manifest"] = _json_metadata(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_checkpoint(path, model=None, map_location="cpu"):
    """Read a MedCore checkpoint or a raw SAM/MedSAM state dictionary.

    Weights may be stored under ``model``, ``model_state_dict``, or ``state_dict``
    if they contain tensors and primitive values. Whole-model pickle
    files are deliberately unsupported. Returns a normalized checkpoint mapping.
    If ``model`` is supplied, weights are loaded strictly; hooks are not installed.
    Compact checkpoints must instead be reconstructed by
    :func:`medcore_pruning.compact.load_compact_checkpoint`.
    """
    raw = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(raw, Mapping):
        raise ValueError("Expected a state dictionary or a checkpoint mapping.")
    state_dict = None
    for key in ("state_dict", "model_state_dict", "model"):
        if key in raw:
            state_dict = raw[key]
            break
    is_raw = state_dict is None and all(isinstance(v, torch.Tensor) for v in raw.values())
    if is_raw:
        state_dict = raw
        source = {}
    else:
        source = raw
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("The checkpoint does not contain a nonempty model state dictionary.")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state_dict.items()):
        raise ValueError("Model state must map parameter names to tensors.")
    version = source.get("format_version", FORMAT_VERSION)
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format version: {version}.")
    payload = {
        "format_version": FORMAT_VERSION,
        "model_type": source.get("model_type", "vit_b"),
        "state_dict": dict(state_dict),
        "head_mask": _mask_tensor(source.get("head_mask"), "head_mask"),
        "neuron_mask": _mask_tensor(source.get("neuron_mask"), "neuron_mask"),
        "metadata": _json_metadata(source.get("metadata")),
    }
    if "compact_manifest" in source:
        payload["compact_manifest"] = _json_metadata(source["compact_manifest"])
    if model is not None:
        if payload.get("compact_manifest") is not None:
            raise ValueError("Use load_compact_checkpoint() to reconstruct compact weights.")
        _validate_model_masks(model, payload["head_mask"], payload["neuron_mask"])
        model.load_state_dict(payload["state_dict"], strict=True)
    return payload
