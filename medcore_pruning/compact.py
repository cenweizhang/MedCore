"""Physically remove masked attention heads and MLP neurons from SAM's encoder."""

import copy

import torch
from torch import nn

from .checkpoints import (
    FORMAT_VERSION,
    _json_metadata,
    _mask_tensor,
    _validate_model_masks,
    load_checkpoint,
    save_checkpoint,
)


def _slice_linear(layer, rows=None, columns=None):
    """Copy a linear layer with selected rows or columns and no pruning hooks."""
    weight = layer.weight.detach()
    if rows is not None:
        weight = weight.index_select(0, rows.to(weight.device))
    if columns is not None:
        weight = weight.index_select(1, columns.to(weight.device))
    result = nn.Linear(
        weight.shape[1], weight.shape[0], bias=layer.bias is not None,
        device=weight.device, dtype=weight.dtype,
    )
    with torch.no_grad():
        result.weight.copy_(weight)
        if layer.bias is not None:
            bias = layer.bias.detach()
            if rows is not None:
                bias = bias.index_select(0, rows.to(bias.device))
            result.bias.copy_(bias)
    result.weight.requires_grad_(layer.weight.requires_grad)
    if layer.bias is not None:
        result.bias.requires_grad_(layer.bias.requires_grad)
    result.train(layer.training)
    return result


def export_compact_model(model, head_mask=None, neuron_mask=None, inplace=False):
    """Return ``(compact_model, manifest)`` with the same masked forward result.

    The residual width, head dimension, attention scale, and relative positional
    embeddings remain unchanged. Q/K/V output rows and attention projection input
    columns are removed together; MLP lin1 rows and lin2 columns are removed
    together. Every block must retain at least one head and one MLP neuron.

    Masks are flattened in block order. If a mask is omitted, all corresponding
    units are retained. The default copies the model; ``inplace=True`` reduces
    peak memory during export and replaces the supplied model's linear layers.
    """
    if getattr(model, "_medcore_compact_manifest", None) is not None:
        raise ValueError("This model is already compact; export from the dense checkpoint.")
    head_mask = _mask_tensor(head_mask, "head_mask")
    neuron_mask = _mask_tensor(neuron_mask, "neuron_mask")
    _validate_model_masks(model, head_mask, neuron_mask)
    blocks = model.image_encoder.blocks
    specifications = []
    head_offset = neuron_offset = 0
    for index, block in enumerate(blocks):
        attn, mlp = block.attn, block.mlp
        heads = attn.num_heads
        inner_dim = attn.qkv.out_features // 3
        if attn.qkv.out_features % 3 or inner_dim % heads:
            raise ValueError(f"Block {index} has inconsistent Q/K/V dimensions.")
        head_dim = inner_dim // heads
        neurons = mlp.lin1.out_features
        keep_heads = (
            torch.arange(heads) if head_mask is None
            else torch.where(head_mask[head_offset:head_offset + heads] == 1)[0]
        )
        keep_neurons = (
            torch.arange(neurons) if neuron_mask is None
            else torch.where(neuron_mask[neuron_offset:neuron_offset + neurons] == 1)[0]
        )
        if keep_heads.numel() == 0 or keep_neurons.numel() == 0:
            raise ValueError(f"Block {index} must retain at least one head and one MLP neuron.")
        specifications.append({
            "original_num_heads": heads,
            "original_mlp_hidden_dim": neurons,
            "embed_dim": attn.qkv.in_features,
            "head_dim": head_dim,
            "num_heads": keep_heads.numel(),
            "mlp_hidden_dim": keep_neurons.numel(),
            "kept_heads": keep_heads.tolist(),
            "kept_neurons": keep_neurons.tolist(),
        })
        head_offset += heads
        neuron_offset += neurons
    compact = model if inplace else copy.deepcopy(model)
    for block, spec in zip(compact.image_encoder.blocks, specifications):
        attn, mlp = block.attn, block.mlp
        head_dim = spec["head_dim"]
        inner_dim = attn.qkv.out_features // 3
        channels = (
            torch.tensor(spec["kept_heads"])[:, None] * head_dim
            + torch.arange(head_dim)[None, :]
        ).flatten()
        qkv_rows = torch.cat([channels + part * inner_dim for part in range(3)])
        attn.qkv = _slice_linear(attn.qkv, rows=qkv_rows)
        attn.proj = _slice_linear(attn.proj, columns=channels)
        attn.num_heads = spec["num_heads"]
        attn.head_mask = None
        kept_neurons = torch.tensor(spec["kept_neurons"])
        mlp.lin1 = _slice_linear(mlp.lin1, rows=kept_neurons)
        mlp.lin2 = _slice_linear(mlp.lin2, columns=kept_neurons)
    manifest = {
        "format_version": FORMAT_VERSION,
        "model_type": getattr(model, "_medcore_model_type", "vit_b"),
        "blocks": specifications,
    }
    compact._medcore_compact_manifest = manifest
    return compact, manifest


def save_compact_checkpoint(path, model, manifest, metadata=None):
    """Save the compact state dictionary and its reconstruction manifest."""
    manifest = _json_metadata(manifest)
    if manifest != getattr(model, "_medcore_compact_manifest", None):
        raise ValueError("The manifest does not match the exported model.")
    return save_checkpoint(path, model, metadata=metadata)


def _masks_from_manifest(model, manifest):
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported compact manifest version.")
    specifications = manifest.get("blocks")
    blocks = model.image_encoder.blocks
    if not isinstance(specifications, list) or len(specifications) != len(blocks):
        raise ValueError("The compact manifest does not match the model's encoder depth.")
    masks = {"heads": [], "neurons": []}
    for index, (block, spec) in enumerate(zip(blocks, specifications)):
        if spec.get("embed_dim") != block.attn.qkv.in_features:
            raise ValueError(f"Block {index} has an incompatible residual width.")
        if spec.get("head_dim") != block.attn.qkv.out_features // (3 * block.attn.num_heads):
            raise ValueError(f"Block {index} has an incompatible attention head dimension.")
        for name, original, count_key, original_key in (
            ("heads", block.attn.num_heads, "num_heads", "original_num_heads"),
            ("neurons", block.mlp.lin1.out_features, "mlp_hidden_dim", "original_mlp_hidden_dim"),
        ):
            kept = spec.get(f"kept_{name}", [])
            if (
                spec.get(original_key) != original
                or not isinstance(kept, list)
                or not kept
                or any(type(unit) is not int or unit < 0 or unit >= original for unit in kept)
                or kept != sorted(set(kept))
                or spec.get(count_key) != len(kept)
            ):
                raise ValueError(f"Block {index} has invalid retained {name} in its manifest.")
            mask = torch.zeros(original)
            mask[kept] = 1
            masks[name].append(mask)
    return torch.cat(masks["heads"]), torch.cat(masks["neurons"])


def load_compact_checkpoint(path, model=None, map_location="cpu"):
    """Reconstruct an exported model and return ``(model, checkpoint_payload)``.

    Pass an unpruned model to reconstruct a custom architecture. Otherwise the
    SAM registry constructs the architecture named by ``model_type``. A newly
    constructed model is placed on ``map_location`` when it is a device or string.
    """
    payload = load_checkpoint(path, map_location=map_location)
    manifest = payload.get("compact_manifest")
    if manifest is None:
        raise ValueError("This is a dense checkpoint; run export.py to compact it first.")
    if manifest.get("model_type") != payload["model_type"]:
        raise ValueError("Checkpoint and compact manifest disagree on the model type.")
    if model is None:
        from segment_anything import sam_model_registry

        model_type = payload["model_type"]
        if model_type not in sam_model_registry:
            raise ValueError(f"Unknown SAM model type: {model_type}.")
        model = sam_model_registry[model_type](checkpoint=None)
        if isinstance(map_location, (str, torch.device)):
            model = model.to(map_location)
    head_mask, neuron_mask = _masks_from_manifest(model, manifest)
    model, _ = export_compact_model(model, head_mask, neuron_mask, inplace=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload
