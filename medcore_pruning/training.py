"""Full-data Dice+BCE fine-tuning and resumable random state."""

import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def segmentation_logits(model, batch, device):
    images = batch["image"].to(device)
    boxes = batch["bbox"].to(device).float()[:, None, :]
    embeddings = model.image_encoder(images)
    with torch.no_grad():
        sparse, dense = model.prompt_encoder(points=None, boxes=boxes, masks=None)
    logits, _ = model.mask_decoder(image_embeddings=embeddings,
        image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense, multimask_output=False)
    return logits


def segmentation_loss(logits, target):
    probabilities = logits.float().sigmoid()
    numerator = (probabilities * target).sum(dim=(-2, -1))
    denominator = probabilities.square().sum(dim=(-2, -1)) + target.square().sum(dim=(-2, -1)) + 1e-6
    dice = (1 - 2 * numerator / denominator).mean()
    return dice + F.binary_cross_entropy_with_logits(logits.float(), target)


def run_epoch(model, loaders, device, optimizer=None, scaler=None, grad_clip=1.0):
    """Return the sample-weighted loss over one training or validation epoch."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for loader in loaders:
        for batch in tqdm(loader, desc="Train" if training else "Validate", leave=False):
            target = batch["mask_256"].to(device).float()
            with torch.set_grad_enabled(training):
                with torch.autocast(device_type=device.type, enabled=scaler is not None):
                    loss = segmentation_loss(segmentation_logits(model, batch, device), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite loss; check input data and training settings.")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
            total += float(loss.detach()) * len(target)
            count += len(target)
    if not count:
        raise ValueError("An empty data loader cannot be trained or evaluated.")
    return total / count


def capture_random_state(generator=None):
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state() if generator is not None else None}


def restore_random_state(state, generator=None):
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None and state["loader"] is not None:
        generator.set_state(state["loader"].cpu())
