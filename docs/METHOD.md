# Method overview

MedCore prunes the image encoder of **MedSAM ViT-B**, using the pretrained **SAM ViT-B** weights as a reference for medical adaptation. The method preserves segmentation quality and boundaries while reducing attention and MLP computation. See the [paper](../document/MedCore_paper.pdf) for the derivation.

## Structured groups

An attention group contains one head's Q/K/V rows and attention output projection columns. An MLP group contains one hidden channel's input weight row and bias, and its output weight column. Shared output biases, prompt encoder dimensions, and mask decoder dimensions are retained.

Boundary displacement depends on both the pruning-induced logit change and the local spatial logit gradient. MedCore therefore measures importance with a segmentation loss that assigns extra weight to morphological boundary pixels.

## Dual-intervention scoring

For each calibration image, compute encoder gradients of the boundary-aware segmentation loss. The diagonal Fisher estimate is

$$
F_i=\frac{1}{N}\sum_{n=1}^{N}
\left(\frac{\partial\mathcal{L}_n}{\partial\theta_i}\right)^2.
$$

Let $\theta^M$ and $\theta^S$ denote MedSAM and SAM parameters, with Fisher estimates $F^M$ and $F^S$. For group $g$, estimate the effects of zeroing and resetting its weights:

$$
\Delta_g^{\mathrm{zero}}=\frac{1}{2}\sum_{i\in g}F_i^M(\theta_i^M)^2,
\qquad
\Delta_g^{\mathrm{reset}}=\frac{1}{2}\sum_{i\in g}
\sqrt{F_i^MF_i^S+\varepsilon_F}(\theta_i^M-\theta_i^S)^2.
$$

Zeroing measures the group's current contribution; resetting measures the contribution of medical adaptation. The geometric mean of Fisher estimates weights the reset intervention. A block-specific mixture combines both scores:

$$
Q_g=\alpha_{b(g)}\Delta_g^{\mathrm{zero}}
 +(1-\alpha_{b(g)})\Delta_g^{\mathrm{reset}}.
$$

The default mixture is $\alpha_b=(1+\rho_b)/2$, where $\rho_b$ is the correlation between the block's dataset-averaged head scores; constant scores use $\rho_b=1$. Fisher estimation uses batch size 1. Groups are ranked separately within the attention and MLP families, with cost exponent $\tau=0$.

## Dataset aggregation and budgets

Compute group scores for each dataset, then aggregate them:

$$
Q_g^{\mathrm{dist}}=\sum_r\pi_rQ_g^{(r)}
 +\beta\operatorname{Var}_r[Q_g^{(r)}].
$$

The weights $\pi_r$ sum to one; the variance is unweighted across datasets. The variance term protects groups whose importance differs across datasets. Encoder blocks with higher Fisher sensitivity receive smaller pruning quotas. Within each block, remove the lowest-scoring groups subject to a minimum retained width. Sparsity budgets use the original full 12-block encoder as their denominator; protected blocks and minimum retained widths cap the achievable pruning.

The [multimodal configuration](../configs/multimodal_stage1.json) applies this aggregation jointly to CVC-ClinicDB, BUSI, and ISIC2018, with 128 calibration images per dataset and equal normalized scoring weights. It produces one shared model for endoscopy, ultrasound, and dermoscopy. The [continuation configuration](../configs/multimodal_last_blocks.json) preserves that dataset split and mixture while pruning the final two blocks. See the [data preparation and commands](../README.md#multimodal-pruning).

## Cascade and recovery

The pruning sequence is **attention heads -> recovery -> MLP channels -> recovery**. Calibration data drive both scoring and short recovery. MLP scores may be recomputed after head recovery with `--recompute_mlp_scores`.

$$
\mathcal{L}_{\mathrm{rec}}=\mathcal{L}_{\mathrm{seg}}
 +\lambda_1\mathcal{L}_{\mathrm{bd}}
 +\lambda_2\mathcal{L}_{\mathrm{feat}}
 +\lambda_3\mathcal{L}_{\mathrm{logit}}
 +\lambda_4\mathcal{L}_{\mathrm{freq}}.
$$

Recovery combines Dice + BCE, boundary supervision, feature and boundary-logit distillation from the checkpoint at the start of the current pruning run, and high-frequency prediction error. Calibration sampling favors masks with higher boundary complexity. All image-encoder parameters can update during recovery, including those in blocks protected from new pruning. Optional fine-tuning uses the full training partition with Dice + BCE and selects a checkpoint by validation loss.

Training checkpoints contain recovered weights, pruning masks, and dataset membership. Export removes masked attention and MLP dimensions and stores the compact architecture. Evaluation uses the held-out test partition and the metrics described in the [README](../README.md#evaluation).

## Sequential pruning

`prune.py` starts from the unpruned MedSAM checkpoint and protects blocks 10-11 by default. `prune_last_blocks.py` starts from its dense recovered checkpoint, retains both attention and MLP masks in blocks 0-9, and allocates new pruning only to blocks 10-11. The source must contain explicit masks and an embedded dataset split, with both final blocks still unpruned. Existing masks remain active during scoring, recovery, and checkpoint saving; previously removed groups cannot return.

Each entry runs the same head-to-MLP cascade and saves its own checkpoint in a separate output directory. The continuation reuses the source calibration membership and, unless overridden, its dataset mixture weights. Its teacher is the masked model at the start of that continuation. Keep the dense checkpoint until all pruning and fine-tuning are complete, then export either result.

The continuation budget is an additional fraction of the original encoder's groups, rather than a cumulative sparsity or a per-block fraction. For example, a head budget of 0.10 corresponds to approximately 60% of the heads in the two eligible blocks before rounding and allocation constraints. The [README](../README.md#pruning) lists the three continuation settings and commands. The provided configurations are runnable experiment templates; exact reported results also depend on data preparation, split membership, pretrained weights, and the training environment.

## Code map

| Component | Implementation |
| --- | --- |
| First-stage and last-block entry points | [prune.py](../prune.py), [prune_last_blocks.py](../prune_last_blocks.py) |
| Boundary-aware Fisher and Cross-Fisher | [fisher.py](../medcore_pruning/fisher.py) |
| Group scores and dataset aggregation | [scoring.py](../medcore_pruning/scoring.py) |
| Block budgets and structured masks | [pruning.py](../medcore_pruning/pruning.py) |
| Calibration recovery | [recovery.py](../medcore_pruning/recovery.py) |
| Data and recorded splits | [dataset.py](../medcore_pruning/dataset.py), [reproducibility.py](../medcore_pruning/reproducibility.py) |
| Checkpoints and compact export | [checkpoints.py](../medcore_pruning/checkpoints.py), [compact.py](../medcore_pruning/compact.py) |
| Region and boundary metrics | [metrics.py](../medcore_pruning/metrics.py) |
