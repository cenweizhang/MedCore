# MedCore

Implementation of **MedCore: Boundary-Preserving Medical Core Pruning for MedSAM**. MedCore combines boundary-aware importance scores, medical adaptation relative to SAM, and short recovery to prune attention heads and MLP channels in the MedSAM ViT-B image encoder.

[Method overview](docs/METHOD.md)

## Repository

```text
prune.py                 Prune the first ten encoder blocks
prune_last_blocks.py     Continue by pruning the final two blocks
finetune.py              Fine-tune either dense masked checkpoint
evaluate.py              Evaluate a dense or compact checkpoint
export.py                Export physically smaller weights
configs/                 Experiment settings
medcore_pruning/         Shared pruning, data, and training code
segment_anything/        SAM ViT-B architecture
docs/METHOD.md           Method and implementation map
document/MedCore_paper.pdf
pyproject.toml           Package metadata and dependencies
uv.lock                  Resolved dependency versions
.python-version          Default Python version for uv
```

## Installation

Use Python 3.10 or newer for the pip installation. The locked uv workflow uses Python 3.12.

### uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then choose one environment from the repository root:

```bash
# NVIDIA GPU on x86-64 Linux or Windows: PyTorch with CUDA 12.8.
uv sync --locked --extra cu128

# CPU environment for inspection, export, and small checks.
uv sync --locked --extra cpu
```

Choose only one of these commands; the extras are mutually exclusive and both select PyTorch 2.8.0. CUDA execution requires a compatible NVIDIA GPU and driver. On Apple silicon macOS, use the CPU option. The PyTorch sources follow the [uv integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

`uv sync` creates `.venv` and installs MedCore in editable mode from the committed `uv.lock`. After syncing, prefix any Python command in this README with `uv run --no-sync` to use that environment and retain the selected PyTorch build:

```bash
uv run --no-sync python prune.py --config configs/multimodal_stage1.json
uv run --no-sync python prune_last_blocks.py --config configs/multimodal_last_blocks.json
```

For CPU execution, append `--device cpu`; the experiment configurations default to `cuda:0`. To change the backend, run the corresponding sync command again. Alternatively, use `uv run --locked --extra cu128 python ...` (or `--extra cpu`) to check and synchronize the selected environment before each command. Commit `uv.lock` and `.python-version`; keep `.venv` out of Git. Update dependency versions intentionally with `uv lock --upgrade`; upgrading PyTorch also requires changing both extra pins in `pyproject.toml`.

### pip

Install PyTorch for your CUDA environment, then run from the repository root:

```bash
python -m pip install -e .
```

Dependencies and the setuptools build backend are declared in `pyproject.toml`; a separate `setup.py` is not required. Pruning at the full 1024-pixel input resolution requires substantial memory; a CUDA GPU is recommended.

## Data and weights

Download MedSAM ViT-B and SAM ViT-B weights from the respective [MedSAM](https://github.com/bowang-lab/MedSAM) and [SAM](https://github.com/facebookresearch/segment-anything) repositories. Put them in `checkpoints/`, or change their paths in the configuration.

Prepare each binary segmentation dataset with matching image and mask filename stems:

```text
data/CVC-ClinicDB/
    images/sample.png
    masks/sample.png
data/CVC-ColonDB/
    images/...
    masks/...
data/Kvasir-SEG/
    images/...
    masks/...
checkpoints/
    medsam_vit_b.pth
    sam_vit_b_01ec64.pth
```

Masks must encode background and foreground as 0/1 or 0/255. Images are resized to 1024 x 1024 and normalized per image. Calibration and evaluation use deterministic boxes derived from ground-truth masks; evaluation therefore measures segmentation with box prompts.

The polyp configurations use three endoscopy datasets. The [multimodal workflow](#multimodal-pruning) combines endoscopy, ultrasound, and dermoscopy in one pruned model. Dataset files and pretrained weights are downloaded separately.

The default split is approximately 60% training, 20% validation, and 20% test, with 128 calibration images drawn from each training partition. Dataset membership is saved in `splits.json` and embedded in the checkpoint. Reuse the manifest for related experiments; use `--data_roots` in the same dataset order when relocating files. These example settings do not by themselves guarantee reproduction of the paper's reported numbers.

## Pruning

Run a single cascade with the standard polyp settings:

```bash
python prune.py --config configs/polyp.json
```

For sequential pruning, run the two entries in order:

```bash
python prune.py --config configs/polyp_stage1.json
python prune_last_blocks.py --config configs/polyp_last_blocks.json
```

The first command prunes blocks 0-9 and saves `outputs/polyp_stage1/checkpoint.pth`. The second loads those recovered weights and masks, preserves the masks of blocks 0-9, and prunes blocks 10-11. It writes a separate `outputs/polyp_last_blocks/checkpoint.pth`. Block indices are zero-based. Protected blocks are excluded from new pruning; their encoder parameters can still update during recovery.

Both commands use **head pruning -> recovery -> MLP pruning -> recovery**, and save weights, masks, configuration, split membership, scores, and validation metrics. The continuation reuses the first checkpoint's calibration membership and dataset mixture weights. It requires a dense masked checkpoint whose final two blocks have not yet been pruned. No intermediate mask-conversion step is needed.

`head_sparsity` and `mlp_sparsity` are pruning budgets relative to the original full 12-block encoder. Protected blocks and minimum retained widths limit the achievable budget. In the continuation, the budget applies only to new removals from the last two blocks:

| Head budget | MLP budget | Approximate fraction removed within the last two blocks |
| --- | --- | --- |
| 0.10 | 0.117 | 60% heads, 70% MLP channels |
| 0.12 | 0.14 | 72% heads, 84% MLP channels |
| 0.14 | 0.158 | 84% heads, 95% MLP channels |

These are aggregate approximations, not exact targets for each block: sensitivity-based allocation, integer rounding, and retained-width limits determine the actual masks. The default continuation uses the first row. Run other settings from the same first-stage checkpoint with distinct output directories:

```bash
python prune_last_blocks.py --config configs/polyp_last_blocks.json --head_sparsity 0.12 --mlp_sparsity 0.14 --output_dir outputs/polyp_last_blocks_medium
python prune_last_blocks.py --config configs/polyp_last_blocks.json --head_sparsity 0.14 --mlp_sparsity 0.158 --output_dir outputs/polyp_last_blocks_high
```

Command-line arguments override JSON settings. Use `python <script>.py --help` for available options.

## Multimodal pruning

The three-modality configuration uses one dataset per modality:

| Dataset | Modality | Prepared root | Calibration images |
| --- | --- | --- | --- |
| CVC-ClinicDB | Endoscopy | `data/CVC-ClinicDB` | 128 |
| BUSI | Breast ultrasound | `data/BUSI` | 128 |
| ISIC2018 | Dermoscopy | `data/ISIC2018` | 128 |

Each root must contain `images/` and `masks/` with matching filename stems, following the format above. Grayscale ultrasound images are converted to three channels by the loader. Keep each dataset in its own directory.

Prepare the segmentation labels as follows:

- **CVC-ClinicDB:** place each endoscopy image and its binary polyp mask in the paired directories.
- **BUSI:** obtain the data from the [authors](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset). For this template, use benign and malignant lesion images and exclude normal images without lesions. Combine all lesion masks for an image using a pixelwise logical OR. Include the category in both output filenames, for example `images/benign_001.png` and `masks/benign_001.png`, to avoid collisions. The [dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6906728/) describes the original category folders and mask naming.
- **ISIC2018:** download the Task 1 training images and segmentation ground truth from the [official archive](https://challenge.isic-archive.com/data/). Rename prepared masks from `ISIC_<id>_segmentation.png` to `ISIC_<id>.png` to match the corresponding `ISIC_<id>.jpg` images. Use segmentation labels, not the Task 3 classification labels.

Preserve matching image/mask dimensions and save masks as binary 0/1 or 0/255 images. This template creates local image-level train/validation/test splits from the prepared data; these are not the official ISIC challenge splits. Each training partition must contain at least 128 images, or `cal_sizes` must be reduced.

Run the first stage and, optionally, continue on the final two blocks:

```bash
python prune.py --config configs/multimodal_stage1.json
python prune_last_blocks.py --config configs/multimodal_last_blocks.json
```

The first stage uses head/MLP budgets of 0.5/0.7 and saves `outputs/multimodal_stage1/checkpoint.pth`. The continuation uses additional budgets of 0.10/0.117 and saves `outputs/multimodal_last_blocks/checkpoint.pth`. It inherits the three datasets, calibration membership, and mixture weights from the first checkpoint. Both stages save validation metrics for each modality and their macro average.

Fisher scores are estimated separately for the three datasets and combined to produce one set of pruning masks. `pi_r: [1, 1, 1]` is normalized to equal scoring weights. Recovery draws from all 384 calibration images using boundary-complexity sampling, so these scoring weights do not enforce equal modality sampling during recovery. The budgets and equal weights are example settings, not a claim of the paper's exact experimental protocol.

The shared fine-tuning, export, and evaluation entries also accept these checkpoints:

```bash
python finetune.py --checkpoint outputs/multimodal_last_blocks/checkpoint.pth --output_dir outputs/multimodal_finetune --num_epochs 20 --use_amp
python export.py --checkpoint outputs/multimodal_finetune/checkpoint_best.pth --output outputs/multimodal_compact.pth
python evaluate.py --checkpoint outputs/multimodal_compact.pth --output outputs/multimodal_test.json
```

## Fine-tuning and export

Optional fine-tuning uses the full training partition and selects the best checkpoint by validation loss. Either pruning checkpoint can be used:

```bash
python finetune.py --checkpoint outputs/polyp_last_blocks/checkpoint.pth --output_dir outputs/polyp_last_blocks_finetune --num_epochs 20 --use_amp
python export.py --checkpoint outputs/polyp_last_blocks_finetune/checkpoint_best.pth --output outputs/polyp_last_blocks_compact.pth
```

Fine-tuning saves `checkpoint_best.pth`, `checkpoint_latest.pth`, and `training_state.pt`. Resume with `--resume path/to/training_state.pt` and the original training settings. Dense checkpoints retain tensor shapes and masks for training; export physically removes masked dimensions and records the compact architecture. Export after all intended pruning and fine-tuning.

## Evaluation

```bash
python evaluate.py --checkpoint outputs/polyp_last_blocks_compact.pth --output outputs/polyp_last_blocks_test.json
```

Evaluation uses the recorded test partition by default and reports Dice, IoU, Boundary F1, and HD95 in resized-image pixels, with per-dataset and macro averages. Dense masked checkpoints can be evaluated with the same command. Keep the held-out test partition separate from configuration and checkpoint selection.


