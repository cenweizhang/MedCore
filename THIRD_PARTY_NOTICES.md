# Third-party notices

MedCore contributions are licensed under the [Apache License, Version 2.0](LICENSE).

## Segment Anything (SAM)

- Source: <https://github.com/facebookresearch/segment-anything>
- Included code: `segment_anything/`
- Copyright: Meta Platforms, Inc. and affiliates. All rights reserved.
- License: Apache-2.0; the upstream license is reproduced in [licenses/SAM-LICENSE](licenses/SAM-LICENSE).

Upstream copyright headers are retained. Local modifications include batched prompt compatibility and checkpoint-loading behavior. MedCore-specific compact export and pruning utilities are implemented in `medcore_pruning/`.

## MedSAM

- Source: <https://github.com/bowang-lab/MedSAM>
- Used for: the pretrained medical model and the medical segmentation workflow.
- License: Apache-2.0; the upstream license is reproduced in [licenses/MedSAM-LICENSE](licenses/MedSAM-LICENSE).
- Reference: Jun Ma, Yuting He, Feifei Li, Lin Han, Chenyu You, and Bo Wang. *Segment Anything in Medical Images*. Nature Communications 15, 654 (2024).

Model checkpoints are downloaded separately from the upstream project. This repository does not distribute them.

## Other materials

Python dependencies retain their respective licenses. Dataset files, externally downloaded model weights, and the paper are subject to their own terms and are not relicensed by the MedCore source-code license.
