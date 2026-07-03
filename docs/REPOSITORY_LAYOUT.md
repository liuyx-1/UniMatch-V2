# SASR Repository Layout

This repository is a cleaned code release for **SASR: Structure-Aware
Semantic Routing for Semi-Supervised Surgical Segmentation**.

## Kept Files

- `unimatch_v2.py`: main semi-supervised trainer with HPTA-MoE, LC-PAM, MGER,
  SMCR/TMRC, and instance-aware loss.
- `test.py`: evaluation entrypoint.
- `supervised.py`, `fixmatch.py`: baseline/reference training entrypoints.
- `configs/`: dataset configs.
- `dataset/`: segmentation dataset and transforms.
- `model/`: DINOv2-DPT backbone, adapters, affinity modules, and MoE blocks.
- `util/`: losses, routing, consistency, edge, affinity, and distributed helpers.
- `scripts/`: runnable train/test/ablation shell scripts.
- `tools/`: dataset preparation, affinity training, plotting, and ablation utilities.
- `docs/SASR_ABLATION_COMMANDS.md`: 16-command ablation plan for Endoscapes-Seg50
  and EndoVis2018 at `RATE=0.10`.

## Removed From Release

- Python caches and notebook checkpoints.
- Experiment outputs, logs, metrics, checkpoints, pretrained weights, and data.
- Large presentation/video/binary artifacts.
- Nested external repositories and legacy unrelated folders.

## Main Ablation Instructions

See `docs/SASR_ABLATION_COMMANDS.md`.
