# S2 R0.10 Reproduction Package

This package contains the current S2 compatibility training entry point, its three dataset configurations, and the fixed `ratio_0.10` manifests used for Cataract1K, EndoVis2018, and EndoScapes Seg50.

Included:

- `s2_legacy/sasr_s2_legacy_compatible.py`: current S2 compatibility entry point.
- `configs/s2_legacy/`: three S2 configurations.
- `splits/ratio_0.10/`: labeled, unlabeled, validation, and test manifests.
- `tools/splits/`: split creation and verification utilities.

Not included:

- Core model implementation files.
- Pretrained backbone, S1/S2 checkpoints, text anchors, raw images, masks, logs, and experiment outputs.

All training runs must provide local paths for `data_root`, backbone checkpoint, S1 checkpoint, and text-anchor checkpoint. Keep each dataset/seed output in a new `save_path`; do not overwrite completed experiments.

For the original 0724 S2 implementation, labeled examples are repeated to `len(unlabeled.ids)` before the two loaders are zipped. This is required so every unlabeled batch is used once per epoch. Any compatible S2 runner must preserve that behavior.
