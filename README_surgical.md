# Surgical Multi-class Segmentation & Stereo Needle Keypoint Estimation

Multi-class surgical segmentation (needle / suture-thread / instruments) with a
DINOv2 + DPT model, plus stereo needle keypoint and 6-DoF pose estimation.
This repository contains the dataset-building, training, testing, and inference
code for the base (fully-supervised) model.

## Installation

```bash
git clone https://github.com/yuxue-liu/<repo>.git
cd <repo>
conda create -n surgseg python=3.10 -y && conda activate surgseg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python scikit-image scipy pillow numpy pyyaml
```

## Pretrained Weights

Place the backbone weights under `pretrained/` and trained checkpoints under `exp/`:

| File | Path | Source |
|------|------|--------|
| DINOv2-base backbone | `pretrained/dinov2_vitb14_pretrain.pth` | DINOv2 release |
| DINOv2-small backbone | `pretrained/dinov2_vits14_pretrain.pth` | DINOv2 release |
| Trained segmentation model | `exp/<name>/best.pth` | project Releases (TODO: add URL) |

## Dataset Structure

```
<DATA_ROOT>/
├── <video>/                          # one folder per video sequence
│   ├── images/<key>/part_xxx/*.jpg   # frames
│   ├── masks/<key>/part_xxx/*.png    # index masks (0=bg, 1=needle, 2=thread, 3=instruments)
│   └── meta.json                     # frame records
└── combined/
    └── splits/r{100,50,30,10}/{labeled,unlabeled,val}.txt
```

Each split line is `<video>/images/...jpg<TAB><video>/masks/...png`, relative to
`<DATA_ROOT>`. Build the combined splits (per-video 8:2 train/val, annotation
ratios 1.0/0.5/0.3/0.1):

```bash
python tools/build_combined_trainset.py --root <DATA_ROOT> \
  --test-ratio 0.2 --ratios 1.0 0.5 0.3 0.1
```

## Training

Fully-supervised (ratio 1.0):

```bash
python train_supervised_basic.py --config configs/surgical_combined.yaml \
  --labeled-id-path <DATA_ROOT>/combined/splits/r100/labeled.txt \
  --val-id-path     <DATA_ROOT>/combined/splits/r100/val.txt \
  --save-path exp/combined_r100_base
```

Small backbone (faster inference, larger batch):

```bash
python train_supervised_basic.py --config configs/surgical_combined_small.yaml \
  --labeled-id-path <DATA_ROOT>/combined/splits/r100/labeled.txt \
  --val-id-path     <DATA_ROOT>/combined/splits/r100/val.txt \
  --save-path exp/combined_r100_small --batch-size 16
```

Training auto-resumes from `<save-path>/latest.pth`.

## Testing

```bash
python test.py --config configs/surgical_combined.yaml \
  --checkpoint exp/<run>/best.pth \
  --id-path <DATA_ROOT>/combined/splits/r100/val.txt \
  --no-bias --eval-size 1024
```

`--eval-size` caps the inference long side (predictions are upsampled back to
the original size); fp16 is enabled by default on CUDA (`--no-amp` to disable).

## Using a Different Dataset

1. Arrange images and index masks as in *Dataset Structure*.
2. In `configs/surgical_combined.yaml` set `data_root`, `nclass`, and (optionally)
   `crop_size`, `batch_size`, `lr`, `epochs`.
3. Add the class-name list to `util/classes.py` under the dataset key.
4. Build the splits with `tools/build_combined_trainset.py`.

## Key Parameters

| Parameter | Location | Meaning |
|-----------|----------|---------|
| `data_root` | config | dataset root directory |
| `nclass` | config | number of classes (incl. background) |
| `crop_size` | config / `--crop-size` | training crop (multiple of 14) |
| `batch_size` | config / `--batch-size` | batch size |
| `lr` | config / `--lr` | learning rate |
| `epochs` | config / `--epochs` | training epochs |
| `--eval-size` | `test.py` | inference long-side cap (avoids OOM) |
| `--no-amp` | train / test | disable fp16 mixed precision |

## Citation

```bibtex
@misc{surgical_seg_keypoints,
  title  = {Surgical Multi-class Segmentation and Stereo Needle Keypoint Estimation},
  author = {Yuxue Liu},
  year   = {2026},
  howpublished = {\url{https://github.com/yuxue-liu}}
}
```
