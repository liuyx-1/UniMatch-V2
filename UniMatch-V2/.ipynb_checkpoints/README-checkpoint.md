# UniMatch V2

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-semantic-segmentation-on-27)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-27?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-semantic-segmentation-on-10)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-10?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-semantic-segmentation-on-22)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-22?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-semantic-segmentation-on-8)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-8?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-semantic-segmentation-on-coco-1)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-coco-1?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-semantic-segmentation-on-coco-2)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-coco-2?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-semantic-segmentation-on-41)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-41?p=unimatch-v2-pushing-the-limit-of-semi)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unimatch-v2-pushing-the-limit-of-semi/semi-supervised-semantic-segmentation-on-42)](https://paperswithcode.com/sota/semi-supervised-semantic-segmentation-on-42?p=unimatch-v2-pushing-the-limit-of-semi)

This codebase contains the official PyTorch implementation of <b>UniMatch V2</b>:

> **[UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation](https://arxiv.org/abs/2410.10777)**</br>
> Lihe Yang, Zhen Zhao, Hengshuang Zhao</br>
> *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2025

<p align="left">
<img src="./docs/framework.png" width=90% height=90% 
class="center">
</p>

**TL;DR:** We upgrade our [UniMatch V1](https://github.com/LiheYoung/UniMatch) by switching the outdated ResNet encoders to the most capable DINOv2 encoders. We unify the image-level and feature-level augmentations into a single learnable stream to challenge the powerful model. Based on this, we further design a Complementary Dropout to craft better dual views.

## Results

**We provide the [training log of each reported value](https://github.com/LiheYoung/UniMatch-V2/blob/main/training-logs). You can refer to them during reproducing. We also provide all the [checkpoints](https://huggingface.co/LiheYoung/UniMatch-V2/tree/main) of our core experiments.**

### Pascal VOC 2012

| Method            |    Encoder  | 1/16 (92) | 1/8 (183) | 1/4 (366) | 1/2 (732) | Full (1464) |
| :---------------: | :---------: | :-------: | :-------: | :-------: | :-------: | :---------: |
| UniMatch V1       | ResNet-101  |   75.2    |   77.2    |    78.8   |    79.9   |     81.2    |
| AllSpark          |    MiT-B5   |   76.1    |   78.4    |    79.8   |    80.8   |     82.1    |
| SemiVL            |  CLIP-Base  |   84.0    |   85.6    |    86.0   |    86.7   |     87.3    |
| **UniMatch V2**   | DINOv2-Base | **86.3**  | **87.9**  | **88.9**  | **90.0**  |   **90.8**  |

### Cityscapes

| Method            |    Encoder  | 1/16 (186)| 1/8 (372) | 1/4 (744) | 1/2 (1488)|
| :---------------: | :---------: | :-------: | :-------: | :-------: | :-------: |
| UniMatch V1       | ResNet-101  |   76.6    |   77.9    |    79.2   |    79.5   |
| AllSpark          |    MiT-B5   |   78.3    |   79.2    |    80.6   |    81.4   |
| SemiVL            |  CLIP-Base  |   77.9    |   79.4    |    80.3   |    80.6   |
| **UniMatch V2**   | DINOv2-Base | **83.6**  | **84.3**  | **84.5**  | **85.1**  |

### ADE20K

| Method            |    Encoder  | 1/64 (316)| 1/32 (631)|1/16 (1263)| 1/8 (2526)|
| :---------------: | :---------: | :-------: | :-------: | :-------: | :-------: |
| UniMatch V1       | ResNet-101  |   21.6    |   28.1    |    31.5   |    34.6   |
| SemiVL            |  CLIP-Base  |   33.7    |   35.1    |    37.2   |    39.4   |
| **UniMatch V2**   | DINOv2-Base | **38.7**  | **45.0**  | **46.7**  | **49.8**  |

### COCO

| Method            |    Encoder  |1/512 (232)|1/256 (463)|1/128 (925)|1/64 (1849)| 1/32 (3697) |
| :---------------: | :---------: | :-------: | :-------: | :-------: | :-------: | :---------: |
| UniMatch V1       | ResNet-101  |   31.9    |   38.9    |    44.4   |    48.2   |     49.8    |
| AllSpark          |    MiT-B5   |   34.1    |   41.7    |    45.5   |    49.6   |     ---     |
| SemiVL            |  CLIP-Base  | **50.1**  |   52.8    |    53.6   |    55.4   |     56.5    |
| **UniMatch V2**   | DINOv2-Base |    47.9   | **55.8**  | **58.7**  | **60.4**  |   **63.3**  |

### Real-World Large-Scale SSS Setting

In addition to the above traditional SSS settings, we also explore a real-world large-scale setting, where substantial images (*e.g.*, 10K) have already been annotated, and meantime much more unlabeled images (*e.g.*, 100K) are available. It is challenging but highly meaningful.

|  Labeled Data (# Img)  |  + Unlabeled Data (# Img)  |   Improvement  |
| :--------------------: | :------------------------: | :------------: |
| COCO (118K)            | COCO Extra (123K)          |66.4 &rarr; 67.1|
| ADE20K (20K)           | COCO Labeled (118K)        |54.1 &rarr; 54.9|
| ADE20K (20K)           | COCO All (118K + 123K)     |54.1 &rarr; 55.7|
| Cityscapes (3K)        | Cityscapes Extra (20K)     |85.2 &rarr; 85.5|


### More Scenarios

We also apply our UniMatch V2 in the scenarios of semi-supervised [**remote sensing change detection**](https://github.com/LiheYoung/UniMatch-V2/blob/main/remote-sensing).

## Getting Started

### Pre-trained Encoders

[DINOv2-Small](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth) | [DINOv2-Base](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth) | [DINOv2-Large](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)

```
├── ./pretrained
    ├── dinov2_small.pth
    ├── dinov2_base.pth
    └── dinov2_large.pth
```

### Datasets

- Pascal: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)
- ADE20K: [images](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) | [annotations](https://drive.google.com/file/d/1f2a4d_mycaI4JCqz-EAVLXVwb6s5EsWa/view?usp=sharing)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip) | [val2017](http://images.cocodataset.org/zips/val2017.zip) | [masks](https://drive.google.com/file/d/166xLerzEEIbU7Mt1UGut-3-VN41FMUb1/view?usp=sharing)

Please modify your dataset path in configuration files.

**The ADE20K and COCO annotations have already been pre-processed by us. You can use them directly.**

```
├── [Your Pascal Path]
    ├── JPEGImages
    └── SegmentationClass
    
├── [Your Cityscapes Path]
    ├── leftImg8bit
    └── gtFine

├── [Your ADE20K Path]
    ├── images
    │   ├── training
    │   └── validation
    └── annotations
        ├── training
        └── validation

├── [Your COCO Path]
    ├── train2017
    ├── val2017
    └── masks
```

## Training

### UniMatch V2

```bash
# use torch.distributed.launch
sh scripts/train.sh <num_gpu> <port>
# to fully reproduce our results, the <num_gpu> should be set as 4 on all four datasets
# otherwise, you need to adjust the learning rate accordingly

# or use slurm
# sh scripts/slurm_train.sh <num_gpu> <port> <partition>
```

To train on other datasets or splits, please modify
``dataset`` and ``split`` in [train.sh](https://github.com/LiheYoung/UniMatch-V2/blob/main/scripts/train.sh).

### FixMatch

Modify the ``method`` from ``'unimatch_v2'`` to ``'fixmatch'`` in [train.sh](https://github.com/LiheYoung/UniMatch-V2/blob/main/scripts/train.sh).

### Supervised Baseline

Modify the ``method`` from ``'unimatch_v2'`` to ``'supervised'`` in [train.sh](https://github.com/LiheYoung/UniMatch-V2/blob/main/scripts/train.sh). 


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{unimatchv2,
  title={UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Zhao, Zhen and Zhao, Hengshuang},
  journal={TPAMI},
  year={2025}
}
```

---

## Surgical Long-Tail Extensions (this fork)

A **minimal** addition on top of the official UniMatch V2: two togglable
modules for surgical semantic segmentation (Endoscapes-Seg50, CholecSeg8k).
Backbone, SSL stream and data pipeline are unchanged. Default backbone is
**DINOv2-Base (ViT-B/14)**.

### Why not Grounding DINO?

GD is detection-pretrained (Objects365 + GoldG) — weak dense per-patch
representation for DPT, and its text branch does not match surgical class
names. We keep DINOv2 as the encoder.

### Two modules

1. **Long-tail debias** (`util/debias.py`): logit adjustment
   `adj = z - tau * log pi_c` on the EMA teacher's pseudo-labels, where
   `pi_c` is an EMA prior over pseudo-label class counts, initialised from
   the labeled split. Updates use **raw** (un-debiased) argmax + a lower
   confidence threshold so tail classes can still contribute.
   References: Menon et al. ICLR 2021, FreeMatch.
2. **Boundary aux head** (`util/boundary_loss.py`): tiny 3x3 conv on top
   of seg-logits, supervised against the Sobel boundary of GT / pseudo
   masks. Three additions over the vanilla version:
   * **BCE + Dice combo** (`dice_weight: 0.5`) — Dice term tracks the
     IoU of the sparse, line-like boundary set; BCE alone tolerates
     scattered-noise predictions.
   * **CutMix-seam suppression** — pixels within `cutmix_edge_thickness`
     of the strong-aug paste outline are excluded from the loss. Without
     this, the Sobel on the cutmixed mask fires on the GT/pseudo seam
     and feeds a spurious boundary target to the head.
   * **Warmup** (`warmup_epochs: 3`) — no boundary loss until the seg
     head produces non-random predictions, so early-epoch noise does
     not flow back through seg-logits into the backbone.

### Datasets

Both datasets share the same processing protocol (Murali et al. 2021 for
Endoscapes: 5 % image-frequency cut-off, instrument merge → `tool`,
`lymph_node` dropped; CholecSeg8k follows the standard 13-class layout).

> **Which Endoscapes subset?**  CAMMA's Endoscapes release ships three
> independent annotation sets over the **same 201 videos** (~58 k 1-fps
> frames). They are NOT interchangeable:
>
> | Subset                | Annotation type | Annotated frames | Classes | Used here?            |
> | --------------------- | --------------- | :--------------: | :-----: | --------------------- |
> | **Endoscapes-Seg50**  | pixel masks     |       493        |    7    | **Yes — this fork**   |
> | Endoscapes-BBox201    | bounding boxes  |     1 933        |    6    | No (detection only)   |
> | Endoscapes-CVS201     | CVS score (img) |    11 090        |  3 (CVS)| No (classification)   |
>
> The 1 933 number you may see quoted in the CAMMA repo is the
> **BBox201** count — box-level labels that cannot be converted to
> pixel masks. Our DPT-based semantic segmentation pipeline uses only
> **Seg50**'s 493 pixel-mask frames. The remaining 1-fps frames (~58 k −
> 493) serve as the optional unlabeled pool for `endoscapes` (full SSL).

| Dataset             | classes | train pool | val   | test | extra unlabeled pool   |
| ------------------- | :-----: | :--------: | :---: | :--: | :--------------------: |
| `endoscapes_seg50`  |    7    |    343     |   76  |  74  | — (Seg50 frames only)  |
| `endoscapes`        |    7    |    343     |   76  |  74  | + 26 314 (1-fps frames)|
| `cholecseg8k`       |   13    |   6 800    | 1 280 |  —   | — (val used for eval)  |

For Endoscapes, "train pool" is the count of **labeled** Seg50 frames
(only frames with pixel masks); the unlabeled stream is sampled
separately. For CholecSeg8k the train pool is fixed and the
labeled / unlabeled split is taken from it at the requested `ratio`.

**`endoscapes` vs `endoscapes_seg50`**: identical class layout and labeled
set; they differ only in the **unlabeled** stream — `endoscapes_seg50`
keeps unlabeled = leftover Seg50 frames (mu ≈ (1-r)/r), while
`endoscapes` reuses the same labeled set but adds the 26 k 1-fps frame
pool as unlabeled, capped at `mu * n_labeled` (default mu = 7).

**How the `endoscapes` (full) split is constructed.** Three rules:

1. **Video-level split is fixed by CAMMA** (120 / 41 / 40 train / val /
   test videos, Murali et al. 2021). We never move a frame across this
   boundary — `pair_by_stem` runs per split independently.
2. **Labeled stream** = `ratio × 343` Seg50 frames (the *only* frames
   with pixel masks); chosen by deterministic `seed=42` shuffle, take
   prefix. **Unlabeled stream** = leftover Seg50 frames **+** 1-fps
   train-video frames that are NOT in Seg50 (CAMMA's `train/` minus
   `train_seg/` = ~26 314). The combined pool is capped at `mu *
   n_labeled` (default mu = 7).
3. **seed = 42** controls (a) which Seg50 frames become labeled, and
   (b) which slice of the 1-fps pool is sampled. The same seed across
   all ratios means labeled sets are NOT strictly nested (10 % labels
   are not necessarily a subset of 25 % labels) — they share the same
   shuffled order, then take different prefixes.

Val and test sets are **always the full Seg50 labeled subset** (76 / 74
frames) — these are evaluation-only and never sub-sampled.

**Per-ratio split sizes (seed 42)**. labeled = `ratio × 343` ; unlabeled
column shows endoscapes_seg50 → endoscapes (capped mu=7) :

| ratio | labeled | unlabeled (`endoscapes_seg50`) | unlabeled (`endoscapes`, mu=7) |
| :---: | :-----: | :----------------------------: | :----------------------------: |
| 0.10  |    34   |              309               |              238               |
| 0.20  |    69   |              274               |              483               |
| 0.25  |    86   |              257               |              602               |
| 0.30  |   103   |              240               |              721               |
| 0.50  |   172   |              171               |             1 204              |

CholecSeg8k per-ratio split sizes (seed 42, total pool = 6 800):

| ratio | labeled | unlabeled |
| :---: | :-----: | :-------: |
| 0.10  |   680   |   6 120   |
| 0.20  |  1 360  |   5 440   |
| 0.25  |  1 700  |   5 100   |
| 0.30  |  2 040  |   4 760   |
| 0.50  |  3 400  |   3 400   |

Generated by `tools/gen_ratio_splits.py` from the existing 0.25 split
(no separate 1-fps pool). Val = 1 280; CholecSeg8k has no held-out test
set in this release, so `test.sh` automatically falls back to val.

### Class layout

```
endoscapes_seg50 / endoscapes (7):
  0 background     1 cystic_plate    2 calot_triangle   3 cystic_artery
  4 cystic_duct    5 gallbladder     6 tool             (255 = ignore)

cholecseg8k (13):
  0 background           1 abdominal_wall     2 liver
  3 gastrointestinal_tract  4 fat              5 grasper
  6 connective_tissue    7 blood              8 cystic_duct
  9 l_hook_electrocautery   10 gallbladder    11 hepatic_vein
  12 liver_ligament       (255 = ignore)
```

The lists above are the source of truth in `util/classes.py`; the trainer
indexes IoU / loss by these names. **Do not reorder** without re-running
the prepare scripts.

### Four reproducible variants

| Variant         | `--debias` | `--boundary` | `--cls-head` |
| --------------- | :--------: | :----------: | :----------: |
| `base`          |            |              |              |
| `debias`        |     v      |              |              |
| `boundary`      |            |      v       |              |
| `cls`           |            |              |       v      |
| `full`          |     v      |      v       |              |
| `debias_cls`    |     v      |              |       v      |
| `boundary_cls`  |            |      v       |       v      |
| `full_cls`      |     v      |      v       |       v      |

The `--cls-head` flag attaches an image-level multi-label auxiliary
head and adds a classification loss on the labeled stream. The head is
**isolated from the segmentation stream** by four guards designed for
the extreme long-tail in surgical data:

1. **ML-Decoder head** (Ridnik et al., WACV 2023, `decoder_type: ml_decoder`):
   per-class learnable query tokens cross-attend to backbone patch
   features. Each class has its own feature channel, so rare classes do
   not compete with common ones for CLS-token capacity. Set
   `decoder_type: mlp` to fall back to a 2-layer MLP on the CLS token.
2. **Asymmetric Loss** (Ben-Baruch et al., ICCV 2021, `loss_type: asl`):
   different focusing for positives vs. negatives, with hard-negative
   probability shifting. Robust to extreme imbalance — replaces the
   prior-weighted BCE that pushed `pos_weight` to >30x on the rarest
   class and destabilised the head. Set `loss_type: bce` for the legacy
   BCE path.
3. **Backbone detach** (`detach_backbone: true`): the cls head's
   gradient does NOT propagate into the backbone. The head learns on
   top of a fixed CLS / patch representation, so even if it fails to
   learn the rarest class it cannot poison the segmentation features.
4. **Warmup** (`warmup_epochs: 5`): no cls loss for the first N epochs,
   so the segmentation head and the EMA prior stabilise before any
   image-level supervision is added.

Test time: the head's predictions are written to `test_metrics_cls.csv`
(per-class AP / F1, plus mAP / mF1).

### Training, testing, plotting

> See **[docs/RUN.md](docs/RUN.md)** for the full run matrix — every
> (dataset × rate × variant) command, log inspection tips, and a
> copy-paste cheat-sheet that runs the entire 40-run ablation in one
> tmux. The summary below is for a quick single-run.

```bash
RATE=0.25 sh scripts/train.sh <NGPU> <PORT> <variant> <dataset>
RATE=0.25 sh scripts/train.sh 1 29500 full endoscapes_seg50
```

Tuning batch / lr / epochs / crop without editing YAML:

```bash
BS=2 LR=5e-6                  sh scripts/train.sh 1 29500 debias endoscapes_seg50
BS=8 LR=2e-5 EPOCHS=120       sh scripts/train.sh 1 29500 full   cholecseg8k
```

Outputs: `${EXP_ROOT}/<dataset>/unimatch_v2_<tag>/` containing
`latest.pth`, `best.pth`, `train_log.csv`, `out.log`.

### Testing

```bash
sh scripts/test.sh <variant> <dataset>
# pass the SAME env vars used during training so the tag matches
BS=2 LR=5e-6 sh scripts/test.sh debias endoscapes_seg50
```

Writes `test_metrics.csv` with per-class IoU / Dice / Prec / Rec / PA / AP.

### Plotting

```bash
sh scripts/plot.sh <dataset>                       # 4-variant comparison
sh scripts/plot.sh endoscapes_seg50 _bs2_lr5e-6    # sweep
```

### Full 4-variant ablation (copy-paste)

Train, evaluate and plot all four variants in one go on Endoscapes-Seg50
with a single 24 GB GPU (BS=2, LR=5e-6 to fit memory; adjust for your card):

```bash
tmux new -s u2s
export DATA_ROOT=/data/test SPLITS=/data/splits EXP_ROOT=/data/exp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for v in base debias boundary full; do
    BS=2 LR=5e-6 sh scripts/train.sh 1 29500 $v endoscapes_seg50
    BS=2 LR=5e-6 sh scripts/test.sh  $v          endoscapes_seg50
done
sh scripts/plot.sh endoscapes_seg50 _bs2_lr5e-6
```

For CholecSeg8k swap the dataset string. For a 40 GB card drop `BS=2 LR=5e-6`
to use the YAML defaults (BS=4 LR=1e-5).

### Crop / transform notes

`crop_size: 490` (= 14 x 35) fits 480-tall surgical frames through the
`resize(0.5, 2.0) -> random crop` pipeline without heavy padding.
ImageNet normalisation kept (required by DINOv2). Strong augs
(ColorJitter 0.5/0.5/0.5/0.25, RandomGrayscale 0.2, GaussianBlur 0.5) are
the official UniMatch V2 setting — appropriate for surgical scenes.

### Migration to a new dataset

1. Add `configs/<your_dataset>.yaml` (copy `endoscapes_seg50.yaml`,
   change `nclass` / `data_root`).
2. Add the class list under the same key in `util/classes.py`.
3. Prepare splits at `${SPLITS}/unimatch_splits_<your_dataset>_0.25_seed42/`:
   `labeled.txt` (`<img>\t<mask>`), `unlabeled.txt` (`<img>` only),
   `val.txt` / `test.txt`.
