# Run Manual - UniMatch-V2 Surgical

> ⚠️ **THIS FILE IS OUTDATED — USE `docs/RUN.html` INSTEAD.**
>
> The commands below still describe the *legacy two-stage* workflow
> (`AFFINITY_WARMSTART` + Stage-1 `train_affinity_per_ds.py`). The current
> mainline uses **single-stage joint text training** via
> `VISUAL_ADAPTER=1 JOINT_TEXT_STAGE=1` — no separate Stage-1 is needed.
>
> Other stale items in this file:
> - `EXP_ROOT` is now `/root/autodl-tmp/exp` (not `/localdisk-tmp/exp`)
> - `SPLITS` is now `/root/autodl-tmp/data/autonomous_surgery/splits` (not `/data/splits`)
> - `VISUAL_ADAPTER`, `JOINT_TEXT_STAGE`, `EDGE_ENHANCE` env switches are not
>   documented below; they are required for current runs.
>
> Treat this file as historical reference only. The canonical, code-aligned
> manual is **`docs/RUN.html`**.

This runbook keeps only the commands needed to train, test, and visualize each dataset. The documented model categories are:

> **Note.** All server-side dataset paths in this document use the new AutoDL data root
> `/root/autodl-tmp/data/autonomous_surgery/public_data`. See the
> [Datasets & Paths](#0-datasets--paths) section below for a single-place
> inventory of every input/output location.

## 0. Datasets & Paths

### 0.1 Surgical datasets (used in the reported experiments)

| Dataset key (config) | `data_root` on server | Preprocessing |
|---|---|---|
| `endoscapes_seg50` | `/root/autodl-tmp/data/autonomous_surgery/public_data/endoscapes_seg50_processed` | Upload processed folder from old `data/test`. |
| `cholecseg8k` | `/root/autodl-tmp/data/autonomous_surgery/public_data/cholecseg8k_processed` | Do not upload processed data; raw data already exists on the target server. Run `tools/prepare_cholecseg8k.py --apply` there. |
| `endovis2018` | `/root/autodl-tmp/data/autonomous_surgery/public_data/endovis2018_processed` | Upload processed folder from old `data/test`. |
| `endovis2017_parts` | `/root/autodl-tmp/data/autonomous_surgery/public_data/endovis2017_parts_processed` | Upload processed folder from old `data/test`. |
| `endovis2017_type` | `/root/autodl-tmp/data/autonomous_surgery/public_data/endovis2017_type_processed` | Upload processed folder from old `data/test`. |
| `needle` | `/root/autodl-tmp/data/autonomous_surgery/public_data/unimatchv2_needle_3class` | Upload processed folder from old `data/test` if used. |

The `data_root` value above is hard-coded into the matching YAML in `configs/<dataset>.yaml` and read by the trainer/tester. The non-surgical configs (`ade20k.yaml`, `cityscapes.yaml`, `coco.yaml`, `pascal.yaml`) still hold `Your/<NAME>/Path` placeholders and are not part of the surgical pipeline.

### 0.2 Splits (labeled / unlabeled / val / test)

```bash
export SPLITS=/data/splits
SPLIT_ROOT=${SPLITS}/unimatch_splits_${DATASET}_${RATE}_seed42
#   labeled.txt   unlabeled.txt   val.txt   [test.txt]
```

Example for `endoscapes_seg50` at rate 0.25:

```text
/data/splits/unimatch_splits_endoscapes_seg50_0.25_seed42/labeled.txt
/data/splits/unimatch_splits_endoscapes_seg50_0.25_seed42/unlabeled.txt
/data/splits/unimatch_splits_endoscapes_seg50_0.25_seed42/val.txt
```

### 0.3 Stage-I (text-affinity) artefacts

| Artefact | Path |
|---|---|
| SigLIP train manifests | `/data/pretrained/siglip_train/` |
| Merged train manifest + `global_classes.json` | `/data/pretrained/siglip_train/merged/` |
| Per-dataset val manifests | `/data/pretrained/siglip_train/merged_val/` |
| Stage-I checkpoint (per dataset) | `/data/pretrained/siglip_train/affinity_per_ds/<ds>/affinity_<ds>.pt` |
| Stage-I visualizations | `/data/pretrained/siglip_train/vis_stage1/<ds>/` |

### 0.4 Backbones & experiment outputs

| Item | Path |
|---|---|
| DINOv2-B/14 pretrained weights | `./pretrained/dinov2_vitb14_pretrain.pth` (in repo root) |
| Stage-II run directory | `${EXP_ROOT}/<dataset>/unimatch_v2_<variant>_r<RATE>[_bs<BS>][_lr<LR>][_ep<EPOCHS>][_cr<CROP>]/` |
| Training log | `<run-dir>/out.log` |
| Best / latest checkpoints | `<run-dir>/best.pth`, `<run-dir>/latest.pth` |
| Stage-II visualizations | `<run-dir>/vis_stage2/` |

`EXP_ROOT` is set to `/localdisk-tmp/exp` in the common setup below.

### 0.5 Upload Code and Datasets to a New AutoDL Server

Use BitaHub as the relay. Replace `BUCKET` with your BitaHub bucket name and
run `bita login` once on both the source machine and the target AutoDL server.

```bash
export BITA_EP=https://www.bitahub.com
export BUCKET=<your_bucket_name>
bita login -u <username> -p <password> -e "$BITA_EP"
```

#### Upload code from the local machine

From the local checkout root:

```bash
cd /mnt/d/study/code/UniMatch-V2
BITA_BK=$BUCKET PREFIX=code/ ./scripts/package_and_upload.sh upload
```

If you only want to create the zip first without uploading:

```bash
./scripts/package_and_upload.sh
```

PowerShell equivalent from Windows:

```powershell
cd D:\study\code\UniMatch-V2
.\scripts\package_and_upload.ps1 -Upload -Bucket <your_bucket_name> -Prefix code/
```

On the new AutoDL server, download and unpack the uploaded code zip:

```bash
mkdir -p /root/autodl-tmp/code
cd /root/autodl-tmp/code

bita download -e "$BITA_EP" -b "$BUCKET" \
  -o code/<uploaded_zip_name>.zip \
  -l /root/autodl-tmp/code/<uploaded_zip_name>.zip

unzip -q /root/autodl-tmp/code/<uploaded_zip_name>.zip -d /root/autodl-tmp/code
cd /root/autodl-tmp/code/UniMatch-V2
chmod +x scripts/*.sh
```

#### Download processed datasets from Bita to the new server

Use this on the new AutoDL server after the processed dataset tarballs have
already been uploaded to Bita. CholecSeg8k is intentionally not downloaded here,
because its raw dataset already exists on the target server and will be
processed there.

```bash
export BITA_EP=https://www.bitahub.com
export BUCKET=<your_bucket_name>
export NEW_DATA_ROOT=/root/autodl-tmp/data/autonomous_surgery/public_data

: "${BITA_EP:?set BITA_EP, e.g. export BITA_EP=https://www.bitahub.com}"
: "${BUCKET:?set BUCKET to your Bita bucket name}"

bita login -u <username> -p <password> -e "$BITA_EP"
mkdir -p "$NEW_DATA_ROOT" /tmp/autonomous_surgery_dataset_pack

for DS in \
  endoscapes_seg50_processed \
  endovis2018_processed \
  endovis2017_parts_processed \
  endovis2017_type_processed \
  unimatchv2_needle_3class
do
  bita download -e "$BITA_EP" -b "$BUCKET" \
    -o datasets/autonomous_surgery/public_data/${DS}.tar.gz \
    -l /tmp/autonomous_surgery_dataset_pack/${DS}.tar.gz || exit 1
  tar -C "$NEW_DATA_ROOT" -xzf /tmp/autonomous_surgery_dataset_pack/${DS}.tar.gz
done
```

Then prepare CholecSeg8k on the target server from the raw dataset that is
already there:

```bash
python tools/prepare_cholecseg8k.py \
  --root <existing_cholecseg8k_raw_root_on_target_server> \
  --out "$NEW_DATA_ROOT/cholecseg8k_processed" \
  --apply
```

Verify the resulting data paths:

```bash
find "$NEW_DATA_ROOT" -maxdepth 2 -type d | sort
grep '^data_root:' configs/endoscapes_seg50.yaml configs/cholecseg8k.yaml \
  configs/endovis2018.yaml configs/endovis2017_parts.yaml configs/endovis2017_type.yaml
```

#### Optional: upload processed datasets from old `data/test` to Bita

Upload all processed surgical datasets except CholecSeg8k. CholecSeg8k raw data
already exists on the target server and should be processed there instead of
uploaded again.

From the machine that has the old `data/test` directory:

```bash
export OLD_DATA_ROOT=/data/test
export PACK_DIR=/tmp/autonomous_surgery_dataset_pack
mkdir -p "$PACK_DIR"

for DS in \
  endoscapes_seg50_processed \
  endovis2018_processed \
  endovis2017_parts_processed \
  endovis2017_type_processed \
  unimatchv2_needle_3class
do
  [ -d "$OLD_DATA_ROOT/$DS" ] || { echo "[skip] missing $OLD_DATA_ROOT/$DS"; continue; }
  tar -C "$OLD_DATA_ROOT" -czf "$PACK_DIR/${DS}.tar.gz" "$DS"
  bita upload -e "$BITA_EP" -b "$BUCKET" \
    -o datasets/autonomous_surgery/public_data/${DS}.tar.gz \
    -l "$PACK_DIR/${DS}.tar.gz"
done
```

| Category | Variant | Meaning |
|---|---|---|
| Official UniMatch-V2 baseline | `base` | Original UniMatch-V2 segmentation model. |
| Text enhancement | `affinity_min` | Adds the trained SigLIP/DINOv2 affinity prior. |
| Boundary branch | `boundary` | Adds the segmentation boundary branch. |
| Bias correction | `debias` | Adds class-prior bias correction. |

Legacy variants that still exist in `scripts/train.sh` are for internal debugging only and are not part of the reported module split.

## 1. Common Setup

Run once in every server shell:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate unimatchv2

cd /root/autodl-tmp/code/UniMatch-V2
chmod +x scripts/*.sh

export DATA_ROOT=/root/autodl-tmp/data/autonomous_surgery/public_data
export SPLITS=/data/splits
export EXP_ROOT=/localdisk-tmp/exp
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$EXP_ROOT"
```

All segmentation commands use this output pattern:

```text
$EXP_ROOT/<dataset>/unimatch_v2_<variant>_r<RATE>[_bs<BS>][_lr<LR>][_ep<EPOCHS>][_cr<CROP>]
```

Testing must use the same `RATE`, `BS`, `LR`, `EPOCHS`, and `CROP` values used for training, because these values are part of the checkpoint directory name.

## 2. Text Enhancement Stage 1

Only `affinity_min` needs this stage. Run it for the text-supported datasets: `endoscapes_seg50`, `cholecseg8k`, `endovis2018`, `endovis2017_parts`, and `endovis2017_type`. `needle` has no text-enhancement path by default.

```bash
RATE=0.25

python tools/build_siglip_manifest.py \
  --splits-root /data/splits \
  --data-roots \
    endoscapes_seg50=$DATA_ROOT/endoscapes_seg50_processed \
    cholecseg8k=$DATA_ROOT/cholecseg8k_processed \
    endovis2018=$DATA_ROOT/endovis2018_processed \
    endovis2017_parts=$DATA_ROOT/endovis2017_parts_processed \
    endovis2017_type=$DATA_ROOT/endovis2017_type_processed \
  --rate $RATE --seed 42 \
  --out /data/pretrained/siglip_train
```

Create validation manifests for Stage-1 visualization and validation:

```bash
for DS in endoscapes_seg50 cholecseg8k endovis2018 endovis2017_parts endovis2017_type; do
  mkdir -p /data/splits/$DS
  ln -sfn /data/splits/unimatch_splits_${DS}_${RATE}_seed42/val.txt /data/splits/$DS/val.txt
done

python tools/build_merged_manifest.py \
  --manifest-dir /data/pretrained/siglip_train \
  --out-dir /data/pretrained/siglip_train/merged

python tools/build_val_manifest.py \
  --splits-root /data/splits \
  --data-root $DATA_ROOT \
  --global-info /data/pretrained/siglip_train/merged/global_classes.json \
  --out-dir /data/pretrained/siglip_train/merged_val
```

Train one affinity checkpoint per dataset:

```bash
python tools/train_affinity_per_ds.py \
  --manifest-dir /data/pretrained/siglip_train \
  --val-dir /data/pretrained/siglip_train/merged_val \
  --out-dir /data/pretrained/siglip_train/affinity_per_ds \
  --model google/siglip2-base-patch16-256 \
  --vision-backbone dinov2 --dinov2-input-size 518 \
  --epochs 40 --bs 16 --lr 1e-3 \
  --lr-vision 1e-5 --lr-text 1e-5 \
  --lambda-patch 1.0 --patch-grid 37 \
  --topk 5 --eval-every 5
```

The checkpoint consumed by `affinity_min` is:

```text
/data/pretrained/siglip_train/affinity_per_ds/<dataset>/affinity_<dataset>.pt
```

## 3. Per-Dataset Commands

The following examples use `RATE=0.25`, `BS=2`, and `LR=5e-6`. Change them only when you want a different labeled ratio or training recipe.

### 3.1 Endoscapes-Seg50

```bash
DS=endoscapes_seg50
RATE=0.25
BS=2
LR=5e-6
AFF=/data/pretrained/siglip_train/affinity_per_ds/$DS/affinity_$DS.pt

# Train
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29500 base $DS
RATE=$RATE BS=$BS LR=$LR AFFINITY_WARMSTART=$AFF sh scripts/train.sh 1 29501 affinity_min $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29502 boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29503 debias $DS

# Test
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh base $DS
RATE=$RATE BS=$BS LR=$LR AFFINITY_WARMSTART=$AFF sh scripts/test.sh affinity_min $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh debias $DS

# Visualize Stage 1 text affinity
python tools/vis_stage1.py --ckpt $AFF --manifest /data/pretrained/siglip_train/merged_val/manifest_${DS}_val.jsonl --out-dir /data/pretrained/siglip_train/vis_stage1/$DS --num-images 10

# Visualize segmentation predictions
RUN=$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}
python tools/vis_stage2.py --ckpt $RUN/best.pth --config configs/$DS.yaml --val-id-path $SPLITS/unimatch_splits_${DS}_${RATE}_seed42/val.txt --out-dir $RUN/vis_stage2 --num-images 10

# Plot training curves
python tools/plot_train_log.py --run base:$EXP_ROOT/$DS/unimatch_v2_base_r${RATE}_bs${BS}_lr${LR}/out.log --run text:$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}/out.log --run boundary:$EXP_ROOT/$DS/unimatch_v2_boundary_r${RATE}_bs${BS}_lr${LR}/out.log --run debias:$EXP_ROOT/$DS/unimatch_v2_debias_r${RATE}_bs${BS}_lr${LR}/out.log --out $EXP_ROOT/$DS/train_curves_${RATE}.png
```

### 3.2 CholecSeg8k

```bash
DS=cholecseg8k
RATE=0.25
BS=2
LR=5e-6
AFF=/data/pretrained/siglip_train/affinity_per_ds/$DS/affinity_$DS.pt

# Train
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29510 base $DS
RATE=$RATE BS=$BS LR=$LR AFFINITY_WARMSTART=$AFF sh scripts/train.sh 1 29511 affinity_min $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29512 boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29513 debias $DS

# Test
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh base $DS
RATE=$RATE BS=$BS LR=$LR AFFINITY_WARMSTART=$AFF sh scripts/test.sh affinity_min $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh debias $DS

# Visualize Stage 1 text affinity
python tools/vis_stage1.py --ckpt $AFF --manifest /data/pretrained/siglip_train/merged_val/manifest_${DS}_val.jsonl --out-dir /data/pretrained/siglip_train/vis_stage1/$DS --num-images 10

# Visualize segmentation predictions
RUN=$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}
python tools/vis_stage2.py --ckpt $RUN/best.pth --config configs/$DS.yaml --val-id-path $SPLITS/unimatch_splits_${DS}_${RATE}_seed42/val.txt --out-dir $RUN/vis_stage2 --num-images 10

# Plot training curves
python tools/plot_train_log.py --run base:$EXP_ROOT/$DS/unimatch_v2_base_r${RATE}_bs${BS}_lr${LR}/out.log --run text:$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}/out.log --run boundary:$EXP_ROOT/$DS/unimatch_v2_boundary_r${RATE}_bs${BS}_lr${LR}/out.log --run debias:$EXP_ROOT/$DS/unimatch_v2_debias_r${RATE}_bs${BS}_lr${LR}/out.log --out $EXP_ROOT/$DS/train_curves_${RATE}.png
```

### 3.3 EndoVis2018

```bash
DS=endovis2018
RATE=0.25
BS=2
LR=5e-6
AFF=/data/pretrained/siglip_train/affinity_per_ds/$DS/affinity_$DS.pt

# Train
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29520 base $DS
RATE=$RATE BS=$BS LR=$LR AFFINITY_WARMSTART=$AFF sh scripts/train.sh 1 29521 affinity_min $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29522 boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29523 debias $DS

# Test
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh base $DS
RATE=$RATE BS=$BS LR=$LR AFFINITY_WARMSTART=$AFF sh scripts/test.sh affinity_min $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh debias $DS

# Visualize Stage 1 text affinity
python tools/vis_stage1.py --ckpt $AFF --manifest /data/pretrained/siglip_train/merged_val/manifest_${DS}_val.jsonl --out-dir /data/pretrained/siglip_train/vis_stage1/$DS --num-images 10

# Visualize segmentation predictions
RUN=$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}
python tools/vis_stage2.py --ckpt $RUN/best.pth --config configs/$DS.yaml --val-id-path $SPLITS/unimatch_splits_${DS}_${RATE}_seed42/val.txt --out-dir $RUN/vis_stage2 --num-images 10

# Plot training curves
python tools/plot_train_log.py --run base:$EXP_ROOT/$DS/unimatch_v2_base_r${RATE}_bs${BS}_lr${LR}/out.log --run text:$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}/out.log --run boundary:$EXP_ROOT/$DS/unimatch_v2_boundary_r${RATE}_bs${BS}_lr${LR}/out.log --run debias:$EXP_ROOT/$DS/unimatch_v2_debias_r${RATE}_bs${BS}_lr${LR}/out.log --out $EXP_ROOT/$DS/train_curves_${RATE}.png
```

### 3.4 EndoVis2017 Parts

```bash
DS=endovis2017_parts
RATE=0.25
BS=2
LR=5e-6
AFF=/data/pretrained/siglip_train/affinity_per_ds/$DS/affinity_$DS.pt

# Train
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29530 base $DS
RATE=$RATE BS=$BS LR=$LR AFFINITY_WARMSTART=$AFF sh scripts/train.sh 1 29531 affinity_min $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29532 boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29533 debias $DS

# Test
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh base $DS
RATE=$RATE BS=$BS LR=$LR AFFINITY_WARMSTART=$AFF sh scripts/test.sh affinity_min $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh debias $DS

# Visualize Stage 1 text affinity
python tools/vis_stage1.py --ckpt $AFF --manifest /data/pretrained/siglip_train/merged_val/manifest_${DS}_val.jsonl --out-dir /data/pretrained/siglip_train/vis_stage1/$DS --num-images 10

# Visualize segmentation predictions
RUN=$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}
python tools/vis_stage2.py --ckpt $RUN/best.pth --config configs/$DS.yaml --val-id-path $SPLITS/unimatch_splits_${DS}_${RATE}_seed42/val.txt --out-dir $RUN/vis_stage2 --num-images 10

# Plot training curves
python tools/plot_train_log.py --run base:$EXP_ROOT/$DS/unimatch_v2_base_r${RATE}_bs${BS}_lr${LR}/out.log --run text:$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}/out.log --run boundary:$EXP_ROOT/$DS/unimatch_v2_boundary_r${RATE}_bs${BS}_lr${LR}/out.log --run debias:$EXP_ROOT/$DS/unimatch_v2_debias_r${RATE}_bs${BS}_lr${LR}/out.log --out $EXP_ROOT/$DS/train_curves_${RATE}.png
```

### 3.5 EndoVis2017 Type

```bash
DS=endovis2017_type
RATE=0.25
BS=2
LR=5e-6
AFF=/data/pretrained/siglip_train/affinity_per_ds/$DS/affinity_$DS.pt

# Train
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29540 base $DS
RATE=$RATE BS=$BS LR=$LR AFFINITY_WARMSTART=$AFF sh scripts/train.sh 1 29541 affinity_min $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29542 boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29543 debias $DS

# Test
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh base $DS
RATE=$RATE BS=$BS LR=$LR AFFINITY_WARMSTART=$AFF sh scripts/test.sh affinity_min $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh debias $DS

# Visualize Stage 1 text affinity
python tools/vis_stage1.py --ckpt $AFF --manifest /data/pretrained/siglip_train/merged_val/manifest_${DS}_val.jsonl --out-dir /data/pretrained/siglip_train/vis_stage1/$DS --num-images 10

# Visualize segmentation predictions
RUN=$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}
python tools/vis_stage2.py --ckpt $RUN/best.pth --config configs/$DS.yaml --val-id-path $SPLITS/unimatch_splits_${DS}_${RATE}_seed42/val.txt --out-dir $RUN/vis_stage2 --num-images 10

# Plot training curves
python tools/plot_train_log.py --run base:$EXP_ROOT/$DS/unimatch_v2_base_r${RATE}_bs${BS}_lr${LR}/out.log --run text:$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}/out.log --run boundary:$EXP_ROOT/$DS/unimatch_v2_boundary_r${RATE}_bs${BS}_lr${LR}/out.log --run debias:$EXP_ROOT/$DS/unimatch_v2_debias_r${RATE}_bs${BS}_lr${LR}/out.log --out $EXP_ROOT/$DS/train_curves_${RATE}.png
```

### 3.6 Needle

`needle` uses the same segmentation scripts but does not have a Stage-1 text checkpoint by default, so run only `base`, `boundary`, and `debias`.

```bash
DS=needle
RATE=0.25
BS=2
LR=5e-6

# Train
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29550 base $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29552 boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/train.sh 1 29553 debias $DS

# Test
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh base $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh boundary $DS
RATE=$RATE BS=$BS LR=$LR sh scripts/test.sh debias $DS

# Visualize segmentation predictions
RUN=$EXP_ROOT/$DS/unimatch_v2_base_r${RATE}_bs${BS}_lr${LR}
python tools/vis_stage2.py --ckpt $RUN/best.pth --config configs/$DS.yaml --val-id-path $SPLITS/unimatch_splits_${DS}_${RATE}_seed42/val.txt --out-dir $RUN/vis_stage2 --num-images 10

# Optional video-style needle visualization
python tools/test_needle_video.py --config configs/needle.yaml --checkpoint $RUN/best.pth --split-file $SPLITS/unimatch_splits_needle_${RATE}_seed42/test.txt --out-root $RUN/video_vis --save-masks

# Plot training curves
python tools/plot_train_log.py --run base:$EXP_ROOT/$DS/unimatch_v2_base_r${RATE}_bs${BS}_lr${LR}/out.log --run boundary:$EXP_ROOT/$DS/unimatch_v2_boundary_r${RATE}_bs${BS}_lr${LR}/out.log --run debias:$EXP_ROOT/$DS/unimatch_v2_debias_r${RATE}_bs${BS}_lr${LR}/out.log --out $EXP_ROOT/$DS/train_curves_${RATE}.png
```

## 4. Quick Checks

Before a full run, use one epoch to check paths and checkpoints:

```bash
RATE=0.25 BS=2 LR=5e-6 EPOCHS=1 sh scripts/train.sh 1 29590 base endoscapes_seg50
RATE=0.25 BS=2 LR=5e-6 EPOCHS=1 sh scripts/test.sh base endoscapes_seg50
```

If testing says the checkpoint is missing, first check that the test command uses the same `RATE`, `BS`, `LR`, `EPOCHS`, and `CROP` values as the training command.
