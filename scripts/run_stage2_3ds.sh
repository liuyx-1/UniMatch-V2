#!/bin/bash
# Sequentially run Stage 2 (UniMatch-V2 segmentation) on 3 datasets:
#   endovis2017_parts, endovis2017_type, endovis2018
# Requires Stage 1 affinity_<ds>.pt to already exist.
#
# Usage:
#   sh scripts/run_stage2_3ds.sh             # default: variant=affinity, rate=0.25
#   VARIANTS="base affinity" sh scripts/run_stage2_3ds.sh
#   RATES="0.10 0.25 0.50"  sh scripts/run_stage2_3ds.sh
#
# Env overrides (any optional):
#   DATASETS=  space-separated list (default 3 datasets)
#   VARIANTS=  default "affinity"  (others: base / debias / boundary / full / affinity_min / affinity)
#   RATES=     default "0.25"      (e.g. "0.10 0.20 0.25 0.30 0.50")
#   BS=        default 2
#   LR=        default 5e-6
#   EPOCHS=    default config value (unset)
#   AFFINITY_DIR= default /data/pretrained/siglip_train/affinity_per_ds
#   STAGE2_LOG_DIR= default /data/code/exp/_stage2_logs

set -e

# ---- Env init ----
# Use POSIX-compatible '.' instead of 'source' so the script works under
# both bash and dash (Debian/Ubuntu /bin/sh -> dash by default).
. /opt/conda/etc/profile.d/conda.sh
conda activate unimatchv2
cd /data/code/UniMatch-V2
chmod +x scripts/*.sh

export DATA_ROOT=${DATA_ROOT:-/data/test}
export SPLITS=${SPLITS:-/data/splits}
export EXP_ROOT=${EXP_ROOT:-/localdisk-tmp/exp}
export EXP_ARCHIVE=${EXP_ARCHIVE:-/data/code/exp}
mkdir -p "$EXP_ROOT" "$EXP_ARCHIVE"

# HF SigLIP + local DINOv2 (Stage-1-trained backbone is loaded automatically when --affinity-warmstart is set)
export HF_HOME=${HF_HOME:-/localdisk-tmp/cache/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export DINOV2_VITB14_PATH=${DINOV2_VITB14_PATH:-/data/code/UniMatch-V2-manifold/pretrained/dinov2_vitb14_pretrain.pth}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Job matrix ----
DATASETS=${DATASETS:-"endovis2017_parts endovis2017_type endovis2018"}
VARIANTS=${VARIANTS:-"affinity"}
RATES=${RATES:-"0.25"}
BS=${BS:-2}
LR=${LR:-5e-6}
AFFINITY_DIR=${AFFINITY_DIR:-/data/pretrained/siglip_train/affinity_per_ds}
STAGE2_LOG_DIR=${STAGE2_LOG_DIR:-/data/code/exp/_stage2_logs}
mkdir -p "$STAGE2_LOG_DIR"

START_PORT=29500
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="$STAGE2_LOG_DIR/run_$TS.summary"
echo "[start] $(date)" > "$SUMMARY"
echo "matrix: datasets=[$DATASETS]  variants=[$VARIANTS]  rates=[$RATES]  bs=$BS  lr=$LR" >> "$SUMMARY"

# ---- Run jobs ----
JOB_IDX=0
for DS in $DATASETS; do
    AFF_PATH="$AFFINITY_DIR/$DS/affinity_$DS.pt"
    for RATE in $RATES; do
        for V in $VARIANTS; do
            JOB_IDX=$((JOB_IDX + 1))
            PORT=$((START_PORT + JOB_IDX))
            JOB_TAG="${DS}_r${RATE}_${V}_bs${BS}_lr${LR}"
            LOG="$STAGE2_LOG_DIR/${TS}_${JOB_TAG}.log"

            echo ""
            echo "==========================================================="
            echo "[$(date '+%H:%M:%S')] job $JOB_IDX: $JOB_TAG"
            echo "  PORT=$PORT  LOG=$LOG"
            echo "==========================================================="

            # Check Stage 1 ckpt for affinity variants
            if [ "$V" = "affinity" ] || [ "$V" = "affinity_min" ]; then
                if [ ! -f "$AFF_PATH" ]; then
                    echo "[SKIP] Stage 1 ckpt missing: $AFF_PATH" | tee -a "$SUMMARY"
                    continue
                fi
                EXPORT_AFF="AFFINITY_WARMSTART=$AFF_PATH"
            else
                EXPORT_AFF=""
            fi

            # Build & run
            CMD="RATE=$RATE BS=$BS LR=$LR $EXPORT_AFF sh scripts/train.sh 1 $PORT $V $DS"
            echo "[cmd] $CMD" | tee -a "$SUMMARY"

            START_T=$SECONDS
            eval $CMD 2>&1 | tee "$LOG"
            DT=$((SECONDS - START_T))

            # Pull final mIoU from train_log.csv if it exists
            CKPT_DIR="$EXP_ROOT/$DS/unimatch_v2_${V}_r${RATE}_bs${BS}_lr${LR}"
            FINAL_LINE=$(tail -1 "$CKPT_DIR/train_log.csv" 2>/dev/null || echo "no-csv")
            echo "[$(date '+%H:%M:%S')] done $JOB_TAG  duration=${DT}s  final=$FINAL_LINE" | tee -a "$SUMMARY"
        done
    done
done

echo ""
echo "==========================================================="
echo "[end] $(date)" | tee -a "$SUMMARY"
echo "Summary: $SUMMARY"
echo "Logs:    $STAGE2_LOG_DIR/"
echo "Ckpts:   $EXP_ROOT/{$DATASETS}/unimatch_v2_*_r*_bs${BS}_lr${LR}/"
echo "==========================================================="
