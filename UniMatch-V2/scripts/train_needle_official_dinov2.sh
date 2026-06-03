#!/usr/bin/env bash
# Train + test Needle (3-class) with the OFFICIAL UniMatch-V2 protocol:
#   Backbone  : DINOv2-B/14   (full fine-tune, NO visual adapter)
#   Decoder   : DPT
#   SSL       : weak/strong consistency + EMA teacher (default UniMatch-V2)
#
# Pretrained weights are expected at:
#   ./pretrained/dinov2_vitb14_pretrain.pth
# Run download_pretrained.sh first if missing.
#
# Usage:
#   sh scripts/train_needle_official_dinov2.sh                 # train + test
#   sh scripts/train_needle_official_dinov2.sh test-only       # skip train
#
# Override defaults via env vars:
#   RATE=0.10  BS=2  LR=1e-5  CROP=490  EPOCHS=80  NGPU=1

set -u
MODE=${1:-full}

DS=needle
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/autonomous_surgery/public_data}
SPLITS=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}
EXP_ROOT=${EXP_ROOT:-/root/autodl-tmp/exp}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Needle uses smaller CROP (490 = 14*35) and lower LR than endoscopic datasets.
RATE=${RATE:-0.10}
BS=${BS:-2}                       # full backbone, small batch (cf. configs/needle.yaml)
LR=${LR:-1e-5}
CROP=${CROP:-490}
EPOCHS=${EPOCHS:-80}
NGPU=${NGPU:-1}
PORT=${PORT:-29550}

TAG="base_r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}"

echo "=================================================================="
echo "[$(date +%H:%M:%S)] Needle · OFFICIAL UniMatch-V2 (DINOv2-B/14)"
echo "                   RATE=$RATE BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS"
echo "                   TAG=$TAG"
echo "                   data_root=$DATA_ROOT/unimatchv2_needle_3class"
echo "=================================================================="

if [ "$MODE" != "test-only" ]; then
    echo
    echo "[$(date +%H:%M:%S)] >>> TRAIN"
    RATE=$RATE BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS TAG=$TAG \
        sh scripts/train.sh $NGPU $PORT base $DS
fi

echo
echo "[$(date +%H:%M:%S)] >>> TEST"
RATE=$RATE BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS TAG=$TAG \
    sh scripts/test.sh                   base $DS

echo
echo "[done] artefacts: $EXP_ROOT/$DS/unimatch_v2_$TAG/"
echo "       train_log.csv  best.pth  test_metrics.csv  out.log"
