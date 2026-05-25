#!/bin/bash
# Usage: sh scripts/test.sh [variant] [dataset]
VARIANT=${1:-full}
DATASET=${2:-endoscapes_seg50}
RATE=${RATE:-0.25}

if [ -z "$TAG" ]; then
    TAG=${VARIANT}_r${RATE}
    [ -n "$BS" ]     && TAG="${TAG}_bs${BS}"
    [ -n "$LR" ]     && TAG="${TAG}_lr${LR}"
    [ -n "$EPOCHS" ] && TAG="${TAG}_ep${EPOCHS}"
    [ -n "$CROP" ]   && TAG="${TAG}_cr${CROP}"
fi

SPLIT_ROOT=${SPLITS:-/data/splits}/unimatch_splits_${DATASET}_${RATE}_seed42
EXP_ROOT=${EXP_ROOT:-/data/exp}
TEST_ID=${SPLIT_ROOT}/test.txt
[ -f "$TEST_ID" ] || TEST_ID=${SPLIT_ROOT}/val.txt

save_path=${EXP_ROOT}/${DATASET}/unimatch_v2_${TAG}
ckpt=${save_path}/best.pth

if [ ! -f "$ckpt" ]; then
    echo "checkpoint not found: $ckpt"
    exit 1
fi

EXTRA_TEST=""
if [ "$VARIANT" = "affinity" ] || [ "$VARIANT" = "affinity_min" ]; then
    if [ -z "$AFFINITY_WARMSTART" ]; then
        echo "ERROR: variant '$VARIANT' requires AFFINITY_WARMSTART env var pointing to affinity_<ds>.pt"
        exit 1
    fi
    EXTRA_TEST="--affinity-warmstart $AFFINITY_WARMSTART"
fi

python test.py \
    --config configs/${DATASET}.yaml \
    --checkpoint $ckpt \
    --id-path $TEST_ID \
    --train-id-path ${SPLIT_ROOT}/labeled.txt \
    --train-freq-cache ${save_path}/train_pixel_freq_test.json \
    --use-ema \
    --ece \
    --ap \
    --save-csv ${save_path}/test_metrics.csv \
    $EXTRA_TEST
