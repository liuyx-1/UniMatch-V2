#!/bin/bash
# Usage: sh scripts/test.sh [variant] [dataset]
# Main documented variants:
#   base         official UniMatch-V2 baseline
#   affinity_min UniMatch-V2 + text enhancement only
#   boundary     UniMatch-V2 + boundary branch only
#   full         UniMatch-V2 + text enhancement + boundary branch
#
# Use the same RATE/BS/LR/EPOCHS/CROP environment variables used for training.
# Text-enhancement tests load affinity_side from the segmentation checkpoint.
# AFFINITY_WARMSTART is optional for legacy checkpoints that did not save it.
VARIANT=${1:-base}
DATASET=${2:-endoscapes_seg50}
RATE=${RATE:-0.25}

if [ -z "$TAG" ]; then
    TAG=${VARIANT}_r${RATE}
    [ -n "$BS" ]     && TAG="${TAG}_bs${BS}"
    [ -n "$LR" ]     && TAG="${TAG}_lr${LR}"
    [ -n "$EPOCHS" ] && TAG="${TAG}_ep${EPOCHS}"
    [ -n "$CROP" ]   && TAG="${TAG}_cr${CROP}"
    if [ -n "${BACKBONE:-}" ]; then
        bb_short=$(echo "$BACKBONE" | awk -F_ '{print $NF}')
        TAG="${TAG}_${bb_short}"
    fi
    [ "${VISUAL_ADAPTER:-0}" = "1" ] && TAG="${TAG}_va"
    [ "${JOINT_TEXT_STAGE:-0}" = "1" ] && TAG="${TAG}_joint"
    [ "${EDGE_ENHANCE:-0}" = "1" ] && TAG="${TAG}_edge"
    [ "${EDGE_ENHANCE:-0}" = "1" ] && [ "${EDGE_REFINER:-0}" = "1" ] && TAG="${TAG}mger"
    [ "${TEMPORAL_CONSISTENCY:-0}" = "1" ] && TAG="${TAG}_tcr"
    [ "${TEMPORAL_CONSISTENCY:-0}" = "1" ] && [ "${TEMPORAL_ORIGINAL:-0}" = "1" ] && TAG="${TAG}v1"
    [ "${TSMDR:-0}" = "1" ] && TAG="${TAG}_tsmdr"
    [ "${TSMDR:-0}" = "1" ] && [ "${TSMDR_SHAPE_ONLY:-0}" = "1" ] && TAG="${TAG}shape"
fi

SPLIT_ROOT=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}/unimatch_splits_${DATASET}_${RATE}_seed42
EXP_ROOT=${EXP_ROOT:-/root/autodl-tmp/exp}
TEST_ID=${SPLIT_ROOT}/test.txt
[ -f "$TEST_ID" ] || TEST_ID=${SPLIT_ROOT}/val.txt

save_path=${EXP_ROOT}/${DATASET}/unimatch_v2_${TAG}
ckpt=${save_path}/best.pth

if [ ! -f "$ckpt" ]; then
    echo "checkpoint not found: $ckpt"
    exit 1
fi

EXTRA_TEST=""
if [ "$VARIANT" = "affinity" ] || [ "$VARIANT" = "affinity_min" ] || [ "$VARIANT" = "full" ] || [ "$VARIANT" = "full_cls" ]; then
    [ -n "$AFFINITY_WARMSTART" ] && EXTRA_TEST="--affinity-warmstart $AFFINITY_WARMSTART"
fi
[ -n "${BACKBONE:-}" ] && EXTRA_TEST="$EXTRA_TEST --backbone $BACKBONE"

# --ap / --ece accumulate per-pixel float arrays over the WHOLE test set in
# host RAM. On large high-res sets (endovis2018, cholecseg8k) this can OOM the
# container. Default ON; set AP=0 and/or ECE=0 to disable.
AP=${AP:-1}
ECE=${ECE:-1}
[ "$AP" = "1" ]  && EXTRA_TEST="$EXTRA_TEST --ap"
[ "$ECE" = "1" ] && EXTRA_TEST="$EXTRA_TEST --ece"

DEVICE=${DEVICE:-cuda:0}

python test.py \
    --config configs/${DATASET}.yaml \
    --checkpoint $ckpt \
    --id-path $TEST_ID \
    --train-id-path ${SPLIT_ROOT}/labeled.txt \
    --train-freq-cache ${save_path}/train_pixel_freq_test.json \
    --use-ema \
    --device $DEVICE \
    --save-csv ${save_path}/test_metrics.csv \
    $EXTRA_TEST
