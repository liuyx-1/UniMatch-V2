#!/bin/bash
# Usage: sh scripts/plot.sh [dataset] [rate]
#   rate: 0.10 | 0.20 | 0.25 | 0.30 | 0.50 (default 0.25)
#
# Plots the 4 variants (base/debias/boundary/full) at the given label rate.
# Pass extra hyperparameter env vars (BS / LR / EPOCHS / CROP) to match the
# train-time tag, e.g.:
#   RATE=0.10 BS=2 LR=5e-6 sh scripts/plot.sh endoscapes_seg50
#
# For arbitrary runs, pass RUNS="name1:csv1 name2:csv2 ..." explicitly.

DATASET=${1:-endoscapes_seg50}
RATE=${2:-${RATE:-0.25}}
EXP_ROOT=${EXP_ROOT:-/data/exp}
ROOT=${EXP_ROOT}/${DATASET}

SUFFIX=""
[ -n "$BS" ]     && SUFFIX="${SUFFIX}_bs${BS}"
[ -n "$LR" ]     && SUFFIX="${SUFFIX}_lr${LR}"
[ -n "$EPOCHS" ] && SUFFIX="${SUFFIX}_ep${EPOCHS}"
[ -n "$CROP" ]   && SUFFIX="${SUFFIX}_cr${CROP}"

if [ -n "$RUNS" ]; then
    RUN_ARGS=""
    for r in $RUNS; do RUN_ARGS="$RUN_ARGS --run $r"; done
    python tools/plot_train_log.py $RUN_ARGS \
        --out ${ROOT}/curves_${DATASET}_r${RATE}_custom.png
else
    python tools/plot_train_log.py \
        --run "v1 base:${ROOT}/unimatch_v2_base_r${RATE}${SUFFIX}/train_log.csv" \
        --run "v2 debias:${ROOT}/unimatch_v2_debias_r${RATE}${SUFFIX}/train_log.csv" \
        --run "v3 boundary:${ROOT}/unimatch_v2_boundary_r${RATE}${SUFFIX}/train_log.csv" \
        --run "v4 full:${ROOT}/unimatch_v2_full_r${RATE}${SUFFIX}/train_log.csv" \
        --out ${ROOT}/curves_${DATASET}_r${RATE}${SUFFIX}.png
fi
