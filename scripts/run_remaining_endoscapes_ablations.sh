#!/usr/bin/env bash
# Serial train + test for the consolidated ablation on endoscapes_seg50.
#
# Variants (proposed method = "full"):
#   1. base       VA only                                       (done)
#   2. +text      VA + text/affinity branch                     (done)
#   3. +edge      VA + RGB-edge dual residuals                   ⏳
#   4. +edge+text VA + text + edge                              ⏳
#   5. full       VA + text + edge + boundary                   ⏳
#
# Usage:
#   sh scripts/run_remaining_endoscapes_ablations.sh
#   # or background:
#   nohup sh scripts/run_remaining_endoscapes_ablations.sh \
#       > /root/autodl-tmp/exp/endoscapes_seg50/ablations_remaining_nohup.log 2>&1 &
#   disown

set -u  # do NOT set -e

# ---------------- hyperparameters ----------------
DS=endoscapes_seg50
RATE=0.10
BS=16
LR=5e-5
CROP=518
EPOCHS=150
NGPU=1

export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/autonomous_surgery/public_data}
export SPLITS=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}
export EXP_ROOT=${EXP_ROOT:-/root/autodl-tmp/exp}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$EXP_ROOT/$DS"

MASTER_LOG=$EXP_ROOT/$DS/ablations_remaining_$(date +%Y%m%d_%H%M).log
echo "[start] $(date) — master log: $MASTER_LOG"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "[config] DS=$DS RATE=$RATE BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS NGPU=$NGPU"

# ---------------- helper ----------------
# run_pair <label> <port> <variant> <TAG> <extra_env...>
run_pair () {
    local LABEL=$1
    local PORT=$2
    local VARIANT=$3
    local TAG=$4
    shift 4
    local EXTRA_ENV="$*"

    echo
    echo "==================================================================="
    echo "[$(date +%H:%M:%S)] >>> TRAIN  $LABEL  (variant=$VARIANT  port=$PORT)"
    echo "                    TAG=$TAG"
    echo "                    extra-env: $EXTRA_ENV"
    echo "==================================================================="
    eval "RATE=$RATE BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS \
          VISUAL_ADAPTER=1 TAG=$TAG \
          $EXTRA_ENV \
          sh scripts/train.sh $NGPU $PORT $VARIANT $DS"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "[!!] TRAIN $LABEL failed (rc=$rc) — skipping test"
        return $rc
    fi

    echo
    echo "==================================================================="
    echo "[$(date +%H:%M:%S)] >>> TEST   $LABEL"
    echo "==================================================================="
    eval "RATE=$RATE BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS \
          VISUAL_ADAPTER=1 TAG=$TAG \
          $EXTRA_ENV \
          sh scripts/test.sh $VARIANT $DS"
}

# ---------------- runs ----------------
# Ports already used: 29500 (base), 29501 (+text). Free: 29502 / 03 / 04.

# row 3 — +edge
run_pair \
    "+edge"  29502  base \
    "base_r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}_va_edge" \
    "EDGE_ENHANCE=1"

# row 4 — +edge+text
run_pair \
    "+edge+text"  29503  affinity_min \
    "affinity_min_r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}_va_joint_edge" \
    "JOINT_TEXT_STAGE=1  EDGE_ENHANCE=1"

# row 5 — full  (VA + text + edge + boundary)
run_pair \
    "full"  29504  full \
    "full_r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}_va_joint_edge_bnd" \
    "JOINT_TEXT_STAGE=1  EDGE_ENHANCE=1"

echo
echo "[done] $(date)  all remaining ablations finished"
echo "Outputs:"
for TAG in \
    "base_r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}_va_edge" \
    "affinity_min_r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}_va_joint_edge" \
    "full_r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}_va_joint_edge_bnd"
do
    echo "  unimatch_v2_$TAG  → $EXP_ROOT/$DS/unimatch_v2_$TAG/"
done
