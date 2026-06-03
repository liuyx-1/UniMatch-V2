#!/usr/bin/env bash
# Test valid ckpts on endoscapes_seg50 (RATE=0.10 BS=16 LR=5e-5 CROP=518 VA).
# After all five tests, build (i) a single metric table PNG and
# (ii) a single ablation-panel visualization PNG.
#
# Usage:
#   sh scripts/test_all_valid_endoscapes.sh

set -u

DS=endoscapes_seg50
RATE=0.10
BS=16
LR=5e-5
CROP=518

export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/autonomous_surgery/public_data}
export SPLITS=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}
export EXP_ROOT=${EXP_ROOT:-/root/autodl-tmp/exp}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

SPLIT=$SPLITS/unimatch_splits_${DS}_${RATE}_seed42
FIGS=$EXP_ROOT/$DS/figs_valid5
mkdir -p $FIGS

LOG=$FIGS/test_all_$(date +%Y%m%d_%H%M).log
echo "[start] $(date) — log: $LOG"

# ---------------- per-ckpt test ----------------
run_test () {
    LABEL=$1; VARIANT=$2; TAG=$3; shift 3
    ENV_VARS="$*"

    echo
    echo "==================================================================="
    echo "[$(date +%H:%M:%S)] TEST  $LABEL  ($VARIANT)"
    echo "                    TAG=$TAG"
    echo "==================================================================="
    eval "RATE=$RATE BS=$BS LR=$LR CROP=$CROP VISUAL_ADAPTER=1 TAG=$TAG \
          $ENV_VARS \
          sh scripts/test.sh $VARIANT $DS"
}

main () {
    run_test "base"      base         "base_r${RATE}_bs${BS}_lr${LR}_cr${CROP}_va"                            ""
    run_test "+text"     affinity_min "affinity_min_r${RATE}_bs${BS}_lr${LR}_cr${CROP}_va_joint"              "JOINT_TEXT_STAGE=1"
    run_test "+boundary" boundary     "boundary_r${RATE}_bs${BS}_lr${LR}_ep150_cr${CROP}_va"                  "EPOCHS=150"
    run_test "full"      full         "full_r${RATE}_bs${BS}_lr${LR}_ep150_cr${CROP}_va_joint_edge_bnd"       "EPOCHS=150 JOINT_TEXT_STAGE=1 EDGE_ENHANCE=1"

    echo
    echo "==================================================================="
    echo "[$(date +%H:%M:%S)] BUILD METRIC TABLE + VIS PANEL"
    echo "==================================================================="

    python tools/build_ablation_table_and_panel.py \
      --run "base:$EXP_ROOT/$DS/unimatch_v2_base_r${RATE}_bs${BS}_lr${LR}_cr${CROP}_va" \
      --run "+text:$EXP_ROOT/$DS/unimatch_v2_affinity_min_r${RATE}_bs${BS}_lr${LR}_cr${CROP}_va_joint" \
      --run "+boundary:$EXP_ROOT/$DS/unimatch_v2_boundary_r${RATE}_bs${BS}_lr${LR}_ep150_cr${CROP}_va" \
      --run "full:$EXP_ROOT/$DS/unimatch_v2_full_r${RATE}_bs${BS}_lr${LR}_ep150_cr${CROP}_va_joint_edge_bnd" \
      --config configs/${DS}.yaml \
      --val-id-path $SPLIT/val.txt \
      --out-dir $FIGS \
      --num-images 6 --seed 42

    echo
    echo "[done] $(date)"
    echo "outputs:"
    echo "  $FIGS/metric_table.png"
    echo "  $FIGS/ablation_panel.png"
}

# POSIX-compatible logging: pipe the whole main() through tee.
# Works with both `sh` and `bash`.
main 2>&1 | tee -a "$LOG"
