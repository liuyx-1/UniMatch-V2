#!/usr/bin/env bash
# ================================================================
# Master ablation script: 5 datasets × 6 variants
#
# Variants:
#   1) base           VA only                              (HPTA)
#   2) +text          VA + LC-PAM                          (HPTA + LC-PAM)
#   3) +edge          VA + EABR-DREC + EABR-MA-CBA         (HPTA + EABR)
#   4) +edge+text     VA + LC-PAM + EABR                    (HPTA + LC-PAM + EABR)
#   5) +boundary      VA + boundary head only
#   6) full           VA + LC-PAM + EABR + boundary
#
# Datasets:
#   endoscapes_seg50   port range 29500-29509
#   cholecseg8k        port range 29510-29519
#   endovis2018        port range 29520-29529
#   endovis2017_parts  port range 29530-29539
#   endovis2017_type   port range 29540-29549
#
# Common hyperparameters (the current main line):
#   RATE=0.10  BS=16  LR=5e-5  CROP=518  EPOCHS=150
#
# Usage:
#   sh scripts/all_dataset_variants.sh                  # run everything
#   sh scripts/all_dataset_variants.sh endoscapes_seg50 # only one dataset
#   sh scripts/all_dataset_variants.sh endovis2018 +text +edge+text   # subset
#
# To dry-run (print commands without executing), set DRYRUN=1.

set -u

# ---------------- common env ----------------
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/autonomous_surgery/public_data}
export SPLITS=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}
export EXP_ROOT=${EXP_ROOT:-/root/autodl-tmp/exp}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RATE=${RATE:-0.10}
BS=${BS:-16}
LR=${LR:-5e-5}
CROP=${CROP:-518}
EPOCHS=${EPOCHS:-150}
NGPU=${NGPU:-1}
DRYRUN=${DRYRUN:-0}

# Per-dataset base port
port_for_ds () {
    case "$1" in
        endoscapes_seg50)   echo 29500 ;;
        cholecseg8k)        echo 29510 ;;
        endovis2018)        echo 29520 ;;
        endovis2017_parts)  echo 29530 ;;
        endovis2017_type)   echo 29540 ;;
        *) echo 29500 ;;
    esac
}

# emit train + test for one (dataset, variant). $1=DS, $2=label, $3=port offset
emit () {
    DS=$1; LBL=$2; OFFSET=$3
    PORT=$(( $(port_for_ds "$DS") + OFFSET ))
    BASE_TAG="r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}"

    case "$LBL" in
      base)
        VARIANT=base
        ENV='VISUAL_ADAPTER=1'
        TAG="base_${BASE_TAG}_va"
        ;;
      +text)
        VARIANT=affinity_min
        ENV='VISUAL_ADAPTER=1 JOINT_TEXT_STAGE=1'
        TAG="affinity_min_${BASE_TAG}_va_joint"
        ;;
      +edge)
        VARIANT=base
        ENV='VISUAL_ADAPTER=1 EDGE_ENHANCE=1'
        TAG="base_${BASE_TAG}_va_edge"
        ;;
      +edge+text)
        VARIANT=affinity_min
        ENV='VISUAL_ADAPTER=1 JOINT_TEXT_STAGE=1 EDGE_ENHANCE=1'
        TAG="affinity_min_${BASE_TAG}_va_joint_edge"
        ;;
      full)
        VARIANT=full
        ENV='VISUAL_ADAPTER=1 JOINT_TEXT_STAGE=1 EDGE_ENHANCE=1'
        TAG="full_${BASE_TAG}_va_joint_edge_bnd"
        ;;
      +boundary)
        VARIANT=boundary
        ENV='VISUAL_ADAPTER=1'
        TAG="boundary_${BASE_TAG}_va"
        ;;
      *) echo "[skip] unknown label: $LBL"; return ;;
    esac

    echo "# ---- ${DS} :: ${LBL}  (variant=${VARIANT}, port=${PORT}) ----"
    TRAIN_CMD="RATE=$RATE BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS $ENV TAG=$TAG sh scripts/train.sh $NGPU $PORT $VARIANT $DS"
    TEST_CMD="RATE=$RATE  BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS $ENV TAG=$TAG sh scripts/test.sh                   $VARIANT $DS"
    echo "$TRAIN_CMD"
    echo "$TEST_CMD"
    if [ "$DRYRUN" = "1" ]; then return; fi
    eval "$TRAIN_CMD" && eval "$TEST_CMD" || echo "[!!] failed: $DS :: $LBL"
}

ALL_DS="endoscapes_seg50 cholecseg8k endovis2018 endovis2017_parts endovis2017_type"
ALL_LBL="base +text +edge +edge+text +boundary full"

# ---------------- CLI ----------------
if [ "$#" -ge 1 ]; then
    DSS="$1"; shift
    LBLS="${*:-$ALL_LBL}"
else
    DSS="$ALL_DS"
    LBLS="$ALL_LBL"
fi

for DS in $DSS; do
  echo
  echo "================================================================="
  echo "  Dataset: $DS"
  echo "================================================================="
  i=0
  for LBL in $LBLS; do
    emit "$DS" "$LBL" "$i"
    i=$((i + 1))
  done
done
