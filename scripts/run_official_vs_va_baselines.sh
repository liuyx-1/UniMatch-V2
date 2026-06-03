#!/usr/bin/env bash
# ================================================================
# Two-row baseline comparison across all 5 surgical datasets:
#   row 1 : "official"  =  UniMatch-V2 baseline (no visual adapter)
#   row 2 : "+VA"       =  Frozen DINOv2 + HPTA (DinoDPTAdapter-HOM-lite)
#
# For each dataset, runs:  train(official) → test(official) → train(VA) → test(VA)
#
# Hyperparameters:
#   Common:    RATE=0.10  LR=5e-5  CROP=518  EPOCHS=150
#   Per-row BS (official trains the full DINOv2-B → smaller BS to fit GPU):
#       official : BS_OFFICIAL = 4   (full ~100M trainable, BS=16 OOM on 24GB)
#       +VA      : BS_VA       = 16  (only ~14M trainable, big batch OK)
#   Override at invocation time, e.g.  BS_OFFICIAL=8 BS_VA=16 sh scripts/...
#
# Port allocation (one pair per dataset; +0=official, +1=VA):
#   endoscapes_seg50    29500 / 29501
#   cholecseg8k         29510 / 29511
#   endovis2018         29520 / 29521
#   endovis2017_parts   29530 / 29531
#   endovis2017_type    29540 / 29541
#
# Usage:
#   sh scripts/run_official_vs_va_baselines.sh
#   sh scripts/run_official_vs_va_baselines.sh endoscapes_seg50
#   sh scripts/run_official_vs_va_baselines.sh endovis2018 endovis2017_parts
#   DRYRUN=1 sh scripts/run_official_vs_va_baselines.sh    # preview only
#
# Background + log:
#   nohup sh scripts/run_official_vs_va_baselines.sh \
#       > /root/autodl-tmp/exp/_baselines_nohup.log 2>&1 &
#   disown
#   tail -f /root/autodl-tmp/exp/_baselines_nohup.log

set -u

# ---------------- common env ----------------
export DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/autonomous_surgery/public_data}
export SPLITS=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}
export EXP_ROOT=${EXP_ROOT:-/root/autodl-tmp/exp}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

RATE=${RATE:-0.10}
BS_OFFICIAL=${BS_OFFICIAL:-4}     # full backbone training: small batch
BS_VA=${BS_VA:-16}                # frozen backbone + adapter: big batch
LR=${LR:-5e-5}
CROP=${CROP:-518}
EPOCHS=${EPOCHS:-150}
NGPU=${NGPU:-1}
DRYRUN=${DRYRUN:-0}

# Per-dataset base port (offset 0 = official, offset 1 = +VA)
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

# emit train + test for one (dataset, label)
# Args: $1=DS  $2=label ("official" | "+VA")
emit () {
    DS=$1
    LBL=$2
    PORT_BASE=$(port_for_ds "$DS")

    case "$LBL" in
      official)
        ENV=""
        PORT=$(( PORT_BASE + 0 ))
        BS=$BS_OFFICIAL
        TAG="base_r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}"
        ;;
      +VA)
        ENV="VISUAL_ADAPTER=1"
        PORT=$(( PORT_BASE + 1 ))
        BS=$BS_VA
        TAG="base_r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}_va"
        ;;
      *) echo "[skip] unknown label: $LBL"; return ;;
    esac

    echo
    echo "==================================================================="
    echo "[$(date +%H:%M:%S)] ${DS} :: ${LBL}  (port=${PORT}  BS=${BS}  TAG=${TAG})"
    echo "==================================================================="

    TRAIN_CMD="RATE=$RATE BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS $ENV TAG=$TAG sh scripts/train.sh $NGPU $PORT base $DS"
    TEST_CMD="RATE=$RATE  BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS $ENV TAG=$TAG sh scripts/test.sh                   base $DS"

    echo "# TRAIN"
    echo "$TRAIN_CMD"
    echo "# TEST"
    echo "$TEST_CMD"

    if [ "$DRYRUN" = "1" ]; then return; fi

    eval "$TRAIN_CMD"
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "[!!] TRAIN failed (rc=$rc): $DS :: $LBL — skipping test"
        return $rc
    fi
    eval "$TEST_CMD" || echo "[!!] TEST failed: $DS :: $LBL"
}

# ---------------- runs ----------------
ALL_DS="endoscapes_seg50 cholecseg8k endovis2018 endovis2017_parts endovis2017_type"

if [ "$#" -ge 1 ]; then
    DSS="$*"
else
    DSS="$ALL_DS"
fi

echo "[start] $(date)"
echo "[config] RATE=$RATE LR=$LR CROP=$CROP EPOCHS=$EPOCHS NGPU=$NGPU"
echo "         BS_OFFICIAL=$BS_OFFICIAL (full backbone)  BS_VA=$BS_VA (frozen + adapter)"
echo "[datasets] $DSS"

for DS in $DSS; do
    echo
    echo "###################################################################"
    echo "#  Dataset: $DS"
    echo "###################################################################"
    emit "$DS" "official"
    emit "$DS" "+VA"
done

echo
echo "[done] $(date)"
echo "Summary of run directories:"
for DS in $DSS; do
    echo "  --- $DS ---"
    echo "    official : $EXP_ROOT/$DS/unimatch_v2_base_r${RATE}_bs${BS_OFFICIAL}_lr${LR}_ep${EPOCHS}_cr${CROP}/"
    echo "    +VA      : $EXP_ROOT/$DS/unimatch_v2_base_r${RATE}_bs${BS_VA}_lr${LR}_ep${EPOCHS}_cr${CROP}_va/"
done
