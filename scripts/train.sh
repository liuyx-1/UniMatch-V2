#!/bin/bash
# Usage:
#   sh scripts/train.sh <num_gpu> <port> [variant] [dataset]
#
# variant  in { base | debias | boundary | tangent | full | cls | debias_cls | boundary_cls | full_cls }
# dataset  in { pascal | cityscapes | ade20k | coco | endoscapes_seg50 | cholecseg8k }
#
# Hyperparameter overrides via env vars (any optional):
#   BS=<int>      batch size per GPU
#   LR=<float>    learning rate
#   EPOCHS=<int>  number of epochs
#   CROP=<int>    crop size (must be patch-multiple: 14 for DINOv2)
#   EXTRA="..."   extra raw flags to forward to unimatch_v2.py
#
# Examples:
#   sh scripts/train.sh 1 29500 full endoscapes_seg50
#   BS=2 LR=5e-6 sh scripts/train.sh 1 29500 debias endoscapes_seg50

NGPU=${1:-1}
PORT=${2:-29500}
VARIANT=${3:-base}
DATASET=${4:-endoscapes_seg50}

METHOD=unimatch_v2
case "$VARIANT" in
    base)            FLAGS="" ;;
    debias)          FLAGS="--debias" ;;
    boundary)        FLAGS="--boundary" ;;
    tangent)         FLAGS="--boundary --tangent --tangent-weight ${TANGENT_WEIGHT:-0.1}" ;;
    cls)             FLAGS="--cls-head" ;;
    full)            FLAGS="--debias --boundary" ;;
    debias_cls)      FLAGS="--debias --cls-head" ;;
    boundary_cls)    FLAGS="--boundary --cls-head" ;;
    full_cls)        FLAGS="--debias --boundary --cls-head" ;;
    affinity)        FLAGS="--debias --boundary --cls-head" ;;
    affinity_min)    FLAGS="" ;;
    *) echo "unknown variant: $VARIANT  (base|debias|boundary|tangent|cls|full|debias_cls|boundary_cls|full_cls|affinity|affinity_min)"; exit 1 ;;
esac

# Stage-1 prior side branch (only when variant in {affinity, affinity_min} and AFFINITY_WARMSTART set)
if [ "$VARIANT" = "affinity" ] || [ "$VARIANT" = "affinity_min" ]; then
    if [ -z "$AFFINITY_WARMSTART" ]; then
        echo "ERROR: variant '$VARIANT' requires AFFINITY_WARMSTART env var pointing to affinity_<ds>.pt"
        exit 1
    fi
    FLAGS="$FLAGS --affinity-warmstart $AFFINITY_WARMSTART"
    [ -n "$AFFINITY_AUX_WEIGHT" ]    && FLAGS="$FLAGS --affinity-aux-weight $AFFINITY_AUX_WEIGHT"
    [ -n "$AFFINITY_FREEZE_WARMUP" ] && FLAGS="$FLAGS --affinity-freeze-warmup $AFFINITY_FREEZE_WARMUP"
    [ -n "$AFFINITY_PLATEAU_EPS" ]   && FLAGS="$FLAGS --affinity-plateau-eps $AFFINITY_PLATEAU_EPS"
    [ -n "$AFFINITY_UNFREEZE_LR_MULT" ] && FLAGS="$FLAGS --affinity-unfreeze-lr-mult $AFFINITY_UNFREEZE_LR_MULT"
fi

[ -n "$BS" ]     && FLAGS="$FLAGS --batch-size $BS"
[ -n "$LR" ]     && FLAGS="$FLAGS --lr $LR"
[ -n "$EPOCHS" ] && FLAGS="$FLAGS --epochs $EPOCHS"
[ -n "$CROP" ]   && FLAGS="$FLAGS --crop-size $CROP"
[ -n "$EXTRA" ]  && FLAGS="$FLAGS $EXTRA"

RATE=${RATE:-0.25}                     # label rate: 0.10 | 0.20 | 0.25 | 0.30 | 0.50
SPLIT_ROOT=${SPLITS:-/data/splits}/unimatch_splits_${DATASET}_${RATE}_seed42
EXP_ROOT=${EXP_ROOT:-/data/exp}

config=configs/${DATASET}.yaml
labeled_id_path=${SPLIT_ROOT}/labeled.txt
unlabeled_id_path=${SPLIT_ROOT}/unlabeled.txt
val_id_path=${SPLIT_ROOT}/val.txt

TAG=${VARIANT}_r${RATE}
[ -n "$BS" ]     && TAG="${TAG}_bs${BS}"
[ -n "$LR" ]     && TAG="${TAG}_lr${LR}"
[ -n "$EPOCHS" ] && TAG="${TAG}_ep${EPOCHS}"
[ -n "$CROP" ]   && TAG="${TAG}_cr${CROP}"
save_path=${EXP_ROOT}/${DATASET}/${METHOD}_${TAG}

mkdir -p "$save_path"

python -m torch.distributed.launch \
    --nproc_per_node=$NGPU \
    --master_addr=localhost \
    --master_port=$PORT \
    ${METHOD}.py \
    --config=$config \
    --labeled-id-path $labeled_id_path \
    --unlabeled-id-path $unlabeled_id_path \
    --val-id-path $val_id_path \
    --save-path $save_path \
    --port $PORT $FLAGS 2>&1 | tee $save_path/out.log
