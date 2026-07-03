#!/bin/bash
# Fully-supervised training (labeled data only, NO unlabeled stream).
#
# Usage:
#   sh scripts/train_sup.sh <num_gpu> <port> <dataset>
#
# Env overrides (mirror scripts/train.sh):
#   RATE=<float>     label rate split to use (default 0.10)
#   FULL=1           train on 100% of train labels (labeled.txt + unlabeled.txt);
#                    requires unlabeled.txt to carry mask paths (2 columns)
#   BS / LR / EPOCHS / CROP
#   VISUAL_ADAPTER=1 use the HPTA visual adapter (match the *_va runs)
#   EVAL_BS=<int>    batched eval (uniform-resolution sets)
#   EVAL_SIZE=<int>  cap eval long side (e.g. 686 for endovis 1280x1024)
#
# Examples:
#   # 10% supervised baseline (same budget as the SSL runs), with HPTA:
#   RATE=0.10 BS=8 LR=5e-5 CROP=490 EPOCHS=150 VISUAL_ADAPTER=1 \
#     EVAL_BS=2 EVAL_SIZE=686 sh scripts/train_sup.sh 1 29540 endovis2018
#
#   # 100% fully-supervised upper bound:
#   FULL=1 BS=8 LR=5e-5 CROP=490 EPOCHS=150 VISUAL_ADAPTER=1 \
#     EVAL_BS=2 EVAL_SIZE=686 sh scripts/train_sup.sh 1 29541 endovis2018

NGPU=${1:-1}
PORT=${2:-29540}
DATASET=${3:-endovis2018}
RATE=${RATE:-0.10}

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

SPLIT_ROOT=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}/unimatch_splits_${DATASET}_${RATE}_seed42
EXP_ROOT=${EXP_ROOT:-/root/autodl-tmp/exp}
config=configs/${DATASET}.yaml
val_id_path=${SPLIT_ROOT}/val.txt
test_id_path=${SPLIT_ROOT}/test.txt

# ---- choose the labeled set ----
labeled_id_path=${SPLIT_ROOT}/labeled.txt
TAGRATE="r${RATE}"
if [ "${FULL:-0}" = "1" ]; then
    labeled_id_path=${SPLIT_ROOT}/sup_full.txt
    cat "${SPLIT_ROOT}/labeled.txt" "${SPLIT_ROOT}/unlabeled.txt" > "$labeled_id_path"
    # sanity: every line must have an image AND a mask path (2 columns)
    if awk 'NF<2{bad=1} END{exit bad}' "$labeled_id_path"; then :; else
        echo "ERROR: $labeled_id_path has image-only lines; unlabeled.txt lacks mask paths."
        echo "       FULL=1 needs masks for all train images. Use RATE-based labeled.txt instead."
        exit 1
    fi
    TAGRATE="r1.00"
fi

FLAGS=""
[ -n "$LR" ]     && FLAGS="$FLAGS --lr $LR"
[ -n "$EPOCHS" ] && FLAGS="$FLAGS --epochs $EPOCHS"
[ -n "$CROP" ]   && FLAGS="$FLAGS --crop-size $CROP"
[ -n "$BS" ]     && FLAGS="$FLAGS --batch-size $BS"
[ "${VISUAL_ADAPTER:-0}" = "1" ] && FLAGS="$FLAGS --visual-adapter"
[ -n "${VISUAL_ADAPTER_REDUCTION:-}" ] && FLAGS="$FLAGS --visual-adapter-reduction $VISUAL_ADAPTER_REDUCTION"
[ -n "${EVAL_BS:-}" ]   && FLAGS="$FLAGS --eval-batch-size $EVAL_BS"
[ -n "${EVAL_SIZE:-}" ] && FLAGS="$FLAGS --eval-size $EVAL_SIZE"
[ -f "$test_id_path" ]  && FLAGS="$FLAGS --test-id-path $test_id_path"

TAG="sup_${TAGRATE}"
[ -n "$BS" ]     && TAG="${TAG}_bs${BS}"
[ -n "$LR" ]     && TAG="${TAG}_lr${LR}"
[ -n "$EPOCHS" ] && TAG="${TAG}_ep${EPOCHS}"
[ -n "$CROP" ]   && TAG="${TAG}_cr${CROP}"
[ "${VISUAL_ADAPTER:-0}" = "1" ] && TAG="${TAG}_va"
TAG=${TAG_OVERRIDE:-$TAG}

save_path=${EXP_ROOT}/${DATASET}/supervised_${TAG}
mkdir -p "$save_path"

echo "[train_sup.sh] dataset=$DATASET labeled=$labeled_id_path save_path=$save_path"

python -m torch.distributed.launch \
    --nproc_per_node=$NGPU \
    --master_addr=localhost \
    --master_port=$PORT \
    supervised.py \
    --config "$config" \
    --labeled-id-path "$labeled_id_path" \
    --val-id-path "$val_id_path" \
    --save-path "$save_path" \
    --port "$PORT" $FLAGS 2>&1 | tee "$save_path/out.log"
