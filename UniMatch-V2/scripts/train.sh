#!/bin/bash
# Usage:
#   sh scripts/train.sh <num_gpu> <port> [variant] [dataset]
#
# Main documented variants:
#   base         official UniMatch-V2 baseline
#   affinity_min UniMatch-V2 + text enhancement only
#   temporal     UniMatch-V2 + temporal consistency only
#   full         UniMatch-V2 + text enhancement + temporal consistency
#
# The standalone boundary branch has been removed. EDGE_ENHANCE still keeps
# its own edge-boundary prediction head when enabled.
#
# dataset examples:
#   endoscapes_seg50 | cholecseg8k | endovis2018 | endovis2017_parts | endovis2017_type | needle
#
# Hyperparameter overrides via env vars (any optional):
#   BS=<int>      batch size per GPU
#   LR=<float>    learning rate
#   EPOCHS=<int>  number of epochs
#   CROP=<int>    crop size (must be patch-multiple: 14 for DINOv2)
#   EXTRA="..."   extra raw flags to forward to unimatch_v2.py
#   EDGE_ENHANCE=1 enables RGB-edge residual adapters
#   VISUAL_ADAPTER=1 enables frozen-DINO DinoDPTAdapter-HOM-lite
#   JOINT_TEXT_STAGE=1 trains the text/affinity branch jointly in the main run
#   AUTO_BS=1     retry with half BS on CUDA OOM (default: 1 when BS is set)
#   MIN_BS=<int>  smallest auto-retry batch size (default: 1)
#
# Examples:
#   sh scripts/train.sh 1 29500 base endoscapes_seg50
#   JOINT_TEXT_STAGE=1 sh scripts/train.sh 1 29501 affinity_min endoscapes_seg50
#   JOINT_TEXT_STAGE=1 EDGE_ENHANCE=1 sh scripts/train.sh 1 29502 full endoscapes_seg50

NGPU=${1:-1}
PORT=${2:-29500}
VARIANT=${3:-base}
DATASET=${4:-endoscapes_seg50}

METHOD=unimatch_v2
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# ---- auto-pick a free port if requested one is busy ----
# Walks PORT, PORT+1, ... up to PORT+PORT_SCAN_MAX (default 30) and picks
# the first one no listener is bound to.  Falls back to a random ephemeral
# port if nothing free is found in that range.
is_port_free() {
    if command -v ss >/dev/null 2>&1; then
        ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${1}\$"
    elif command -v netstat >/dev/null 2>&1; then
        ! netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${1}\$"
    elif command -v lsof >/dev/null 2>&1; then
        ! lsof -iTCP:${1} -sTCP:LISTEN -P -n >/dev/null 2>&1
    else
        # last resort: python bind probe
        python - <<PY 2>/dev/null
import socket, sys
s = socket.socket()
try:
    s.bind(('127.0.0.1', ${1})); s.close(); sys.exit(0)
except OSError:
    sys.exit(1)
PY
        [ $? -eq 0 ]
    fi
}
pick_free_port() {
    start=$1
    span=${PORT_SCAN_MAX:-30}
    p=$start
    end=$((start + span))
    while [ "$p" -lt "$end" ]; do
        if is_port_free "$p"; then
            echo "$p"
            return 0
        fi
        p=$((p + 1))
    done
    # random ephemeral fallback
    python - <<'PY'
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0))
print(s.getsockname()[1]); s.close()
PY
}
ORIG_PORT=$PORT
NEW_PORT=$(pick_free_port "$PORT")
if [ -n "$NEW_PORT" ] && [ "$NEW_PORT" != "$PORT" ]; then
    echo "[train.sh] port $ORIG_PORT busy -> switching to $NEW_PORT"
    PORT=$NEW_PORT
fi

case "$VARIANT" in
    base)            FLAGS="" ;;
    temporal)        FLAGS="--temporal-consistency" ;;
    cls)             FLAGS="--cls-head" ;;
    affinity)        FLAGS="--cls-head" ;;
    affinity_min)    FLAGS="" ;;
    full)            FLAGS="--temporal-consistency" ;;
    full_cls)        FLAGS="--temporal-consistency --cls-head" ;;
    *) echo "unknown variant: $VARIANT  supported: base|temporal|affinity_min|cls|affinity|full|full_cls"; exit 1 ;;
esac

# Text-enhancement side branch. The documented text-only variant is
# affinity_min. A legacy combined text variant also uses this path.
if [ "$VARIANT" = "affinity" ] || [ "$VARIANT" = "affinity_min" ] || [ "$VARIANT" = "full" ] || [ "$VARIANT" = "full_cls" ]; then
    if [ -z "$AFFINITY_WARMSTART" ] && [ "${JOINT_TEXT_STAGE:-0}" != "1" ]; then
        echo "ERROR: variant '$VARIANT' requires either JOINT_TEXT_STAGE=1 or AFFINITY_WARMSTART=/path/affinity_<ds>.pt"
        exit 1
    fi
    [ -n "$AFFINITY_WARMSTART" ] && FLAGS="$FLAGS --affinity-warmstart $AFFINITY_WARMSTART"
    [ -n "$AFFINITY_AUX_WEIGHT" ]    && FLAGS="$FLAGS --affinity-aux-weight $AFFINITY_AUX_WEIGHT"
    [ -n "$AFFINITY_FREEZE_WARMUP" ] && FLAGS="$FLAGS --affinity-freeze-warmup $AFFINITY_FREEZE_WARMUP"
    [ -n "$AFFINITY_PLATEAU_EPS" ]   && FLAGS="$FLAGS --affinity-plateau-eps $AFFINITY_PLATEAU_EPS"
    [ -n "$AFFINITY_UNFREEZE_LR_MULT" ] && FLAGS="$FLAGS --affinity-unfreeze-lr-mult $AFFINITY_UNFREEZE_LR_MULT"
fi

[ -n "$LR" ]     && FLAGS="$FLAGS --lr $LR"
[ -n "$EPOCHS" ] && FLAGS="$FLAGS --epochs $EPOCHS"
[ -n "$CROP" ]   && FLAGS="$FLAGS --crop-size $CROP"
[ "${VISUAL_ADAPTER:-0}" = "1" ] && FLAGS="$FLAGS --visual-adapter"
[ -n "${VISUAL_ADAPTER_REDUCTION:-}" ] && FLAGS="$FLAGS --visual-adapter-reduction $VISUAL_ADAPTER_REDUCTION"
[ -n "${VISUAL_ADAPTER_DROPOUT:-}" ] && FLAGS="$FLAGS --visual-adapter-dropout $VISUAL_ADAPTER_DROPOUT"
[ "${JOINT_TEXT_STAGE:-0}" = "1" ] && FLAGS="$FLAGS --joint-text-stage"
[ -n "${JOINT_TEXT_LR_MULT:-}" ] && FLAGS="$FLAGS --joint-text-lr-mult $JOINT_TEXT_LR_MULT"
[ -n "${JOINT_TEXT_WARMUP:-}" ] && FLAGS="$FLAGS --joint-text-warmup $JOINT_TEXT_WARMUP"
[ "${EDGE_ENHANCE:-0}" = "1" ] && FLAGS="$FLAGS --edge-enhance"
[ -n "${EDGE_BOUNDARY_WEIGHT:-}" ] && FLAGS="$FLAGS --edge-boundary-weight $EDGE_BOUNDARY_WEIGHT"
[ -n "${EDGE_CONSISTENCY_WEIGHT:-}" ] && FLAGS="$FLAGS --edge-consistency-weight $EDGE_CONSISTENCY_WEIGHT"
[ -n "${EDGE_WARMUP:-}" ] && FLAGS="$FLAGS --edge-warmup $EDGE_WARMUP"
# ---- Mask-Guided Edge Refiner (MGER) ----
[ "${EDGE_REFINER:-0}" = "1" ] && FLAGS="$FLAGS --edge-refiner"
[ -n "${EDGE_REFINER_WEIGHT:-}" ] && FLAGS="$FLAGS --edge-refiner-weight $EDGE_REFINER_WEIGHT"
[ -n "${EDGE_REFINER_THICKNESS:-}" ] && FLAGS="$FLAGS --edge-refiner-thickness $EDGE_REFINER_THICKNESS"
[ -n "${EDGE_REFINER_POS_WEIGHT:-}" ] && FLAGS="$FLAGS --edge-refiner-pos-weight $EDGE_REFINER_POS_WEIGHT"
[ "${EDGE_REFINER_CHECKPOINT:-0}" = "1" ] && FLAGS="$FLAGS --edge-refiner-checkpoint"
[ -n "${EMA_DECAY_CAP:-}" ] && FLAGS="$FLAGS --ema-decay-cap $EMA_DECAY_CAP"
[ -n "${BACKBONE:-}" ] && FLAGS="$FLAGS --backbone $BACKBONE"
[ -n "${BACKBONE_CHECKPOINT:-}" ] && FLAGS="$FLAGS --backbone-checkpoint $BACKBONE_CHECKPOINT"
# ---- Temporal Consistency Regularization (TCR, DA-VSN style) ----
[ "${TEMPORAL_CONSISTENCY:-0}" = "1" ] && FLAGS="$FLAGS --temporal-consistency"
[ -n "${TEMPORAL_WEIGHT:-}" ] && FLAGS="$FLAGS --temporal-weight $TEMPORAL_WEIGHT"
[ -n "${TEMPORAL_WARMUP:-}" ] && FLAGS="$FLAGS --temporal-warmup $TEMPORAL_WARMUP"
[ "${TEMPORAL_ORIGINAL:-0}" = "1" ] && FLAGS="$FLAGS --temporal-original"
# ---- TS-MDR (Text-Semantic Morphology Dynamic Router) ----
[ "${TSMDR:-0}" = "1" ] && FLAGS="$FLAGS --tsmdr-enabled"
[ -n "${TSMDR_WEIGHT:-}" ] && FLAGS="$FLAGS --tsmdr-weight $TSMDR_WEIGHT"
[ -n "${TSMDR_WARMUP:-}" ] && FLAGS="$FLAGS --tsmdr-warmup $TSMDR_WARMUP"
[ -n "${TSMDR_RADIUS:-}" ] && FLAGS="$FLAGS --tsmdr-radius $TSMDR_RADIUS"
[ "${TSMDR_USE_EDGE:-0}" = "1" ] && FLAGS="$FLAGS --tsmdr-use-edge"
[ "${TSMDR_SHAPE_ONLY:-0}" = "1" ] && FLAGS="$FLAGS --tsmdr-shape-only"
[ -n "$EXTRA" ]  && FLAGS="$FLAGS $EXTRA"

RATE=${RATE:-0.25}                     # label rate: 0.10 | 0.20 | 0.25 | 0.30 | 0.50
SPLIT_ROOT=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}/unimatch_splits_${DATASET}_${RATE}_seed42
EXP_ROOT=${EXP_ROOT:-/root/autodl-tmp/exp}

config=configs/${DATASET}.yaml
labeled_id_path=${SPLIT_ROOT}/labeled.txt
unlabeled_id_path=${SPLIT_ROOT}/unlabeled.txt
val_id_path=${SPLIT_ROOT}/val.txt

USER_TAG=${TAG:-}
make_tag() {
    cur_bs=$1
    if [ -n "$USER_TAG" ]; then
        printf '%s\n' "$USER_TAG"
        return
    fi
    tag=${VARIANT}_r${RATE}
    [ -n "$cur_bs" ] && tag="${tag}_bs${cur_bs}"
    [ -n "$LR" ]     && tag="${tag}_lr${LR}"
    [ -n "$EPOCHS" ] && tag="${tag}_ep${EPOCHS}"
    [ -n "$CROP" ]   && tag="${tag}_cr${CROP}"
    if [ -n "${BACKBONE:-}" ]; then
        # short alias for backbone size in TAG (small / base / large / giant)
        bb_short=$(echo "$BACKBONE" | awk -F_ '{print $NF}')
        tag="${tag}_${bb_short}"
    fi
    [ "${VISUAL_ADAPTER:-0}" = "1" ] && tag="${tag}_va"
    [ "${JOINT_TEXT_STAGE:-0}" = "1" ] && tag="${tag}_joint"
    [ "${EDGE_ENHANCE:-0}" = "1" ] && tag="${tag}_edge"
    [ "${EDGE_ENHANCE:-0}" = "1" ] && [ "${EDGE_REFINER:-0}" = "1" ] && tag="${tag}mger"
    [ "${TEMPORAL_CONSISTENCY:-0}" = "1" ] && tag="${tag}_tcr"
    [ "${TEMPORAL_CONSISTENCY:-0}" = "1" ] && [ "${TEMPORAL_ORIGINAL:-0}" = "1" ] && tag="${tag}v1"
    [ "${TSMDR:-0}" = "1" ] && tag="${tag}_tsmdr"
    [ "${TSMDR:-0}" = "1" ] && [ "${TSMDR_SHAPE_ONLY:-0}" = "1" ] && tag="${tag}shape"
    printf '%s\n' "$tag"
}

is_oom_log() {
    log_file=$1
    grep -Eiq 'CUDA out of memory|out of memory|torch\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED|DefaultCPUAllocator|MemoryError' "$log_file"
}

AUTO_BS=${AUTO_BS:-1}
MIN_BS=${MIN_BS:-1}
cur_bs=${BS:-}

while :; do
    TAG=$(make_tag "$cur_bs")
    save_path=${EXP_ROOT}/${DATASET}/${METHOD}_${TAG}
    mkdir -p "$save_path"

    run_flags="$FLAGS"
    [ -n "$cur_bs" ] && run_flags="$run_flags --batch-size $cur_bs"

    LAUNCH_CMD="python -m torch.distributed.launch \
        --nproc_per_node=$NGPU \
        --master_addr=localhost \
        --master_port=$PORT \
        ${METHOD}.py \
        --config=$config \
        --labeled-id-path $labeled_id_path \
        --unlabeled-id-path $unlabeled_id_path \
        --val-id-path $val_id_path \
        --save-path $save_path \
        --port $PORT $run_flags"

    echo "[train.sh] launch TAG=$TAG BS=${cur_bs:-yaml} save_path=$save_path"
    eval "$LAUNCH_CMD" > "$save_path/out.log" 2>&1
    status=$?
    cat "$save_path/out.log"
    [ "$status" -eq 0 ] && exit 0

    if [ "$AUTO_BS" != "1" ] || [ -z "$cur_bs" ] || ! is_oom_log "$save_path/out.log"; then
        exit "$status"
    fi
    next_bs=$((cur_bs / 2))
    [ "$next_bs" -lt "$MIN_BS" ] && exit "$status"
    echo "[train.sh] CUDA OOM detected at BS=$cur_bs; retrying with BS=$next_bs"
    cur_bs=$next_bs
done
