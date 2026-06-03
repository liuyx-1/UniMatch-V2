#!/usr/bin/env bash
# Train + test Needle (3-class) with a ResNet backbone — the ORIGINAL
# UniMatch-V1 protocol (pre-DINOv2 era).
#
# Prereqs (run once):
#   1. Pretrained ResNet weights under ./pretrained/  (see download block at top)
#   2. A ResNet-flavoured config at configs/needle_resnet.yaml  (auto-created
#      below if missing — points configs/needle.yaml to ResNet backbone)
#
# Usage:
#   sh scripts/train_needle_resnet.sh                 # train + test
#   sh scripts/train_needle_resnet.sh test-only
#   BACKBONE=resnet101 sh scripts/train_needle_resnet.sh
#
# Override env vars: RATE BS LR CROP EPOCHS NGPU BACKBONE

set -u
MODE=${1:-full}

DS=needle
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/autonomous_surgery/public_data}
SPLITS=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}
EXP_ROOT=${EXP_ROOT:-/root/autodl-tmp/exp}
PRETRAINED=${PRETRAINED:-./pretrained}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

BACKBONE=${BACKBONE:-resnet50}        # resnet50 | resnet101
RATE=${RATE:-0.10}
BS=${BS:-8}                           # ResNet-50 fits BS=8 at 512 on 24 GB
LR=${LR:-1e-3}                        # original UniMatch ResNet LR (much higher than DINOv2's)
CROP=${CROP:-512}                     # standard ResNet UniMatch crop
EPOCHS=${EPOCHS:-80}
NGPU=${NGPU:-1}
PORT=${PORT:-29560}

# ---------------- download instructions ----------------
PT_R50=$PRETRAINED/resnet50.pth
PT_R101=$PRETRAINED/resnet101.pth

case "$BACKBONE" in
    resnet50)  PT=$PT_R50  ;;
    resnet101) PT=$PT_R101 ;;
    *) echo "[err] unknown BACKBONE=$BACKBONE (use resnet50 or resnet101)"; exit 1 ;;
esac

if [ ! -f "$PT" ]; then
    echo "[!] pretrained backbone not found: $PT"
    echo
    echo "    Download with ONE of the following:"
    echo
    echo "    # (A) Torchvision ImageNet weights (recommended, IMAGENET1K_V2):"
    echo "    mkdir -p $PRETRAINED"
    echo "    python - <<PY"
    echo "    import torch, torchvision.models as M"
    echo "    sd = M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2).state_dict()  # or resnet101"
    echo "    torch.save(sd, '$PT')"
    echo "    PY"
    echo
    echo "    # (B) Original UniMatch released checkpoints:"
    echo "    wget -O $PT_R50  https://github.com/LiheYoung/UniMatch/releases/download/v0.1/resnet50.pth"
    echo "    wget -O $PT_R101 https://github.com/LiheYoung/UniMatch/releases/download/v0.1/resnet101.pth"
    echo
    echo "    # (C) MMSegmentation 'imagenet' converted checkpoints (BGR-ordered):"
    echo "    wget -O $PT_R50  https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth"
    echo "    wget -O $PT_R101 https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth"
    exit 1
fi

# ---------------- write a ResNet-flavoured needle config ----------------
RES_CFG=configs/needle_${BACKBONE}.yaml
if [ ! -f "$RES_CFG" ]; then
    echo "[info] creating $RES_CFG from configs/needle.yaml"
    python - <<PY
import yaml, sys
with open('configs/needle.yaml') as f:
    cfg = yaml.safe_load(f)
# Swap backbone to ResNet
cfg['model']                 = 'deeplabv3plus'        # ResNet-compatible head (UniMatch v1 default)
cfg['backbone']              = '${BACKBONE}'
cfg['backbone_checkpoint']   = '${PT}'
cfg.pop('lock_backbone', None)
cfg['lr']                    = ${LR}
cfg['lr_multi']              = 10.0
cfg['batch_size']            = ${BS}
cfg['crop_size']             = ${CROP}
cfg['epochs']                = ${EPOCHS}
# ResNet backbones do not need patch-14 alignment
with open('${RES_CFG}', 'w') as f:
    yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)
print('[ok] wrote ${RES_CFG}')
PY
fi

TAG="base_${BACKBONE}_r${RATE}_bs${BS}_lr${LR}_ep${EPOCHS}_cr${CROP}"
SPLIT_ROOT=$SPLITS/unimatch_splits_${DS}_${RATE}_seed42

echo "=================================================================="
echo "[$(date +%H:%M:%S)] Needle · UniMatch with $BACKBONE backbone"
echo "                   config=$RES_CFG  pretrained=$PT"
echo "                   RATE=$RATE BS=$BS LR=$LR CROP=$CROP EPOCHS=$EPOCHS"
echo "                   TAG=$TAG"
echo "=================================================================="

LAUNCH () {
    SCRIPT=$1; shift
    python -m torch.distributed.launch \
        --nproc_per_node=$NGPU --master_addr=localhost --master_port=$PORT \
        $SCRIPT \
        --config=$RES_CFG \
        --labeled-id-path  $SPLIT_ROOT/labeled.txt \
        --unlabeled-id-path $SPLIT_ROOT/unlabeled.txt \
        --val-id-path      $SPLIT_ROOT/val.txt \
        --save-path        $EXP_ROOT/$DS/unimatch_v2_$TAG \
        --port $PORT "$@"
}

if [ "$MODE" != "test-only" ]; then
    echo
    echo "[$(date +%H:%M:%S)] >>> TRAIN"
    mkdir -p "$EXP_ROOT/$DS/unimatch_v2_$TAG"
    LAUNCH unimatch_v2.py
fi

echo
echo "[$(date +%H:%M:%S)] >>> TEST"
TEST_ID=$SPLIT_ROOT/test.txt
[ -f "$TEST_ID" ] || TEST_ID=$SPLIT_ROOT/val.txt
python test.py \
    --config $RES_CFG \
    --checkpoint $EXP_ROOT/$DS/unimatch_v2_$TAG/best.pth \
    --id-path $TEST_ID \
    --train-id-path $SPLIT_ROOT/labeled.txt \
    --use-ema --ece --ap \
    --save-csv $EXP_ROOT/$DS/unimatch_v2_$TAG/test_metrics.csv

echo
echo "[done] artefacts: $EXP_ROOT/$DS/unimatch_v2_$TAG/"
