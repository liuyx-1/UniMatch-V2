#!/bin/bash
# Cross-Modal Adapter (PR 2025) — sequential Stage-I + Stage-II on endoscapes_seg50.
#
# Usage:
#     bash scripts/run_cma_endoscapes_seg50.sh [RATIOS...]
#     CMA_R=8 CMA_DS=32 bash scripts/run_cma_endoscapes_seg50.sh 0.10
set -Eeuo pipefail

# ── env ────────────────────────────────────────────────────────────────
if [ "${CONDA_DEFAULT_ENV:-}" != "unimatchv2" ]; then
    for cand in /opt/conda/etc/profile.d/conda.sh /root/miniconda3/etc/profile.d/conda.sh \
                /root/anaconda3/etc/profile.d/conda.sh \
                /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh \
                /root/autodl-tmp/conda/etc/profile.d/conda.sh ; do
        if [ -f "$cand" ]; then source "$cand"; conda activate unimatchv2; break; fi
    done
    if [ "${CONDA_DEFAULT_ENV:-}" != "unimatchv2" ]; then
        [ -x /root/autodl-tmp/envs/unimatchv2/bin/python ] && \
            export PATH=/root/autodl-tmp/envs/unimatchv2/bin:$PATH
    fi
fi

export DATA_ROOT=/root/autodl-tmp/data/autonomous_surgery/public_data
export SPLITS=/root/autodl-tmp/data/autonomous_surgery/splits
export EXP_ROOT=/root/autodl-tmp/exp
export SIGLIP=/root/autodl-tmp/data/autonomous_surgery/siglip_train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/autodl-tmp/code/UniMatch-V2

# ── adapter hyperparameters (env-overrideable) ─────────────────────────
DS=endoscapes_seg50
CMA_R=${CMA_R:-8}              # bottleneck (paper: 8)
CMA_DS=${CMA_DS:-32}           # shared up dim (paper: 16-32)
CMA_INIT_STD=${CMA_INIT_STD:-0.01}
CMA_DROP=${CMA_DROP:-0.1}

# Stage-I
S1_EPOCHS=${S1_EPOCHS:-80}
S1_BS=${S1_BS:-16}
S1_LR=${S1_LR:-3e-3}
S1_LR_VISION=${S1_LR_VISION:-2e-5}
S1_LR_TEXT=${S1_LR_TEXT:-2e-5}
S1_OUT=$SIGLIP/affinity_per_ds_cma

# Stage-II
S2_BS=${S2_BS:-16}
S2_LR=${S2_LR:-5e-6}
S2_CROP=${S2_CROP:-518}
S2_PORT=30523                                      # avoid clash with hyp_path runs
RESET_CMA_WEIGHTS=${RESET_CMA_WEIGHTS:-1}          # 1 = delete old Stage-I/II weights before running
EDGE_ENHANCE=${EDGE_ENHANCE:-1}
EDGE_BOUNDARY_WEIGHT=${EDGE_BOUNDARY_WEIGHT:-0.05}
EDGE_CONSISTENCY_WEIGHT=${EDGE_CONSISTENCY_WEIGHT:-0.01}
EDGE_WARMUP=${EDGE_WARMUP:-0}

LOGDIR=$EXP_ROOT/_logs
mkdir -p "$LOGDIR" "$EXP_ROOT" "$S1_OUT"

RATIOS=( "$@" )
[ ${#RATIOS[@]} -eq 0 ] && RATIOS=( 0.10 0.30 0.50 1.00 )

AFF=$S1_OUT/$DS/affinity_$DS.pt

ts(){ date '+%Y-%m-%d %H:%M:%S'; }
log(){ echo "[$(ts)] $*"; }
hdr(){ echo; echo "================================================="; echo "  $*"; echo "================================================="; }
fail(){ echo "[FATAL] $*" >&2; exit 1; }
trap 'echo "[ERROR] line $LINENO exited with status $? while running: $BASH_COMMAND" >&2' ERR

hdr "Pre-flight"
python -c "
from model.cross_modal_adapter import CrossModalAdapter, CMAFusionBlock
from model.affinity_cls import AffinityClassifier
from util.affinity_side import AffinitySideBranch
print('  imports OK')
print('  CMA defaults: r=$CMA_R d_s=$CMA_DS dropout=$CMA_DROP init_std=$CMA_INIT_STD')
" || fail "import smoke failed"

[ -f "$SIGLIP/manifest_${DS}.jsonl" ] || fail "missing $SIGLIP/manifest_${DS}.jsonl"
[ -f "$SIGLIP/merged_val/manifest_${DS}_val.jsonl" ] || fail "missing val manifest"

pkill -9 -f 'torch.distributed.launch' 2>/dev/null || true
pkill -9 -f 'unimatch_v2.py'           2>/dev/null || true
pkill -9 -f 'tools/train_affinity_per_ds.py' 2>/dev/null || true
sleep 2

# ── Stage I ────────────────────────────────────────────────────────────
hdr "STAGE-I  CMA  ds=$DS  r=$CMA_R  d_s=$CMA_DS  epochs=$S1_EPOCHS"
if [ "$RESET_CMA_WEIGHTS" = "1" ] && [ -d "$S1_OUT/$DS" ]; then
    log "RESET_CMA_WEIGHTS=1: deleting previous Stage-I output dir $S1_OUT/$DS"
    rm -rf "$S1_OUT/$DS"
    mkdir -p "$S1_OUT"
fi
if [ -f "$AFF" ]; then
    METRIC=$(python -c "import torch; ck=torch.load('$AFF',map_location='cpu',weights_only=False); print(ck.get('affinity_metric',''))")
    if [ "$METRIC" = "cma" ]; then
        log "ckpt already exists with metric='cma' — skipping Stage-I"
        SKIP_S1=1
    else
        log "ckpt has metric='$METRIC' (not cma) → re-training"
        SKIP_S1=0
    fi
else
    SKIP_S1=0
fi

S1_LOG=$LOGDIR/stage1_${DS}_cma_$(date +%Y%m%d_%H%M%S).log
if [ "$SKIP_S1" = "0" ]; then
    log "writing Stage-I log to $S1_LOG"
    log "fresh Stage-I start: epochs=1..$S1_EPOCHS lr=$S1_LR lr_vision=$S1_LR_VISION lr_text=$S1_LR_TEXT"
    python tools/train_affinity_per_ds.py \
        --manifest-dir "$SIGLIP" \
        --val-dir      "$SIGLIP/merged_val" \
        --out-dir      "$S1_OUT" \
        --datasets     "$DS" \
        --model        google/siglip2-base-patch16-256 \
        --vision-backbone dinov2 --dinov2-input-size 518 \
        --epochs  "$S1_EPOCHS"  --bs "$S1_BS"  --lr "$S1_LR" \
        --lr-vision "$S1_LR_VISION" --lr-text "$S1_LR_TEXT" \
        --lambda-patch 1.0 --patch-grid 37 \
        --topk 5 --eval-every 5 \
        --affinity-metric cma \
        --cma-reduction         "$CMA_R" \
        --cma-share-dim         "$CMA_DS" \
        --cma-adapter-dropout   "$CMA_DROP" \
        --cma-adapter-init-std  "$CMA_INIT_STD" \
        --cma-non-linearity     quick_gelu \
        --cma-fuse-dropout      "$CMA_DROP" \
        --cma-tau-init          0.07 \
        2>&1 | tee "$S1_LOG"
    [ -f "$AFF" ] || fail "Stage-I done but $AFF not found"
fi

hdr "STAGE-I  CKPT VERIFY"
python - <<PY || fail "Stage-I ckpt verification failed"
import torch, sys
ck=torch.load("$AFF", map_location='cpu', weights_only=False)
print("  metric          :", ck.get('affinity_metric'))
print("  cma_block_state :", 'present' if ck.get('cma_block_state') else 'MISSING')
print("  r (reduction)   :", ck.get('cma_reduction'))
print("  d_s (share_dim) :", ck.get('cma_share_dim'))
print("  non_linearity   :", ck.get('cma_non_linearity'))
cst=ck.get('cma_block_state', {})
must=['adapter','text_features_raw','log_tau','up_proj','rezero_gamma','cls_bias']
miss=[k for k in must if k not in cst]
if miss: print('  MISSING keys:', miss); sys.exit(1)
print('  all required fields present ✓')
PY

# ── Stage II per ratio ─────────────────────────────────────────────────
RESULTS_TSV=$LOGDIR/results_${DS}_cma.tsv
echo -e "ratio\tmIoU\tDice\tfwIoU\trun_dir" > "$RESULTS_TSV"

OFFSET=0
for R in "${RATIOS[@]}"; do
    PORT=$(( S2_PORT + OFFSET ))
    OFFSET=$(( OFFSET + 1 ))
    TAG=affinity_min_cma_r${R}_bs${S2_BS}_lr${S2_LR}_cr${S2_CROP}
    [ "$EDGE_ENHANCE" = "1" ] && TAG=${TAG}_edge
    RUN=$EXP_ROOT/$DS/unimatch_v2_${TAG}
    S2_LOG=$LOGDIR/stage2_${DS}_cma_r${R}_$(date +%Y%m%d_%H%M%S).log
    if [ "$RESET_CMA_WEIGHTS" = "1" ] && [ -d "$RUN" ]; then
        log "RESET_CMA_WEIGHTS=1: deleting previous Stage-II run $RUN"
        rm -rf "$RUN"
    fi

    hdr "STAGE-II TRAIN  ratio=$R  port=$PORT"
    RATE=$R BS=$S2_BS LR=$S2_LR CROP=$S2_CROP \
        EDGE_ENHANCE="$EDGE_ENHANCE" \
        EDGE_BOUNDARY_WEIGHT="$EDGE_BOUNDARY_WEIGHT" \
        EDGE_CONSISTENCY_WEIGHT="$EDGE_CONSISTENCY_WEIGHT" \
        EDGE_WARMUP="$EDGE_WARMUP" \
        AFFINITY_WARMSTART="$AFF" TAG="$TAG" \
        bash scripts/train.sh 1 "$PORT" affinity_min "$DS" \
        2>&1 | tee "$S2_LOG"

    [ -f "$RUN/best.pth" ] || fail "Stage-II r=$R done but $RUN/best.pth not found"

    hdr "STAGE-II TEST   ratio=$R"
    RATE=$R BS=$S2_BS LR=$S2_LR CROP=$S2_CROP \
        AFFINITY_WARMSTART="$AFF" TAG="$TAG" \
        bash scripts/test.sh affinity_min "$DS" \
        2>&1 | tee -a "$S2_LOG"

    CSV=$RUN/test_metrics.csv
    if [ -f "$CSV" ]; then
        MIOU=$(awk -F, '/^summary,miou_present/{printf "%.2f",100*$3}' "$CSV")
        DICE=$(awk -F, '/^summary,mdice_present/{printf "%.2f",100*$3}' "$CSV")
        FWIOU=$(awk -F, '/^summary,fwiou/{printf "%.2f",100*$3}' "$CSV")
        echo -e "${R}\t${MIOU}\t${DICE}\t${FWIOU}\t${RUN}" >> "$RESULTS_TSV"
        log "  r=$R  mIoU=${MIOU}%  Dice=${DICE}%  fwIoU=${FWIOU}%"
    else
        echo -e "${R}\t-\t-\t-\t${RUN}" >> "$RESULTS_TSV"
    fi
    pkill -9 -f 'torch.distributed.launch' 2>/dev/null || true
    pkill -9 -f 'unimatch_v2.py'           2>/dev/null || true
    sleep 2
done

hdr "SUMMARY  CMA on $DS  (r=$CMA_R, d_s=$CMA_DS)"
column -ts $'\t' "$RESULTS_TSV"
echo
log "logs:        $LOGDIR"
log "ckpt:        $AFF"
log "Stage-II runs: $EXP_ROOT/$DS/unimatch_v2_affinity_min_cma_r*"
log "DONE"
