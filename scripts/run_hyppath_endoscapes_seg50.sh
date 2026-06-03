#!/bin/bash
# ============================================================================
#  Sequential pipeline: hyperbolic_pathway (改进 1+4 完整版) on endoscapes_seg50
#
#  Stage-I:   train one hyperbolic-pathway affinity ckpt (40 epochs).
#  Stage-II:  for each labeled ratio, train + test affinity_min using that ckpt.
#
#  Stage-II is launched ONLY after Stage-I writes its checkpoint AND we verify
#  the ckpt contains a populated `hyp_pathway_state` field.  Each step's stdout
#  is teed to its own log under $EXP_ROOT/_logs/ so the master log stays clean.
#
#  Usage:
#      cd /root/autodl-tmp/code/UniMatch-V2
#      bash scripts/run_hyppath_endoscapes_seg50.sh [RATIOS...]
#
#  Examples:
#      bash scripts/run_hyppath_endoscapes_seg50.sh                     # all 4 ratios
#      bash scripts/run_hyppath_endoscapes_seg50.sh 0.10                # only 10 %
#      bash scripts/run_hyppath_endoscapes_seg50.sh 0.10 0.30           # 10 % + 30 %
# ============================================================================
set -Eeuo pipefail

# ─── conda env ──────────────────────────────────────────────────────────────
# If already in the right env (e.g. running from an interactive shell that
# already did `conda activate unimatchv2`) we skip activation entirely.
if [ "${CONDA_DEFAULT_ENV:-}" != "unimatchv2" ]; then
    CONDA_SH=""
    for cand in \
        /opt/conda/etc/profile.d/conda.sh \
        /root/miniconda3/etc/profile.d/conda.sh \
        /root/anaconda3/etc/profile.d/conda.sh \
        /root/autodl-tmp/miniconda3/etc/profile.d/conda.sh \
        /root/autodl-tmp/conda/etc/profile.d/conda.sh \
        $HOME/miniconda3/etc/profile.d/conda.sh \
        $HOME/anaconda3/etc/profile.d/conda.sh ; do
        if [ -f "$cand" ]; then
            CONDA_SH=$cand; break
        fi
    done
    if [ -n "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        conda activate unimatchv2
        echo "[env] activated unimatchv2 via $CONDA_SH"
    else
        # Fall back: just use the env's python directly via $PATH
        if [ -x /root/autodl-tmp/envs/unimatchv2/bin/python ]; then
            export PATH=/root/autodl-tmp/envs/unimatchv2/bin:$PATH
            echo "[env] using /root/autodl-tmp/envs/unimatchv2/bin via PATH"
        else
            echo "[FATAL] cannot locate unimatchv2 conda env" >&2
            exit 1
        fi
    fi
else
    echo "[env] already in CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
fi
echo "[env] python = $(which python)"
echo "[env] torch  = $(python -c 'import torch; print(torch.__version__)')"

# ─── paths ──────────────────────────────────────────────────────────────────
export DATA_ROOT=/root/autodl-tmp/data/autonomous_surgery/public_data
export SPLITS=/root/autodl-tmp/data/autonomous_surgery/splits
export EXP_ROOT=/root/autodl-tmp/exp
export SIGLIP=/root/autodl-tmp/data/autonomous_surgery/siglip_train
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/autodl-tmp/code/UniMatch-V2
chmod +x scripts/*.sh

LOGDIR=$EXP_ROOT/_logs
mkdir -p "$LOGDIR" "$EXP_ROOT"

# ─── hyperparameters ────────────────────────────────────────────────────────
DS=endoscapes_seg50

# Stage-I
S1_EPOCHS=40
S1_BS=16
S1_LR=1e-3
S1_OUT=$SIGLIP/affinity_per_ds_hyp_path           # ← formal ckpt location

# Stage-II
S2_BS=8
S2_LR=5e-6
S2_CROP=224
S2_PORT=30503                                      # base port; per-ratio +offset
# Allow override via CLI args; default = all four
RATIOS=( "$@" )
if [ ${#RATIOS[@]} -eq 0 ]; then
    RATIOS=( 0.10 0.30 0.50 1.00 )
fi

AFF=$S1_OUT/$DS/affinity_$DS.pt

# ─── helpers ────────────────────────────────────────────────────────────────
ts()   { date '+%Y-%m-%d %H:%M:%S'; }
log()  { echo "[$(ts)] $*"; }
hdr()  { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }
fail() { echo; echo "[FATAL] $*" >&2; exit 1; }

# Trap any unexpected error and dump where it was
trap 'echo; echo "[ERROR] line $LINENO exited with status $? while running: $BASH_COMMAND" >&2' ERR

# ─── pre-flight ─────────────────────────────────────────────────────────────
hdr "Pre-flight"

log "checking source files…"
for f in util/hyperbolic.py model/hyperbolic_fusion.py model/affinity_cls.py \
         util/affinity_side.py tools/train_affinity_per_ds.py \
         model/semseg/dpt.py unimatch_v2.py test.py \
         scripts/train.sh scripts/test.sh; do
    [ -f "$f" ] || fail "missing source file: $f"
done

log "checking imports…"
python - <<'PY' || fail "import smoke failed"
from util.hyperbolic import expmap0, logmap0, mobius_add, hyp_weighted_mean, HypLinear, HypMLP
from model.hyperbolic_fusion import HyperbolicFusionBlock
from model.affinity_cls import AffinityClassifier
from util.affinity_side import AffinitySideBranch
print('  imports OK')
PY

log "checking Stage-I manifests exist…"
[ -f "$SIGLIP/manifest_${DS}.jsonl" ] \
    || fail "missing $SIGLIP/manifest_${DS}.jsonl — build it with build_siglip_manifest.py first"
[ -f "$SIGLIP/merged/global_classes.json" ] \
    || fail "missing $SIGLIP/merged/global_classes.json — run build_merged_manifest.py"
[ -f "$SIGLIP/merged_val/manifest_${DS}_val.jsonl" ] \
    || fail "missing $SIGLIP/merged_val/manifest_${DS}_val.jsonl — run build_val_manifest.py"

log "checking splits exist for the requested ratios…"
for R in "${RATIOS[@]}"; do
    D=$SPLITS/unimatch_splits_${DS}_${R}_seed42
    for fn in labeled.txt unlabeled.txt val.txt; do
        [ -f "$D/$fn" ] || fail "missing $D/$fn"
    done
done

log "checking DINOv2 backbone weights…"
[ -f /root/autodl-tmp/code/UniMatch-V2/pretrained/dinov2_vitb14_pretrain.pth ] \
    || fail "missing pretrained/dinov2_vitb14_pretrain.pth"

log "killing any stale unimatch_v2.py / launcher processes…"
pkill -9 -f 'torch.distributed.launch' 2>/dev/null || true
pkill -9 -f 'unimatch_v2.py'           2>/dev/null || true
sleep 2

# ─── Stage I ────────────────────────────────────────────────────────────────
hdr "STAGE-I  hyperbolic_pathway  ds=$DS  epochs=$S1_EPOCHS"

if [ -f "$AFF" ]; then
    log "Stage-I ckpt already exists at $AFF"
    log "  → checking metric field…"
    METRIC=$(python -c "
import torch
ck=torch.load('$AFF',map_location='cpu',weights_only=False)
print(ck.get('affinity_metric',''))
")
    if [ "$METRIC" = "hyperbolic_pathway" ]; then
        log "  ckpt is hyperbolic_pathway — skipping Stage-I training"
        SKIP_S1=1
    else
        log "  existing ckpt has metric='$METRIC' (not hyperbolic_pathway) → re-training"
        SKIP_S1=0
    fi
else
    SKIP_S1=0
fi

S1_LOG=$LOGDIR/stage1_${DS}_hyppath_$(date +%Y%m%d_%H%M%S).log
if [ "$SKIP_S1" = "0" ]; then
    log "writing Stage-I log to $S1_LOG"
    mkdir -p "$S1_OUT"
    python tools/train_affinity_per_ds.py \
        --manifest-dir "$SIGLIP" \
        --val-dir      "$SIGLIP/merged_val" \
        --out-dir      "$S1_OUT" \
        --datasets     "$DS" \
        --model        google/siglip2-base-patch16-256 \
        --vision-backbone dinov2 --dinov2-input-size 518 \
        --epochs   "$S1_EPOCHS" \
        --bs       "$S1_BS" \
        --lr       "$S1_LR" \
        --lr-vision 1e-5 --lr-text 1e-5 \
        --lambda-patch 1.0 --patch-grid 37 \
        --topk 5 --eval-every 5 \
        --affinity-metric hyperbolic_pathway \
        --hyperbolic-dim 128 \
        --hyperbolic-curvature-init 1.0 \
        --hyperbolic-temperature-init 0.1 \
        --hyperbolic-distance-scale 1.0 \
        --hyperbolic-pathway-layers 2 \
        --hyperbolic-pathway-alpha-init -3.0 \
        2>&1 | tee "$S1_LOG"
    [ -f "$AFF" ] || fail "Stage-I finished but ckpt $AFF not found"
fi

hdr "STAGE-I  CKPT VERIFY"
python - <<PY || fail "Stage-I ckpt verification failed"
import torch, sys
F="$AFF"
ck=torch.load(F, map_location='cpu', weights_only=False)
print(f"  metric            : {ck.get('affinity_metric')}")
print(f"  hyp_pathway_state : {'present' if ck.get('hyp_pathway_state') else 'MISSING'}")
print(f"  pathway layers    : {ck.get('hyp_pathway_layers')}")
print(f"  pathway alpha init: {ck.get('hyp_pathway_alpha_init')}")
pst = ck.get('hyp_pathway_state', {})
must = ['visual_proj','v_pre_ln','T_h_param','rho','log_tau_h','hyp_up','alpha_bias','cls_bias']
missing = [k for k in must if k not in pst]
if missing:
    print('  MISSING pathway keys:', missing)
    sys.exit(1)
print('  all required pathway fields present ✓')
PY

# ─── Stage II per ratio ─────────────────────────────────────────────────────
RESULTS_TSV=$LOGDIR/results_${DS}_hyppath.tsv
echo -e "ratio\tmIoU\tDice\tfwIoU\trun_dir" > "$RESULTS_TSV"

OFFSET=0
for R in "${RATIOS[@]}"; do
    PORT=$(( S2_PORT + OFFSET ))
    OFFSET=$(( OFFSET + 1 ))

    TAG=affinity_min_hyppath_r${R}_bs${S2_BS}_lr${S2_LR}_cr${S2_CROP}
    RUN=$EXP_ROOT/$DS/unimatch_v2_${TAG}
    S2_LOG=$LOGDIR/stage2_${DS}_r${R}_$(date +%Y%m%d_%H%M%S).log

    hdr "STAGE-II TRAIN  ratio=$R  port=$PORT"
    log "TAG     = $TAG"
    log "log     = $S2_LOG"

    RATE=$R BS=$S2_BS LR=$S2_LR CROP=$S2_CROP \
        AFFINITY_WARMSTART="$AFF" TAG="$TAG" \
        sh scripts/train.sh 1 "$PORT" affinity_min "$DS" \
        2>&1 | tee "$S2_LOG"

    [ -f "$RUN/best.pth" ] || fail "Stage-II r=$R finished but $RUN/best.pth not found"

    hdr "STAGE-II TEST   ratio=$R"
    RATE=$R BS=$S2_BS LR=$S2_LR CROP=$S2_CROP \
        AFFINITY_WARMSTART="$AFF" TAG="$TAG" \
        sh scripts/test.sh affinity_min "$DS" \
        2>&1 | tee -a "$S2_LOG"

    CSV=$RUN/test_metrics.csv
    if [ -f "$CSV" ]; then
        MIOU=$(awk -F, '/^summary,miou_present/{printf "%.2f",100*$3}' "$CSV")
        DICE=$(awk -F, '/^summary,mdice_present/{printf "%.2f",100*$3}' "$CSV")
        FWIOU=$(awk -F, '/^summary,fwiou/{printf "%.2f",100*$3}' "$CSV")
        echo -e "${R}\t${MIOU}\t${DICE}\t${FWIOU}\t${RUN}" >> "$RESULTS_TSV"
        log "  r=$R  mIoU=${MIOU}%  Dice=${DICE}%  fwIoU=${FWIOU}%"
    else
        log "  r=$R  (test_metrics.csv missing)"
        echo -e "${R}\t-\t-\t-\t${RUN}" >> "$RESULTS_TSV"
    fi

    pkill -9 -f 'torch.distributed.launch' 2>/dev/null || true
    pkill -9 -f 'unimatch_v2.py'           2>/dev/null || true
    sleep 2
done

# ─── final summary ──────────────────────────────────────────────────────────
hdr "SUMMARY  hyperbolic_pathway on $DS"
column -ts $'\t' "$RESULTS_TSV"
echo
log "all logs        : $LOGDIR"
log "results table   : $RESULTS_TSV"
log "Stage-I ckpt    : $AFF"
log "Stage-II runs   : $EXP_ROOT/$DS/unimatch_v2_affinity_min_hyppath_r*"
echo
log "DONE"
