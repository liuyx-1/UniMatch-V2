#!/usr/bin/env bash
# Three-module ablation (HPTA + LC-PAM + EDGE/MGER) + TCR, anchor datasets.
# TS-MDR and the instance-aware loss are explicitly disabled in every row here
# (--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled) so this sweep
# isolates exactly the four rows of Table abl: base / +text / +edge / +full.
#
# BS=12 empirically confirmed safe on this 24GB RTX 3090 (peak ~22.4-22.7GB
# through both training and full-resolution eval, epochs 1-4 of the "full"
# row). Same BS used for all rows for a fair comparison; lighter rows (no
# edge/temporal) will have MORE headroom than the "full" row that was probed.
#
# Runs sequentially (single GPU) via wait_gpu_free, same pattern as
# scripts/run_surgical_bcp_fixed_all.sh. Safe to leave running unattended.
set -u

cd /root/autodl-tmp/code/UniMatch-V2_local
export PATH=/root/autodl-tmp/envs/unimatchv2/bin:$PATH
source /etc/network_turbo 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

BS=12
EPOCHS=80
NGPU=1
EXTRA_COMMON='--no-tsmdr --no-instance-loss --no-instance-loss-unlabeled'
mkdir -p logs

wait_gpu_free() {
  local need_free=${1:-3}
  local ok=0
  while true; do
    local used
    used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END{print s+0}')
    if [ "$used" -lt 1000 ]; then
      ok=$((ok + 1))
      echo "[$(date '+%F %T')] GPU free check $ok/$need_free: ${used} MiB"
      [ "$ok" -ge "$need_free" ] && return 0
    else
      ok=0
      echo "[$(date '+%F %T')] GPU busy: ${used} MiB; waiting"
    fi
    sleep 60
  done
}

run_row () {
  local dataset=$1 variant=$2 tag=$3 port=$4
  shift 4
  local extra_env="$*"
  echo
  echo "=== TRAIN dataset=${dataset} variant=${variant} tag=${tag} start $(date) ==="
  wait_gpu_free 3
  eval "RATE=0.10 BS=$BS EPOCHS=$EPOCHS EXTRA='${EXTRA_COMMON}' TAG=${tag} ${extra_env} \
        bash scripts/train.sh $NGPU $port $variant $dataset"
  echo "=== DONE dataset=${dataset} variant=${variant} tag=${tag} $(date) ==="
}

# ---- Endoscapes-Seg50 (r=0.10) ----
# NOTE: the "full" row (row 4) for this dataset was already launched
# separately (probe_full_bs12, port 29601) BEFORE this script started —
# do not relaunch it here, or it'll double-train the same tag.
# Table abl is CUMULATIVE: "+text" = HPTA+LC-PAM, "+edge" = HPTA+LC-PAM+EDGE.
run_row endoscapes_seg50 base          ablation_base_r0.10          29610
run_row endoscapes_seg50 affinity_min  ablation_text_r0.10          29611 VISUAL_ADAPTER=1 JOINT_TEXT_STAGE=1
run_row endoscapes_seg50 base          ablation_edge_r0.10          29612 VISUAL_ADAPTER=1 JOINT_TEXT_STAGE=1 EDGE_ENHANCE=1 EDGE_REFINER=1

# ---- EndoVis2018 (r=0.10) ----
run_row endovis2018 base          ablation_base_r0.10          29620
run_row endovis2018 affinity_min  ablation_text_r0.10          29621 VISUAL_ADAPTER=1 JOINT_TEXT_STAGE=1
run_row endovis2018 base          ablation_edge_r0.10          29622 VISUAL_ADAPTER=1 JOINT_TEXT_STAGE=1 EDGE_ENHANCE=1 EDGE_REFINER=1
run_row endovis2018 full          ablation_full_r0.10          29623 VISUAL_ADAPTER=1 JOINT_TEXT_STAGE=1 EDGE_ENHANCE=1 EDGE_REFINER=1

echo
echo "[$(date '+%F %T')] module ablation queue finished"
