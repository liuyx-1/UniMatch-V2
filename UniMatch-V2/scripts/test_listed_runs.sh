#!/usr/bin/env bash
# Test all run dirs listed below in parallel (3 at a time) and aggregate.
# Adjust the heredoc below to add / remove runs.
#
# Usage:
#   sh scripts/test_listed_runs.sh                  # full run
#   sh scripts/test_listed_runs.sh --skip-existing  # don't re-test if csv exists
#   sh scripts/test_listed_runs.sh --skip-smoke     # drop _ep1 smoke runs

set -u

DS=${DS:-endoscapes_seg50}
EXP_ROOT=${EXP_ROOT:-/root/autodl-tmp/exp}
SPLITS=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}
WORKERS=${WORKERS:-3}
DEVICE=${DEVICE:-cuda:0}

OUT=$EXP_ROOT/$DS/_batch_test
mkdir -p "$OUT"

python tools/test_runs_batch.py \
    --exp-root   "$EXP_ROOT/$DS" \
    --splits-root "$SPLITS" \
    --dataset    "$DS" \
    --workers    "$WORKERS" \
    --device     "$DEVICE" \
    --out-dir    "$OUT" \
    "$@" \
    --runs - <<'EOF'
unimatch_v2_affinity_min_cma_r0.10_bs8_lr5e-6_cr224
unimatch_v2_affinity_min_r0.10_bs16_lr5e-5_cr518_va_joint
unimatch_v2_affinity_min_r0.10_bs16_lr5e-6_cr518
unimatch_v2_affinity_min_r0.10_bs16_lr5e-6_cr518_va_joint
unimatch_v2_affinity_min_r0.10_bs2_lr5e-6_cr518
unimatch_v2_affinity_min_r0.10_bs2_lr5e-6_ep1_cr224
unimatch_v2_affinity_min_r0.10_bs4_lr5e-6_cr518
unimatch_v2_affinity_min_r0.10_bs8_lr5e-6_cr224
unimatch_v2_affinity_min_r0.10_bs8_lr5e-6_cr518
unimatch_v2_affinity_min_r0.10_bs8_lr5e-6_ep1_cr224
unimatch_v2_affinity_min_r0.10_va_joint_edge
unimatch_v2_affinity_min_r0.1_bs16_lr5e-6_cr518_va_joint
unimatch_v2_base_r0.10_bs16_lr5e-5_cr518_va
unimatch_v2_base_r0.10_bs8_lr5e-6_cr224
unimatch_v2_base_r0.10_va_edge
unimatch_v2_boundary_r0.10_bs16_lr5e-5_ep150_cr518_va
unimatch_v2_boundary_r0.10_bs8_lr5e-5_ep150_cr518_va
unimatch_v2_boundary_r0.10_va
unimatch_v2_debias_r0.10_bs16_lr5e-5_ep150_cr518_va
unimatch_v2_full_r0.10_bs16_lr1e-4_ep150_cr518_va_joint_edge_bnd_dbs
unimatch_v2_full_r0.10_bs16_lr1e-6_ep150_cr518_va_joint_edge_bnd_dbs
unimatch_v2_full_r0.10_bs16_lr2e-6_ep150_cr518_va_joint_edge_bnd_dbs
unimatch_v2_full_r0.10_bs16_lr5e-5_ep150_cr518_va
unimatch_v2_full_r0.10_bs16_lr5e-5_ep150_cr518_va_joint_edge_bnd_dbs
unimatch_v2_full_r0.10_va
EOF

echo
echo "[done] tables at $OUT/{batch_table.md,csv,tex}"
