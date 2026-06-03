#!/bin/bash
# Convert <processed>/splits/  →  $SPLITS/unimatch_splits_<ds>_<rate>_seed42/
#
# Handles two layouts:
#   (a) per-rate sub-dirs:  <processed>/splits/<NNN>/{labeled,unlabeled}.txt
#                           with val.txt/test.txt in the parent.
#   (b) single-rate flat:   <processed>/splits/{labeled,unlabeled,val[,test]}.txt
#       (cholecseg8k uses this; only one rate per --apply call)
#
# Creates symlinks (relative) so re-running the prepare scripts updates everything.
#
# Usage:
#   sh tools/link_unimatch_splits.sh                   # all datasets, all rates
#   sh tools/link_unimatch_splits.sh endoscapes_seg50  # one dataset
set -e

DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/data/autonomous_surgery/public_data}
SPLITS=${SPLITS:-/root/autodl-tmp/data/autonomous_surgery/splits}
SEED=${SEED:-42}
RATES=${RATES:-"0.10 0.20 0.25 0.30 0.50"}

mkdir -p "$SPLITS"

# Dataset key  →  processed-dir name
declare -A DS2DIR=(
  [endoscapes_seg50]=endoscapes_seg50_processed
  [cholecseg8k]=cholecseg8k_processed
  [endovis2018]=endovis2018_processed
  [endovis2017_parts]=endovis2017_parts_processed
  [endovis2017_type]=endovis2017_type_processed
)

ALL="endoscapes_seg50 cholecseg8k endovis2018 endovis2017_parts endovis2017_type"
ONLY=${1:-}
DATASETS=${ONLY:-$ALL}

for DS in $DATASETS; do
    sub="${DS2DIR[$DS]}"
    src_root="$DATA_ROOT/$sub/splits"
    if [ ! -d "$src_root" ]; then
        echo "[skip] $DS — no $src_root (run prepare_$DS.py --apply first)"
        continue
    fi
    echo "===== $DS ====="
    for r in $RATES; do
        rate_pct=$(awk -v r=$r 'BEGIN{printf "%03d", r*100}')   # 0.25 → 025
        tgt="$SPLITS/unimatch_splits_${DS}_${r}_seed${SEED}"
        mkdir -p "$tgt"

        # ---- labeled.txt / unlabeled.txt: try per-rate sub-dir first, then flat ----
        if [ -f "$src_root/$rate_pct/labeled.txt" ]; then
            ln -sfn "$src_root/$rate_pct/labeled.txt"   "$tgt/labeled.txt"
            ln -sfn "$src_root/$rate_pct/unlabeled.txt" "$tgt/unlabeled.txt"
            src_label="per-rate"
        elif [ -f "$src_root/labeled.txt" ]; then
            # Flat layout (cholecseg8k): only this rate is available
            ln -sfn "$src_root/labeled.txt"   "$tgt/labeled.txt"
            ln -sfn "$src_root/unlabeled.txt" "$tgt/unlabeled.txt"
            src_label="flat (rate-agnostic)"
        else
            echo "  rate=$r  [warn] no labeled.txt found under $src_root for this rate"
            continue
        fi

        # ---- val.txt and (if present) test.txt ----
        ln -sfn "$src_root/val.txt" "$tgt/val.txt"
        [ -f "$src_root/test.txt" ] && ln -sfn "$src_root/test.txt" "$tgt/test.txt"

        n_lab=$(wc -l < "$tgt/labeled.txt"   2>/dev/null || echo 0)
        n_unl=$(wc -l < "$tgt/unlabeled.txt" 2>/dev/null || echo 0)
        n_val=$(wc -l < "$tgt/val.txt"       2>/dev/null || echo 0)
        printf "  rate=%-5s  src=%-22s  labeled=%-5d  unlabeled=%-6d  val=%-5d  → %s\n" \
               "$r" "$src_label" "$n_lab" "$n_unl" "$n_val" "$tgt"
    done
done

echo
echo "[done] verify with: sh scripts/verify_splits.sh"
