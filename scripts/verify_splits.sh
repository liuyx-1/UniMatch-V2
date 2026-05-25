#!/bin/bash
# Print labeled / unlabeled counts for every (dataset, ratio) combination
# under $SPLITS. Missing files show as '-' (no shell errors).
#
# Usage:
#   sh scripts/verify_splits.sh                       # all 3 datasets
#   sh scripts/verify_splits.sh endoscapes_seg50      # one dataset

SPLITS=${SPLITS:-/data/splits}
ONLY=${1:-}
RATIOS="0.10 0.20 0.25 0.30 0.50"

count() { [ -f "$1" ] && wc -l < "$1" || echo "-"; }

datasets="endoscapes_seg50 endoscapes cholecseg8k"
[ -n "$ONLY" ] && datasets="$ONLY"

printf '%-22s %-6s %8s %10s %10s %10s\n' DATASET RATIO LABELED UNLABELED VAL TEST
printf '%s\n' '----------------------------------------------------------------------'
for ds in $datasets; do
    for r in $RATIOS; do
        D=$SPLITS/unimatch_splits_${ds}_${r}_seed42
        if [ -d "$D" ]; then
            printf '%-22s %-6s %8s %10s %10s %10s\n' \
                "$ds" "$r" \
                "$(count "$D/labeled.txt")" \
                "$(count "$D/unlabeled.txt")" \
                "$(count "$D/val.txt")" \
                "$(count "$D/test.txt")"
        else
            printf '%-22s %-6s %8s\n' "$ds" "$r" "MISSING"
        fi
    done
    echo
done
