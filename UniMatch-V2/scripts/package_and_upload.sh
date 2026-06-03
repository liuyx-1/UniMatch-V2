#!/usr/bin/env bash
# Bash variant of package_and_upload.ps1 — same defaults, same layout.
#
# Usage:
#   ./scripts/package_and_upload.sh           # pack only
#   ./scripts/package_and_upload.sh upload    # pack + bita upload
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="$(basename "$REPO_ROOT")"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_ROOT}/../_pack"
STAGING="${OUT_DIR}/${REPO_NAME}"
ZIP="${OUT_DIR}/${REPO_NAME}_${STAMP}.zip"

BITA_EP="${BITA_EP:-https://www.bitahub.com}"
BITA_BK="${BITA_BK:-b20260514152010197qfyxbs}"
PREFIX="${PREFIX:-code/}"

mkdir -p "$OUT_DIR"
rm -rf "$STAGING"
mkdir -p "$STAGING"

echo "[pack] repo  = $REPO_ROOT"
echo "[pack] out   = $ZIP"

# rsync with excludes (same set as the PS script)
rsync -a --delete \
    --exclude='__pycache__/' --exclude='.git/' --exclude='.idea/' \
    --exclude='.vscode/' --exclude='.pytest_cache/' --exclude='.mypy_cache/' \
    --exclude='_pack/' --exclude='exp/' --exclude='experiments/' \
    --exclude='training-logs/' --exclude='runs/' --exclude='tb_logs/' \
    --exclude='wandb/' \
    --exclude='*.pth' --exclude='*.pt' --exclude='*.ckpt' \
    --exclude='*.zip' --exclude='*.tar' --exclude='*.tar.gz' \
    --exclude='*.log' --exclude='*.tmp' --exclude='*.swp' \
    --exclude='*.png' --exclude='*.jpg' --exclude='.DS_Store' \
    --exclude='*.pdf' \
    --exclude='splits/coco/' --exclude='splits/ade20k/' \
    --exclude='splits/cityscapes/' --exclude='splits/pascal/' \
    "$REPO_ROOT/" "$STAGING/"

( cd "$OUT_DIR" && rm -f "$ZIP" && zip -qr "$ZIP" "$REPO_NAME" )
SIZE=$(du -h "$ZIP" | cut -f1)
echo "[pack] zipped $SIZE -> $ZIP"

print_unpack() {
    echo ""
    echo "On the new AutoDL server, unpack with:"
    echo "  mkdir -p /root/autodl-tmp/code && cd /root/autodl-tmp/code && rm -rf ${REPO_NAME}.old && \\"
    echo "    [ -d ${REPO_NAME} ] && mv ${REPO_NAME} ${REPO_NAME}.old; \\"
    echo "    unzip -q /root/autodl-tmp/code/$(basename "$ZIP") -d /root/autodl-tmp/code"
}

if [[ "${1:-}" != "upload" ]]; then
    echo "[done] pass 'upload' as first arg to push to bitahub."
    echo "[scp] fallback: scp this zip to /root/autodl-tmp/code/ before unpacking."
    print_unpack
    exit 0
fi

command -v bita >/dev/null || { echo "bita CLI not in PATH"; exit 1; }

echo "[upload] $ZIP -> $BITA_BK/$PREFIX"
n=0
until bita upload -e "$BITA_EP" -b "$BITA_BK" -o "$PREFIX" -l "$ZIP"; do
    n=$((n+1))
    if [[ $n -ge 5 ]]; then echo "[upload] giving up after $n attempts"; exit 1; fi
    echo "[upload] attempt $n failed; retrying in 30s..."
    sleep 30
done
echo "[upload] OK"

print_unpack
