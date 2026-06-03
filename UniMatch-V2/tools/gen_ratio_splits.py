"""Generate (ratio, seed) UniMatchV2 split dirs from an existing full pool.

Takes ONE existing reference split dir (typically the 0.25_seed42 one) that
contains:
    <ref>/labeled.txt     # <img>\t<mask>  per line
    <ref>/unlabeled.txt   # <img>          per line (one-token lines)
    <ref>/val.txt         # <img>\t<mask>
    <ref>/test.txt        # optional

Builds a deterministic pool = labeled_pool (labeled.txt entries) merged with
the labeled-style versions of unlabeled.txt's images.  For the latter we
INFER a mask path by string-replacing 'images/' with 'masks/' (typical
UniMatchV2 surgical convention).  Pool entries are sorted then shuffled
with seed42, after which the first ``ratio`` fraction becomes labeled and
the rest unlabeled.

Outputs are written to:
    <out_root>/unimatch_splits_<dataset>_<ratio>_seed<seed>/
        labeled.txt   unlabeled.txt   val.txt   [test.txt]

Usage:
    python tools/gen_ratio_splits.py \\
        --ref /data/splits/unimatch_splits_cholecseg8k_0.25_seed42 \\
        --dataset cholecseg8k \\
        --out-root /data/splits \\
        --ratios 0.10 0.20 0.25 0.30 0.50 \\
        --seed 42
"""
from __future__ import annotations
import argparse
import random
import re
from pathlib import Path


def _read_lines(p: Path):
    return [ln.strip() for ln in p.read_text(encoding='utf-8').splitlines() if ln.strip()]


def _split_line(ln: str):
    parts = ln.split('\t') if '\t' in ln else ln.split(' ')
    return parts[0], (parts[1] if len(parts) > 1 else None)


def _img_to_mask(img: str) -> str:
    # Heuristics: 'images/' -> 'masks/'; '.jpg' -> '.png'; 'cholecseg8k/' -> 'cholecseg8k_coco_semantic/'
    m = img
    m = m.replace('/images/', '/masks/')
    if '/cholecseg8k/' in m and '/masks/' not in m:
        m = m.replace('/cholecseg8k/', '/cholecseg8k_coco_semantic/')
    m = re.sub(r'\.(jpg|jpeg)$', '.png', m, flags=re.IGNORECASE)
    return m


def build_pool(ref: Path):
    pool = []      # list of (img, mask) — mask never None
    seen = set()
    for ln in _read_lines(ref / 'labeled.txt'):
        img, mask = _split_line(ln)
        if mask is None:
            mask = _img_to_mask(img)
        if img not in seen:
            pool.append((img, mask)); seen.add(img)
    ul = ref / 'unlabeled.txt'
    if ul.exists():
        for ln in _read_lines(ul):
            img, mask = _split_line(ln)
            if mask is None:
                mask = _img_to_mask(img)
            if img not in seen:
                pool.append((img, mask)); seen.add(img)
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ref', type=Path, required=True,
                    help='reference split dir to harvest the full pool from')
    ap.add_argument('--dataset', type=str, required=True,
                    help='dataset short name (used in output dir name)')
    ap.add_argument('--out-root', type=Path, required=True,
                    help='where to write unimatch_splits_<dataset>_<r>_seed<s>/')
    ap.add_argument('--ratios', type=float, nargs='+',
                    default=[0.10, 0.20, 0.25, 0.30, 0.50])
    ap.add_argument('--seed',   type=int, default=42)
    args = ap.parse_args()

    pool = build_pool(args.ref)
    pool.sort()                        # deterministic order before shuffling
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    n_total = len(pool)
    print(f'[pool] {n_total} (image, mask) pairs from {args.ref}')

    # carry val / test through unchanged
    val_lines  = _read_lines(args.ref / 'val.txt')  if (args.ref / 'val.txt').exists()  else []
    test_lines = _read_lines(args.ref / 'test.txt') if (args.ref / 'test.txt').exists() else []

    for r in args.ratios:
        n_lab = max(1, int(round(n_total * r)))
        lab  = sorted(pool[:n_lab])
        unl  = sorted(pool[n_lab:])
        rstr = f'{r:.2f}'
        dest = args.out_root / f'unimatch_splits_{args.dataset}_{rstr}_seed{args.seed}'
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'labeled.txt').write_text(
            '\n'.join(f'{i}\t{m}' for i, m in lab) + '\n', encoding='utf-8')
        (dest / 'unlabeled.txt').write_text(
            '\n'.join(i for i, _ in unl) + '\n', encoding='utf-8')
        if val_lines:
            (dest / 'val.txt').write_text('\n'.join(val_lines) + '\n', encoding='utf-8')
        if test_lines:
            (dest / 'test.txt').write_text('\n'.join(test_lines) + '\n', encoding='utf-8')
        print(f'[r={rstr}]  labeled={len(lab):>5d}  unlabeled={len(unl):>5d}  -> {dest}')


if __name__ == '__main__':
    main()
