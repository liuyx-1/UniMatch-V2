"""Generate label-ratio (labeled/unlabeled) splits by scanning an already-built
UniMatch-V2 processed dataset directory.

Robust alternative to gen_ratio_splits.py for datasets whose split lines use
RELATIVE paths like "images/train/x.png" (the 'images/'->'masks/' string-infer
in gen_ratio_splits silently fails on those). Here we pair each image with its
mask by the SAME stem under masks/<split>/, asserting existence.

Expected layout (data_root):
    <data_root>/images/{train,val,test}/<stem>.<ext>
    <data_root>/masks/{train,val,test}/<stem>.png

Writes, paths RELATIVE to data_root:
    <splits_root>/unimatch_splits_<dataset>_<r>_seed<seed>/
        labeled.txt    "images/train/x.png<TAB>masks/train/x.png"
        unlabeled.txt  "images/train/x.png"
        val.txt        all val frames   (img<TAB>mask)
        test.txt       all test frames  (img<TAB>mask)
  * r < 1.0 : labeled = first r of seeded-shuffled TRAIN; unlabeled = the rest.
  * r >= 1.0: labeled = unlabeled = ALL train (full-label SSL upper bound).

Usage:
    python tools/make_ratio_splits_from_processed.py \
        --data-root /.../endovis2018_scene_processed \
        --splits-root /.../splits \
        --dataset endovis2018_scene \
        --ratios 0.10 0.30 0.50 1.00 --seed 42
"""
from __future__ import annotations
import argparse, random
from pathlib import Path

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')


def list_pairs(data_root: Path, split: str):
    """Return sorted [(img_rel, mask_rel)] for one split; assert masks exist."""
    img_dir = data_root / 'images' / split
    msk_dir = data_root / 'masks' / split
    if not img_dir.is_dir():
        return []
    pairs, missing = [], 0
    for f in sorted(img_dir.iterdir()):
        if f.suffix not in IMG_EXTS or not f.is_file():
            continue
        mask = msk_dir / (f.stem + '.png')
        if not mask.exists():
            missing += 1
            if missing <= 5:
                print(f'[warn] no mask for {f.name} (expected {mask})')
            continue
        pairs.append((f'images/{split}/{f.name}', f'masks/{split}/{mask.name}'))
    if missing:
        print(f'[warn] {split}: {missing} images without masks (skipped)')
    return pairs


def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--splits-root', type=Path, required=True)
    ap.add_argument('--dataset', type=str, required=True)
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.10, 0.30, 0.50, 1.00])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    train = list_pairs(args.data_root, 'train')
    val   = list_pairs(args.data_root, 'val')
    test  = list_pairs(args.data_root, 'test')
    if not train:
        raise RuntimeError(f'no train images found under {args.data_root}/images/train')
    # Withheld-test datasets (e.g. EndoVis2017) have no usable test masks:
    # follow common practice and report validation as test.
    if not test and val:
        print('[info] no test frames with masks -> reusing val as test (withheld-test convention)')
        test = val
    print(f'[pool] train={len(train)} val={len(val)} test={len(test)}')

    shuffled = train[:]
    random.Random(args.seed).shuffle(shuffled)
    n = len(shuffled)

    for r in args.ratios:
        if r >= 1.0:
            lab, unl = sorted(shuffled), sorted(shuffled)
        else:
            k = max(1, int(round(n * r)))
            lab, unl = sorted(shuffled[:k]), sorted(shuffled[k:])
        dest = args.splits_root / f'unimatch_splits_{args.dataset}_{r:.2f}_seed{args.seed}'
        dest.mkdir(parents=True, exist_ok=True)
        write_lines(dest / 'labeled.txt',   [f'{i}\t{m}' for i, m in lab])
        write_lines(dest / 'unlabeled.txt', [i for i, _ in unl])
        write_lines(dest / 'val.txt',       [f'{i}\t{m}' for i, m in val])
        write_lines(dest / 'test.txt',      [f'{i}\t{m}' for i, m in test])
        print(f'[r={r:.2f}] labeled={len(lab)} unlabeled={len(unl)} '
              f'val={len(val)} test={len(test)} -> {dest}')


if __name__ == '__main__':
    main()
