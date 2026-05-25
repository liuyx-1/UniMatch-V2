"""Prepare EndoVis2018 robotic scene segmentation dataset for UniMatch-V2.

Input layout (after unzipping the 4 release zips + test zip):
    <root>/
      labels.json                         # RGB -> class name mapping
      miccai_challenge_release_1/seq_{1,2,3,4}/
        left_frames/*.png                  # 1280x1024 RGB frames
        labels/*.png                       # 1280x1024 RGBA masks (alpha=255)
      miccai_challenge_release_2/seq_{5,6,7}/    # NOTE seq_8 not in official release
      miccai_challenge_release_3/seq_{9,10,11,12}/
      miccai_challenge_release_4/seq_{13,14,15,16}/
      images/seq_{1..16}/                  # optional consolidated dir (may be partial)
      test/test_data/seq_{1..4}/           # test zip nests one level
        left_frames/*.png
        labels/*.png                       # released post-challenge

Total: 15 train sequences (149 frames each = 2235 frames),
       4 test sequences (250 frames each = 1000 frames).

Output layout (under --out):
    out/
      remap_plan.json
      dataset_info.txt
      endovis2018_classes.txt              # copy-paste line for util/classes.py
      images/{train,val,test}/<seqN>_<frame>.png      # symlinks to originals
      masks/{train,val,test}/<seqN>_<frame>.png       # remapped single-channel
      splits/
        val.txt   test.txt
        010/ 020/ 025/ 030/ 050/ 100/
          labeled.txt    unlabeled.txt

Sequence-level train/val split prevents leakage. Default val = [5, 10, 15]
(spread across the 16-sequence range).

Class layout: 12 classes (0..11). Unknown RGB -> 255 (ignore).

Usage:
    # Dry-run analysis
    python tools/prepare_endovis2018.py \\
        --root /data/test/endovis2018/EndoVis2018/data

    # Apply
    python tools/prepare_endovis2018.py \\
        --root /data/test/endovis2018/EndoVis2018/data \\
        --out  /data/test/endovis2018_processed \\
        --apply --ratios 10 20 25 30 50 --seed 42
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


# Class layout — 12 contiguous classes, order matches labels.json
ENDOVIS2018_CLASSES = [
    'background-tissue',
    'instrument-shaft',
    'instrument-clasper',
    'instrument-wrist',
    'kidney-parenchyma',
    'covered-kidney',
    'thread',
    'clamps',
    'suturing-needle',
    'suction-instrument',
    'small-intestine',
    'ultrasound-probe',
]


def load_color_lut(labels_json_path: Path) -> Dict[Tuple[int, int, int], int]:
    """Parse labels.json into {(R,G,B): class_id} where id == position in
    ENDOSCAPES2018_CLASSES."""
    data = json.loads(labels_json_path.read_text(encoding='utf-8'))
    # labels.json is a list of {"name": str, "color": [R, G, B]} OR a dict
    # ── 尝试两种格式以兼容 mli0603 的 labels.json
    name_to_color: Dict[str, Tuple[int, int, int]] = {}
    if isinstance(data, dict):
        for name, color in data.items():
            name_to_color[name] = tuple(int(c) for c in color)
    elif isinstance(data, list):
        for entry in data:
            name = entry.get('name') or entry.get('class')
            color = entry.get('color') or entry.get('rgb')
            name_to_color[name] = tuple(int(c) for c in color)
    else:
        raise ValueError(f'Unexpected labels.json format: {type(data)}')

    lut: Dict[Tuple[int, int, int], int] = {}
    for idx, cname in enumerate(ENDOVIS2018_CLASSES):
        if cname not in name_to_color:
            sys.exit(f'class "{cname}" missing from labels.json')
        lut[name_to_color[cname]] = idx
    return lut


def rgba_to_id(arr: np.ndarray, lut: Dict[Tuple[int, int, int], int]) -> np.ndarray:
    """Convert (H, W, 4) RGBA or (H, W, 3) RGB to (H, W) uint8 class id.
    Unknown colors → 255 (ignore)."""
    if arr.ndim == 3:
        rgb = arr[..., :3]
    else:
        sys.exit(f'mask is single-channel; expected RGB/RGBA: shape={arr.shape}')
    H, W = rgb.shape[:2]
    out = np.full((H, W), 255, dtype=np.uint8)
    for color, cid in lut.items():
        mask = (rgb[..., 0] == color[0]) & \
               (rgb[..., 1] == color[1]) & \
               (rgb[..., 2] == color[2])
        out[mask] = cid
    return out


def _collect_pairs(seq_root: Path, with_labels: bool):
    """Return list of (img, label_or_None) under <seq_root>/left_frames/."""
    frames_dir = seq_root / 'left_frames'
    labels_dir = seq_root / 'labels' if with_labels else None
    if not frames_dir.is_dir():
        return []
    pairs = []
    for img in sorted(frames_dir.glob('*.png')):
        if labels_dir is not None:
            lab = labels_dir / img.name
            if not lab.exists():
                continue
            pairs.append((img, lab))
        else:
            pairs.append((img, None))
    return pairs


def discover_sequences(root: Path) -> Tuple[Dict[int, list], Dict[int, list]]:
    """Return ({train_seq: pair_list}, {test_seq: pair_list}).

    Scans MULTIPLE candidate parent directories for seq_* subdirs because the
    canonical CAMMA layout puts the actual data in
    ``miccai_challenge_release_{1..4}/seq_*/``, with a possible secondary
    consolidated ``images/`` directory that may be partially populated.

    For each seq id we pick the source dir that yields the most (img, mask)
    pairs — so even if ``images/`` has stale partial copies, the release dirs
    win as long as they have more complete data.
    """
    train: Dict[int, list] = {}
    test:  Dict[int, list] = {}

    # All candidate parent dirs to scan for TRAIN sequences
    train_parents = []
    if (root / 'images').is_dir():
        train_parents.append(root / 'images')
    for p in sorted(root.glob('miccai_challenge*release*')):
        if p.is_dir():
            train_parents.append(p)

    # All candidate parent dirs to scan for TEST sequences.
    # CAMMA test zip unpacks to data/test/test_data/seq_{1..4}/ (extra
    # nesting), so we add both data/test/ and data/test/test_data/ as
    # candidate parents. Anything else matching *test* is also scanned.
    test_parents = []
    if (root / 'test').is_dir():
        test_parents.append(root / 'test')
    if (root / 'test' / 'test_data').is_dir():
        test_parents.append(root / 'test' / 'test_data')
    for p in sorted(root.glob('*test*')):
        if p.is_dir() and p not in test_parents:
            test_parents.append(p)

    def scan(parents, store):
        for parent in parents:
            for d in sorted(parent.iterdir()):
                if not d.is_dir() or not d.name.startswith('seq_'):
                    continue
                try:
                    n = int(d.name.split('_')[1])
                except ValueError:
                    continue
                pairs = _collect_pairs(d, with_labels=True)
                if not pairs:                      # try image-only (test sets)
                    pairs = _collect_pairs(d, with_labels=False)
                if not pairs:
                    continue
                # Keep the source with the most pairs for this seq id
                if n not in store or len(pairs) > len(store[n]):
                    store[n] = pairs

    scan(train_parents, train)
    scan(test_parents,  test)

    # Sanity log: which source won for each seq
    return train, test


def analyze(root: Path):
    print(f'\n=== Discovering sequences under {root} ===')
    # show which parent dirs we will scan
    train_parents = [str((root / 'images').relative_to(root))] if (root / 'images').is_dir() else []
    train_parents += [str(p.relative_to(root)) for p in sorted(root.glob('miccai_challenge*release*')) if p.is_dir()]
    test_parents  = []
    if (root / 'test').is_dir():
        test_parents.append(str((root / 'test').relative_to(root)))
    if (root / 'test' / 'test_data').is_dir():
        test_parents.append(str((root / 'test' / 'test_data').relative_to(root)))
    print(f'[scan] train parents: {train_parents}')
    print(f'[scan] test  parents: {test_parents}')

    train, test = discover_sequences(root)
    print(f'\n[train] {len(train)} sequences:')
    for n, p in sorted(train.items()):
        src = p[0][0].parent.parent.relative_to(root)
        print(f'  seq_{n:2d}  {len(p):4d} frames    from {src}')
    print(f'\n[test] {len(test)} sequences:')
    for n, p in sorted(test.items()):
        labeled = sum(1 for _, m in p if m is not None)
        src = p[0][0].parent.parent.relative_to(root)
        print(f'  seq_{n:2d}  {len(p):4d} frames  ({labeled} with labels)  from {src}')

    print(f'\nTotal train frames: {sum(len(p) for p in train.values())}')
    print(f'Total test  frames: {sum(len(p) for p in test.values())}')
    return train, test


def apply_split(train, test, root, out, args):
    out.mkdir(parents=True, exist_ok=True)
    lut = load_color_lut(root / 'labels.json')
    print(f'\n[lut] loaded {len(lut)} colors from labels.json')

    # ---- Sequence-level train/val split ----
    val_seq_set = set(args.val_sequences)
    train_seq_ids = sorted(s for s in train.keys() if s not in val_seq_set)
    val_seq_ids = sorted(s for s in train.keys() if s in val_seq_set)
    test_seq_ids = sorted(test.keys())

    print(f'\n[seq-split] train seqs: {train_seq_ids}')
    print(f'[seq-split] val   seqs: {val_seq_ids}')
    print(f'[seq-split] test  seqs: {test_seq_ids}')

    # ---- Symlink images + remap masks ----
    def process_seqs(seq_ids, source, split_name):
        img_out = out / 'images' / split_name
        mask_out = out / 'masks' / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        n_img = n_mask = 0
        records = []   # [(img_rel, mask_rel)] for the split file
        for sid in seq_ids:
            for img_path, mask_path in source[sid]:
                stem = f'seq{sid:02d}_{img_path.stem}'
                # image: symlink only (saves disk)
                dst_img = img_out / f'{stem}.png'
                if not dst_img.exists():
                    try:
                        dst_img.symlink_to(img_path.resolve())
                    except OSError:
                        import shutil
                        shutil.copy2(img_path, dst_img)
                n_img += 1
                # mask: remap then save single-channel PNG
                mask_rel = None
                if mask_path is not None:
                    dst_mask = mask_out / f'{stem}.png'
                    if not dst_mask.exists():
                        arr = np.array(Image.open(mask_path))
                        ids = rgba_to_id(arr, lut)
                        Image.fromarray(ids).save(dst_mask)
                    n_mask += 1
                    mask_rel = f'masks/{split_name}/{stem}.png'
                records.append((f'images/{split_name}/{stem}.png', mask_rel))
        print(f'[{split_name}] images={n_img}  masks={n_mask}')
        return records

    train_records = process_seqs(train_seq_ids, train, 'train')
    val_records   = process_seqs(val_seq_ids,   train, 'val')
    test_records  = process_seqs(test_seq_ids,  test,  'test')

    # ---- Write val/test split files ----
    split_dir = out / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    val_lines = sorted(f'{i}\t{m}' for i, m in val_records if m)
    (split_dir / 'val.txt').write_text('\n'.join(val_lines) + '\n', encoding='utf-8')
    print(f'[split] val.txt   {len(val_lines)} lines')
    if any(m for _, m in test_records):
        test_lines = sorted(f'{i}\t{m}' for i, m in test_records if m)
        (split_dir / 'test.txt').write_text('\n'.join(test_lines) + '\n', encoding='utf-8')
        print(f'[split] test.txt  {len(test_lines)} lines (labels present)')
    else:
        # test labels not released — write image-only test.txt for visualisation
        test_lines = sorted(i for i, _ in test_records)
        (split_dir / 'test.txt').write_text('\n'.join(test_lines) + '\n', encoding='utf-8')
        print(f'[split] test.txt  {len(test_lines)} lines (NO labels — image-only)')

    # ---- Per-ratio train splits (labeled / unlabeled) ----
    rng = random.Random(args.seed)
    train_pairs = [r for r in train_records if r[1] is not None]
    rng.shuffle(train_pairs)
    n_total = len(train_pairs)
    for ratio_pct in args.ratios:
        n_lab = max(1, int(round(n_total * ratio_pct / 100.0)))
        lab, unl = train_pairs[:n_lab], train_pairs[n_lab:]
        sub = split_dir / f'{ratio_pct:03d}'
        sub.mkdir(parents=True, exist_ok=True)
        (sub / 'labeled.txt').write_text(
            '\n'.join(sorted(f'{i}\t{m}' for i, m in lab)) + '\n', encoding='utf-8')
        (sub / 'unlabeled.txt').write_text(
            '\n'.join(sorted(i for i, _ in unl)) + '\n', encoding='utf-8')
        print(f'[ratio {ratio_pct:>3d}%]  labeled={len(lab):>4d}  unlabeled={len(unl):>5d}')

    # ---- Plan + classes snippet ----
    plan = {
        'source_root': str(root),
        'val_sequences': sorted(val_seq_ids),
        'test_sequences': sorted(test_seq_ids),
        'train_sequences': sorted(train_seq_ids),
        'class_names': ENDOVIS2018_CLASSES,
        'nclass': len(ENDOVIS2018_CLASSES),
        'ignore_index': 255,
        'color_lut': {f'{r},{g},{b}': cid for (r, g, b), cid in lut.items()},
        'ratios': list(args.ratios),
        'seed': int(args.seed),
        'frame_counts': {
            'train': sum(len(train[s]) for s in train_seq_ids),
            'val':   sum(len(train[s]) for s in val_seq_ids),
            'test':  sum(len(test[s]) for s in test_seq_ids),
        },
    }
    (out / 'remap_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    (out / 'endovis2018_classes.txt').write_text(
        f"'endovis2018': {ENDOVIS2018_CLASSES},\n", encoding='utf-8')
    info = out / 'dataset_info.txt'
    with open(info, 'w', encoding='utf-8') as f:
        f.write(f'EndoVis2018 processed from: {root}\n')
        f.write(f'Reference: Allan et al., "2018 Robotic Scene Segmentation '
                f'Challenge", arXiv 2001.11190\n')
        f.write(f'Total released: 19 sequences = 15 training + 4 '
                f'"validation" (== our test).\n')
        f.write(f'nclass = {len(ENDOVIS2018_CLASSES)}  ignore_index = 255\n\n')
        for i, n in enumerate(ENDOVIS2018_CLASSES):
            f.write(f'  {i}: {n}\n')
        f.write('\nSequence-level splits (no leakage):\n')
        f.write(f'  train seqs ({len(train_seq_ids)}): {train_seq_ids}     '
                f'<- subset of challenge train (15 seqs - val held-out)\n')
        f.write(f'  val   seqs ({len(val_seq_ids)}): {val_seq_ids}     '
                f'<- subset of challenge train, held out for our hp tuning\n')
        f.write(f'  test  seqs ({len(test_seq_ids)}): {test_seq_ids}     '
                f'<- challenge "validation" set (test_data/), used as our test\n')
        f.write(f'\nNote: seq_8 is NOT in CAMMA\'s public release.\n')
    print(f'\n[info]  wrote {info}')
    print(f'[plan]  wrote {out / "remap_plan.json"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True,
                    help='inner data dir (containing labels.json + images/ + test/)')
    ap.add_argument('--out',  type=Path, default=None,
                    help='output directory (used with --apply)')
    ap.add_argument('--ratios', type=int, nargs='+', default=[10, 20, 25, 30, 50],
                    help='labeled ratios (percent) for train split')
    ap.add_argument('--val-sequences', type=int, nargs='+',
                    default=[5, 10, 15],
                    help='train sequences reserved for validation '
                          '(prevents leakage). Default: spread across 16 seqs.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f'--root {args.root} not a directory')
    if not (args.root / 'labels.json').exists():
        sys.exit(f'missing {args.root}/labels.json')

    train, test = analyze(args.root)
    if args.apply:
        if args.out is None:
            sys.exit('--apply requires --out')
        apply_split(train, test, args.root, args.out, args)
    else:
        print('\n[dry-run] add --apply --out <dir> to write outputs.')


if __name__ == '__main__':
    main()
