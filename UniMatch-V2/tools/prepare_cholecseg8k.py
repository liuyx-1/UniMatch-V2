"""Prepare CholecSeg8k for UniMatchV2-style SSL training.

Source layout (Mendeley Data release; each clip dir holds 4 PNGs per frame):
    <root>/videoXX/videoXX_NNNNN/frame_<N>_endo.png                 # RGB image
    <root>/videoXX/videoXX_NNNNN/frame_<N>_endo_mask.png            # watershed gray-indexed label
    <root>/videoXX/videoXX_NNNNN/frame_<N>_endo_color_mask.png      # RGB visualization (ignored)
    <root>/videoXX/videoXX_NNNNN/frame_<N>_endo_watershed_mask.png  # boundary sketch  (ignored)

Output layout (under --out):
    out/images/videoXX/videoXX_NNNNN/frame_<N>.png      # symlink to *_endo.png
    out/masks/videoXX/videoXX_NNNNN/frame_<N>.png       # single-channel uint8, values in {0..12, 255}
    out/splits/labeled.txt                              # rate-25% of TRAIN-video frames
    out/splits/unlabeled.txt                            # remaining TRAIN-video frames
    out/splits/val.txt                                  # ALL frames from val videos
    out/dataset_info.txt                                # human summary
    out/remap_plan.json                                 # full record of class remap + split

Each split line is:
    <img_rel_path>\t<mask_rel_path>      (labeled / val)
    <img_rel_path>                       (unlabeled)

Class space (13 classes; matches util/classes.py['cholecseg8k']):
    0  background
    1  abdominal_wall
    2  liver
    3  gastrointestinal_tract
    4  fat
    5  grasper
    6  connective_tissue
    7  blood
    8  cystic_duct
    9  l_hook_electrocautery
    10 gallbladder
    11 hepatic_vein
    12 liver_ligament

Watershed gray palette in *_endo_mask.png  →  contiguous class id
(per the official CholecSeg8k README; use --analyze first to verify against your copy):
    50 → 0   (Black Background)
    11 → 1   (Abdominal Wall)
    21 → 2   (Liver)
    13 → 3   (Gastrointestinal Tract)
    12 → 4   (Fat)
    31 → 5   (Grasper)
    23 → 6   (Connective Tissue)
    24 → 7   (Blood)
    25 → 8   (Cystic Duct)
    32 → 9   (L-hook Electrocautery)
    22 → 10  (Gallbladder)
    33 → 11  (Hepatic Vein)
    5  → 12  (Liver Ligament)
Any pixel value not in this table is written as 255 (ignore index).

Video split (Twinanda-style, also used by recent SSL papers):
    val   = {video12, video20, video48, video55}      ~25% of clips
    train = all other 13 videos                       ~75% of clips
The labeled/unlabeled partition inside the train pool is a seed-42 random
shuffle followed by the first `--rate` fraction being kept as labeled.

Usage:
    # 1) Analyze only — scans a sample of masks and prints found pixel values
    python tools/prepare_cholecseg8k.py --root /data/test/cholecseg8k --analyze

    # 2) Apply: remap masks, build images symlinks, write splits
    python tools/prepare_cholecseg8k.py \\
        --root /data/test/cholecseg8k \\
        --out  /data/test/cholecseg8k_processed \\
        --rate 0.25 --seed 42 --apply

    # 3) Generate other label rates from the same train pool
    python tools/prepare_cholecseg8k.py \\
        --root /data/test/cholecseg8k \\
        --out  /data/test/cholecseg8k_processed \\
        --rate 0.10 --seed 42 --apply --splits-only
"""
from __future__ import annotations
import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
CLASS_NAMES = [
    'background',           # 0
    'abdominal_wall',       # 1
    'liver',                # 2
    'gastrointestinal_tract',  # 3
    'fat',                  # 4
    'grasper',              # 5
    'connective_tissue',    # 6
    'blood',                # 7
    'cystic_duct',          # 8
    'l_hook_electrocautery',  # 9
    'gallbladder',          # 10
    'hepatic_vein',         # 11
    'liver_ligament',       # 12
]
N_CLASSES = len(CLASS_NAMES)

# Default watershed gray-value → contiguous class id (per CholecSeg8k README).
DEFAULT_PALETTE: dict[int, int] = {
    50: 0,   # Background
    11: 1,   # Abdominal Wall
    21: 2,   # Liver
    13: 3,   # Gastrointestinal Tract
    12: 4,   # Fat
    31: 5,   # Grasper
    23: 6,   # Connective Tissue
    24: 7,   # Blood
    25: 8,   # Cystic Duct
    32: 9,   # L-hook Electrocautery
    22: 10,  # Gallbladder
    33: 11,  # Hepatic Vein
    5:  12,  # Liver Ligament
}
IGNORE_INDEX = 255

# Val videos per Twinanda et al. video-level split.
DEFAULT_VAL_VIDEOS = {'video12', 'video20', 'video48', 'video55'}


# ---------------------------------------------------------------------------
FRAME_RE = re.compile(r'^frame_(\d+)_endo\.png$', re.IGNORECASE)


def discover_frames(root: Path) -> list[tuple[str, str, Path, Path]]:
    """Return [(video_id, clip_id, image_path, mask_path), ...] sorted.

    video_id  e.g. "video01"
    clip_id   e.g. "video01_00080"
    """
    out = []
    if not root.exists():
        sys.exit(f'[err] root does not exist: {root}')
    for vdir in sorted(root.iterdir()):
        if not vdir.is_dir() or not vdir.name.lower().startswith('video'):
            continue
        for cdir in sorted(vdir.iterdir()):
            if not cdir.is_dir():
                continue
            for img in sorted(cdir.iterdir()):
                m = FRAME_RE.match(img.name)
                if not m:
                    continue
                stem = f'frame_{m.group(1)}_endo'
                mask = cdir / f'{stem}_mask.png'
                if not mask.exists():
                    continue
                out.append((vdir.name, cdir.name, img, mask))
    return out


def analyze(frames, sample: int = 200, palette: dict[int, int] = DEFAULT_PALETTE):
    """Scan a sample of masks; print unique pixel values & coverage by palette."""
    rng = random.Random(0)
    sub = rng.sample(frames, k=min(sample, len(frames)))
    found: Counter[int] = Counter()
    per_video: dict[str, Counter[int]] = defaultdict(Counter)
    for v, _, _, mp in sub:
        arr = np.array(Image.open(mp))
        if arr.ndim == 3:
            arr = arr[..., 0]               # take a single channel; watershed is gray-coded
        vals, cnts = np.unique(arr, return_counts=True)
        for val, c in zip(vals.tolist(), cnts.tolist()):
            found[int(val)] += int(c)
            per_video[v][int(val)] += int(c)

    print(f'\n[analyze] sampled {len(sub)} masks  ({len(frames)} total)')
    total = sum(found.values())
    print(f'[analyze] {len(found)} distinct gray values, {total:,} pixels scanned')
    print('  val   | count        | %    | mapped class')
    print('  ------+--------------+------+--------------------------')
    for val, c in sorted(found.items()):
        cls = palette.get(val)
        cls_name = CLASS_NAMES[cls] if cls is not None else f'(unmapped → {IGNORE_INDEX})'
        marker = '✓' if cls is not None else '✗'
        print(f'  {val:>5} | {c:>12,} | {100*c/total:5.2f} | {marker} {cls_name}')

    unmapped = sorted(v for v in found if v not in palette)
    if unmapped:
        print(f'\n[warn] {len(unmapped)} values not in the default palette: {unmapped}')
        print('       Edit DEFAULT_PALETTE in this script (or pass --palette-json) before --apply.')

    missing_classes = [palette[v] for v in palette if palette[v] not in
                        {palette.get(g, -1) for g in found}]
    if missing_classes:
        names = [CLASS_NAMES[c] for c in sorted(set(missing_classes))]
        print(f'[note] classes never observed in the sample: {names}  '
              '(rare classes may still appear in the full set)')


# ---------------------------------------------------------------------------
def build_remap_lut(palette: dict[int, int]) -> np.ndarray:
    """Build a length-256 LUT: raw gray value → contiguous class id (255 = ignore)."""
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for raw, cls in palette.items():
        if not (0 <= raw <= 255):
            raise ValueError(f'palette key {raw} outside 0..255')
        if not (0 <= cls < N_CLASSES):
            raise ValueError(f'palette value {cls} outside 0..{N_CLASSES-1}')
        lut[raw] = cls
    return lut


def write_mask(src: Path, dst: Path, lut: np.ndarray) -> Counter[int]:
    arr = np.array(Image.open(src))
    if arr.ndim == 3:
        arr = arr[..., 0]
    new = lut[arr]                             # uint8, values in {0..12, 255}
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(new, mode='L').save(dst, optimize=True)
    vals, cnts = np.unique(new, return_counts=True)
    return Counter({int(v): int(c) for v, c in zip(vals, cnts)})


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy:
        import shutil
        shutil.copy2(src, dst)
    else:
        # Windows requires admin for symlinks; fall back to hardlink, then copy.
        try:
            os.symlink(src, dst)
        except (OSError, NotImplementedError):
            try:
                os.link(src, dst)
            except OSError:
                import shutil
                shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
def build_splits(frames_rel: list[tuple[str, str, str]],
                 val_videos: set[str],
                 rate: float, seed: int):
    """frames_rel: [(video_id, img_rel, mask_rel), ...] over the WHOLE dataset.
    Returns (labeled, unlabeled, val) as lists of (img_rel, mask_rel|None)."""
    train_pool, val_list = [], []
    for v, img, mask in frames_rel:
        (val_list if v in val_videos else train_pool).append((img, mask))
    rng = random.Random(seed)
    train_pool.sort()                          # deterministic before shuffle
    rng.shuffle(train_pool)
    n_lab = max(1, int(round(len(train_pool) * float(rate))))
    labeled   = train_pool[:n_lab]
    unlabeled = train_pool[n_lab:]
    val_list.sort()
    return labeled, unlabeled, val_list


def write_split_files(out: Path, labeled, unlabeled, val_list, rate, seed):
    sd = out / 'splits'
    sd.mkdir(parents=True, exist_ok=True)
    with (sd / 'labeled.txt').open('w', encoding='utf-8') as f:
        for img, mask in labeled:
            f.write(f'{img}\t{mask}\n')
    with (sd / 'unlabeled.txt').open('w', encoding='utf-8') as f:
        for img, _ in unlabeled:
            f.write(f'{img}\n')
    with (sd / 'val.txt').open('w', encoding='utf-8') as f:
        for img, mask in val_list:
            f.write(f'{img}\t{mask}\n')
    print(f'[split] wrote labeled={len(labeled)}  unlabeled={len(unlabeled)}  '
          f'val={len(val_list)}  (rate={rate}, seed={seed})')


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True,
                    help='raw CholecSeg8k root, e.g. /data/test/cholecseg8k')
    ap.add_argument('--out',  type=Path, default=None,
                    help='output processed root; required for --apply')
    ap.add_argument('--rate', type=float, default=0.25,
                    help='labeled fraction within the TRAIN-video pool (default 0.25)')
    ap.add_argument('--seed', type=int,   default=42)
    ap.add_argument('--val-videos', type=str, nargs='+', default=sorted(DEFAULT_VAL_VIDEOS),
                    help=f'video ids for validation (default: {sorted(DEFAULT_VAL_VIDEOS)})')
    ap.add_argument('--palette-json', type=Path, default=None,
                    help='optional JSON dict {raw_gray: class_id}; overrides DEFAULT_PALETTE')
    ap.add_argument('--analyze', action='store_true',
                    help='scan a mask sample, print observed gray values, exit')
    ap.add_argument('--analyze-sample', type=int, default=200)
    ap.add_argument('--apply',  action='store_true',
                    help='actually write remapped masks, image links, and splits')
    ap.add_argument('--splits-only', action='store_true',
                    help='skip mask remap + image linking; only rewrite split files '
                         '(use after a first --apply to add more rates)')
    ap.add_argument('--copy', action='store_true',
                    help='copy images instead of symlink/hardlink')
    args = ap.parse_args()

    palette = DEFAULT_PALETTE
    if args.palette_json is not None:
        palette = {int(k): int(v) for k, v in json.loads(args.palette_json.read_text()).items()}

    frames = discover_frames(args.root)
    if not frames:
        sys.exit(f'[err] no frames found under {args.root}  (expected videoXX/clip/frame_N_endo.png)')
    videos = sorted({v for v, *_ in frames})
    print(f'[scan] {len(frames):,} (image, mask) pairs across {len(videos)} videos')
    print(f'       videos: {videos}')

    if args.analyze:
        analyze(frames, sample=args.analyze_sample, palette=palette)
        return

    if not args.apply:
        print('\n[dry-run] re-run with --apply to write outputs. No files modified.')
        print(f'          would write to: {args.out}')
        return

    if args.out is None:
        sys.exit('[err] --apply requires --out')

    val_videos = set(args.val_videos)
    missing = val_videos - set(videos)
    if missing:
        print(f'[warn] val-videos not found in scan: {sorted(missing)}')

    lut = build_remap_lut(palette)
    frames_rel: list[tuple[str, str, str]] = []
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    img_total = Counter()

    if not args.splits_only:
        for i, (vid, clip, src_img, src_mask) in enumerate(frames):
            stem = src_img.stem.replace('_endo', '')        # frame_<N>
            rel_img  = f'images/{vid}/{clip}/{stem}.png'
            rel_mask = f'masks/{vid}/{clip}/{stem}.png'
            link_or_copy(src_img, out / rel_img, copy=args.copy)
            counts = write_mask(src_mask, out / rel_mask, lut)
            img_total += counts
            frames_rel.append((vid, rel_img, rel_mask))
            if (i + 1) % 500 == 0:
                print(f'  [progress] {i+1:,}/{len(frames):,}')
    else:
        # Recover the relative paths from the existing output layout.
        for vid, clip, src_img, _ in frames:
            stem = src_img.stem.replace('_endo', '')
            rel_img  = f'images/{vid}/{clip}/{stem}.png'
            rel_mask = f'masks/{vid}/{clip}/{stem}.png'
            if not (out / rel_img).exists():
                sys.exit(f'[err] --splits-only but image missing: {out/rel_img}\n'
                         '       Run a full --apply first.')
            frames_rel.append((vid, rel_img, rel_mask))

    labeled, unlabeled, val_list = build_splits(
        frames_rel, val_videos, rate=args.rate, seed=args.seed)
    write_split_files(out, labeled, unlabeled, val_list, args.rate, args.seed)

    # Provenance
    plan = {
        'palette': {int(k): int(v) for k, v in palette.items()},
        'ignore_index': IGNORE_INDEX,
        'n_classes': N_CLASSES,
        'class_names': CLASS_NAMES,
        'val_videos': sorted(val_videos),
        'train_videos': sorted(v for v in videos if v not in val_videos),
        'rate': args.rate,
        'seed': args.seed,
        'n_total_frames': len(frames),
        'n_labeled': len(labeled),
        'n_unlabeled': len(unlabeled),
        'n_val': len(val_list),
    }
    (out / 'remap_plan.json').write_text(json.dumps(plan, indent=2))

    if img_total:
        total = sum(img_total.values())
        lines = ['CholecSeg8k processed dataset', '=' * 32,
                 f'frames total : {len(frames):,}',
                 f'train videos : {len(plan["train_videos"])}  ({plan["train_videos"]})',
                 f'val   videos : {len(plan["val_videos"])}  ({plan["val_videos"]})',
                 f'labeled (r={args.rate}) : {len(labeled):,}',
                 f'unlabeled            : {len(unlabeled):,}',
                 f'val                  : {len(val_list):,}',
                 '',
                 'pixel-frequency (remapped mask, including ignore):']
        for cid in range(N_CLASSES):
            c = img_total.get(cid, 0)
            lines.append(f'  {cid:>2} {CLASS_NAMES[cid]:<24} {c:>14,} ({100*c/total:5.2f}%)')
        ig = img_total.get(IGNORE_INDEX, 0)
        lines.append(f'  {IGNORE_INDEX:>2} ignore                  {ig:>14,} ({100*ig/total:5.2f}%)')
        (out / 'dataset_info.txt').write_text('\n'.join(lines))
        print('\n' + '\n'.join(lines[-(N_CLASSES + 4):]))

    print(f'\n[done] wrote {out}')
    print(f'       point configs/cholecseg8k.yaml -> data_root: {out}')


if __name__ == '__main__':
    main()
