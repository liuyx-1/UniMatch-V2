"""Prepare EndoVis2017 Robotic Instrument Segmentation dataset for UniMatch-V2.

Reference: Allan et al., "2017 Robotic Instrument Segmentation Challenge",
arXiv 1902.06426. Two segmentation tasks supported:

  * --task parts   (4 classes): background / shaft / wrist / clasper
  * --task type    (8 classes): background + 7 instrument types

Per-frame ground truth is stored as MULTIPLE binary-like PNGs under
``ground_truth/<Tool>_labels/frameXXX.png``. Each PNG has pixel values:
    0  : background (this tool not present here)
    10 : shaft       (this tool's shaft)
    20 : wrist       (this tool's wrist)
    30 : clasper     (this tool's clasper)
    40 : whole-instrument region for "Other_labels" (ultrasound / specials)

We merge all per-tool PNGs into ONE single-channel multi-class mask per
frame, with the chosen task's class IDs. Conflicts (rare) are resolved by
processing order: specific instruments win over "Other".

EndoVis2017 raw frames are 1920x1080. The official challenge crops to
1280x1024 at offset (320, 28). We do this crop by default; pass --no-crop
to keep raw resolution.

Input layout (after unzipping the training + 3 testing zips):
    <root>/
      instrument_dataset_{1..8}/                            # train (8 sequences x 225 frames)
        left_frames/frameNNN.png
        right_frames/frameNNN.png
        ground_truth/<ToolName>_labels/frameNNN.png
        mappings.json (optional)
      test_1_4/instrument_dataset_{1..4}/                   # test 1-4 (4 x 75 frames)
      test_5_8/instrument_dataset_{1..4}/                   # test 5-8 (4 x 75 frames)
      test_9_10/instrument_dataset_{1..2}/                  # test 9-10 (2 x 300 frames)

Output layout (under --out):
    out/
      remap_plan.json
      dataset_info.txt
      endovis2017_classes.txt
      images/{train,val,test}/<tag>_<frame>.png
      masks/{train,val,test}/<tag>_<frame>.png
      splits/
        val.txt   test.txt
        010/ 020/ 025/ 030/ 050/ 100/
          labeled.txt    unlabeled.txt

Sequence-level train/val split prevents leakage. Default val = [2, 7]
(spread across the 8 train datasets).

Usage:
    # Dry-run
    python tools/prepare_endovis2017.py \\
        --root /data/test/endovis2017/EndoVis2017/data --task parts

    # Apply (parts task)
    python tools/prepare_endovis2017.py \\
        --root /data/test/endovis2017/EndoVis2017/data \\
        --out  /data/test/endovis2017_parts_processed \\
        --task parts --apply --val-datasets 2 7 \\
        --ratios 10 20 25 30 50 --seed 42

    # Apply (type task)
    python tools/prepare_endovis2017.py \\
        --root /data/test/endovis2017/EndoVis2017/data \\
        --out  /data/test/endovis2017_type_processed \\
        --task type --apply --val-datasets 2 7 \\
        --ratios 10 20 25 30 50 --seed 42
"""
from __future__ import annotations
import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


# ---------- Class definitions per task ----------
PARTS_CLASSES = ['background', 'shaft', 'wrist', 'clasper']

TYPE_CLASSES = [
    'background',
    'Bipolar Forceps',
    'Prograsp Forceps',
    'Large Needle Driver',
    'Vessel Sealer',
    'Grasping Retractor',
    'Monopolar Curved Scissors',
    'Ultrasound Probe',
]

# Tool subdir name (normalized) -> type class id (for --task type)
# Key matched by lower-cased canonical name; "Left_/Right_" prefix stripped first.
TYPE_NAME_TO_ID = {
    'bipolar_forceps':            1,
    'maryland_bipolar_forceps':   1,    # Maryland is a subtype of Bipolar
    'prograsp_forceps':           2,
    'large_needle_driver':        3,
    'vessel_sealer':              4,
    'grasping_retractor':         5,
    'monopolar_curved_scissors':  6,
    'ultrasound_probe':           7,
    'other':                      7,    # "Other_labels" subdir lumped with ultrasound
}

# Raw pixel value -> parts class id (for --task parts)
PARTS_VALUE_TO_ID = {
    10: 1,    # shaft
    20: 2,    # wrist
    30: 3,    # clasper
    # 40 (Other whole-instrument) -> 255 (ignore) for parts task
}

# Official crop: 1920x1080 -> 1280x1024 at offset (320, 28)
CROP_OFFSET = (320, 28)
CROP_SIZE   = (1280, 1024)


# ---------- Helpers ----------
def _normalize_tool_name(subdir_name: str) -> str:
    """E.g. 'Left_Prograsp_Forceps_labels' -> 'prograsp_forceps'."""
    s = subdir_name
    if s.endswith('_labels'):
        s = s[: -len('_labels')]
    s = re.sub(r'^(Left|Right)_', '', s, flags=re.IGNORECASE)
    return s.lower()


def _to_2d(arr: np.ndarray) -> np.ndarray:
    """Normalize a label PNG array to (H, W) grayscale. EndoVis2017 labels
    are nominally grayscale but some datasets save them as RGB(A) where all
    channels are identical — take channel 0."""
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _build_parts_mask(label_files: List[Tuple[Path, str]], h: int, w: int) -> np.ndarray:
    """Merge multiple per-tool label PNGs into a single parts-mask [H, W].
    Pixel rule: take max (later writes win for ties)."""
    out = np.zeros((h, w), dtype=np.uint8)
    for path, _tool_name in label_files:
        arr = _to_2d(np.array(Image.open(path)))
        if arr.shape != (h, w):
            arr = _to_2d(np.array(Image.open(path).resize((w, h), Image.NEAREST)))
        for raw, cid in PARTS_VALUE_TO_ID.items():
            out[arr == raw] = cid
        # value 40 (Other whole-instrument) -> ignore in parts task
        ignore = (arr == 40)
        out[ignore & (out == 0)] = 255
    return out


def _build_type_mask(label_files: List[Tuple[Path, str]], h: int, w: int) -> np.ndarray:
    """Merge per-tool labels into a type-mask [H, W]. Specific tools win
    over 'Other' via processing-order priority."""
    out = np.zeros((h, w), dtype=np.uint8)
    # Process specific tools first (so they win when overlapping with "Other")
    others = []
    specifics = []
    for path, tool_name in label_files:
        if tool_name == 'other':
            others.append((path, tool_name))
        else:
            specifics.append((path, tool_name))

    for path, tool_name in specifics + others:
        cid = TYPE_NAME_TO_ID.get(tool_name)
        if cid is None:
            continue
        arr = _to_2d(np.array(Image.open(path)))
        if arr.shape != (h, w):
            arr = _to_2d(np.array(Image.open(path).resize((w, h), Image.NEAREST)))
        # Any non-zero pixel in this tool's mask -> assign this tool's class id
        mask = (arr > 0)
        out[mask] = cid
    return out


def _apply_crop(img: Image.Image) -> Image.Image:
    """Crop 1920x1080 -> 1280x1024 at (320, 28). Returns img unchanged if
    already cropped or different sized."""
    if img.size == (1920, 1080):
        x, y = CROP_OFFSET
        w, h = CROP_SIZE
        return img.crop((x, y, x + w, y + h))
    return img


# ---------- Discovery ----------
def _list_datasets_under(parent: Path) -> List[Tuple[str, Path]]:
    """Return [(name, path)] for every instrument_dataset_* under parent.
    Searches both direct children and one nesting level (e.g. test_1_4/instrument_dataset_1)."""
    found = []
    if not parent.is_dir():
        return found
    for d in sorted(parent.iterdir()):
        if d.is_dir() and d.name.startswith('instrument_dataset_'):
            found.append((d.name, d))
    return found


def discover(root: Path):
    """Return ([(train_tag, ds_path)...], [(test_tag, ds_path)...])."""
    train: List[Tuple[str, Path]] = []
    test: List[Tuple[str, Path]] = []

    # Train: instrument_dataset_* directly under root (after unzipping the 2
    # training zips). Tag: ds<N>
    for name, p in _list_datasets_under(root):
        # extract number suffix
        m = re.match(r'instrument_dataset_(\d+)', name)
        if not m: continue
        n = int(m.group(1))
        train.append((f'ds{n:02d}', p))

    # Test: scan test_*/instrument_dataset_*. Tag by parent + number.
    test_parents = [d for d in sorted(root.iterdir())
                     if d.is_dir() and d.name.startswith('test_')]
    for tp in test_parents:
        tp_idx = tp.name                              # e.g. test_1_4
        for name, p in _list_datasets_under(tp):
            m = re.match(r'instrument_dataset_(\d+)', name)
            if not m: continue
            n = int(m.group(1))
            # tag: t<parent_first_idx><local_idx>
            #   test_1_4/instrument_dataset_1 -> t1_1, instrument_dataset_4 -> t1_4
            #   test_5_8/instrument_dataset_1 -> t5_1, ...
            #   test_9_10/instrument_dataset_1 -> t9_1
            base = tp_idx.replace('test_', '').split('_')[0]
            test.append((f't{base}_{n}', p))

    return train, test


def collect_frames(ds_path: Path) -> List[Tuple[Path, Path]]:
    """Return [(img_path, list_of_label_files)] for one instrument_dataset_*.
    label_files is a list, not joined yet (so caller can run task-specific merger)."""
    img_dir = ds_path / 'left_frames'
    gt_dir = ds_path / 'ground_truth'
    if not img_dir.is_dir():
        return []
    tool_subdirs = sorted([d for d in gt_dir.iterdir() if d.is_dir()]) if gt_dir.is_dir() else []
    pairs = []
    for img in sorted(img_dir.glob('*.png')):
        # gather all per-tool labels for this frame name
        label_files = []
        for td in tool_subdirs:
            lab = td / img.name
            if lab.exists():
                label_files.append((lab, _normalize_tool_name(td.name)))
        pairs.append((img, label_files))
    return pairs


# ---------- Analysis & application ----------
def analyze(root: Path):
    print(f'\n=== Discovering EndoVis2017 under {root} ===')
    train, test = discover(root)
    print(f'\n[train] {len(train)} datasets:')
    for tag, p in train:
        n = len(list((p / 'left_frames').glob('*.png'))) if (p / 'left_frames').is_dir() else 0
        gt_dirs = sorted([d.name for d in (p / 'ground_truth').iterdir()
                          if d.is_dir()]) if (p / 'ground_truth').is_dir() else []
        rel = p.relative_to(root)
        print(f'  {tag:>6}  {n:4d} frames    from {rel}')
        for gd in gt_dirs:
            print(f'           tool: {gd}')
    print(f'\n[test] {len(test)} datasets:')
    for tag, p in test:
        n = len(list((p / 'left_frames').glob('*.png'))) if (p / 'left_frames').is_dir() else 0
        rel = p.relative_to(root)
        print(f'  {tag:>6}  {n:4d} frames    from {rel}')

    total_train = sum(len(list((p / 'left_frames').glob('*.png')))
                       for _, p in train if (p / 'left_frames').is_dir())
    total_test = sum(len(list((p / 'left_frames').glob('*.png')))
                      for _, p in test if (p / 'left_frames').is_dir())
    print(f'\nTotal train frames: {total_train}')
    print(f'Total test  frames: {total_test}')
    return train, test


def apply_split(train, test, root: Path, out: Path, args):
    out.mkdir(parents=True, exist_ok=True)
    classes = PARTS_CLASSES if args.task == 'parts' else TYPE_CLASSES
    nclass = len(classes)
    merge = _build_parts_mask if args.task == 'parts' else _build_type_mask
    print(f'\n[task] {args.task}   nclass={nclass}')

    # Sequence-level train/val split (by ds number)
    val_set = set(args.val_datasets)
    train_split = [(tag, p) for tag, p in train
                    if int(re.match(r'ds(\d+)', tag).group(1)) not in val_set]
    val_split   = [(tag, p) for tag, p in train
                    if int(re.match(r'ds(\d+)', tag).group(1)) in val_set]
    test_split  = test
    print(f'[seq-split] train: {[t for t,_ in train_split]}')
    print(f'[seq-split] val  : {[t for t,_ in val_split]}')
    print(f'[seq-split] test : {[t for t,_ in test_split]}')

    def process(ds_list, split_name):
        img_out = out / 'images' / split_name
        mask_out = out / 'masks' / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)
        n_img = n_mask = 0
        records = []
        for tag, ds_path in ds_list:
            pairs = collect_frames(ds_path)
            for img_path, label_files in pairs:
                stem = f'{tag}_{img_path.stem}'
                # symlink (or crop+save) image
                dst_img = img_out / f'{stem}.png'
                if not dst_img.exists():
                    if args.crop and Image.open(img_path).size == (1920, 1080):
                        # need to actually crop, can't symlink
                        Image.open(img_path).crop((CROP_OFFSET[0], CROP_OFFSET[1],
                                                     CROP_OFFSET[0] + CROP_SIZE[0],
                                                     CROP_OFFSET[1] + CROP_SIZE[1])).save(dst_img)
                    else:
                        try:
                            dst_img.symlink_to(img_path.resolve())
                        except OSError:
                            import shutil; shutil.copy2(img_path, dst_img)
                n_img += 1
                # merge labels
                mask_rel = None
                if label_files:
                    dst_mask = mask_out / f'{stem}.png'
                    if not dst_mask.exists():
                        # use raw resolution; merge then crop
                        sample = np.array(Image.open(label_files[0][0]))
                        h, w = sample.shape[:2]
                        merged = merge(label_files, h, w)
                        merged_img = Image.fromarray(merged)
                        if args.crop and merged_img.size == (1920, 1080):
                            merged_img = merged_img.crop((CROP_OFFSET[0], CROP_OFFSET[1],
                                                            CROP_OFFSET[0] + CROP_SIZE[0],
                                                            CROP_OFFSET[1] + CROP_SIZE[1]))
                        merged_img.save(dst_mask)
                    n_mask += 1
                    mask_rel = f'masks/{split_name}/{stem}.png'
                records.append((f'images/{split_name}/{stem}.png', mask_rel))
        print(f'[{split_name}] images={n_img}  masks={n_mask}')
        return records

    train_records = process(train_split, 'train')
    val_records   = process(val_split,   'val')
    test_records  = process(test_split,  'test')

    # ---- Split files ----
    split_dir = out / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)
    val_lines = sorted(f'{i}\t{m}' for i, m in val_records if m)
    (split_dir / 'val.txt').write_text('\n'.join(val_lines) + '\n', encoding='utf-8')
    print(f'[split] val.txt   {len(val_lines)} lines')
    if any(m for _, m in test_records):
        test_lines = sorted(f'{i}\t{m}' for i, m in test_records if m)
        (split_dir / 'test.txt').write_text('\n'.join(test_lines) + '\n', encoding='utf-8')
        print(f'[split] test.txt  {len(test_lines)} lines')
    else:
        test_lines = sorted(i for i, _ in test_records)
        (split_dir / 'test.txt').write_text('\n'.join(test_lines) + '\n', encoding='utf-8')
        print(f'[split] test.txt  {len(test_lines)} lines (NO labels — image-only)')

    # Per-ratio train splits
    rng = random.Random(args.seed)
    train_pairs = [r for r in train_records if r[1] is not None]
    rng.shuffle(train_pairs)
    n_total = len(train_pairs)
    for r in args.ratios:
        n_lab = max(1, int(round(n_total * r / 100.0)))
        lab, unl = train_pairs[:n_lab], train_pairs[n_lab:]
        sub = split_dir / f'{r:03d}'
        sub.mkdir(parents=True, exist_ok=True)
        (sub / 'labeled.txt').write_text(
            '\n'.join(sorted(f'{i}\t{m}' for i, m in lab)) + '\n', encoding='utf-8')
        (sub / 'unlabeled.txt').write_text(
            '\n'.join(sorted(i for i, _ in unl)) + '\n', encoding='utf-8')
        print(f'[ratio {r:>3d}%]  labeled={len(lab):>4d}  unlabeled={len(unl):>5d}')

    plan = {
        'source_root': str(root),
        'task': args.task,
        'val_datasets': sorted(args.val_datasets),
        'class_names': classes,
        'nclass': nclass,
        'ignore_index': 255,
        'crop': bool(args.crop),
        'crop_offset': list(CROP_OFFSET),
        'crop_size':   list(CROP_SIZE),
        'parts_value_to_id': {str(k): v for k, v in PARTS_VALUE_TO_ID.items()},
        'type_name_to_id':   TYPE_NAME_TO_ID,
        'ratios': list(args.ratios),
        'seed': int(args.seed),
    }
    (out / 'remap_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    key = f'endovis2017_{args.task}'
    (out / 'endovis2017_classes.txt').write_text(
        f"'{key}': {classes},\n", encoding='utf-8')

    info = out / 'dataset_info.txt'
    with open(info, 'w', encoding='utf-8') as f:
        f.write(f'EndoVis2017 ({args.task} task) processed from: {root}\n')
        f.write(f'Reference: Allan et al., "2017 Robotic Instrument Segmentation '
                f'Challenge", arXiv 1902.06426\n')
        f.write(f'nclass = {nclass}  ignore_index = 255\n\n')
        for i, n in enumerate(classes):
            f.write(f'  {i}: {n}\n')
        f.write(f'\nSequence-level splits:\n')
        f.write(f'  train datasets ({len(train_split)}): {[t for t,_ in train_split]}\n')
        f.write(f'  val   datasets ({len(val_split)}): {[t for t,_ in val_split]}\n')
        f.write(f'  test  datasets ({len(test_split)}): {[t for t,_ in test_split]}\n')
        f.write(f'\nCrop: {"enabled (1920x1080 -> 1280x1024 at 320,28)" if args.crop else "DISABLED"}\n')
    print(f'\n[info]  wrote {info}')
    print(f'[plan]  wrote {out / "remap_plan.json"}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out',  type=Path, default=None)
    ap.add_argument('--task', choices=['parts', 'type'], default='parts',
                    help='parts (4 cls: bg/shaft/wrist/clasper) | type (8 cls)')
    ap.add_argument('--val-datasets', type=int, nargs='+', default=[2, 7],
                    help='train dataset numbers reserved for validation')
    ap.add_argument('--ratios', type=int, nargs='+', default=[10, 20, 25, 30, 50])
    ap.add_argument('--crop', action='store_true', default=True,
                    help='crop 1920x1080 -> 1280x1024 at offset (320, 28)')
    ap.add_argument('--no-crop', dest='crop', action='store_false')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f'--root {args.root} not a directory')

    train, test = analyze(args.root)
    if args.apply:
        if args.out is None:
            sys.exit('--apply requires --out')
        apply_split(train, test, args.root, args.out, args)
    else:
        print('\n[dry-run] add --apply --out <dir> to write outputs.')


if __name__ == '__main__':
    main()
