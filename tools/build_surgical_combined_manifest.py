"""Build Stage-1 affinity manifests for the surgical_combined (needle/thread/clamps) set.

train_affinity_per_ds.py consumes two files for this dataset:
    <manifest-dir>/manifest_surgical_combined.jsonl       (train, from labeled.txt)
    <val-dir>/manifest_surgical_combined_val.jsonl         (val,   from val.txt)

Each record: {"dataset", "image" (abs), "mask" (abs), "classes" (present orig ids),
"class_names"}. `classes` lists every class id (incl. background 0) with
>= --min-pixels pixels; train_affinity_per_ds drops background via CLASS_NAMES,
so a leading 0 here is harmless. This matches the schema produced by
tools/build_siglip_manifest.py / build_val_manifest.py.

Why a dedicated script: surgical_combined splits live at a NON-standard location
(surgical_seg/combined/splits/r*/{labeled,val}.txt) and have no merged global
remap, so they do not fit build_siglip_manifest.py's
`unimatch_splits_{ds}_{rate}_seed{seed}` convention nor build_val_manifest.py's
`--global-info` dependency. This builder takes the split files directly.

Usage (server):
    python tools/build_surgical_combined_manifest.py \
        --data-root    /root/autodl-tmp/data/surgical_seg \
        --labeled-file /root/autodl-tmp/data/surgical_seg/combined/splits/r100/labeled.txt \
        --val-file     /root/autodl-tmp/data/surgical_seg/combined/splits/r100/val.txt \
        --manifest-dir /root/autodl-tmp/data/autonomous_surgery/siglip_train \
        --val-dir      /root/autodl-tmp/data/autonomous_surgery/siglip_train/merged_val \
        --min-pixels 32
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DS = 'surgical_combined'
CLASS_NAMES = ['background', 'needle', 'thread', 'clamps']   # mirrors util/classes.py
NCLASS = len(CLASS_NAMES)
DEFAULT_MIN_PIXELS = 32          # mirror cls_target_from_mask / build_siglip_manifest


def _split_line(line: str):
    parts = line.split('\t') if '\t' in line else line.split()
    return parts[0], (parts[1] if len(parts) > 1 else '')


def _present_classes(mask_path: Path, min_pixels: int):
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]                      # safety: drop alpha/RGB if any
    bincnt = np.bincount(arr.ravel(), minlength=256)
    return [c for c in range(NCLASS) if bincnt[c] >= min_pixels]


def build(split_file: Path, data_root: Path, min_pixels: int):
    recs = []
    lines = [ln.strip() for ln in split_file.read_text(encoding='utf-8').splitlines() if ln.strip()]
    n_skip = 0
    for ln in lines:
        img_rel, mask_rel = _split_line(ln)
        if not mask_rel:
            n_skip += 1
            continue                            # image-only line (no mask) -> skip
        img_abs = (data_root / img_rel).resolve()
        mask_abs = (data_root / mask_rel).resolve()
        if not img_abs.exists() or not mask_abs.exists():
            print(f'  ! missing pair, skip: {img_rel}')
            n_skip += 1
            continue
        present = _present_classes(mask_abs, min_pixels)
        if not present or present == [0]:
            n_skip += 1
            continue                            # background-only -> nothing to learn
        recs.append({
            'dataset':     DS,
            'image':       str(img_abs),
            'mask':        str(mask_abs),
            'classes':     present,
            'class_names': [CLASS_NAMES[c] for c in present],
        })
    return recs, n_skip


def _dump(recs, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=Path, required=True,
                    help='root that relative img/mask paths in the split files resolve against '
                         '(e.g. /root/autodl-tmp/data/surgical_seg)')
    ap.add_argument('--labeled-file', type=Path, required=True,
                    help='train split (labeled.txt) -> manifest_surgical_combined.jsonl')
    ap.add_argument('--val-file', type=Path, default=None,
                    help='val split (val.txt) -> manifest_surgical_combined_val.jsonl')
    ap.add_argument('--manifest-dir', type=Path, required=True,
                    help='output dir for the TRAIN manifest')
    ap.add_argument('--val-dir', type=Path, default=None,
                    help='output dir for the VAL manifest (default: --manifest-dir)')
    ap.add_argument('--min-pixels', type=int, default=DEFAULT_MIN_PIXELS)
    args = ap.parse_args()

    train_recs, n_skip = build(args.labeled_file, args.data_root, args.min_pixels)
    tpath = args.manifest_dir / f'manifest_{DS}.jsonl'
    _dump(train_recs, tpath)
    print(f'[train] {len(train_recs)} records (skipped {n_skip}) -> {tpath}')

    if args.val_file:
        val_dir = args.val_dir or args.manifest_dir
        val_recs, v_skip = build(args.val_file, args.data_root, args.min_pixels)
        vpath = val_dir / f'manifest_{DS}_val.jsonl'
        _dump(val_recs, vpath)
        print(f'[val]   {len(val_recs)} records (skipped {v_skip}) -> {vpath}')


if __name__ == '__main__':
    main()
