"""Build a Stage-I manifest for the merged endovis2017 (parts + type).

Each frame in `endovis2017_parts_processed` shares the same image as the
matching frame in `endovis2017_type_processed`.  We read BOTH masks per
frame and emit a single jsonl record whose `classes` field is the union
of the present parts (re-indexed 1..3) and types (re-indexed 4..10),
with background dropped.

Output (one file):
    <out-dir>/manifest_endovis2017_merged.jsonl

Output line format (one record per frame):
    {"img": "<abs path or relative to data_root>",
     "classes": [<merged class IDs present in either mask>],
     "ds": "endovis2017_merged"}

Usage:
    python tools/build_endovis2017_merged_manifest.py \
        --parts-root /root/autodl-tmp/data/.../endovis2017_parts_processed \
        --type-root  /root/autodl-tmp/data/.../endovis2017_type_processed \
        --split-root /root/autodl-tmp/data/.../splits \
        --rate 0.25 --seed 42 \
        --out-dir   /root/autodl-tmp/data/.../siglip_train

Stage-I trainer command remains unchanged; just pass
    --datasets endovis2017_merged
and ensure util.classes.CLASSES contains the matching name.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
from PIL import Image

# repo import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.classes import CLASSES


def _parse_split(p: Path):
    """Return list of img_rel (drop mask column if present)."""
    if not p.is_file(): return []
    out = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln: continue
        parts = ln.split('\t') if '\t' in ln else ln.split()
        out.append(parts[0])
    return out


def _load_mask(p: Path) -> np.ndarray:
    a = np.asarray(Image.open(p))
    if a.ndim == 3: a = a[..., 0]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts-root', type=Path, required=True)
    ap.add_argument('--type-root',  type=Path, required=True)
    ap.add_argument('--split-root', type=Path, required=True,
                    help='dir holding unimatch_splits_endovis2017_parts_<rate>_seed42/')
    ap.add_argument('--rate', type=str, default='0.25')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--split', type=str, default='train',
                    choices=['train', 'val'],
                    help='train → labeled+unlabeled (training manifest);  '
                          'val → val.txt (validation manifest, name suffixed _val)')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged_names = CLASSES['endovis2017_merged']            # [bg, 3 parts, 7 types]
    parts_names  = CLASSES['endovis2017_parts']             # [bg, 3 parts]
    type_names   = CLASSES['endovis2017_type']              # [bg, 7 types]

    # Map raw mask values  →  merged class id (0 = bg)
    # parts mask: 0..3 → merged {0, 1, 2, 3}
    # type  mask: 0..7 → merged {0, 4, 5, 6, 7, 8, 9, 10}
    parts_to_merged = {i: i for i in range(len(parts_names))}                   # 0..3
    type_to_merged  = {0: 0, **{i: i + len(parts_names) - 1                       # +3
                                 for i in range(1, len(type_names))}}

    # Source split list — parts and type share the same image set, use parts'
    SPLIT = args.split_root / f'unimatch_splits_endovis2017_parts_{args.rate}_seed{args.seed}'
    if args.split == 'train':
        img_rels = _parse_split(SPLIT / 'labeled.txt') + _parse_split(SPLIT / 'unlabeled.txt')
        out_path = args.out_dir / 'manifest_endovis2017_merged.jsonl'
    else:
        img_rels = _parse_split(SPLIT / 'val.txt')
        out_path = args.out_dir / 'manifest_endovis2017_merged_val.jsonl'
    print(f'[input] {len(img_rels)} frames from {SPLIT.name} / {args.split}')
    n_with_classes = 0
    with open(out_path, 'w') as fout:
        for rel in img_rels:
            p_mask = args.parts_root / rel.replace('images/', 'masks/')
            t_mask = args.type_root  / rel.replace('images/', 'masks/')
            if not (p_mask.exists() and t_mask.exists()):
                continue

            classes_present = set()

            am = _load_mask(p_mask)
            for v in np.unique(am):
                if v == 255: continue
                mid = parts_to_merged.get(int(v))
                if mid is not None and mid != 0:
                    classes_present.add(mid)

            tm = _load_mask(t_mask)
            for v in np.unique(tm):
                if v == 255: continue
                mid = type_to_merged.get(int(v))
                if mid is not None and mid != 0:
                    classes_present.add(mid)

            rec = {'img': rel, 'classes': sorted(classes_present),
                   'ds': 'endovis2017_merged'}
            fout.write(json.dumps(rec) + '\n')
            if classes_present: n_with_classes += 1

    print(f'[done] wrote {out_path}  ({n_with_classes} non-empty frames out of {len(img_rels)})')

    # Tiny class histogram for sanity
    counts = {n: 0 for n in merged_names[1:]}
    with open(out_path) as fin:
        for ln in fin:
            r = json.loads(ln)
            for c in r['classes']:
                counts[merged_names[c]] += 1
    print('[hist] images per class:')
    for n, k in counts.items():
        print(f'  {n:<28} {k}')


if __name__ == '__main__':
    main()
