"""Build (image, class-list) manifests for SigLIP-2 contrastive fine-tuning.

For each of 5 surgical/endoscopic datasets at the 0.10 label-rate split:
  1. read labeled.txt   -> (img_rel, mask_rel) pairs
  2. for each image, scan mask, decide which classes have >= min_pixels (default 32)
  3. record a manifest entry  {dataset, image, mask, classes, class_names, has_rare_class}
  4. compute per-class "image count" within this dataset
  5. mark rare classes (image-count ratio < rare_threshold, default 0.10)
  6. dump JSONL manifests + class_stats JSONs + a merged manifest_all.jsonl

Output layout (under --out, default pretrained/siglip_train/):
    manifest_<dataset>.jsonl      # one record per image, one dataset
    manifest_all.jsonl            # concatenated across all datasets
    class_stats_<dataset>.json    # {class_name: image_count, ...} + rare list
    rare_classes.json             # {dataset: [rare_class_names], ...}
    sample_weights_<dataset>.json # per-image weight for WeightedRandomSampler
    summary.txt                   # human-readable stats

Usage (AutoDL paths):
    python tools/build_siglip_manifest.py \\
        --splits-root /root/autodl-tmp/data/autonomous_surgery/splits \\
        --data-roots endoscapes_seg50=/root/autodl-tmp/data/autonomous_surgery/public_data/endoscapes_seg50_processed \\
                     cholecseg8k=/root/autodl-tmp/data/autonomous_surgery/public_data \\
                     endovis2018=/root/autodl-tmp/data/autonomous_surgery/public_data/endovis2018_processed \\
                     endovis2017_parts=/root/autodl-tmp/data/autonomous_surgery/public_data/endovis2017_parts_processed \\
                     endovis2017_type=/root/autodl-tmp/data/autonomous_surgery/public_data/endovis2017_type_processed \\
        --rate 0.25 --seed 42 \\
        --out /root/autodl-tmp/data/autonomous_surgery/siglip_train
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------
# Class name registry (mirrors util/classes.py, kept local to keep this
# script self-contained and importable without pulling in DDP/torch deps)
# ----------------------------------------------------------------------
CLASS_NAMES = {
    'endoscapes_seg50': [
        'background-tissue', 'cystic_plate', 'calot_triangle',
        'cystic_artery', 'cystic_duct', 'gallbladder', 'tool',
    ],
    'cholecseg8k': [
        'background', 'abdominal_wall', 'liver', 'gastrointestinal_tract',
        'fat', 'grasper', 'connective_tissue', 'blood', 'cystic_duct',
        'l_hook_electrocautery', 'gallbladder', 'hepatic_vein',
        'liver_ligament',
    ],
    'endovis2018': [
        'background-tissue', 'instrument-shaft', 'instrument-clasper',
        'instrument-wrist', 'kidney-parenchyma', 'covered-kidney',
        'thread', 'clamps', 'suturing-needle', 'suction-instrument',
        'small-intestine', 'ultrasound-probe',
    ],
    'endovis2017_parts': ['background', 'shaft', 'wrist', 'clasper'],
    'endovis2017_type': [
        'background', 'Bipolar Forceps', 'Prograsp Forceps',
        'Large Needle Driver', 'Vessel Sealer', 'Grasping Retractor',
        'Monopolar Curved Scissors', 'Ultrasound Probe',
    ],
    # Surgical multi-video set; mirrors util/classes.py. Note: its splits live
    # outside the unimatch_splits_{ds}_{rate}_seed{seed} convention, so build its
    # manifests with tools/build_surgical_combined_manifest.py, not this script.
    'surgical_combined': ['background', 'needle', 'thread', 'clamps'],
}

# Per-dataset hard-prompt template (used later in soft prompt training as
# the initial position; CoOp will replace most tokens with learned ones).
PROMPT_TEMPLATE = {
    'endoscapes_seg50':  '{cls} during laparoscopic cholecystectomy',
    'cholecseg8k':       '{cls} in a cholecystectomy surgery',
    'endovis2018':       '{cls} during robotic kidney surgery',
    'endovis2017_parts': 'the {cls} part of a surgical instrument',
    'endovis2017_type':  'a {cls} surgical instrument',
    'surgical_combined': '{cls} during robotic surgical suturing',
}

# Default thresholds (mirror cls_target_from_mask in util/cls_head.py)
DEFAULT_MIN_PIXELS = 32
DEFAULT_RARE_RATIO = 0.10        # class present in < 10% of labeled imgs -> rare


# ----------------------------------------------------------------------
def _split_line(line: str) -> Tuple[str, str]:
    parts = line.split('\t') if '\t' in line else line.split(' ')
    return parts[0], (parts[1] if len(parts) > 1 else '')


def _classes_in_mask(mask_path: Path, nclass: int,
                       min_pixels: int = DEFAULT_MIN_PIXELS,
                       ignore_index: int = 255) -> List[int]:
    """Return sorted list of class ids that have >= min_pixels in this mask."""
    arr = np.array(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]                          # safety: drop alpha/RGB if any
    bincnt = np.bincount(arr.ravel(), minlength=256)
    present = []
    for c in range(nclass):
        if bincnt[c] >= min_pixels:
            present.append(c)
    return present


def process_dataset(ds_name: str, data_root: Path, split_file: Path,
                     min_pixels: int, rare_ratio: float
                     ) -> Tuple[List[Dict], Dict[str, int], List[str]]:
    """Process one dataset's labeled split.

    Returns:
      records:        list of manifest entries (dicts)
      class_counts:   {class_name: num_images_containing_it}
      rare_classes:   list of class names judged rare
    """
    nclass = len(CLASS_NAMES[ds_name])
    cnames = CLASS_NAMES[ds_name]

    print(f'\n[{ds_name}] reading {split_file}')
    if not split_file.exists():
        print(f'  ! split file missing, skipping')
        return [], {}, []

    lines = [ln.strip() for ln in split_file.read_text(encoding='utf-8').splitlines() if ln.strip()]
    records: List[Dict] = []
    class_image_count = defaultdict(int)

    for ln in lines:
        img_rel, mask_rel = _split_line(ln)
        if not mask_rel:
            continue
        img_abs = (data_root / img_rel).resolve()
        mask_abs = (data_root / mask_rel).resolve()
        if not img_abs.exists() or not mask_abs.exists():
            print(f'  ! missing pair, skip: {img_rel}')
            continue

        present_ids = _classes_in_mask(mask_abs, nclass, min_pixels)
        if not present_ids:
            continue                                # purely empty mask, skip
        present_names = [cnames[c] for c in present_ids]
        for n in present_names:
            class_image_count[n] += 1
        records.append({
            'dataset':     ds_name,
            'image':       str(img_abs),
            'mask':        str(mask_abs),
            'classes':     present_ids,
            'class_names': present_names,
            'has_rare_class': False,                # filled later
        })

    # ---- decide rare classes ----
    n_imgs = len(records)
    rare = []
    for c in cnames:
        if c in class_image_count:
            ratio = class_image_count[c] / max(n_imgs, 1)
            if ratio < rare_ratio:
                rare.append(c)
    rare_set = set(rare)

    # ---- mark has_rare_class on each record ----
    for r in records:
        if any(n in rare_set for n in r['class_names']):
            r['has_rare_class'] = True

    print(f'  records: {n_imgs}  rare classes: {rare}')
    return records, dict(class_image_count), rare


def compute_sample_weights(records: List[Dict],
                            class_counts: Dict[str, int],
                            rare_classes: List[str],
                            rare_boost: float = 3.0,
                            alpha: float = 0.5) -> List[float]:
    """Per-image WeightedRandomSampler weight.

    base_w   = sum over c in image.class_names of (1 / class_counts[c])^alpha
    boost_w  = rare_boost if image contains any rare class else 1.0
    weight   = base_w * boost_w

    alpha=0.5 mild bias toward rare classes; rare_boost=3.0 extra kick
    for images containing any class in rare list.
    """
    rare_set = set(rare_classes)
    out = []
    for r in records:
        base = sum(1.0 / max(class_counts.get(c, 1), 1) ** alpha
                    for c in r['class_names'])
        boost = rare_boost if any(n in rare_set for n in r['class_names']) else 1.0
        out.append(base * boost)
    # normalise so sum==len (keeps WeightedRandomSampler effective sample size)
    s = sum(out)
    if s > 0:
        out = [w * len(out) / s for w in out]
    return out


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits-root', type=Path, required=True,
                    help='e.g. /root/autodl-tmp/data/autonomous_surgery/splits (contains unimatch_splits_*_<rate>_seed<seed>/)')
    ap.add_argument('--data-roots', nargs='+', required=True,
                    metavar='DSNAME=PATH',
                    help='one DSNAME=PATH per dataset; PATH is data_root used to '
                          'resolve relative img/mask paths in labeled.txt')
    ap.add_argument('--rate', type=str, default='0.10',
                    help='label rate string (default 0.10)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-pixels', type=int, default=DEFAULT_MIN_PIXELS)
    ap.add_argument('--rare-ratio', type=float, default=DEFAULT_RARE_RATIO,
                    help='class is rare if it appears in < rare_ratio fraction '
                          'of labeled images')
    ap.add_argument('--rare-boost', type=float, default=3.0,
                    help='per-image extra sampling weight if image contains any rare class')
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='exponent for inverse-frequency base weight')
    ap.add_argument('--out', type=Path, required=True,
                    help='output directory for manifests + stats')
    args = ap.parse_args()

    # parse data-roots
    data_roots = {}
    for spec in args.data_roots:
        if '=' not in spec:
            sys.exit(f'--data-roots entry must be DSNAME=PATH, got: {spec}')
        k, v = spec.split('=', 1)
        if k not in CLASS_NAMES:
            sys.exit(f'unknown dataset name: {k} (known: {list(CLASS_NAMES)})')
        data_roots[k] = Path(v)

    args.out.mkdir(parents=True, exist_ok=True)

    all_records = []
    all_rare_summary = {}
    summary_lines = []
    summary_lines.append(f'SigLIP-2 manifest build summary')
    summary_lines.append(f'  rate={args.rate}  seed={args.seed}')
    summary_lines.append(f'  min_pixels={args.min_pixels}  rare_ratio={args.rare_ratio}')
    summary_lines.append('=' * 60)

    for ds_name, data_root in data_roots.items():
        split_dir = args.splits_root / f'unimatch_splits_{ds_name}_{args.rate}_seed{args.seed}'
        split_file = split_dir / 'labeled.txt'

        records, class_counts, rare = process_dataset(
            ds_name, data_root, split_file,
            min_pixels=args.min_pixels, rare_ratio=args.rare_ratio,
        )
        if not records:
            continue

        # weights
        weights = compute_sample_weights(records, class_counts, rare,
                                           rare_boost=args.rare_boost, alpha=args.alpha)

        # dump per-dataset manifest
        mpath = args.out / f'manifest_{ds_name}.jsonl'
        with open(mpath, 'w', encoding='utf-8') as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        print(f'  -> {mpath}  ({len(records)} records)')

        # stats
        spath = args.out / f'class_stats_{ds_name}.json'
        spath.write_text(json.dumps({
            'n_images': len(records),
            'n_classes': len(CLASS_NAMES[ds_name]),
            'class_image_count': class_counts,
            'class_image_ratio': {c: round(v / max(len(records), 1), 4)
                                    for c, v in class_counts.items()},
            'rare_classes': rare,
            'prompt_template': PROMPT_TEMPLATE.get(ds_name, '{cls}'),
        }, ensure_ascii=False, indent=2), encoding='utf-8')

        # weights for sampler
        wpath = args.out / f'sample_weights_{ds_name}.json'
        wpath.write_text(json.dumps(weights), encoding='utf-8')

        all_records.extend(records)
        all_rare_summary[ds_name] = rare

        summary_lines.append(f'\n[{ds_name}]')
        summary_lines.append(f'  images: {len(records)}')
        summary_lines.append(f'  rare:   {rare}')
        for c in CLASS_NAMES[ds_name]:
            cnt = class_counts.get(c, 0)
            ratio = cnt / max(len(records), 1)
            tag = ' (rare)' if c in set(rare) else ''
            summary_lines.append(f'    {c:<28s}  {cnt:>4d} imgs ({ratio:.2%}){tag}')

    # merged manifest
    merged = args.out / 'manifest_all.jsonl'
    with open(merged, 'w', encoding='utf-8') as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'\n[merged] {merged}  total {len(all_records)} records')

    # rare summary
    (args.out / 'rare_classes.json').write_text(
        json.dumps(all_rare_summary, ensure_ascii=False, indent=2), encoding='utf-8')

    # text summary
    (args.out / 'summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')
    print(f'[summary] {args.out / "summary.txt"}')


if __name__ == '__main__':
    main()
