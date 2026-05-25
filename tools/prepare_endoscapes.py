"""Prepare full Endoscapes dataset for UniMatchV2-style SSL training.

Applies the protocol from Murali et al. (arXiv 2112.13815):
  1. Ignore classes appearing in <5% of images in ANY of train/val/test splits.
  2. Merge all instrument tip + shaft sub-classes into a single 'tools' class.
  3. Ignore the lymph_node class.

Expected source layout (after extracting CAMMA-public/Endoscapes):
    <root>/seg_label_map.txt
    <root>/train_seg/<frame_id>.png        # seg masks
    <root>/train/<frame_id>.jpg            # all 1-fps frames (incl. unlabeled)
    <root>/{val,test,val_seg,test_seg}/    # likewise

Output layout (under --out):
    out/remap_plan.json                    # full record of class decisions
    out/dataset_info.txt                   # human-readable summary
    out/images/{train,val,test}/*.jpg      # symlinks to originals (no copy)
    out/masks/{train,val,test}/*.png       # REMAPPED single-channel masks
    out/splits/labeled.txt                 # train frames WITH mask
    out/splits/unlabeled.txt               # train frames WITHOUT mask
    out/splits/val.txt
    out/splits/test.txt

Each line in a split file:
    <img_rel_path>\t<mask_rel_path>           (labeled / val / test)
    <img_rel_path>                            (unlabeled — mask col empty)

Then point the YAML `data_root` at <out> directly.

Usage:
    # 1) Analyze only — prints proposed plan, NO writes
    python tools/prepare_endoscapes.py --root /data/test/endoscapes/endoscapes

    # 2) Apply the plan
    python tools/prepare_endoscapes.py --root /data/test/endoscapes/endoscapes \
        --out /data/test/endoscapes_processed --apply
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# ----------------------------------------------------------------------
INSTRUMENT_KEYWORDS = [
    'grasper', 'bipolar', 'hook', 'lhook', 'l-hook', 'l_hook',
    'scissors', 'clipper', 'irrigator', 'instrument', 'tool',
    'tip', 'shaft', 'wrist', 'specbag',
]
LYMPH_KEYWORDS = ['lymph']


def detect_role(name: str) -> str:
    n = name.lower().replace(' ', '').replace('-', '').replace('_', '')
    if any(k.replace(' ', '').replace('-', '').replace('_', '') in n for k in LYMPH_KEYWORDS):
        return 'lymph_node'
    if any(k.replace(' ', '').replace('-', '').replace('_', '') in n for k in INSTRUMENT_KEYWORDS):
        return 'instrument'
    if n in ('background', 'bg', 'void', 'unknown', 'unlabeled'):
        return 'background'
    return 'normal'


# ----------------------------------------------------------------------
def read_label_map(path: Path) -> dict[int, str]:
    """Tolerant parser. Supports:
        (a) '<id> <name>' per line  (id from first column)
        (b) '<name>' per line       (id = line index, 0-based)
    Lines starting with '#' or blank lines are skipped.
    """
    raw_lines = [ln for ln in path.read_text(encoding='utf-8').splitlines()
                  if ln.strip() and not ln.strip().startswith('#')]
    out = {}
    used_idx_format = True
    for raw in raw_lines:
        m = re.match(r'^\s*(\d+)\s*[:\s,]\s*(\S.+?)\s*$', raw)
        if not m:
            used_idx_format = False
            break
        out[int(m.group(1))] = m.group(2).strip().strip('"').strip("'")
    if not used_idx_format or not out:
        # Fall back: one name per line, id = line index
        out = {}
        for i, raw in enumerate(raw_lines):
            out[i] = raw.strip().strip('"').strip("'")
    return out


def pair_by_stem(img_dir: Path, mask_dir: Path, full_img_dir: Path | None = None):
    """Return ([labeled (img,mask) pairs], [unlabeled img-only]) for one split.

    img_dir       — split's labeled-image dir (e.g. train_seg/, contains .jpg)
    mask_dir      — flat mask dir for all splits (e.g. semseg/, contains .png)
    full_img_dir  — optional larger image pool (e.g. train/); frames here whose
                    stem is NOT in img_dir are treated as unlabeled.
    """
    img_exts = {'.jpg', '.jpeg', '.png'}
    labeled_imgs = {p.stem: p for p in img_dir.iterdir()
                     if p.is_file() and p.suffix.lower() in img_exts}
    mask_by_stem = {p.stem: p for p in mask_dir.iterdir()
                     if p.is_file() and p.suffix.lower() == '.png'}

    pairs = []
    for stem, ip in sorted(labeled_imgs.items()):
        mp = mask_by_stem.get(stem)
        if mp is not None:
            pairs.append((ip, mp))

    unlabeled = []
    if full_img_dir and full_img_dir.is_dir():
        for ip in full_img_dir.iterdir():
            if not ip.is_file() or ip.suffix.lower() not in img_exts:
                continue
            if ip.stem not in labeled_imgs:
                unlabeled.append(ip)
    return pairs, unlabeled


def class_presence(pairs, ignore_index: int = 255):
    """Count number of masks containing each class id (excluding ignore_index)."""
    present = defaultdict(int)
    for _, mp in pairs:
        arr = np.array(Image.open(mp))
        if arr.ndim == 3:
            arr = arr[..., 0]
        for c in np.unique(arr):
            c = int(c)
            if c == ignore_index:
                continue
            present[c] += 1
    return len(pairs), dict(present)


# ----------------------------------------------------------------------
def analyze(root: Path, threshold_pct: float = 5.0):
    label_map = read_label_map(root / 'seg_label_map.txt')
    print(f'\n=== seg_label_map.txt ({len(label_map)} classes) ===')
    for cid, nm in sorted(label_map.items()):
        print(f'  {cid:3d}  {nm:<30s}  ({detect_role(nm)})')

    # Mask layout: flat `semseg/` shared across splits; per-split labeled
    # subset given by `<split>_seg/` (which contains JPG IMAGES, not masks);
    # full sampled pool in `<split>/`.
    mask_dir = root / 'semseg'
    if not mask_dir.is_dir():
        sys.exit(f'missing {mask_dir} — expected flat directory of seg masks')

    splits = {}
    for s in ['train', 'val', 'test']:
        labeled_img_dir = root / f'{s}_seg'
        full_img_dir = root / s
        if not labeled_img_dir.is_dir():
            print(f'\n[warn] missing {labeled_img_dir}, skipping split={s}')
            continue
        pairs, unlab = pair_by_stem(labeled_img_dir, mask_dir, full_img_dir)
        n_pairs, freq = class_presence(pairs)
        splits[s] = {'pairs': pairs, 'unlabeled': unlab, 'freq': freq, 'n': n_pairs}
        print(f'\n[{s}] labeled={len(pairs)} frames with masks, '
              f'unlabeled={len(unlab)} frames without masks '
              f'(from {labeled_img_dir.name}/ vs {full_img_dir.name}/)')

    # 5% rule applied to image-level frequency.
    # Only consider classes that exist in seg_label_map.txt; anything else
    # (255 ignore, stray noise pixels) stays as 255 in the remapped masks.
    valid_ids = set(label_map.keys())
    all_ids = set()
    for s in splits.values():
        for cid in s['freq']:
            if cid in valid_ids:
                all_ids.add(cid)
            else:
                print(f'[note] orig id {cid} not in label_map — treated as ignore')

    print(f'\n=== Image-level class frequency (cut-off {threshold_pct}% in any split) ===')
    print(f'{"id":>3}  {"name":<28s}  {"role":<11s}  ', end='')
    for s in splits:
        print(f'{s+" %":>8s} ', end='')
    print(' decision')

    drop_freq, lymph_ids, instr_ids, keep_ids = [], [], [], []
    bg_id = 0

    for cid in sorted(all_ids):
        name = label_map.get(cid, f'cls{cid}')
        role = detect_role(name)
        pcts = []
        for s in splits.values():
            pct = 100.0 * s['freq'].get(cid, 0) / max(s['n'], 1)
            pcts.append(pct)
        min_pct = min(pcts) if pcts else 0.0

        if role == 'lymph_node':
            decision = 'drop (lymph_node rule)'; lymph_ids.append(cid)
        elif role == 'instrument':
            decision = 'merge -> tools';        instr_ids.append(cid)
        elif role == 'background':
            decision = 'keep (background)';     keep_ids.append(cid)
        elif min_pct < threshold_pct:
            decision = f'drop (<{threshold_pct}%)'; drop_freq.append(cid)
        else:
            decision = 'keep';                  keep_ids.append(cid)

        print(f'{cid:>3d}  {name:<28s}  {role:<11s}  ', end='')
        for p in pcts:
            print(f'{p:>7.2f}% ', end='')
        print(f' {decision}')

    # Build contiguous new IDs
    new_id = {}
    new_names = []
    next_id = 0
    # background first
    bg_kept = None
    for cid in keep_ids:
        name = label_map.get(cid, f'cls{cid}')
        if detect_role(name) == 'background':
            bg_kept = cid
            break
    if bg_kept is None:
        bg_kept = bg_id
    new_id[bg_kept] = next_id
    new_names.append(label_map.get(bg_kept, 'background'))
    next_id += 1
    # other kept classes
    for cid in keep_ids:
        if cid == bg_kept:
            continue
        new_id[cid] = next_id
        new_names.append(label_map.get(cid, f'cls{cid}'))
        next_id += 1
    # tools (if any instruments)
    if instr_ids:
        for cid in instr_ids:
            new_id[cid] = next_id
        new_names.append('tools')
        next_id += 1
    # dropped → 255 (ignore)
    drop_set = sorted(set(drop_freq + lymph_ids))

    plan = {
        'threshold_pct': threshold_pct,
        'original_label_map': label_map,
        'per_split_frequency': {s: {'n': v['n'], 'freq': v['freq']} for s, v in splits.items()},
        'remap': {int(k): int(v) for k, v in new_id.items()},
        'drop_to_ignore': drop_set,
        'instruments_merged': instr_ids,
        'lymph_node_dropped': lymph_ids,
        'new_class_names': new_names,
        'nclass_new': next_id,
    }

    print('\n=== Final class layout ===')
    print(f'  nclass = {next_id}, ignore_index = 255')
    for new_c, name in enumerate(new_names):
        origs = sorted([k for k, v in new_id.items() if v == new_c])
        print(f'  {new_c}: {name:<30s}  <- orig {origs}')
    if drop_set:
        print(f'  ignored ({len(drop_set)}): {drop_set}')

    return plan, splits


# ----------------------------------------------------------------------
def apply(plan, splits, root: Path, out: Path, args):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'remap_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(f'\n[plan] wrote {out / "remap_plan.json"}')

    # 256-entry LUT
    lut = np.full(256, 255, dtype=np.uint8)
    for orig, new in plan['remap'].items():
        lut[int(orig)] = int(new)

    for s, info in splits.items():
        img_out = out / 'images' / s
        mask_out = out / 'masks' / s
        img_out.mkdir(parents=True, exist_ok=True)
        mask_out.mkdir(parents=True, exist_ok=True)

        # All images become symlinks (no copy)
        n_img = 0
        for ip in {p for p, _ in info['pairs']} | set(info['unlabeled']):
            dst = img_out / ip.name
            if dst.exists() or dst.is_symlink():
                continue
            try:
                dst.symlink_to(ip.resolve())
            except OSError:
                # fall back to copy if symlink not supported
                import shutil
                shutil.copy2(ip, dst)
            n_img += 1

        # Remap masks
        n_mask = 0
        for _, mp in info['pairs']:
            arr = np.array(Image.open(mp))
            if arr.ndim == 3:
                arr = arr[..., 0]
            new_arr = lut[arr]
            Image.fromarray(new_arr.astype(np.uint8)).save(mask_out / mp.name)
            n_mask += 1
        print(f'[{s}] images={n_img}  masks_remapped={n_mask}')

    # Splits — produce val/test + multiple labeled-ratio variants of train.
    import random
    split_dir = out / 'splits'
    split_dir.mkdir(parents=True, exist_ok=True)

    def labeled_line(ip: Path, mp: Path, split_name: str) -> str:
        return f'images/{split_name}/{ip.name}\tmasks/{split_name}/{mp.stem}.png'

    def unlabeled_line(ip: Path, split_name: str) -> str:
        return f'images/{split_name}/{ip.name}'

    # val / test (always full)
    if 'val' in splits:
        lines = [labeled_line(ip, mp, 'val') for ip, mp in splits['val']['pairs']]
        (split_dir / 'val.txt').write_text('\n'.join(sorted(lines)) + '\n', encoding='utf-8')
        print(f'[split] val.txt: {len(lines)} lines')
    if 'test' in splits:
        lines = [labeled_line(ip, mp, 'test') for ip, mp in splits['test']['pairs']]
        (split_dir / 'test.txt').write_text('\n'.join(sorted(lines)) + '\n', encoding='utf-8')
        print(f'[split] test.txt: {len(lines)} lines')

    # train — generate ratio subdirs
    train_pairs = list(splits['train']['pairs'])              # 343 labeled
    train_unlab_all = list(splits['train']['unlabeled'])      # 26k+ unlabeled

    rng = random.Random(args.seed)
    pairs_shuffled = list(train_pairs)
    rng.shuffle(pairs_shuffled)
    unlab_shuffled = list(train_unlab_all)
    rng.shuffle(unlab_shuffled)

    n_full = len(pairs_shuffled)
    for ratio_pct in args.ratios:
        n_lab = max(1, int(round(n_full * ratio_pct / 100.0)))
        lab_pairs = pairs_shuffled[:n_lab]
        leftover_pairs = pairs_shuffled[n_lab:]
        leftover_imgs = [p for p, _ in leftover_pairs]
        if args.no_pool:
            # Endoscapes-Seg50 convention: unlabeled = leftover Seg50 only
            unlab_imgs = leftover_imgs
        else:
            # Endoscapes-full: leftover Seg50 + the 1-fps unlabeled pool
            unlab_imgs = leftover_imgs + unlab_shuffled
            # Cap by mu * n_lab if mu > 0
            if args.mu and args.mu > 0:
                cap = int(args.mu * n_lab)
                unlab_imgs = unlab_imgs[:cap]

        sub = split_dir / f'{ratio_pct:03d}'
        sub.mkdir(parents=True, exist_ok=True)
        lab_lines = sorted(labeled_line(ip, mp, 'train') for ip, mp in lab_pairs)
        unl_lines = sorted(unlabeled_line(ip, 'train') for ip in unlab_imgs)
        (sub / 'labeled.txt').write_text('\n'.join(lab_lines) + '\n', encoding='utf-8')
        (sub / 'unlabeled.txt').write_text('\n'.join(unl_lines) + '\n', encoding='utf-8')
        print(f'[split] ratio={ratio_pct:>3d}%:  '
              f'labeled={len(lab_lines):>4d}  unlabeled={len(unl_lines):>5d}  '
              f'(mu={len(unl_lines)/max(len(lab_lines),1):.1f})')

    # Human-readable summary
    info_path = out / 'dataset_info.txt'
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f'Endoscapes dataset prepared from: {root}\n')
        f.write(f'Number of classes (incl. background): {plan["nclass_new"]}\n')
        f.write(f'Ignore index: 255\n\n')
        f.write('Class layout:\n')
        for new_c, name in enumerate(plan['new_class_names']):
            origs = sorted([int(k) for k, v in plan['remap'].items() if int(v) == new_c])
            f.write(f'  {new_c}: {name:<30s}  <- orig {origs}\n')
        f.write(f'\nDropped to ignore: {plan["drop_to_ignore"]}\n')
        f.write(f'Lymph node ids dropped: {plan["lymph_node_dropped"]}\n')
        f.write(f'Instrument ids merged into tools: {plan["instruments_merged"]}\n\n')
        f.write('Splits:\n')
        for s, info in splits.items():
            f.write(f'  {s}: labeled={len(info["pairs"])}, '
                    f'unlabeled={len(info.get("unlabeled", []))}\n')
    print(f'\n[info] wrote {info_path}')


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True,
                    help='inner endoscapes dir (containing seg_label_map.txt + train_seg/...)')
    ap.add_argument('--out',  type=Path, default=None,
                    help='output directory (only used with --apply)')
    ap.add_argument('--threshold', type=float, default=5.0,
                    help='%% image-frequency cut-off in any split (default 5.0)')
    ap.add_argument('--ratios', type=int, nargs='+', default=[10, 20, 25, 30, 50, 100],
                    help='labeled ratios (in %%) to produce splits for (default: 10 20 25 30 50 100)')
    ap.add_argument('--mu', type=float, default=7.0,
                    help='unlabeled = mu * labeled (FixMatch convention). '
                          '0 disables the cap (use all unlabeled). default 7')
    ap.add_argument('--no-pool', action='store_true',
                    help='IGNORE the 1-fps full-Endoscapes unlabeled pool. '
                          'Unlabeled = leftover Seg50 labeled frames only. '
                          'Use this for the Endoscapes-Seg50 convention '
                          '(mu = (1-r)/r). Without it, the script uses the '
                          'full ~26k 1-fps pool (Endoscapes-full setup).')
    ap.add_argument('--seed', type=int, default=42,
                    help='random seed for labeled/unlabeled shuffling')
    ap.add_argument('--apply', action='store_true',
                    help='actually write outputs; without this only analysis is shown')
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f'--root {args.root} not a directory')
    if not (args.root / 'seg_label_map.txt').exists():
        sys.exit(f'missing {args.root}/seg_label_map.txt')

    plan, splits = analyze(args.root, args.threshold)

    if args.apply:
        if args.out is None:
            sys.exit('--apply requires --out')
        apply(plan, splits, args.root, args.out, args)
    else:
        print('\n[dry-run] add --apply --out <dir> to write outputs.')


if __name__ == '__main__':
    main()
