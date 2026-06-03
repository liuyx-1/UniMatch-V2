"""Build per-dataset val manifests + merged val manifest from val.txt splits.

Reads <splits-root>/<ds>/val.txt entries like:
    images/val/seq_1_frame123.png  masks/val/seq_1_frame123.png

For each entry: open the mask, extract unique class ids, build a record like
manifest_<ds>.jsonl. Then apply the same remap (from global_classes.json) to
build manifest_merged_val.jsonl that train_dense_align_merged.py can consume.

Usage (AutoDL paths):
    python tools/build_val_manifest.py \\
        --splits-root /root/autodl-tmp/data/autonomous_surgery/splits \\
        --data-root   /root/autodl-tmp/data/autonomous_surgery/public_data \\
        --global-info /root/autodl-tmp/data/autonomous_surgery/siglip_train/merged/global_classes.json \\
        --out-dir     /root/autodl-tmp/data/autonomous_surgery/siglip_train/merged_val

Expects data_root layout:
    <data-root>/<ds>_processed/{images,masks}/{train,val,test}/*.png  (or .jpg)
or, when ds is endoscapes_seg50, '<ds>_processed' is the standard name.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
from PIL import Image


# data_root subdir aliases (some datasets use a different processed dir name)
DATA_ROOT_NAMES = {
    'endoscapes_seg50':  ['endoscapes_seg50_processed', 'endoscapes_seg50'],
    'cholecseg8k':       ['cholecseg8k_processed', 'cholecseg8k'],
    'endovis2018':       ['endovis2018_processed', 'endovis2018'],
    'endovis2017_parts': ['endovis2017_parts_processed', 'endovis2017_parts'],
    'endovis2017_type':  ['endovis2017_type_processed', 'endovis2017_type'],
}


def _split_line(line):
    parts = line.strip().split('\t') if '\t' in line else line.strip().split(' ')
    return parts[0], (parts[1] if len(parts) > 1 else None)


def _find_data_root(data_root: Path, ds: str) -> Path | None:
    for name in DATA_ROOT_NAMES.get(ds, [ds]):
        p = data_root / name
        if p.is_dir():
            return p
    return None


def build_one_ds_val(splits_root: Path, data_root: Path, ds: str):
    val_txt = splits_root / ds / 'val.txt'
    if not val_txt.exists():
        print(f'  [skip] {ds}: no val.txt at {val_txt}')
        return []
    droot = _find_data_root(data_root, ds)
    if droot is None:
        print(f'  [skip] {ds}: data_root not found')
        return []
    recs = []
    def _resolve(rel):
        if rel.startswith('/'):
            return Path(rel)
        # try multiple roots: <data>/<ds>_processed/<rel>, then <data>/<rel>
        for base in [droot, data_root]:
            p = base / rel
            if p.exists(): return p
        return droot / rel   # default, may not exist

    for ln in val_txt.read_text(encoding='utf-8').splitlines():
        if not ln.strip(): continue
        img_rel, mask_rel = _split_line(ln)
        img_p = _resolve(img_rel)
        if mask_rel is None:
            stem = Path(img_rel).stem
            mask_p = _resolve(f'masks/val/{stem}.png')
        else:
            mask_p = _resolve(mask_rel)
        if not img_p.exists() or not mask_p.exists():
            continue
        try:
            mask = np.array(Image.open(mask_p))
            if mask.ndim == 3: mask = mask[..., 0]
        except Exception as e:
            print(f'  [warn] cannot read {mask_p}: {e}'); continue
        present = sorted(int(c) for c in np.unique(mask).tolist() if int(c) < 250)
        recs.append({
            'image':   str(img_p),
            'mask':    str(mask_p),
            'classes': present,
        })
    print(f'  {ds}: {len(recs)} val records')
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits-root', type=Path, required=True)
    ap.add_argument('--data-root',   type=Path, required=True)
    ap.add_argument('--global-info', type=Path, required=True,
                    help='global_classes.json produced by build_merged_manifest.py')
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads(args.global_info.read_text(encoding='utf-8'))
    remap_per_ds = info['remap']
    rare_global = set(info.get('rare_global', []))
    G = len(info['names'])

    out_jsonl = args.out_dir / 'manifest_merged_val.jsonl'
    n_total = 0; n_rare = 0
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for ds in remap_per_ds.keys():
            recs = build_one_ds_val(args.splits_root, args.data_root, ds)
            # also dump per-ds val manifest for inspection
            per_ds_path = args.out_dir / f'manifest_{ds}_val.jsonl'
            with open(per_ds_path, 'w', encoding='utf-8') as g:
                for r in recs:
                    g.write(json.dumps(r) + '\n')
            # write to merged val with remap
            for r in recs:
                gids = sorted({int(remap_per_ds[ds][str(int(c))])
                                for c in r['classes']
                                if remap_per_ds[ds].get(str(int(c))) is not None})
                if not gids: continue
                has_rare = any(g in rare_global for g in gids)
                f.write(json.dumps({
                    'image':           r['image'],
                    'mask':            r['mask'],
                    'dataset':         ds,
                    'classes_global':  gids,
                    'classes_orig':    [int(c) for c in r['classes']],
                    'has_rare_class':  has_rare,
                    'sample_weight':   1.0,
                }) + '\n')
                n_total += 1
                if has_rare: n_rare += 1

    print(f'\n[write] {out_jsonl}  ({n_total} records, {n_rare} with rare class)')


if __name__ == '__main__':
    main()
