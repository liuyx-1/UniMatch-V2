"""Remap EndoVis 2018 12-class masks to the 7-class "Scene" task used by
Surg-SegFormer (CVPR'25 / arXiv 2024).

Old class set (12, in endovis2018_processed):
    0  background-tissue
    1  instrument-shaft           ┐
    2  instrument-clasper         ├─→  merged into single class "Robotic Instrument Part"
    3  instrument-wrist           ┘
    4  kidney-parenchyma
    5  covered-kidney
    6  thread                     ─→  ignore (255)
    7  clamps                     ─→  ignore (255)
    8  suturing-needle
    9  suction-instrument         ─→  ignore (255)
   10  small-intestine
   11  ultrasound-probe

New class set (7, used by Surg-SegFormer Task 1):
    0  background-tissue
    1  robotic-instrument-part
    2  kidney-parenchyma
    3  covered-kidney
    4  suturing-needle
    5  small-intestine
    6  ultrasound-probe

Output layout mirrors the input (images/, masks/, splits/), so the new
dir can be plugged in as a fresh dataset:

    <out_root>/
        images/{train,val,test}/...   (symlinks to source images)
        masks/{train,val,test}/...    (remapped uint8 PNG)
        splits/{labeled,unlabeled,val,test}.txt   (paths rewritten to
                                                   point at out_root)

Usage:
    python tools/remap_endovis2018_scene.py \
        --src /root/autodl-tmp/data/autonomous_surgery/public_data/endovis2018_processed \
        --dst /root/autodl-tmp/data/autonomous_surgery/public_data/endovis2018_scene_processed
"""
from __future__ import annotations
import argparse, os, shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ----- old → new class id mapping ------------------------------------
REMAP = {
    0: 0,                  # background-tissue
    1: 1, 2: 1, 3: 1,      # shaft / clasper / wrist  → robotic-instrument-part
    4: 2,                  # kidney-parenchyma
    5: 3,                  # covered-kidney
    6: 255, 7: 255,        # thread / clamps          → ignore
    8: 4,                  # suturing-needle
    9: 255,                # suction-instrument       → ignore
    10: 5,                 # small-intestine
    11: 6,                 # ultrasound-probe
}
NEW_NCLASS = 7


def apply_remap(arr: np.ndarray) -> np.ndarray:
    """Vectorised lookup; values not in REMAP stay as 255 (ignore)."""
    out = np.full(arr.shape, 255, dtype=np.uint8)
    for old, new in REMAP.items():
        out[arr == old] = new
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True,
                    help='source endovis2018_processed dir')
    ap.add_argument('--dst', type=Path, required=True,
                    help='destination endovis2018_scene_processed dir')
    ap.add_argument('--symlink-images', action='store_true', default=True,
                    help='use symlinks for images (saves space)')
    ap.add_argument('--copy-images', dest='symlink_images', action='store_false')
    args = ap.parse_args()

    src, dst = args.src, args.dst
    assert src.is_dir(), f"source not a dir: {src}"
    dst.mkdir(parents=True, exist_ok=True)

    # ----- images: symlink or copy ----------------------------------
    for split in ('train', 'val', 'test'):
        s = src / 'images' / split
        if not s.is_dir(): continue
        d = dst / 'images' / split
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted(s.iterdir()):
            tgt = d / f.name
            if tgt.exists() or tgt.is_symlink(): tgt.unlink()
            if args.symlink_images:
                tgt.symlink_to(f.resolve())
            else:
                shutil.copy2(f, tgt)
        print(f'[images] {split}: {sum(1 for _ in d.iterdir())} files')

    # ----- masks: remap pixel values ---------------------------------
    n_total, n_skip = 0, 0
    histo_old = np.zeros(256, dtype=np.int64)
    histo_new = np.zeros(256, dtype=np.int64)
    for split in ('train', 'val', 'test'):
        s = src / 'masks' / split
        if not s.is_dir(): continue
        d = dst / 'masks' / split
        d.mkdir(parents=True, exist_ok=True)
        for f in sorted(s.iterdir()):
            try:
                a = np.asarray(Image.open(f))
            except Exception as e:
                print(f'[skip] {f}: {e}'); n_skip += 1; continue
            if a.ndim == 3: a = a[..., 0]
            histo_old += np.bincount(a.ravel(), minlength=256)
            b = apply_remap(a)
            histo_new += np.bincount(b.ravel(), minlength=256)
            Image.fromarray(b).save(d / f.name)
            n_total += 1
        print(f'[masks]  {split}: {sum(1 for _ in d.iterdir())} remapped')
    print(f'[masks]  total remapped={n_total}  skipped={n_skip}')

    # ----- splits: copy lines, no path rewrite needed ----------------
    sp_src = src / 'splits'
    sp_dst = dst / 'splits'
    if sp_src.is_dir():
        shutil.copytree(sp_src, sp_dst, dirs_exist_ok=True)
        print(f'[splits] copied {sp_src.name} → {sp_dst.name}')

    # ----- print histograms before / after ---------------------------
    print('\n[hist] old → new pixel counts (px > 0):')
    print(f'  {"old_id":>6} {"old_count":>14}   {"new_id":>6} {"new_count":>14}')
    for v in range(12):
        if histo_old[v] == 0: continue
        new_v = REMAP[v]
        print(f'  {v:>6d} {int(histo_old[v]):>14,d}   {new_v:>6d} '
              f'{int(histo_new[new_v]) if new_v != 255 else int(histo_new[255]):>14,d}')
    print(f'  ignore (255): old={int(histo_old[255]):,d}  new={int(histo_new[255]):,d}')


if __name__ == '__main__':
    main()
