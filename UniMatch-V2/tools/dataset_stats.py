"""Per-dataset class / video / split statistics.

For each surgical dataset, prints:
  - class names (from util.classes.CLASSES)
  - per-class image count (#imgs containing ≥1 pixel of class c)
  - per-class pixel share (% of total non-ignore pixels)
  - train / val / test image counts
  - which video / sequence IDs are assigned to val and test

Usage:
    export SPLITS=/root/autodl-tmp/data/autonomous_surgery/splits
    export DATA_ROOT=/root/autodl-tmp/data/autonomous_surgery/public_data
    python tools/dataset_stats.py
    python tools/dataset_stats.py --datasets endoscapes_seg50 cholecseg8k
    python tools/dataset_stats.py --rate 0.25 --sample 0   # 0 = full scan; >0 = subsample
"""
from __future__ import annotations
import argparse, os, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

# repo root → util.classes
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.classes import CLASSES

IGNORE_INDEX = 255
DEFAULT_DATASETS = ["endoscapes_seg50", "cholecseg8k",
                    "endovis2018", "endovis2017_parts", "endovis2017_type"]


# ----- video / sequence id extractor per dataset ---------------------------
def video_of(ds: str, rel_path: str) -> str:
    """Extract the natural grouping ID (video / sequence) from a relative path."""
    if ds == "cholecseg8k":
        m = re.search(r"(video\d+)", rel_path)
        return m.group(1) if m else "?"
    if ds == "endovis2018":
        m = re.search(r"(seq_?\d+)", rel_path)
        return m.group(1) if m else "?"
    if ds in ("endovis2017_parts", "endovis2017_type"):
        m = re.search(r"(instrument_dataset_\d+|dataset_\d+)", rel_path)
        return m.group(1) if m else "?"
    if ds == "endoscapes_seg50":
        # endoscapes file names are usually <video_id>_<frame>.jpg
        stem = Path(rel_path).stem
        m = re.match(r"^(\d+)_", stem)
        return f"video{m.group(1)}" if m else "?"
    return "?"


# ----- split file reading --------------------------------------------------
def parse_split(p: Path):
    if not p.is_file(): return []
    rows = []
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if not ln: continue
        parts = ln.split("\t") if "\t" in ln else ln.split()
        rows.append((parts[0], parts[1] if len(parts) > 1 else None))
    return rows


def resolve(rel: str, root: Path) -> Path:
    p = Path(rel); return p if p.is_absolute() else root/p


# ----- mask scanner --------------------------------------------------------
def scan_masks(entries, root: Path, n_cls: int, sample: int = 0):
    """Return (img_count[c], pix_count[c], total_pix)."""
    if sample > 0 and len(entries) > sample:
        np.random.seed(42)
        idx = np.random.choice(len(entries), sample, replace=False)
        entries = [entries[i] for i in idx]

    img_cnt = np.zeros(n_cls, dtype=np.int64)
    pix_cnt = np.zeros(n_cls, dtype=np.int64)
    total   = 0
    n_done  = 0
    n_skip  = 0
    for img, msk in entries:
        if msk is None: n_skip += 1; continue
        p = resolve(msk, root)
        if not p.exists(): n_skip += 1; continue
        try:
            a = np.asarray(Image.open(p))
        except Exception:
            n_skip += 1; continue
        if a.ndim == 3: a = a[..., 0]   # if accidentally RGB
        valid = a != IGNORE_INDEX
        total += int(valid.sum())
        # bincount over valid pixels
        flat = a[valid].ravel()
        bc = np.bincount(flat, minlength=n_cls)
        if bc.shape[0] > n_cls:    # there may be values beyond declared classes
            bc = bc[:n_cls]
        pix_cnt += bc
        for c in range(n_cls):
            if bc[c] > 0: img_cnt[c] += 1
        n_done += 1
    return img_cnt, pix_cnt, total, n_done, n_skip


# ----- pretty print --------------------------------------------------------
def pct(num, den): return 100.0 * num / max(den, 1)

def print_class_table(class_names, img_cnt, pix_cnt, total, n_imgs):
    print(f'  {"id":>3} {"class":28} {"#imgs":>7} {"img%":>6}  '
          f'{"pixels":>14} {"pix%":>7}')
    print("  " + "-"*72)
    for c, name in enumerate(class_names):
        print(f'  {c:>3} {name[:28]:28} '
              f'{int(img_cnt[c]):>7d} {pct(img_cnt[c], n_imgs):>5.1f}%  '
              f'{int(pix_cnt[c]):>14,} {pct(pix_cnt[c], total):>6.2f}%')


def print_video_split(ds, val_entries, test_entries):
    val_vids  = sorted({video_of(ds, e[0]) for e in val_entries})
    test_vids = sorted({video_of(ds, e[0]) for e in test_entries})
    print(f'  val  videos ({len(val_vids):>3}): {", ".join(val_vids) if val_vids else "—"}')
    print(f'  test videos ({len(test_vids):>3}): {", ".join(test_vids) if test_vids else "—"}')


# ----- main ----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--rate", type=str, default="0.25",
                    help="ratio dir to use as the labeled+unlabeled training pool")
    ap.add_argument("--sample", type=int, default=0,
                    help=">0 → subsample N masks per split to speed up pixel stats")
    args = ap.parse_args()

    SPLITS    = Path(os.environ.get("SPLITS",
                  "/root/autodl-tmp/data/autonomous_surgery/splits"))
    DATA_ROOT = Path(os.environ.get("DATA_ROOT",
                  "/root/autodl-tmp/data/autonomous_surgery/public_data"))
    print(f'[paths] SPLITS={SPLITS}\n        DATA_ROOT={DATA_ROOT}\n        rate={args.rate}  sample={args.sample}\n')

    for ds in args.datasets:
        root  = DATA_ROOT / f"{ds}_processed"
        sdir  = SPLITS / f"unimatch_splits_{ds}_{args.rate}_seed42"
        class_names = CLASSES.get(ds)
        if class_names is None:
            print(f'== {ds} : no entry in util.classes.CLASSES, skip =='); continue
        n_cls = len(class_names)

        print("=" * 90)
        print(f'== {ds}   ({n_cls} classes)   data_root={root}')
        print("=" * 90)

        lab  = parse_split(sdir/"labeled.txt")
        unl  = parse_split(sdir/"unlabeled.txt")
        val  = parse_split(sdir/"val.txt")
        test = parse_split(sdir/"test.txt")
        train = lab + unl
        print(f'  splits:   labeled={len(lab):>5}   unlabeled={len(unl):>5}   '
              f'train(L+U)={len(train):>5}   val={len(val):>5}   test={len(test):>5}\n')

        # Train-set class statistics (scan labeled masks; unlabeled has masks too)
        print('  ── Train pool (labeled+unlabeled) class statistics ──')
        img_cnt, pix_cnt, total, n_done, n_skip = scan_masks(
            train, root, n_cls, sample=args.sample)
        print(f'  scanned {n_done}/{len(train)} masks  (skipped {n_skip})\n')
        print_class_table(class_names, img_cnt, pix_cnt, total, n_done)

        # Val-set class statistics
        if val:
            print('\n  ── Val class statistics ──')
            v_img, v_pix, v_tot, v_n, v_skip = scan_masks(val, root, n_cls, sample=args.sample)
            print(f'  scanned {v_n}/{len(val)} masks  (skipped {v_skip})\n')
            print_class_table(class_names, v_img, v_pix, v_tot, v_n)

        # Video-level split assignment
        print('\n  ── Video / sequence assignment ──')
        print_video_split(ds, val, test)
        print()


if __name__ == "__main__":
    main()
