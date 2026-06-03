"""Compute per-class statistics for each surgical dataset and emit
a 3-line (booktabs) table per dataset, in Markdown / LaTeX / CSV.

For each dataset, scans every mask listed in the union of the available
splits (labeled / unlabeled / val / test) under
   ${SPLITS}/unimatch_splits_${DS}_${RATE}_seed42/
and accumulates:
   - n_imgs_with_class[c]   : # frames containing ≥1 pixel of class c
   - pixel_count[c]         : total pixels of class c
   - total_imgs, total_pixels (denominators)

Emits:
   <out_dir>/<DS>_class_stats.csv
   <out_dir>/<DS>_class_stats.md
   <out_dir>/<DS>_class_stats.tex      (booktabs three-line)
   <out_dir>/ALL_class_stats.tex       (concatenated for paper appendix)

Usage:
   python tools/dataset_stats_table.py \\
       --ds endoscapes_seg50 cholecseg8k endovis2018 endovis2017_parts endovis2017_type needle \\
       --rate 0.10 \\
       --data-root /root/autodl-tmp/data/autonomous_surgery/public_data \\
       --splits-root /root/autodl-tmp/data/autonomous_surgery/splits \\
       --out-dir /root/autodl-tmp/exp/_dataset_stats
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.classes import CLASSES


# Per-dataset data-root suffix in the AutoDL layout
DS_DIR = {
    'endoscapes_seg50':  'endoscapes_seg50_processed',
    'cholecseg8k':       'cholecseg8k_processed',
    'endovis2018':       'endovis2018_processed',
    'endovis2017_parts': 'endovis2017_parts_processed',
    'endovis2017_type':  'endovis2017_type_processed',
    'needle':            'unimatchv2_needle_3class',
}


def scan_dataset(data_root: Path, split_files: List[Path], nclass: int, ignore=255):
    """Iterate every (image,mask) line in the union of split_files and accumulate
    per-class image counts and pixel counts."""
    seen_pairs = set()
    n_imgs = 0
    img_count = np.zeros(nclass, dtype=np.int64)
    pix_count = np.zeros(nclass, dtype=np.int64)
    total_pix = 0

    for sf in split_files:
        if not sf.exists(): continue
        for line in sf.read_text().splitlines():
            line = line.strip()
            if not line: continue
            parts = line.split('\t')
            if len(parts) < 2: continue
            img_rel, msk_rel = parts[0], parts[1]
            if (img_rel, msk_rel) in seen_pairs: continue
            seen_pairs.add((img_rel, msk_rel))

            msk_path = data_root / msk_rel
            if not msk_path.exists():
                print(f'  [warn] missing {msk_path}'); continue
            arr = np.array(Image.open(msk_path))
            valid = arr != ignore
            ids, cnt = np.unique(arr[valid], return_counts=True)
            for c, n in zip(ids.tolist(), cnt.tolist()):
                if 0 <= c < nclass:
                    img_count[c] += 1
                    pix_count[c] += int(n)
            total_pix += int(valid.sum())
            n_imgs += 1

    return n_imgs, img_count, pix_count, total_pix


def emit_csv(out: Path, ds: str, class_names: List[str], n_imgs, img_count, pix_count, total_pix):
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_id', 'class_name', 'n_images_with_class',
                    'img_share_pct', 'pixel_count', 'pixel_share_pct'])
        for c, name in enumerate(class_names):
            img_share = img_count[c] / max(1, n_imgs) * 100
            pix_share = pix_count[c] / max(1, total_pix) * 100
            w.writerow([c, name, int(img_count[c]),
                        f'{img_share:.2f}', int(pix_count[c]), f'{pix_share:.4f}'])
        w.writerow([])
        w.writerow(['total', 'all', n_imgs, '100.00', total_pix, '100.0000'])


def emit_markdown(out: Path, ds: str, class_names, n_imgs, img_count, pix_count, total_pix):
    lines = [
        f'### {ds}  ({n_imgs} images, {total_pix:,} valid pixels)',
        '',
        '| ID | Class | #imgs | img share (%) | pixels | pixel share (%) |',
        '|---:|:---|---:|---:|---:|---:|',
    ]
    for c, name in enumerate(class_names):
        img_share = img_count[c] / max(1, n_imgs) * 100
        pix_share = pix_count[c] / max(1, total_pix) * 100
        lines.append(
            f'| {c} | {name} | {int(img_count[c]):,} | {img_share:.2f} | '
            f'{int(pix_count[c]):,} | {pix_share:.4f} |'
        )
    lines.append('')
    out.write_text('\n'.join(lines), encoding='utf-8')


def emit_latex(out: Path, ds: str, class_names, n_imgs, img_count, pix_count, total_pix):
    safe_ds = ds.replace('_', '\\_')
    cap = (f'Per-class statistics on \\textbf{{{safe_ds}}}: '
           f'{n_imgs} images and {total_pix:,} valid pixels in total.')
    lab = f'tab:stats_{ds}'
    rows = []
    for c, name in enumerate(class_names):
        safe_name = name.replace('_', '\\_')
        img_share = img_count[c] / max(1, n_imgs) * 100
        pix_share = pix_count[c] / max(1, total_pix) * 100
        rows.append(f'{c} & {safe_name} & {int(img_count[c]):,} & '
                    f'{img_share:.2f} & {int(pix_count[c]):,} & {pix_share:.4f} \\\\')
    body = '\n'.join(rows)
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{{cap}}}
\\label{{{lab}}}
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\renewcommand{{\\arraystretch}}{{1.10}}
\\begin{{tabular}}{{clrrrr}}
\\toprule
\\textbf{{ID}} & \\textbf{{Class}} & \\textbf{{\\#imgs}} & \\textbf{{img share (\\%)}}
              & \\textbf{{pixels}}        & \\textbf{{pixel share (\\%)}} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    out.write_text(tex, encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', nargs='+', required=True,
                    help='dataset keys (must exist in util/classes.py and DS_DIR)')
    ap.add_argument('--rate', type=str, default='0.10',
                    help='split rate used when locating split dirs')
    ap.add_argument('--data-root',   type=Path, required=True)
    ap.add_argument('--splits-root', type=Path, required=True)
    ap.add_argument('--out-dir',     type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_tex = []
    for ds in args.ds:
        if ds not in CLASSES:
            print(f'[skip] {ds}: missing from util/classes.py'); continue
        if ds not in DS_DIR:
            print(f'[skip] {ds}: missing DS_DIR entry'); continue
        class_names = CLASSES[ds]
        nclass = len(class_names)
        data_root = args.data_root / DS_DIR[ds]
        split_dir = args.splits_root / f'unimatch_splits_{ds}_{args.rate}_seed42'
        split_files = [split_dir / n for n in ('labeled.txt', 'unlabeled.txt',
                                                'val.txt', 'test.txt')]

        print(f'\n[scan] {ds}  data_root={data_root}  splits={split_dir}')
        n_imgs, img_count, pix_count, total_pix = scan_dataset(
            data_root, split_files, nclass)
        print(f'       images={n_imgs}  pixels={total_pix:,}')

        emit_csv     (args.out_dir / f'{ds}_class_stats.csv',
                      ds, class_names, n_imgs, img_count, pix_count, total_pix)
        emit_markdown(args.out_dir / f'{ds}_class_stats.md',
                      ds, class_names, n_imgs, img_count, pix_count, total_pix)
        tex_path = args.out_dir / f'{ds}_class_stats.tex'
        emit_latex   (tex_path,
                      ds, class_names, n_imgs, img_count, pix_count, total_pix)
        all_tex.append(tex_path.read_text(encoding='utf-8'))

    # combined
    if all_tex:
        combined = args.out_dir / 'ALL_class_stats.tex'
        combined.write_text('\n\n'.join(all_tex), encoding='utf-8')
        print(f'\n[combined] wrote {combined}')


if __name__ == '__main__':
    main()
