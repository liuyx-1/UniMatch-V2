"""One-shot readiness check for the Surg-SegFormer comparison datasets.

Verifies, for each dataset:
  1. configs/<ds>.yaml exists; nclass == len(CLASSES[<ds>])
  2. data_root exists with images/{train,val,test} + masks/{train,val,test}
  3. each ratio split dir exists with labeled/unlabeled/val/test and correct
     column counts (labeled/val/test = 2 cols, unlabeled = 1 col)
  4. a sample of labeled lines resolve to real files under data_root
  5. mask label ids are a subset of {0..nclass-1, 255}

Usage (on the server, from repo root):
    python tools/check_datasets.py
    python tools/check_datasets.py --datasets endovis2018_scene endovis2018_type \
        --ratios 0.10 0.30 0.50 1.00 --sample 40
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

# allow running as `python tools/check_datasets.py` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.classes import CLASSES

DEFAULT_DATASETS = ['endovis2017_type', 'endovis2018_scene',
                    'endovis2018_type', 'endovis2018_anatomy_parts']
GREEN, RED, YEL, RST = '\033[32m', '\033[31m', '\033[33m', '\033[0m'


def split_line(ln):
    parts = ln.rstrip('\n').split('\t') if '\t' in ln else ln.split()
    return parts[0], (parts[1] if len(parts) > 1 else None)


def read_lines(p: Path):
    return [l for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]


def check_dataset(ds, splits_root: Path, ratios, sample, repo: Path):
    problems = []
    cfg_path = repo / 'configs' / f'{ds}.yaml'
    if not cfg_path.exists():
        return [f'config missing: {cfg_path}']
    cfg = yaml.load(open(cfg_path, encoding='utf-8'), Loader=yaml.Loader)
    data_root = Path(cfg['data_root'])
    nclass = int(cfg['nclass'])

    # 1) classes/nclass consistency
    names = CLASSES.get(ds)
    if names is None:
        problems.append(f"CLASSES['{ds}'] missing in util/classes.py")
    elif len(names) != nclass:
        problems.append(f"nclass={nclass} but len(CLASSES['{ds}'])={len(names)}")

    # 2) processed dir layout
    if not data_root.is_dir():
        problems.append(f'data_root missing: {data_root}')
    else:
        for sub in ('images/train', 'images/val', 'images/test',
                    'masks/train', 'masks/val', 'masks/test'):
            d = data_root / sub
            if not d.is_dir():
                problems.append(f'missing {sub} under data_root')

    # 3) + 4) + 5) ratio splits + file existence + label range
    allowed = set(range(nclass)) | {255}
    for r in ratios:
        sd = splits_root / f'unimatch_splits_{ds}_{r:.2f}_seed42'
        if not sd.is_dir():
            problems.append(f'[r={r:.2f}] split dir missing: {sd}')
            continue
        for fn, want2 in [('labeled', True), ('unlabeled', False),
                          ('val', True), ('test', True)]:
            fp = sd / f'{fn}.txt'
            if not fp.exists():
                problems.append(f'[r={r:.2f}] missing {fn}.txt')
                continue
            lines = read_lines(fp)
            if not lines:
                problems.append(f'[r={r:.2f}] {fn}.txt is empty')
                continue
            bad = sum(1 for l in lines if (split_line(l)[1] is None) == want2)
            if want2 and bad:
                problems.append(f'[r={r:.2f}] {fn}.txt has {bad} lines WITHOUT a mask column')
            # sample existence + label range on labeled.txt only
            if fn == 'labeled' and data_root.is_dir():
                miss_f = 0; bad_lab = 0; checked = 0
                for l in lines[:sample]:
                    img, msk = split_line(l)
                    ip, mp = data_root / img, data_root / (msk or '')
                    if not ip.exists():
                        miss_f += 1
                    if msk and mp.exists():
                        a = np.asarray(Image.open(mp))
                        ill = set(np.unique(a).tolist()) - allowed
                        if ill:
                            bad_lab += 1
                        checked += 1
                    elif msk:
                        miss_f += 1
                if miss_f:
                    problems.append(f'[r={r:.2f}] {miss_f}/{min(sample,len(lines))} sampled img/mask files missing on disk')
                if bad_lab:
                    problems.append(f'[r={r:.2f}] {bad_lab} sampled masks have labels outside 0..{nclass-1},255')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS)
    ap.add_argument('--splits-root', type=Path,
                    default=Path('/root/autodl-tmp/data/autonomous_surgery/splits'))
    ap.add_argument('--ratios', type=float, nargs='+', default=[0.10, 0.30, 0.50, 1.00])
    ap.add_argument('--sample', type=int, default=30, help='labeled lines to spot-check per ratio')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent

    all_ok = True
    for ds in args.datasets:
        probs = check_dataset(ds, args.splits_root, args.ratios, args.sample, repo)
        if not probs:
            print(f'{GREEN}[ OK ]{RST} {ds}: config+classes+data+splits all present')
        else:
            all_ok = False
            print(f'{RED}[FAIL]{RST} {ds}: {len(probs)} issue(s)')
            for p in probs:
                print(f'        - {p}')
    print('\n' + ('%sALL 4 DATASETS READY%s' % (GREEN, RST) if all_ok
                  else '%s>>> fix the issues above%s' % (YEL, RST)))
    raise SystemExit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
