"""Plot training status (loss + metrics) for the supervised baselines.

Two input modes:

  # 1. from the CSV that train_supervised_basic.py now writes
  python tools/plot_training.py --csv exp/combined_r100_small/train_log.csv

  # 2. from a captured stdout/nohup log (older runs with no CSV).
  #    Parses the '[eval] ep=.. mIoU(pres)=.. mDice(pres)=.. PixAcc=..' lines.
  #    (Per-epoch loss is only available from the CSV, not the stdout log.)
  python tools/plot_training.py --log train.out

Output: a PNG next to the input (or --out PATH).
"""
from __future__ import annotations

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float('nan')


def read_csv(path):
    rows = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                rows.setdefault(k, []).append(_f(v))
    return rows


# matches: [eval] ep=3/80 [val] mIoU(pres)=41.20 mDice(pres)=56.00 PixAcc=94.93 | mIoU(all)=39.1 ...
_EVAL = re.compile(
    r'ep=(\d+)/\d+.*?\[val\]\s*mIoU\(pres\)=([\d.]+)\s*mDice\(pres\)=([\d.]+)\s*PixAcc=([\d.]+)')


def read_log(path):
    rows = {'epoch': [], 'val_mIoU': [], 'val_mDice': [], 'val_PixAcc': []}
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = _EVAL.search(line)
            if m:
                rows['epoch'].append(_f(m.group(1)))
                rows['val_mIoU'].append(_f(m.group(2)))
                rows['val_mDice'].append(_f(m.group(3)))
                rows['val_PixAcc'].append(_f(m.group(4)))
    return rows


def plot(rows, out, title):
    ep = rows.get('epoch', [])
    has_loss = 'train_loss' in rows and any(v == v for v in rows['train_loss'])
    n = 2 if has_loss else 1
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.4))
    axes = axes if n > 1 else [axes]

    k = 0
    if has_loss:
        ax = axes[k]; k += 1
        ax.plot(ep, rows['train_loss'], '-o', color='#b8413b', ms=3, label='train loss')
        ax.set_xlabel('epoch'); ax.set_ylabel('loss'); ax.set_title('Training loss')
        ax.grid(alpha=.3)
        if 'lr' in rows:
            ax2 = ax.twinx()
            ax2.plot(ep, rows['lr'], '--', color='#888', lw=1, label='lr')
            ax2.set_ylabel('lr')
        ax.legend(loc='upper right')

    ax = axes[k]
    for key, c, lab in [('val_mIoU', '#2e7d57', 'val mIoU'),
                        ('val_mDice', '#4a9', 'val mDice'),
                        ('val_PixAcc', '#b9892f', 'val PixAcc'),
                        ('test_mIoU', '#7a6cb0', 'test mIoU')]:
        if key in rows and any(v == v for v in rows[key]):
            ax.plot(ep, rows[key], '-o', color=c, ms=3, label=lab)
    ax.set_xlabel('epoch'); ax.set_ylabel('score (%)'); ax.set_title('Validation / test metrics')
    ax.grid(alpha=.3); ax.legend(loc='lower right')

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    print(f'saved -> {out}  ({len(ep)} eval points)')


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--csv')
    g.add_argument('--log')
    p.add_argument('--out', default=None)
    p.add_argument('--title', default=None)
    a = p.parse_args()

    src = a.csv or a.log
    rows = read_csv(a.csv) if a.csv else read_log(a.log)
    if not rows.get('epoch'):
        raise SystemExit(f'no epoch data parsed from {src}')
    out = a.out or os.path.splitext(src)[0] + '_curve.png'
    plot(rows, out, a.title or os.path.dirname(os.path.abspath(src)).split(os.sep)[-1])


if __name__ == '__main__':
    main()
