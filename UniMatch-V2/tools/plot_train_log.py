"""Plot losses + mIoU curves from one or more train_log.csv files.

Each ``--run name:path`` is a labeled training run. Output is a multi-panel
PNG with six panels: loss, components, mIoU, mIoU_ema, H/B/T keep ratio,
and tail keep-ratio comparison across runs.

Example:
    python tools/plot_train_log.py \
        --run "v1 base:/data/exp/endo_v1_base/train_log.csv" \
        --run "v2 text:/data/exp/endo_v2_text/train_log.csv" \
        --run "v3 boundary:/data/exp/endo_v3_boundary/train_log.csv" \
        --run "v4 full:/data/exp/endo_v4_full/train_log.csv" \
        --out /data/exp/endoscapes_curves.png
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True,
                    help='name:path/to/train_log.csv (repeatable)')
    p.add_argument('--out', type=str, required=True)
    return p.parse_args()


def load_csv(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for r in rd:
            rows.append({k: float(v) if v.replace('.', '', 1).replace('-', '', 1).isdigit() else v
                          for k, v in r.items()})
    return rows


def _col(rows, key, default=0.0):
    return [r.get(key, default) if r.get(key, default) != '' else default for r in rows]


def main():
    args = parse_args()
    runs = []
    for spec in args.run:
        name, path = spec.split(':', 1)
        runs.append((name, load_csv(path)))

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Panel 1: total loss
    ax = axes[0, 0]
    for name, rows in runs:
        ax.plot(_col(rows, 'epoch'), _col(rows, 'loss_total'), label=name, linewidth=1.5)
    ax.set_xlabel('epoch'); ax.set_ylabel('total loss')
    ax.set_title('Total training loss'); ax.grid(True, alpha=0.3); ax.legend()

    # Panel 2: loss components (for the first run only — separate panel if more)
    ax = axes[0, 1]
    name, rows = runs[0]
    ep = _col(rows, 'epoch')
    for key in ('loss_x', 'loss_s', 'loss_b', 'loss_tan', 'loss_c'):
        ys = [r.get(key, 0.0) for r in rows]
        if any(y > 0 for y in ys):
            ax.plot(ep, ys, label=key, linewidth=1.5)
    ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.set_title(f'Loss components — {name}'); ax.grid(True, alpha=0.3); ax.legend()

    # Panel 3: mIoU + mIoU_ema
    ax = axes[1, 0]
    for name, rows in runs:
        ax.plot(_col(rows, 'epoch'), _col(rows, 'mIoU'), label=f'{name}', linewidth=1.5)
    ax.set_xlabel('epoch'); ax.set_ylabel('mIoU (%)')
    ax.set_title('Validation mIoU (student)'); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1, 1]
    for name, rows in runs:
        ax.plot(_col(rows, 'epoch'), _col(rows, 'mIoU_ema'), label=f'{name}', linewidth=1.5)
    ax.set_xlabel('epoch'); ax.set_ylabel('mIoU (%)')
    ax.set_title('Validation mIoU (EMA teacher)'); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[0, 2]
    name, rows = runs[0]
    ep = _col(rows, 'epoch')
    if rows and 'keep_ratio_head' in rows[0]:
        ax.plot(ep, _col(rows, 'keep_ratio_head'), label='head', linewidth=1.5)
        ax.plot(ep, _col(rows, 'keep_ratio_body'), label='body', linewidth=1.5)
        ax.plot(ep, _col(rows, 'keep_ratio_tail'), label='tail', linewidth=1.5)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('epoch'); ax.set_ylabel('keep ratio')
        ax.set_title(f'Pseudo-label keep ratio — {name}')
        ax.grid(True, alpha=0.3); ax.legend()
    else:
        ax.text(0.5, 0.5, 'no keep_ratio_* columns', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Pseudo-label keep ratio — N/A')

    ax = axes[1, 2]
    any_tail = False
    for name, rows in runs:
        if rows and 'keep_ratio_tail' in rows[0]:
            ax.plot(_col(rows, 'epoch'), _col(rows, 'keep_ratio_tail'),
                    label=name, linewidth=1.5)
            any_tail = True
    if any_tail:
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('epoch'); ax.set_ylabel('tail keep ratio')
        ax.set_title('Tail-class pseudo-label keep ratio')
        ax.grid(True, alpha=0.3); ax.legend()
    else:
        ax.text(0.5, 0.5, 'no keep_ratio_tail in any run', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Tail keep ratio — N/A')

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f'[plot] saved {args.out}')


if __name__ == '__main__':
    main()
