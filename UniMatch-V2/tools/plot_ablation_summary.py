"""Plot ablation summary figures for the surgical segmentation paper.

Two outputs:
  1. <out_dir>/loss_curves.png
       Per-variant loss curves (loss_total / loss_x / loss_s / loss_b / loss_aff)
       read from each variant's train_log.csv.
  2. <out_dir>/iou_boxplot.png
       Per-variant per-class IoU box plot read from each variant's
       test_metrics.csv (the per-class rows, not the summary rows).

Usage:
    python tools/plot_ablation_summary.py \
        --run base:/root/autodl-tmp/exp/endoscapes_seg50/unimatch_v2_base_r0.10_bs16_lr5e-5_cr518_va \
        --run text:/root/autodl-tmp/exp/endoscapes_seg50/unimatch_v2_affinity_min_r0.10_bs16_lr5e-5_cr518_va_joint \
        --run edge:/root/autodl-tmp/exp/endoscapes_seg50/unimatch_v2_base_r0.10_bs16_lr5e-5_ep150_cr518_va_edge \
        --run edge+text:/root/autodl-tmp/exp/endoscapes_seg50/unimatch_v2_affinity_min_r0.10_bs16_lr5e-5_ep150_cr518_va_joint_edge \
        --run bias:/root/autodl-tmp/exp/endoscapes_seg50/unimatch_v2_debias_r0.10_bs16_lr5e-5_ep150_cr518_va \
        --run full:/root/autodl-tmp/exp/endoscapes_seg50/unimatch_v2_full_r0.10_bs16_lr5e-5_ep150_cr518_va_joint_edge_bnd_dbs \
        --out-dir /root/autodl-tmp/exp/endoscapes_seg50/figs
"""
from __future__ import annotations
import argparse, csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------- IO helpers -----------------------------
def parse_train_log(csv_path: Path) -> Dict[str, np.ndarray]:
    """Return {column: 1-D array indexed by epoch}."""
    if not csv_path.exists():
        print(f"[warn] missing {csv_path}")
        return {}
    with open(csv_path) as f:
        rd = csv.DictReader(f)
        rows = [r for r in rd if r and r.get('epoch') not in (None, '')]
    cols = {}
    for k in rows[0].keys():
        try:
            cols[k] = np.asarray([float(r[k]) for r in rows], dtype=np.float32)
        except (ValueError, KeyError):
            pass
    return cols


def parse_test_metrics(csv_path: Path) -> Dict[str, Dict[str, float]]:
    """Return {class_name: {'iou': float, 'dice': float}} from per-class rows."""
    if not csv_path.exists():
        print(f"[warn] missing {csv_path}")
        return {}
    per_cls = {}
    with open(csv_path) as f:
        rd = csv.reader(f)
        header = next(rd, None)
        if not header:
            return {}
        try:
            i_name = header.index('name')
            i_iou  = header.index('iou')
            i_dice = header.index('dice')
        except ValueError:
            return {}
        for row in rd:
            if not row or row[0] in ('summary', 'bias') or not row[0].strip().isdigit():
                continue
            try:
                per_cls[row[i_name]] = {
                    'iou':  float(row[i_iou]),
                    'dice': float(row[i_dice]),
                }
            except (ValueError, IndexError):
                continue
    return per_cls


# ----------------------------- plot 1: loss curves -----------------------------
LOSS_KEYS = ['loss_total', 'loss_x', 'loss_s', 'loss_b', 'loss_aff']

def plot_loss_curves(runs: List[Tuple[str, Path]], out_path: Path):
    keys_present = []
    data = {}                                # data[label][key] = arr
    for label, run_dir in runs:
        cols = parse_train_log(run_dir / 'train_log.csv')
        if not cols:
            continue
        data[label] = cols
        for k in LOSS_KEYS:
            if k in cols and float(np.nanmax(cols[k])) > 0 and k not in keys_present:
                keys_present.append(k)

    if not data:
        print('[loss] nothing to plot'); return

    n = len(keys_present)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharex=True)
    if n == 1: axes = [axes]
    cmap = plt.get_cmap('tab10')

    for ax, key in zip(axes, keys_present):
        for ci, (label, cols) in enumerate(data.items()):
            if key not in cols: continue
            y = cols[key]
            x = cols.get('epoch', np.arange(len(y)))
            ax.plot(x, y, lw=1.4, color=cmap(ci % 10), label=label)
        ax.set_title(key.replace('_', r'\_'), fontsize=11)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.grid(True, alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel('Loss', fontsize=10)
    axes[-1].legend(loc='upper right', fontsize=8, frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f'[loss] saved {out_path} (+.pdf)')
    plt.close(fig)


# ----------------------------- plot 2: IoU + Dice boxplot -----------------------------
def _box_one_ax(ax, series_dict, common_classes, metric_key, ylabel, cmap):
    labels = list(series_dict.keys())
    box_data = []
    for lbl in labels:
        vals = [series_dict[lbl][c][metric_key] for c in common_classes
                if c in series_dict[lbl]]
        box_data.append(vals)

    bp = ax.boxplot(box_data, labels=labels, widths=0.55,
                    patch_artist=True, showmeans=True,
                    medianprops=dict(color='black', lw=1.2),
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=5),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(cmap(i % 10)); patch.set_alpha(0.55); patch.set_edgecolor('black')

    rng = np.random.RandomState(0)
    for i, vals in enumerate(box_data):
        xs = rng.normal(i + 1, 0.06, size=len(vals))
        ax.scatter(xs, vals, s=18, color=cmap(i % 10),
                   edgecolors='black', linewidths=0.4, alpha=0.85, zorder=3)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    ax.tick_params(axis='x', labelsize=10, rotation=15)
    ax.tick_params(axis='y', labelsize=9)


def plot_iou_dice_boxplot(runs: List[Tuple[str, Path]], out_path: Path,
                          exclude_classes=('background',)):
    series_dict = OrderedDict()
    common_classes = None
    for label, run_dir in runs:
        per_cls = parse_test_metrics(run_dir / 'test_metrics.csv')
        if not per_cls:
            continue
        names = [n for n in per_cls if n not in exclude_classes]
        if common_classes is None:
            common_classes = set(names)
        else:
            common_classes &= set(names)
        series_dict[label] = {n: per_cls[n] for n in per_cls if n not in exclude_classes}

    if not series_dict:
        print('[box] nothing to plot'); return
    if not common_classes:
        common_classes = set().union(*(set(s.keys()) for s in series_dict.values()))
    common_classes = sorted(common_classes)

    fig, (ax_iou, ax_dice) = plt.subplots(1, 2, figsize=(2.6 * len(series_dict) + 4.0, 4.6),
                                          sharey=True)
    cmap = plt.get_cmap('tab10')

    _box_one_ax(ax_iou,  series_dict, common_classes, 'iou',  'Per-class IoU',  cmap)
    _box_one_ax(ax_dice, series_dict, common_classes, 'dice', 'Per-class Dice', cmap)

    ax_iou.set_title(f'IoU on {len(common_classes)} foreground classes',  fontsize=11)
    ax_dice.set_title(f'Dice on {len(common_classes)} foreground classes', fontsize=11)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    print(f'[box] saved {out_path} (+.pdf)  | n_classes={len(common_classes)}  '
          f'variants={list(series_dict.keys())}')
    plt.close(fig)


# alias for backward compat
plot_iou_boxplot = plot_iou_dice_boxplot


# ----------------------------- CLI -----------------------------
def parse_run(s: str) -> Tuple[str, Path]:
    if ':' not in s:
        raise argparse.ArgumentTypeError(f'expected LABEL:PATH, got {s!r}')
    label, path = s.split(':', 1)
    return label, Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=parse_run, action='append', required=True,
                    help='LABEL:RUN_DIR (repeat). Each RUN_DIR must contain '
                         'train_log.csv and test_metrics.csv')
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--exclude-classes', nargs='*', default=['background'],
                    help='class names excluded from the IoU box plot')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curves(args.run,    args.out_dir / 'loss_curves.png')
    plot_iou_dice_boxplot(args.run, args.out_dir / 'iou_dice_boxplot.png',
                          exclude_classes=tuple(args.exclude_classes))


if __name__ == '__main__':
    main()
