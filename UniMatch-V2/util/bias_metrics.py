"""Class-bias metrics for long-tail semantic segmentation evaluation.

Indicators (matches the methodology doc):
  1) Head / Body / Tail mIoU + gap         (`hbt_*`, `gap_HT`)
  2) Predicted vs GT class-distribution    (`KL_pred_vs_gt`, `TV_pred_vs_gt`)
  3) Recall coefficient-of-variation       (`recall_CV`)
  4) (pseudo-label keep ratios — see unimatch_v2.py training log, not here)
  5) Expected Calibration Error            (`ECE`, `ECE_head`, `ECE_tail`)

All routines are framework-free numpy so they can be reused from notebooks.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------
def compute_train_pixel_freq(id_path: str, data_root: str,
                              nclass: int, ignore_index: int = 255,
                              cache_path: Optional[str] = None) -> np.ndarray:
    """Pixel frequency vector pi_c over labeled training masks.

    Loads each mask once. Caches to `cache_path` (JSON) if given.
    """
    import json, os
    from PIL import Image

    if cache_path and os.path.isfile(cache_path):
        with open(cache_path, 'r') as f:
            d = json.load(f)
        if d.get('nclass') == nclass and len(d.get('counts', [])) == nclass:
            counts = np.asarray(d['counts'], dtype=np.float64)
            return counts / max(counts.sum(), 1.0)

    counts = np.zeros(nclass, dtype=np.int64)
    with open(id_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    for ln in lines:
        parts = ln.split('\t') if '\t' in ln else ln.split()
        if len(parts) < 2:
            continue
        mask_rel = parts[1]
        mask = np.array(Image.open(os.path.join(data_root, mask_rel)))
        if mask.ndim == 3:
            mask = mask[..., 0]
        valid = (mask != ignore_index) & (mask < nclass)
        if valid.any():
            counts += np.bincount(mask[valid].ravel().astype(np.int64),
                                  minlength=nclass)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({'nclass': nclass, 'counts': counts.tolist()}, f)
    total = counts.sum()
    return counts.astype(np.float64) / max(total, 1.0)


# ----------------------------------------------------------------------
def split_head_body_tail(train_freq: np.ndarray,
                         exclude: Optional[Iterable[int]] = None,
                         n_groups: int = 3) -> Dict[str, List[int]]:
    """Partition class indices by descending train frequency into 3 groups.

    With ``n_groups=3`` returns dict with keys ``head`` / ``body`` / ``tail``,
    sizes ceil(C'/3), ceil(C'/3), remaining; where C' = #non-excluded classes.
    Excluded indices are absent from all three groups.
    """
    assert n_groups == 3, "only 3-way split supported for now"
    exclude = set(exclude or [])
    classes = [c for c in range(len(train_freq)) if c not in exclude]
    classes.sort(key=lambda c: -train_freq[c])
    n = len(classes)
    k = (n + n_groups - 1) // n_groups
    return {
        'head': classes[:k],
        'body': classes[k:2 * k],
        'tail': classes[2 * k:],
    }


# ----------------------------------------------------------------------
def _safe_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) over a discrete distribution, with smoothing on q only."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64) + eps
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


# ----------------------------------------------------------------------
class ECEAccumulator:
    """Online Expected Calibration Error (Naeini et al. 2015) over pixels.

    Bins by predicted top-1 confidence. ``update`` accepts flat numpy arrays
    of shape [N] each: confidence in [0,1], correctness in {0,1}, class id
    of the prediction (for head/tail breakdowns).
    """

    def __init__(self, n_bins: int = 15):
        self.n_bins = int(n_bins)
        self.conf_sum = np.zeros(n_bins, dtype=np.float64)
        self.correct_sum = np.zeros(n_bins, dtype=np.float64)
        self.count = np.zeros(n_bins, dtype=np.int64)
        self.class_conf_sum: Dict[int, np.ndarray] = {}
        self.class_correct_sum: Dict[int, np.ndarray] = {}
        self.class_count: Dict[int, np.ndarray] = {}

    def update(self, confidence: np.ndarray, correct: np.ndarray,
               pred_cls: Optional[np.ndarray] = None) -> None:
        confidence = np.clip(confidence.astype(np.float64), 0.0, 1.0 - 1e-9)
        correct = correct.astype(np.float64)
        bins = np.floor(confidence * self.n_bins).astype(np.int64)
        np.add.at(self.conf_sum, bins, confidence)
        np.add.at(self.correct_sum, bins, correct)
        np.add.at(self.count, bins, 1)
        if pred_cls is not None:
            pred_cls = pred_cls.astype(np.int64)
            for c in np.unique(pred_cls):
                sel = pred_cls == c
                cbins = bins[sel]
                if int(c) not in self.class_count:
                    self.class_conf_sum[int(c)] = np.zeros(self.n_bins, dtype=np.float64)
                    self.class_correct_sum[int(c)] = np.zeros(self.n_bins, dtype=np.float64)
                    self.class_count[int(c)] = np.zeros(self.n_bins, dtype=np.int64)
                np.add.at(self.class_conf_sum[int(c)], cbins, confidence[sel])
                np.add.at(self.class_correct_sum[int(c)], cbins, correct[sel])
                np.add.at(self.class_count[int(c)], cbins, 1)

    def compute(self) -> float:
        N = self.count.sum()
        if N == 0:
            return 0.0
        avg_conf = self.conf_sum / np.maximum(self.count, 1)
        acc = self.correct_sum / np.maximum(self.count, 1)
        weights = self.count / N
        return float(np.sum(weights * np.abs(acc - avg_conf)))

    def compute_for_classes(self, class_ids: Sequence[int]) -> float:
        if not self.class_count:
            return float('nan')
        conf_sum = np.zeros(self.n_bins, dtype=np.float64)
        corr_sum = np.zeros(self.n_bins, dtype=np.float64)
        cnt = np.zeros(self.n_bins, dtype=np.int64)
        for c in class_ids:
            c = int(c)
            if c not in self.class_count:
                continue
            conf_sum += self.class_conf_sum[c]
            corr_sum += self.class_correct_sum[c]
            cnt += self.class_count[c]
        N = cnt.sum()
        if N == 0:
            return float('nan')
        avg_conf = conf_sum / np.maximum(cnt, 1)
        acc = corr_sum / np.maximum(cnt, 1)
        return float(np.sum((cnt / N) * np.abs(acc - avg_conf)))


# ----------------------------------------------------------------------
def compute_bias_metrics(cm: np.ndarray,
                          train_freq: np.ndarray,
                          pred_freq: Optional[np.ndarray] = None,
                          ece: Optional[ECEAccumulator] = None,
                          exclude: Optional[Iterable[int]] = None,
                          n_groups: int = 3,
                          ) -> Dict:
    """Bundle of class-bias diagnostics derived from a confusion matrix.

    Args:
      cm:         [C,C] confusion matrix (rows = GT, cols = pred).
      train_freq: [C] pixel frequency over labeled training masks.
      pred_freq:  [C] pixel frequency of model predictions on the eval set.
                  If None, derived from ``cm.sum(axis=0)``.
      ece:        optional ECEAccumulator with pixels accumulated during eval.
      exclude:    classes excluded from group splits and means.

    Returns:
      dict with the indicators listed in the module docstring plus
      ``hbt`` (the group assignment) and ``per_group`` raw stats.
    """
    exclude = set(exclude or [])
    nclass = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(np.float64)

    iou = tp / np.maximum(tp + fp + fn, 1e-12)
    recall = tp / np.maximum(tp + fn, 1e-12)

    if pred_freq is None:
        total_pred = cm.sum(axis=0).astype(np.float64)
        pred_freq = total_pred / max(total_pred.sum(), 1.0)
    gt_freq = support / max(support.sum(), 1.0)

    hbt = split_head_body_tail(train_freq, exclude=exclude, n_groups=n_groups)

    def _grp_mean(values, idxs):
        idxs = [c for c in idxs if support[c] > 0]
        return float(np.mean(values[idxs])) if idxs else float('nan')

    per_group = {g: {
        'classes': hbt[g],
        'mIoU': _grp_mean(iou, hbt[g]),
        'recall': _grp_mean(recall, hbt[g]),
        'train_freq_sum': float(train_freq[hbt[g]].sum()) if hbt[g] else 0.0,
    } for g in ('head', 'body', 'tail')}

    gap_HT = per_group['head']['mIoU'] - per_group['tail']['mIoU'] \
        if not (np.isnan(per_group['head']['mIoU']) or np.isnan(per_group['tail']['mIoU'])) \
        else float('nan')

    # distribution distance — restrict to non-excluded classes, renormalise
    keep = np.array([c not in exclude for c in range(nclass)])
    p_norm = gt_freq[keep] / max(gt_freq[keep].sum(), 1e-12)
    q_norm = pred_freq[keep] / max(pred_freq[keep].sum(), 1e-12)
    kl_pg = _safe_kl(q_norm, p_norm)
    tv_pg = _tv(p_norm, q_norm)

    rec_valid = recall[[c for c in range(nclass) if c not in exclude and support[c] > 0]]
    if rec_valid.size > 0 and rec_valid.mean() > 0:
        recall_cv = float(rec_valid.std() / rec_valid.mean())
    else:
        recall_cv = float('nan')

    out = {
        'hbt': hbt,
        'per_group': per_group,
        'gap_HT': gap_HT,
        'KL_pred_vs_gt': kl_pg,
        'TV_pred_vs_gt': tv_pg,
        'recall_CV': recall_cv,
        'gt_freq': gt_freq,
        'pred_freq': pred_freq,
        'train_freq': np.asarray(train_freq, dtype=np.float64),
    }
    if ece is not None:
        out['ECE'] = ece.compute()
        out['ECE_head'] = ece.compute_for_classes(hbt['head'])
        out['ECE_body'] = ece.compute_for_classes(hbt['body'])
        out['ECE_tail'] = ece.compute_for_classes(hbt['tail'])
    return out


# ----------------------------------------------------------------------
def format_bias_report(bm: Dict, class_names: Sequence[str]) -> str:
    """Pretty-print block for the console."""
    lines = []
    lines.append('\n' + '=' * 100)
    lines.append('Class-bias diagnostics (head / body / tail by train-pixel frequency)')
    lines.append('-' * 100)
    for g in ('head', 'body', 'tail'):
        pg = bm['per_group'][g]
        names = ', '.join(class_names[c] for c in pg['classes'])
        lines.append(f"  {g:<5}  classes=[{names}]")
        lines.append(f"         train-pixel-share = {pg['train_freq_sum']*100:6.2f}%   "
                     f"mIoU = {pg['mIoU']*100:6.2f}%   recall = {pg['recall']*100:6.2f}%")
    lines.append('-' * 100)
    lines.append(f"  gap_HT  (mIoU_head - mIoU_tail)         = {bm['gap_HT']*100:+.2f}%")
    lines.append(f"  KL(Pred || GT) over class distribution  = {bm['KL_pred_vs_gt']:.4f}")
    lines.append(f"  TV(GT, Pred)   over class distribution  = {bm['TV_pred_vs_gt']:.4f}")
    lines.append(f"  CV(recall) across non-excluded classes  = {bm['recall_CV']:.4f}")
    if 'ECE' in bm:
        lines.append(f"  ECE  (overall)                          = {bm['ECE']:.4f}")
        lines.append(f"  ECE  (head / body / tail)               = "
                     f"{bm['ECE_head']:.4f} / {bm['ECE_body']:.4f} / {bm['ECE_tail']:.4f}")
    lines.append('=' * 100 + '\n')
    return '\n'.join(lines)


def bias_metrics_csv_rows(bm: Dict) -> List[List]:
    """Rows to append to test_metrics.csv summary block."""
    rows = [
        ['bias', 'gap_HT', bm['gap_HT']],
        ['bias', 'KL_pred_vs_gt', bm['KL_pred_vs_gt']],
        ['bias', 'TV_pred_vs_gt', bm['TV_pred_vs_gt']],
        ['bias', 'recall_CV', bm['recall_CV']],
    ]
    for g in ('head', 'body', 'tail'):
        pg = bm['per_group'][g]
        rows.append(['bias', f'mIoU_{g}', pg['mIoU']])
        rows.append(['bias', f'recall_{g}', pg['recall']])
        rows.append(['bias', f'classes_{g}', ';'.join(str(c) for c in pg['classes'])])
        rows.append(['bias', f'train_freq_{g}', pg['train_freq_sum']])
    if 'ECE' in bm:
        rows.append(['bias', 'ECE', bm['ECE']])
        rows.append(['bias', 'ECE_head', bm['ECE_head']])
        rows.append(['bias', 'ECE_body', bm['ECE_body']])
        rows.append(['bias', 'ECE_tail', bm['ECE_tail']])
    return rows
