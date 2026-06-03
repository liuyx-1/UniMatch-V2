"""Build, for an ablation list, BOTH:
  (1) <out>/metric_table.png — single PNG table of summary + per-class IoU
  (2) <out>/ablation_panel.png — single PNG grid: rows = N random val images,
      cols = [Image, GT, pred_variant_1, ..., pred_variant_k]

Loading mirrors test.py:
  - strips DDP 'module.' prefix
  - auto-detects + enables visual_adapter
  - auto-detects + builds edge_seg_adapter
  - auto-detects + rebuilds AffinitySideBranch (with edge_text_adapter if present)
  - runs forward via the same affinity-fusion path as test.py

Usage:
    python tools/build_ablation_table_and_panel.py \\
        --run base:$RUN_BASE \\
        --run +text:$RUN_TEXT \\
        --run +bias:$RUN_BIAS \\
        --run +boundary:$RUN_BND \\
        --run +bnd+bias:$RUN_FULL \\
        --config configs/endoscapes_seg50.yaml \\
        --val-id-path $SPLIT/val.txt \\
        --out-dir $EXP/figs \\
        --num-images 6 --seed 42
"""
from __future__ import annotations
import argparse, csv, sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.semseg.dpt import DPT
from util.classes import CLASSES
from util.edge_enhance import EdgeSegResidualAdapter, rgb_edge_prior, edge_to_tokens


# ============================================================
# 1. Metric table — parse test_metrics.csv from each run dir
# ============================================================
SUMMARY_KEYS = [
    ('miou_all',      'mIoU'),
    ('mdice_all',     'mDice'),
    ('mAP_present',   'mAP'),
    ('pa',            'PA'),
    ('fwiou',         'fwIoU'),
]
BIAS_KEYS = [
    ('mIoU_head',  'mIoU_h'),
    ('mIoU_body',  'mIoU_b'),
    ('mIoU_tail',  'mIoU_t'),
    ('gap_HT',     'gap_HT'),
]


def parse_test_metrics(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    per_cls, summary, bias = {}, {}, {}
    with open(path) as f:
        rd = csv.reader(f); header = next(rd, None)
        if not header: return {}
        for row in rd:
            if not row: continue
            tag = row[0].strip()
            if tag == 'summary' and len(row) >= 3:
                try: summary[row[1]] = float(row[2])
                except Exception: pass
            elif tag == 'bias' and len(row) >= 3:
                try: bias[row[1]] = float(row[2])
                except Exception: bias[row[1]] = row[2]
            elif tag.isdigit():
                try: per_cls[row[1]] = float(row[3])
                except Exception: pass
    return {'per_cls': per_cls, 'summary': summary, 'bias': bias}


def build_metric_table(runs: List[Tuple[str, Path]], out_path: Path,
                       class_names: List[str]):
    parsed = []
    fg_classes = [c for c in class_names if not c.lower().startswith('background')]
    for lbl, run_dir in runs:
        m = parse_test_metrics(run_dir / 'test_metrics.csv')
        if not m: print(f'[table] skip empty: {run_dir}'); continue
        parsed.append((lbl, m))
    if not parsed:
        print('[table] nothing to plot'); return

    sum_hdr  = [h[1] for h in SUMMARY_KEYS]
    bias_hdr = [h[1] for h in BIAS_KEYS]
    cls_hdr  = [c[:10] for c in fg_classes]
    columns  = ['Variant'] + sum_hdr + bias_hdr + cls_hdr

    rows = []
    for lbl, m in parsed:
        row = [lbl]
        for k, _ in SUMMARY_KEYS:
            v = m['summary'].get(k);  row.append(f'{v*100:.2f}' if isinstance(v,float) else '—')
        for k, _ in BIAS_KEYS:
            v = m['bias'].get(k);     row.append(f'{v*100:.2f}' if isinstance(v,float) else '—')
        for c in fg_classes:
            v = m['per_cls'].get(c);  row.append(f'{v*100:.2f}' if isinstance(v,float) else '—')
        rows.append(row)

    def col_best(j):
        vals = []
        for r in rows:
            try: vals.append(float(r[j]))
            except: vals.append(-1)
        return int(np.argmax(vals)) if max(vals) > -1 else -1

    nrows, ncols = len(rows), len(columns)
    fig_w = 0.55 * ncols + 1.5
    fig_h = 0.45 * (nrows + 2) + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')

    tbl = ax.table(cellText=rows, colLabels=columns,
                   loc='center', cellLoc='center', colLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)

    for j in range(ncols):
        c = tbl[(0, j)]; c.set_facecolor('#2c3e50'); c.set_text_props(color='white', weight='bold')
    for i in range(1, nrows + 1):
        tbl[(i, 0)].set_text_props(weight='bold')
        if i % 2 == 0:
            for j in range(ncols): tbl[(i, j)].set_facecolor('#f6f8fa')
    for j in range(1, ncols):
        bi = col_best(j)
        if bi >= 0:
            tbl[(bi + 1, j)].set_text_props(weight='bold', color='#b8860b')

    plt.title('Endoscapes-Seg50 · per-variant metrics (%)', fontsize=12, pad=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'[table] saved {out_path} (+.pdf)')


# ============================================================
# 2. Ablation visualization panel — true forward path
# ============================================================
def palette(n):
    cmap = plt.get_cmap('tab20', max(20, n))
    return np.array([cmap(i)[:3] for i in range(n)])

def colorise(mask, pal, ignore=255):
    out = np.zeros((*mask.shape, 3), dtype=np.float32)
    for c in range(len(pal)):
        out[mask == c] = pal[c]
    out[mask == ignore] = 0.0
    return out

def preprocess(img, crop=518):
    target = (crop // 14) * 14
    img_resized = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    return torch.from_numpy(norm).permute(2, 0, 1).float(), arr, target

def build_model(cfg, device):
    cfgs = {
        'small': dict(encoder_size='small', features=64,  out_channels=[48,96,192,384]),
        'base':  dict(encoder_size='base',  features=128, out_channels=[96,192,384,768]),
        'large': dict(encoder_size='large', features=256, out_channels=[256,512,1024,1024]),
    }
    bb = cfg['backbone'].split('_')[-1]
    return DPT(**cfgs[bb], nclass=int(cfg['nclass'])).to(device)


# ---------- ckpt loading (mirrors test.py) ----------
def _strip_prefix(sd: Dict[str, torch.Tensor],
                  prefixes=('module.', '_orig_mod.')) -> Dict[str, torch.Tensor]:
    """Strip BOTH DDP ('module.') and torch.compile ('_orig_mod.') prefixes.
    Iteratively, because torch.compile-wrapped DDP can stack: 'module._orig_mod.foo'."""
    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]; changed = True
        out[nk] = v
    return out


def load_ckpt_into_model(model, ck_path: Path, device: str, use_ema: bool = True):
    """Replicates the loading flow of test.py main() up to model.eval().
    Returns (model, affinity_side or None, has_edge_seg: bool)."""
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    sd = ck.get('model_ema' if use_ema else 'model',
                ck.get('model', ck.get('model_ema', {})))
    sd = _strip_prefix(sd)

    has_va = any(k.startswith('visual_adapter.') or '.visual_adapter.' in k for k in sd)
    has_edge_seg = any(k.endswith('edge_seg_adapter.gamma') or '.edge_seg_adapter.' in k
                       for k in sd)

    if has_va and hasattr(model, 'enable_visual_adapter'):
        # Infer reduction from down-projection shape if available
        va_cfg = ck.get('visual_adapter') or {}
        va_reduction = int(va_cfg.get('reduction', 8))
        va_dropout = float(va_cfg.get('dropout', 0.0))
        down_key = next((k for k in sd if k.endswith('visual_adapter.blocks.0.down.weight')), None)
        if down_key is not None:
            hidden = int(sd[down_key].shape[0])
            if hidden > 0:
                va_reduction = max(1, int(model.backbone.embed_dim) // hidden)
        model.enable_visual_adapter(reduction=va_reduction, dropout=va_dropout)

    if has_edge_seg:
        model.edge_seg_adapter = EdgeSegResidualAdapter(model.backbone.embed_dim)

    msg = model.load_state_dict(sd, strict=False)

    # Allow auxiliary modules to remain unexpected
    aux_prefixes = ('affinity_side', 'boundary_head', 'cls_head')
    unexpected_core = [k for k in msg.unexpected_keys if not k.startswith(aux_prefixes)]
    print(f'    [model] VA={has_va} edge_seg={has_edge_seg}  '
          f'missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} '
          f'(non-aux unexpected={len(unexpected_core)})')
    if msg.missing_keys[:3]:
        print(f'           missing[:3]={msg.missing_keys[:3]}')
    if unexpected_core[:3]:
        print(f'           unexp[:3]  ={unexpected_core[:3]}')

    # ---- affinity_side ----
    affinity_side = None
    if 'affinity_side' in ck:
        try:
            from util.affinity_side import AffinitySideBranch
            affinity_side = AffinitySideBranch(
                ck['affinity_side'],
                n_orig_classes=int(model.head.scratch.output_conv[-1].out_channels),
                class_names=None,           # ckpt already carries class anchors
                dataset='unknown',
            )
            if ck['affinity_side'].get('edge_text_adapter') is not None:
                affinity_side.enable_edge_enhance()
            affinity_side.load_state_dict_from_save(ck['affinity_side'])
            affinity_side = affinity_side.to(device).eval()
            print('    [affinity] side branch reconstructed and loaded')
        except Exception as e:
            print(f'    [affinity] failed: {e}')
            affinity_side = None

    model = model.to(device).eval()
    return model, affinity_side, has_edge_seg


# ---------- forward replicating test.py's affinity-fusion path ----------
@torch.no_grad()
def forward_predict(model, affinity_side, has_edge_seg, x: torch.Tensor,
                    patch_size: int = 14) -> torch.Tensor:
    """Mirrors test.py::collect_predictions for a single image.
    x: [1, 3, H, W] already preprocessed, on device. Returns logits at
    decoder's native output resolution."""
    edge_in = rgb_edge_prior(x) if has_edge_seg else None
    need_patches = affinity_side is not None

    if need_patches:
        # DPT supports return_cls/edge_prior natively; let it handle everything
        logits, _cls_tok, patches = model(x, return_cls=True, edge_prior=edge_in)
        ph = x.shape[-2] // patch_size
        pw = x.shape[-1] // patch_size
        side = affinity_side(
            patches,
            logits.shape[-2], logits.shape[-1],
            patch_hw=(ph, pw),
            edge_prior=edge_in,
        )
        # If affinity uses hyperbolic/cma re-decode with x_aware
        if getattr(affinity_side, 'affinity_metric', '') in ('hyperbolic_pathway', 'cma'):
            x_aware = getattr(affinity_side, 'last_x_aware', None)
            if x_aware is not None:
                if has_edge_seg and hasattr(model, 'edge_seg_adapter'):
                    x_aware, _ = model.edge_seg_adapter(
                        x_aware, edge_to_tokens(edge_in, ph, pw), ph, pw)
                feats, _, _, _ = model.encode(x, return_cls=False)
                feats = list(model.adapt_features(feats, ph, pw))
                feats[-1] = x_aware
                logits = model.decode(feats, ph, pw, comp_drop=False)
        logits = affinity_side.fuse(logits, side)
    else:
        logits = model(x, edge_prior=edge_in)
    return logits


def build_panel(runs: List[Tuple[str, Path]], cfg, val_id_path: Path,
                out_path: Path, num_images: int, seed: int,
                device='cuda', crop=518, use_ema=True):
    nclass = int(cfg['nclass'])
    class_names = CLASSES[cfg['dataset']]
    pal = palette(nclass)
    data_root = cfg.get('data_root', '')

    # Pick random images
    lines = [l.strip().split('\t') for l in open(val_id_path).read().splitlines() if l.strip()]
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(lines), size=min(num_images, len(lines)), replace=False)

    raw01s, gts, tensors, targets = [], [], [], []
    for idx in chosen:
        img_rel, msk_rel = lines[idx][0], lines[idx][1]
        img = Image.open(f"{data_root}/{img_rel}").convert('RGB')
        msk = Image.open(f"{data_root}/{msk_rel}")
        x, raw01, tgt = preprocess(img, crop)
        gt = np.array(msk.resize((tgt, tgt), Image.NEAREST))
        raw01s.append(raw01); gts.append(gt); tensors.append(x); targets.append(tgt)

    preds_per_run = {}
    for lbl, run_dir in runs:
        ck_path = run_dir / 'best.pth'
        if not ck_path.exists():
            print(f'[panel] missing {ck_path}'); continue
        print(f'[panel] >>> {lbl}  ({run_dir.name})')
        model = build_model(cfg, device)
        model, aff, has_edge = load_ckpt_into_model(model, ck_path, device, use_ema=use_ema)
        preds = []
        for x, tgt in zip(tensors, targets):
            logits = forward_predict(model, aff, has_edge, x.unsqueeze(0).to(device))
            if isinstance(logits, (tuple, list)): logits = logits[0]
            if logits.shape[-1] != tgt:
                logits = F.interpolate(logits, size=(tgt, tgt), mode='bilinear', align_corners=False)
            preds.append(logits.argmax(1)[0].cpu().numpy())
        preds_per_run[lbl] = preds
        del model
        if aff is not None: del aff
        if device.startswith('cuda'): torch.cuda.empty_cache()

    # Compose grid
    ordered = [lbl for lbl, _ in runs if lbl in preds_per_run]
    n_cols = 2 + len(ordered)
    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    if n_rows == 1: axes = axes[None, :]
    if n_cols == 1: axes = axes[:, None]

    for row in range(n_rows):
        axes[row, 0].imshow(raw01s[row]);                    axes[row, 0].axis('off')
        axes[row, 1].imshow(colorise(gts[row], pal));         axes[row, 1].axis('off')
        for ci, lbl in enumerate(ordered):
            axes[row, 2 + ci].imshow(colorise(preds_per_run[lbl][row], pal))
            axes[row, 2 + ci].axis('off')
        if row == 0:
            axes[row, 0].set_title('Image',  fontsize=11)
            axes[row, 1].set_title('GT',     fontsize=11)
            for ci, lbl in enumerate(ordered):
                axes[row, 2 + ci].set_title(lbl, fontsize=11)

    fg_idx = [i for i, n in enumerate(class_names) if not n.lower().startswith('background')]
    patches_legend = [mpatches.Patch(color=pal[i], label=class_names[i]) for i in fg_idx]
    fig.legend(handles=patches_legend, loc='lower center',
               ncol=min(len(patches_legend), 7),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.savefig(out_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'[panel] saved {out_path} (+.pdf)')


# ============================================================
# CLI
# ============================================================
def parse_run(s: str) -> Tuple[str, Path]:
    if ':' not in s: raise argparse.ArgumentTypeError(f'expected LABEL:PATH, got {s!r}')
    lbl, path = s.split(':', 1)
    return lbl, Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run',         type=parse_run, action='append', required=True)
    ap.add_argument('--config',      type=Path, required=True)
    ap.add_argument('--val-id-path', type=Path, required=True)
    ap.add_argument('--out-dir',     type=Path, required=True)
    ap.add_argument('--num-images',  type=int, default=6)
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--device',      type=str, default='cuda')
    ap.add_argument('--crop',        type=int, default=518)
    ap.add_argument('--no-ema',      action='store_true',
                    help='use raw model weights instead of model_ema')
    ap.add_argument('--only-table',  action='store_true')
    ap.add_argument('--only-panel',  action='store_true')
    args = ap.parse_args()

    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    class_names = CLASSES[cfg['dataset']]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.only_panel:
        build_metric_table(args.run, args.out_dir / 'metric_table.png', class_names)
    if not args.only_table:
        build_panel(args.run, cfg, args.val_id_path,
                    args.out_dir / 'ablation_panel.png',
                    num_images=args.num_images, seed=args.seed,
                    device=args.device, crop=args.crop, use_ema=not args.no_ema)


if __name__ == '__main__':
    main()
