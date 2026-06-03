"""Random N-image qualitative panel.

Per image: 4 columns
  1. raw image
  2. raw image + GT mask overlay + GT class legend
  3. predicted mask + predicted class legend
  4. t-SNE of the LAST visual feature BEFORE the DPT head
     (post visual-adapter if --visual-adapter is on), coloured by GT class

Usage:
    python tools/vis_pred_random6.py \
        --ckpt $RUN/best.pth \
        --config configs/endoscapes_seg50.yaml \
        --val-id-path $SPLIT/val.txt \
        --out $RUN/vis_random6.png \
        --num-images 6 --seed 42 --use-ema --visual-adapter
"""
from __future__ import annotations
import argparse, sys, inspect
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.semseg.dpt import DPT
from util.classes import CLASSES


# ---------- colour utils ----------
def palette(n):
    cmap = plt.get_cmap('tab20', max(20, n))
    return np.array([cmap(i)[:3] for i in range(n)])


def colorise(mask, pal, ignore=255):
    out = np.zeros((*mask.shape, 3), dtype=np.float32)
    for c in range(len(pal)):
        out[mask == c] = pal[c]
    out[mask == ignore] = 0.0
    return out


def overlay(img01, mask, pal, alpha=0.5, ignore=255):
    color = colorise(mask, pal, ignore=ignore)
    blend = img01 * (1 - alpha) + color * alpha
    blend[mask == ignore] = img01[mask == ignore]
    return np.clip(blend, 0, 1)


# ---------- io ----------
def preprocess(img, crop=518):
    target = (crop // 14) * 14
    img = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    return torch.from_numpy(norm).permute(2, 0, 1).float(), arr, target


# ---------- model ----------
def build_model(cfg, device):
    cfgs = {
        'small': dict(encoder_size='small', features=64,  out_channels=[48, 96, 192, 384]),
        'base':  dict(encoder_size='base',  features=128, out_channels=[96, 192, 384, 768]),
        'large': dict(encoder_size='large', features=256, out_channels=[256, 512, 1024, 1024]),
    }
    bb = cfg['backbone'].split('_')[-1]
    return DPT(**cfgs[bb], nclass=int(cfg['nclass'])).to(device)


def legend_patches(class_names, present_ids, pal):
    return [mpatches.Patch(color=pal[i], label=class_names[i])
            for i in present_ids if 0 <= i < len(class_names)]


# ---------- pre-DPT features ----------
def encode_pre_dpt(model, x):
    """Return (feat_last [N, D], patch_h, patch_w).

    feat_last is the LAST visual feature tensor that is fed into the DPT head,
    AFTER the visual adapter if it is enabled.
    """
    features, _, ph, pw = model.encode(x)            # list of [B, N, D]
    if getattr(model, 'visual_adapter', None) is not None:
        features = model.adapt_features(features, ph, pw)
    feat_last = features[-1]                          # [1, N, D]
    return feat_last[0].detach().cpu().numpy(), ph, pw


# ---------- t-SNE panel ----------
def tsne_panel(ax, feat, gt_full, ph, pw, class_names, pal, nclass,
               samples_per_class=180, max_total=1500, seed=42):
    """feat: (N, D) at patch grid (ph, pw); gt_full: (H, W) full-res GT mask."""
    from sklearn.manifold import TSNE
    gt_pil = Image.fromarray(gt_full.astype(np.uint8))
    gt_patch = np.array(gt_pil.resize((pw, ph), Image.NEAREST)).reshape(-1)
    assert feat.shape[0] == gt_patch.shape[0], (feat.shape, gt_patch.shape)
    rng = np.random.RandomState(seed)

    keep_idx = []
    for c in np.unique(gt_patch):
        if c == 255 or c >= nclass:
            continue
        ids = np.where(gt_patch == c)[0]
        take = min(len(ids), samples_per_class)
        keep_idx.append(rng.choice(ids, take, replace=False))
    if not keep_idx:
        ax.text(0.5, 0.5, '(no valid GT)', ha='center', va='center')
        ax.axis('off')
        return
    keep_idx = np.concatenate(keep_idx)
    if len(keep_idx) > max_total:
        keep_idx = rng.choice(keep_idx, max_total, replace=False)

    X = feat[keep_idx]
    y = gt_patch[keep_idx]

    tsne_kwargs = dict(
        n_components=2, init='pca', random_state=seed,
        perplexity=min(30, max(5, len(X) // 5)),
    )
    sig = inspect.signature(TSNE.__init__).parameters
    if 'max_iter' in sig:
        tsne_kwargs['max_iter'] = 1000
    elif 'n_iter' in sig:
        tsne_kwargs['n_iter'] = 1000
    try:
        Z = TSNE(**tsne_kwargs).fit_transform(X)
    except Exception as e:
        ax.text(0.5, 0.5, f'(tsne fail)\n{e}', ha='center', va='center', fontsize=7)
        ax.axis('off')
        return

    for c in np.unique(y):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=8, c=[pal[int(c)]],
                   label=class_names[int(c)], edgecolors='none', alpha=0.85)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc='lower left', fontsize=6, framealpha=0.85)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',        type=Path, required=True)
    ap.add_argument('--config',      type=Path, required=True)
    ap.add_argument('--val-id-path', type=Path, required=True)
    ap.add_argument('--out',         type=Path, required=True)
    ap.add_argument('--num-images',  type=int, default=6)
    ap.add_argument('--seed',        type=int, default=42)
    ap.add_argument('--crop',        type=int, default=518)
    ap.add_argument('--device',      type=str, default='cuda')
    ap.add_argument('--use-ema',     action='store_true')
    ap.add_argument('--visual-adapter',           action='store_true',
                    help='enable VA before loading the checkpoint so adapter weights load')
    ap.add_argument('--visual-adapter-reduction', type=int,   default=8)
    ap.add_argument('--visual-adapter-dropout',   type=float, default=0.0)
    ap.add_argument('--alpha',                    type=float, default=0.5)
    ap.add_argument('--tsne-samples-per-class',   type=int,   default=180)
    ap.add_argument('--tsne-max-total',           type=int,   default=1500)
    args = ap.parse_args()

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    nclass = int(cfg['nclass'])
    class_names = CLASSES[cfg['dataset']]
    pal = palette(nclass)

    # Build model and optionally enable visual adapter so its weights load.
    model = build_model(cfg, args.device)
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)

    has_va_in_ck = any(
        k.startswith('visual_adapter.') or '.visual_adapter.' in k
        for k in (ck.get('model_ema', ck.get('model', ck)) or {}).keys()
    )
    if args.visual_adapter or has_va_in_ck:
        va_cfg = ck.get('visual_adapter') or {}
        va_reduction = int(va_cfg.get('reduction', args.visual_adapter_reduction))
        va_dropout = float(va_cfg.get('dropout',  args.visual_adapter_dropout))
        model.enable_visual_adapter(reduction=va_reduction, dropout=va_dropout)
        print(f'[visual-adapter] enabled  reduction={va_reduction} dropout={va_dropout}')

    sd = ck.get('model_ema' if args.use_ema else 'model', ck)
    msg = model.load_state_dict(sd, strict=False)
    print(f'[load] missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}')
    model.eval()

    # Choose images.
    lines = [l.strip().split('\t')
             for l in open(args.val_id_path).read().splitlines() if l.strip()]
    rng = np.random.RandomState(args.seed)
    chosen = rng.choice(len(lines), size=min(args.num_images, len(lines)), replace=False)
    chosen = sorted(chosen.tolist())

    n = len(chosen)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[None, :]
    data_root = cfg.get('data_root', '')

    for row, idx in enumerate(chosen):
        parts = lines[idx]
        img_rel, msk_rel = parts[0], parts[1]
        img = Image.open(f"{data_root}/{img_rel}").convert('RGB')
        msk = Image.open(f"{data_root}/{msk_rel}")
        x, raw01, tgt = preprocess(img, args.crop)
        gt = np.array(msk.resize((tgt, tgt), Image.NEAREST))
        x_dev = x.unsqueeze(0).to(args.device)

        with torch.no_grad():
            # Segmentation prediction (standard forward).
            logits = model(x_dev)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.shape[-1] != tgt:
                logits = torch.nn.functional.interpolate(
                    logits, size=(tgt, tgt), mode='bilinear', align_corners=False)
            pred = logits.argmax(1)[0].cpu().numpy()

            # Pre-DPT visual feature for t-SNE.
            feat_last, ph, pw = encode_pre_dpt(model, x_dev)

        gt_ids   = sorted({int(c) for c in np.unique(gt)   if c != 255 and c < nclass})
        pred_ids = sorted({int(c) for c in np.unique(pred) if c < nclass})

        axes[row, 0].imshow(raw01);                                  axes[row, 0].axis('off')
        axes[row, 1].imshow(overlay(raw01, gt, pal, args.alpha));    axes[row, 1].axis('off')
        axes[row, 2].imshow(colorise(pred, pal));                    axes[row, 2].axis('off')
        tsne_panel(axes[row, 3], feat_last, gt, ph, pw,
                   class_names, pal, nclass,
                   samples_per_class=args.tsne_samples_per_class,
                   max_total=args.tsne_max_total, seed=args.seed)

        if row == 0:
            axes[row, 0].set_title('Image',                fontsize=12)
            axes[row, 1].set_title('Image + GT mask',      fontsize=12)
            axes[row, 2].set_title('Prediction',           fontsize=12)
            axes[row, 3].set_title('t-SNE (pre-DPT feat)', fontsize=12)

        axes[row, 1].legend(handles=legend_patches(class_names, gt_ids, pal),
                            loc='lower left', fontsize=7, framealpha=0.85)
        axes[row, 2].legend(handles=legend_patches(class_names, pred_ids, pal),
                            loc='lower left', fontsize=7, framealpha=0.85)

    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=160, bbox_inches='tight')
    print(f'[saved] {args.out}')


if __name__ == '__main__':
    main()
