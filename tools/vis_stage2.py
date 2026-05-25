"""Stage 2 visualisation: 4-column grid per random val image.

  col 1: original image
  col 2: predicted segmentation mask (coloured)
  col 3: per-image t-SNE of DPT pre-classifier features, labelled by GT class
  col 4: ground-truth segmentation mask (coloured + class legend)

The pre-classifier features are captured via a forward hook on the
input of `model.head.scratch.output_conv[-1]` (the final 1x1 Conv2d
classifier). For each pixel we sample its [features] vector for t-SNE,
sub-sampled to keep the embedding tractable.

Usage:
    python tools/vis_stage2.py \\
        --ckpt /data/exp/endovis2018/unimatch_v2_affinity_r0.25_bs2_lr5e-6/best.pth \\
        --config configs/endovis2018.yaml \\
        --val-id-path /data/splits/unimatch_splits_endovis2018_0.25_seed42/val.txt \\
        --out-dir /data/exp/endovis2018/unimatch_v2_affinity_r0.25_bs2_lr5e-6/vis_stage2 \\
        --num-images 10 --seed 42
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.semseg.dpt import DPT
from util.classes import CLASSES


def _palette(n):
    base = plt.get_cmap('tab20', max(20, n))
    return np.array([base(i)[:3] for i in range(n)])


def _colorise(mask: np.ndarray, palette: np.ndarray, ignore=255):
    out = np.zeros((*mask.shape, 3), dtype=np.float32)
    for c in range(len(palette)):
        out[mask == c] = palette[c]
    out[mask == ignore] = 0.0
    return out


def _preprocess(img: Image.Image, crop: int = 518):
    """Resize to a patch-multiple of 14, ImageNet normalise."""
    target = (crop // 14) * 14
    img = img.resize((target, target), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).float(), target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',         type=Path, required=True)
    ap.add_argument('--config',       type=Path, required=True)
    ap.add_argument('--val-id-path',  type=Path, required=True)
    ap.add_argument('--out-dir',      type=Path, required=True)
    ap.add_argument('--num-images',   type=int, default=10)
    ap.add_argument('--seed',         type=int, default=42)
    ap.add_argument('--device',       type=str, default='cuda')
    ap.add_argument('--crop',         type=int, default=518)
    ap.add_argument('--tsne-samples', type=int, default=1500,
                    help='max pixels per image to send into t-SNE')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    nclass = int(cfg['nclass'])
    class_names = CLASSES[cfg['dataset']]
    palette = _palette(nclass)

    # Build DPT model
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'base':  {'encoder_size': 'base',  'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    bb_key = cfg['backbone'].split('_')[-1]
    model = DPT(**{**model_configs[bb_key], 'nclass': nclass}).to(args.device)

    # Load checkpoint
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('model', ck)
    # Strip 'module.' prefix if from DDP
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[load] {args.ckpt}  missing={len(missing)}  unexpected={len(unexpected)}')
    model.eval()

    # Hook: capture pre-classifier features
    pre_cls_feats = []
    def _hook(mod, inp, out):
        # inp is a tuple; inp[0] = [B, F, H, W] features into 1x1 conv
        pre_cls_feats.append(inp[0].detach().cpu())
    handle = model.head.scratch.output_conv[-1].register_forward_hook(_hook)

    # Read val ids
    lines = [l.strip() for l in args.val_id_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(lines)
    lines = lines[:args.num_images]

    data_root = cfg['data_root']
    from sklearn.manifold import TSNE
    import inspect

    for i, line in enumerate(lines):
        parts = line.split('\t') if '\t' in line else line.split(' ')
        img_rel  = parts[0]
        mask_rel = parts[1] if len(parts) > 1 else None
        img_path = os.path.join(data_root, img_rel)
        if mask_rel is None:
            stem = Path(img_rel).stem
            mask_path = os.path.join(data_root, 'masks', 'val', f'{stem}.png')
        else:
            mask_path = os.path.join(data_root, mask_rel)
        try:
            img_pil  = Image.open(img_path).convert('RGB')
            mask     = np.array(Image.open(mask_path))
            if mask.ndim == 3: mask = mask[..., 0]
        except Exception as e:
            print(f'  [skip] {img_path}: {e}'); continue

        W0, H0 = img_pil.size
        pre_cls_feats.clear()
        pix, side = _preprocess(img_pil, crop=args.crop)
        pix = pix.unsqueeze(0).to(args.device)

        with torch.no_grad():
            logits = model(pix)                        # [1, C, H, W]
        pred = logits.argmax(dim=1)[0].cpu().numpy()    # [H, W] @ model resolution
        pred_up = np.array(Image.fromarray(pred.astype(np.uint8)).resize((W0, H0), Image.NEAREST))
        pred_rgb = _colorise(pred_up, palette)

        # GT to RGB
        gt_rgb = _colorise(mask, palette)

        # Pre-classifier features: [1, F, Hf, Wf]
        if not pre_cls_feats:
            print(f'  [warn] no hook output for {img_path}'); continue
        feat = pre_cls_feats[-1][0].numpy()             # [F, Hf, Wf]
        F_dim, Hf, Wf = feat.shape
        # Downsample GT to (Hf, Wf) for per-pixel labels
        gt_small = np.array(Image.fromarray(mask).resize((Wf, Hf), Image.NEAREST))
        labels = gt_small.reshape(-1).astype(np.int64)
        feats_flat = feat.transpose(1, 2, 0).reshape(-1, F_dim)

        # Sub-sample for t-SNE (balanced per class)
        keep_idx = []
        per_cls = max(50, args.tsne_samples // max(1, len(np.unique(labels[labels != 255]))))
        for c in np.unique(labels):
            if c == 255: continue
            ids = np.where(labels == c)[0]
            if len(ids) > per_cls:
                ids = rng.choice(ids, per_cls, replace=False)
            keep_idx.extend(ids.tolist())
        if not keep_idx:
            print(f'  [warn] no valid GT pixels for {img_path}'); continue
        keep_idx = np.array(keep_idx)
        Xs = feats_flat[keep_idx]
        ys = labels[keep_idx]

        try:
            tsne_kwargs = dict(n_components=2, init='pca', random_state=args.seed,
                                perplexity=min(30.0, max(5.0, len(Xs) / 50.0)))
            sig = inspect.signature(TSNE.__init__).parameters
            if 'max_iter' in sig: tsne_kwargs['max_iter'] = 1000
            elif 'n_iter' in sig: tsne_kwargs['n_iter'] = 1000
            Z = TSNE(**tsne_kwargs).fit_transform(Xs)
        except Exception as e:
            print(f'  [tsne fail] {e}'); continue

        # Plot 4 columns
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img_pil); axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].imshow(pred_rgb); axes[1].set_title('Predicted mask'); axes[1].axis('off')

        for c in sorted(set(ys.tolist())):
            sel = (ys == c)
            axes[2].scatter(Z[sel, 0], Z[sel, 1], c=[palette[c]], s=8, alpha=0.7,
                              label=class_names[c] if c < len(class_names) else str(c),
                              edgecolors='none')
        axes[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                        fontsize=6, framealpha=0.85)
        axes[2].set_title('DPT pre-classifier features t-SNE')
        axes[2].set_xticks([]); axes[2].set_yticks([])

        axes[3].imshow(gt_rgb); axes[3].set_title('GT mask'); axes[3].axis('off')
        present = [c for c in sorted(set(mask.flatten().tolist()))
                   if c < nclass and c != 255]
        legend_handles = [mpatches.Patch(color=palette[c], label=class_names[c])
                          for c in present]
        if legend_handles:
            axes[3].legend(handles=legend_handles, loc='lower center',
                            bbox_to_anchor=(0.5, -0.18),
                            fontsize=6, ncol=min(4, len(legend_handles)), framealpha=0.85)

        fig.suptitle(Path(img_rel).stem, fontsize=10)
        plt.tight_layout()
        out_p = args.out_dir / f'vis_{i:02d}.png'
        plt.savefig(out_p, dpi=110, bbox_inches='tight'); plt.close()
        print(f'  [{i+1}/{len(lines)}] -> {out_p}')

    handle.remove()
    print(f'\n[done] saved {args.num_images} visualisations to {args.out_dir}')


if __name__ == '__main__':
    main()
