"""Stage 1 visualisation: 4-column grid per random val image.

  col 1: original image
  col 2: predicted affinity heatmap (per-pixel argmax over class affinities, coloured)
  col 3: per-image t-SNE of patch latents (after Stage-1 visual_proj), labelled by GT class
  col 4: ground-truth segmentation mask (coloured + class legend)

Usage:
    python tools/vis_stage1.py \\
        --ckpt     /data/pretrained/siglip_train/affinity_per_ds/endovis2018/affinity_endovis2018.pt \\
        --manifest /data/pretrained/siglip_train/merged_val/manifest_endovis2018_val.jsonl \\
        --out-dir  /data/pretrained/siglip_train/affinity_per_ds/endovis2018/vis_stage1 \\
        --num-images 10 --seed 42
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.affinity_cls import AffinityClassifier


def _palette(n):
    base = plt.get_cmap('tab20', max(20, n))
    return np.array([base(i)[:3] for i in range(n)])


def _colorise(mask: np.ndarray, palette: np.ndarray, ignore=255):
    out = np.zeros((*mask.shape, 3), dtype=np.float32)
    for c in range(len(palette)):
        out[mask == c] = palette[c]
    out[mask == ignore] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',     type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--out-dir',  type=Path, required=True)
    ap.add_argument('--model',    type=str, default='google/siglip2-base-patch16-256')
    ap.add_argument('--num-images', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--tsne-perplexity', type=float, default=30.0)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 1 model
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    class_names = ckpt['class_names']
    orig_class_names = ckpt['orig_class_names']
    orig_to_new = {int(k): int(v) for k, v in ckpt['orig_to_new'].items()}
    C_new = len(class_names)

    m = AffinityClassifier(
        args.model, class_names,
        d_latent=int(ckpt.get('d_latent', 128)),
        topk=int(ckpt.get('topk', 5)),
        vision_backbone=ckpt.get('vision_backbone', 'dinov2'),
        dinov2_input_size=int(ckpt.get('dinov2_input_size') or 518),
        freeze_vision=True, freeze_text=True,
    ).to(args.device).eval()
    m.visual_proj.load_state_dict(ckpt['visual_proj'])
    m.log_tau.data.copy_(ckpt['log_tau'].float())
    m.cls_bias.data.copy_(ckpt['cls_bias'].float())
    m.T_anchor.data.copy_(ckpt['T_anchor'].float())
    if 'dinov2_state_dict' in ckpt:
        m._dinov2.load_state_dict(ckpt['dinov2_state_dict'], strict=False)
    m._anchor_initialised = True
    print(f'[load] {args.ckpt}  classes={C_new}  ({class_names})')

    # Build new-id -> name map; we'll label t-SNE points by NEW id
    new_id_to_name = {v: orig_class_names[k] for k, v in orig_to_new.items()}
    palette_new = _palette(C_new)

    # Pick random val images
    recs = [json.loads(l) for l in args.manifest.read_text(encoding='utf-8').splitlines() if l.strip()]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(recs)
    recs = recs[:args.num_images]
    print(f'[manifest] sampled {len(recs)} images')

    from sklearn.manifold import TSNE

    for i, rec in enumerate(recs):
        try:
            img = Image.open(rec['image']).convert('RGB')
            mask_full = np.array(Image.open(rec['mask']))
            if mask_full.ndim == 3: mask_full = mask_full[..., 0]
        except Exception as e:
            print(f'  [skip] {rec["image"]}: {e}'); continue
        W, H = img.size
        pix = m.preprocess(img).unsqueeze(0).to(args.device)

        with torch.no_grad():
            out = m(pix)
        dense = out['dense']                                       # [1, C, H_p, W_p]
        patches = out['patches'][0].cpu().numpy()                  # [N, 128]
        side = out['patch_grid']

        # col 2: affinity heatmap (argmax over classes per pixel, upsampled)
        argmax_map = dense.argmax(dim=1)[0].cpu().numpy()          # [H_p, W_p] in new-id space
        argmax_up = np.array(Image.fromarray(argmax_map.astype(np.uint8))
                              .resize((W, H), Image.NEAREST))
        argmax_rgb = _colorise(argmax_up, palette_new)

        # col 3: per-image t-SNE of patch latents, coloured by GT class
        mask_p = np.array(Image.fromarray(mask_full).resize((side, side), Image.NEAREST))
        labels = np.full(side * side, -1, dtype=np.int32)
        for orig_id, new_id in orig_to_new.items():
            labels[mask_p.flatten() == int(orig_id)] = int(new_id)

        # col 4: GT mask remapped to new-id colours (bg shown black)
        gt_new = np.full(mask_full.shape, 255, dtype=np.uint8)
        for orig_id, new_id in orig_to_new.items():
            gt_new[mask_full == int(orig_id)] = int(new_id)
        gt_rgb = _colorise(gt_new, palette_new)

        # t-SNE
        try:
            tsne_kwargs = dict(n_components=2, init='pca', random_state=args.seed)
            import inspect
            sig = inspect.signature(TSNE.__init__).parameters
            tsne_kwargs['perplexity'] = float(min(args.tsne_perplexity, max(5, len(patches) // 50)))
            if 'max_iter' in sig: tsne_kwargs['max_iter'] = 1000
            elif 'n_iter' in sig: tsne_kwargs['n_iter'] = 1000
            Z = TSNE(**tsne_kwargs).fit_transform(patches)
        except Exception as e:
            print(f'  [tsne fail] {e}'); continue

        # Plot 4 columns
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(img); axes[0].set_title('Original'); axes[0].axis('off')
        axes[1].imshow(argmax_rgb); axes[1].set_title('Affinity argmax'); axes[1].axis('off')

        # t-SNE scatter, only labelled (GT-known) patches with colour by class
        valid = labels >= 0
        if valid.any():
            for c in sorted(set(labels[valid].tolist())):
                sel = (labels == c)
                axes[2].scatter(Z[sel, 0], Z[sel, 1], c=[palette_new[c]],
                                  s=8, alpha=0.7, label=new_id_to_name.get(c, str(c)),
                                  edgecolors='none')
            axes[2].legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                            fontsize=6, framealpha=0.85)
        # also show ignore-class patches in grey
        if (~valid).any():
            axes[2].scatter(Z[~valid, 0], Z[~valid, 1], c='lightgrey', s=4,
                              alpha=0.3, label='(bg/ignore)')
        axes[2].set_title('Patch latents t-SNE'); axes[2].set_xticks([]); axes[2].set_yticks([])

        axes[3].imshow(gt_rgb); axes[3].set_title('GT mask'); axes[3].axis('off')
        # Add a class colour legend below GT
        present_ids = sorted(set(orig_to_new[c] for c in np.unique(mask_full)
                                  if int(c) in orig_to_new))
        legend_handles = [mpatches.Patch(color=palette_new[c],
                                          label=new_id_to_name.get(c, str(c)))
                          for c in present_ids]
        if legend_handles:
            axes[3].legend(handles=legend_handles, loc='lower center',
                            bbox_to_anchor=(0.5, -0.18),
                            fontsize=6, ncol=min(4, len(legend_handles)), framealpha=0.85)

        fig.suptitle(f'[{rec.get("dataset", "?")}] {Path(rec["image"]).stem}', fontsize=10)
        plt.tight_layout()
        out_p = args.out_dir / f'vis_{i:02d}.png'
        plt.savefig(out_p, dpi=110, bbox_inches='tight'); plt.close()
        print(f'  [{i+1}/{len(recs)}] -> {out_p}')

    print(f'\n[done] saved {args.num_images} visualisations to {args.out_dir}')


if __name__ == '__main__':
    main()
