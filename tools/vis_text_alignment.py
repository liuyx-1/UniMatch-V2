"""Cross-modal alignment visualisation for the text-enhance (改进 1 + 改进 4) module.

Generates 5 figures from a Stage-I affinity checkpoint (cosine OR hyperbolic):

  1. word_predictions.png    — N random val images with top-k predicted class
                                words + ground-truth presence
  2. tsne_global.png         — t-SNE of ALL patch latents (subsampled) +
                                text anchors, coloured by class
  3. poincare_disk.png       — 2-D Poincaré disk projection of patches +
                                anchors (hyperbolic ckpts only); for cosine
                                ckpts this falls back to a cosine UMAP/PCA view
  4. topk_retrieval.png      — for each class, the top-k patches across the
                                val set with highest affinity → ranks the
                                "most class-c looking" patches in the data
  5. affinity_heatmaps.png   — per-class dense affinity heatmap overlaid on
                                random val images (one row per image, one
                                column per class)

Usage:
    python tools/vis_text_alignment.py \\
        --ckpt     /root/autodl-tmp/data/.../affinity_per_ds_hyp/endoscapes_seg50/affinity_endoscapes_seg50.pt \\
        --manifest /root/autodl-tmp/data/.../merged_val/manifest_endoscapes_seg50_val.jsonl \\
        --data-root /root/autodl-tmp/data/autonomous_surgery/public_data/endoscapes_seg50_processed \\
        --out-dir  /root/autodl-tmp/data/.../vis_text_alignment/endoscapes_seg50_hyp \\
        --num-images 12 --topk 5 --max-patches-tsne 10000
"""
from __future__ import annotations
import argparse, json, os, sys, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.affinity_cls import AffinityClassifier
from util.hyperbolic import expmap0, poincare_distance


# ─── small helpers ────────────────────────────────────────────────────
def _palette(n):
    base = plt.get_cmap('tab20', max(20, n))
    return np.array([base(i)[:3] for i in range(n)])


def _load_image_pil(path):
    return Image.open(path).convert('RGB')


def _save(fig, p):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'[save] {p}')


def _resolve(rel, root: Path):
    p = Path(rel); return p if p.is_absolute() else root / p


# ─── model loading ─────────────────────────────────────────────────────
def build_model(ckpt_path: Path, device: str) -> tuple[AffinityClassifier, dict]:
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    class_names = ckpt['class_names']
    metric = ckpt.get('affinity_metric', 'cosine')
    print(f'[ckpt] metric={metric}  d_latent={ckpt.get("d_latent",128)}  '
          f'C\'={len(class_names)}  vision_backbone={ckpt.get("vision_backbone","siglip")}')

    common = dict(
        model_name=ckpt['model'], class_names=class_names,
        d_latent=int(ckpt.get('d_latent', 128)),
        topk=int(ckpt.get('topk', 5)),
        pooling_mode=ckpt.get('pooling_mode', 'topk'),
        threshold_init=float(ckpt.get('threshold_init', 0.3)),
        threshold_learnable=bool(ckpt.get('threshold_learnable', False)),
        threshold_gamma=float(ckpt.get('threshold_gamma', 0.1)),
        min_selected=int(ckpt.get('min_selected', 1)),
        hard_threshold_eval=bool(ckpt.get('hard_threshold_eval', True)),
        vision_backbone=ckpt.get('vision_backbone', 'siglip'),
        dinov2_input_size=int(ckpt.get('dinov2_input_size') or 518),
        freeze_vision=False, freeze_text=False,
    )
    if metric == 'hyperbolic':
        common.update(
            affinity_metric='hyperbolic',
            hyperbolic_dim=int(ckpt.get('hyperbolic_dim', 128)),
            hyperbolic_curvature_init=1.0,
            hyperbolic_temperature_init=0.1,
            hyperbolic_distance_scale=float(ckpt.get('hyperbolic_distance_scale', 1.0)),
            hyperbolic_eps=float(ckpt.get('hyperbolic_eps', 1e-5)),
        )

    m = AffinityClassifier(**common).to(device).eval()
    m.visual_proj.load_state_dict(ckpt['visual_proj'])
    m.text_proj.load_state_dict(ckpt['text_proj'])
    m.T_anchor.copy_(ckpt['T_anchor'])
    m.log_tau.data.copy_(ckpt['log_tau'])
    m.cls_bias.data.copy_(ckpt['cls_bias'])
    if metric == 'hyperbolic':
        if ckpt.get('hyperbolic_v_pre_ln') is not None:
            m.v_pre_ln.load_state_dict(ckpt['hyperbolic_v_pre_ln'])
        if ckpt.get('hyperbolic_t_pre_ln') is not None:
            m.t_pre_ln.load_state_dict(ckpt['hyperbolic_t_pre_ln'])
        if ckpt.get('hyperbolic_rho') is not None:
            m.rho.data.copy_(ckpt['hyperbolic_rho'])
        if ckpt.get('hyperbolic_log_tau_h') is not None:
            m.log_tau_h.data.copy_(ckpt['hyperbolic_log_tau_h'])
    if ckpt.get('threshold') is not None:
        m.pooling.theta.data.copy_(ckpt['threshold'])
    m._anchor_initialised = True
    return m, ckpt


# ─── data iter ────────────────────────────────────────────────────────
def iter_val_records(manifest: Path, data_root: Path, n: int, seed: int):
    recs = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
    rng = random.Random(seed)
    rng.shuffle(recs)
    return recs[:n]


def forward_image(m, pil_img, device):
    with torch.no_grad():
        x = m.preprocess(pil_img).unsqueeze(0).to(device)
        out = m(x)
    return out


# ─── 1. word predictions ──────────────────────────────────────────────
def viz_word_predictions(m, recs, data_root, class_names, out_path,
                         num_images=12, topk=5, device='cuda'):
    cols = 2
    rows = math.ceil(num_images / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*7, rows*4))
    axes = np.atleast_2d(axes).reshape(-1)

    for idx, r in enumerate(recs[:num_images]):
        ax = axes[idx]
        img = _load_image_pil(_resolve(r['img'], data_root))
        out = forward_image(m, img, device)
        probs = torch.sigmoid(out['cls']).cpu().numpy()[0]
        order = np.argsort(-probs)[:topk]
        gt = set(r.get('classes', []))

        ax.imshow(img); ax.axis('off')
        lines = []
        for k, c in enumerate(order):
            mark = '★' if c in gt else ' '
            lines.append(f'{mark} {class_names[c]:<22}  p={probs[c]:.2f}')
        gt_names = ', '.join(class_names[c] for c in sorted(gt)) or '(none)'
        ax.set_title(f"Top-{topk} predicted:\n" + "\n".join(lines) +
                      f"\nGT: {gt_names}", fontsize=8, loc='left',
                      family='monospace')
    for j in range(num_images, len(axes)):
        axes[j].axis('off')
    fig.suptitle('Word-level (image-tag) predictions on val images\n'
                  '(★ = predicted class is in ground truth)', fontsize=12, y=1.005)
    _save(fig, out_path)


# ─── 2. global t-SNE ──────────────────────────────────────────────────
def viz_tsne_global(m, recs, data_root, class_names, out_path,
                    max_patches=10000, perplexity=30, device='cuda'):
    from sklearn.manifold import TSNE
    metric = m.affinity_metric

    pat_lat, pat_label = [], []          # collected patch latents + nearest-class label
    print(f'[tsne] collecting patches from {len(recs)} images …')
    for r in recs:
        out = forward_image(m, _load_image_pil(_resolve(r['img'], data_root)), device)
        # patches is the *post-projection* latent (cosine: L2-normed; hyperbolic: pre-expmap0)
        patches = out['patches'][0].cpu()                        # [N, d]
        # Pseudo-label each patch by its argmax affinity class
        A = out['affinity'][0].cpu()                              # [N, C']
        lab = A.argmax(dim=-1).numpy()
        pat_lat.append(patches.numpy()); pat_label.append(lab)

    P = np.concatenate(pat_lat, 0); L = np.concatenate(pat_label, 0)
    if len(P) > max_patches:
        idx = np.random.RandomState(0).choice(len(P), max_patches, replace=False)
        P, L = P[idx], L[idx]

    # text anchors as extra points
    T = m.T_anchor.detach().cpu().numpy()                          # [C', d]
    XY_in = np.concatenate([P, T], 0)
    print(f'[tsne] running on {len(XY_in)} points (perplexity={perplexity}) …')
    XY = TSNE(n_components=2, perplexity=perplexity, init='pca',
              random_state=0, metric='cosine' if metric=='cosine' else 'euclidean'
              ).fit_transform(XY_in)
    P2 = XY[:len(P)]; T2 = XY[len(P):]

    C = len(class_names); pal = _palette(C)
    fig, ax = plt.subplots(figsize=(8, 7))
    for c in range(C):
        sel = L == c
        ax.scatter(P2[sel,0], P2[sel,1], s=4, alpha=.3, c=[pal[c]], label=class_names[c])
    # anchors with larger star markers
    for c in range(C):
        ax.scatter(T2[c,0], T2[c,1], s=260, marker='*', edgecolor='black',
                   linewidths=1.4, c=[pal[c]])
        ax.annotate(class_names[c], (T2[c,0], T2[c,1]), fontsize=8, ha='left', va='bottom')
    ax.set_title(f't-SNE of patch latents + text anchors  ({metric})')
    ax.legend(loc='lower right', fontsize=7, framealpha=.85, markerscale=2)
    ax.set_xticks([]); ax.set_yticks([])
    _save(fig, out_path)


# ─── 3. Poincaré disk projection (hyperbolic only; PCA fallback for cosine) ─
def viz_poincare_disk(m, recs, data_root, class_names, out_path,
                      max_patches=4000, device='cuda'):
    metric = m.affinity_metric
    pat_h, pat_label = [], []
    print(f'[poincare] collecting patches from {len(recs)} images …')
    for r in recs:
        with torch.no_grad():
            x = m.preprocess(_load_image_pil(_resolve(r['img'], data_root))).unsqueeze(0).to(device)
            tokens = m._encode_patches(x)
            if metric == 'hyperbolic':
                z = m.v_pre_ln(m.visual_proj(tokens))
                p_h = expmap0(z, c=m.get_curvature(), eps=m.hyperbolic_eps)[0]   # [N, d]
                A = m(x)['affinity'][0]
            else:
                p_h = F.normalize(m.visual_proj(tokens), dim=-1)[0]
                A = m(x)['affinity'][0]
        pat_h.append(p_h.cpu().numpy()); pat_label.append(A.argmax(dim=-1).cpu().numpy())

    P = np.concatenate(pat_h, 0); L = np.concatenate(pat_label, 0)
    if len(P) > max_patches:
        idx = np.random.RandomState(0).choice(len(P), max_patches, replace=False)
        P, L = P[idx], L[idx]
    T = m.T_anchor.detach().cpu().numpy()
    X = np.concatenate([P, T], 0)

    # 2-D embedding: PCA → keep top 2 dims as the disk coords (preserves
    # radial structure for typical d≤128 cases). For full geodesic fidelity
    # you'd use a hyperbolic-specific MDS; PCA is the lightweight stand-in.
    from sklearn.decomposition import PCA
    XY = PCA(n_components=2, random_state=0).fit_transform(X)
    # clip to ball of radius 0.95
    norms = np.linalg.norm(XY, axis=1, keepdims=True)
    max_norm = 0.95
    XY = XY / np.maximum(norms, 1e-6) * np.minimum(norms, max_norm)

    P2, T2 = XY[:len(P)], XY[len(P):]
    C = len(class_names); pal = _palette(C)
    fig, ax = plt.subplots(figsize=(8, 8))
    # disk boundary
    theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', lw=.7, alpha=.5)
    for c in range(C):
        sel = L == c
        ax.scatter(P2[sel,0], P2[sel,1], s=4, alpha=.35, c=[pal[c]], label=class_names[c])
    for c in range(C):
        ax.scatter(T2[c,0], T2[c,1], s=300, marker='*', edgecolor='black',
                   linewidths=1.5, c=[pal[c]])
        ax.annotate(class_names[c], (T2[c,0], T2[c,1]), fontsize=8)
    ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    title = ('Poincaré disk projection (hyperbolic)' if metric=='hyperbolic'
             else 'PCA disk (cosine; for reference)')
    ax.set_title(title)
    ax.legend(loc='lower right', fontsize=7, framealpha=.85, markerscale=2)
    _save(fig, out_path)


# ─── 4. Top-k patch retrieval per class ───────────────────────────────
def viz_topk_retrieval(m, recs, data_root, class_names, out_path,
                        k=6, max_imgs_scan=80, device='cuda'):
    """For each class, show the k patches across the val set whose affinity score is highest."""
    C = len(class_names)
    # store (score, img_path, patch_xy, patch_size)
    best = [[] for _ in range(C)]
    print(f'[topk] scanning {min(len(recs), max_imgs_scan)} images …')
    for r in recs[:max_imgs_scan]:
        path = _resolve(r['img'], data_root)
        img = _load_image_pil(path)
        with torch.no_grad():
            x = m.preprocess(img).unsqueeze(0).to(device)
            out = m(x)
            A = out['affinity'][0].cpu().numpy()                  # [N, C']
            grid = int(out['patch_grid'])
        ow, oh = img.size
        ps_x = ow / grid; ps_y = oh / grid
        for c in range(C):
            for i in range(A.shape[0]):
                py, px = divmod(i, grid)
                x0 = int(px * ps_x); y0 = int(py * ps_y)
                x1 = int((px+1) * ps_x); y1 = int((py+1) * ps_y)
                best[c].append((float(A[i,c]), str(path), (x0,y0,x1,y1)))
        # trim per class to keep memory bounded
        if any(len(b) > 4*k*max_imgs_scan for b in best):
            for c in range(C):
                best[c].sort(key=lambda t: -t[0])
                best[c] = best[c][:k]
    # final trim
    for c in range(C):
        best[c].sort(key=lambda t: -t[0])
        best[c] = best[c][:k]

    fig, axes = plt.subplots(C, k, figsize=(k*1.6, C*1.6))
    if C == 1: axes = axes.reshape(1, -1)
    if k == 1: axes = axes.reshape(-1, 1)
    for c in range(C):
        for j in range(k):
            ax = axes[c,j]; ax.axis('off')
            if j >= len(best[c]): continue
            score, p, (x0,y0,x1,y1) = best[c][j]
            crop = _load_image_pil(p).crop((x0,y0,x1,y1))
            ax.imshow(crop)
            if j == 0:
                ax.set_ylabel(class_names[c], rotation=0, ha='right',
                              va='center', fontsize=8)
            ax.set_title(f'{score:.2f}', fontsize=7)
    fig.suptitle(f'Top-{k} patches retrieved by each text anchor', fontsize=11, y=1.005)
    fig.tight_layout()
    _save(fig, out_path)


# ─── 5. Per-class dense affinity heatmaps ─────────────────────────────
def viz_affinity_heatmaps(m, recs, data_root, class_names, out_path,
                           num_images=6, device='cuda'):
    C = len(class_names)
    fig, axes = plt.subplots(num_images, C+1, figsize=((C+1)*1.9, num_images*1.9))
    if num_images == 1: axes = axes.reshape(1, -1)
    for i, r in enumerate(recs[:num_images]):
        img = _load_image_pil(_resolve(r['img'], data_root))
        out = forward_image(m, img, device)
        dense = out['dense'][0].cpu().numpy()                 # [C', H_p, W_p]
        # column 0 = original
        axes[i,0].imshow(img); axes[i,0].axis('off')
        if i == 0: axes[i,0].set_title('image', fontsize=8)
        for c in range(C):
            ax = axes[i, c+1]
            # softmax across classes for color saturation
            d = dense[c]
            d = (d - d.min()) / (d.max() - d.min() + 1e-6)
            ax.imshow(img); ax.imshow(d, alpha=0.55, cmap='magma',
                                       extent=(0, img.size[0], img.size[1], 0))
            ax.axis('off')
            if i == 0: ax.set_title(class_names[c][:14], fontsize=7)
    fig.suptitle('Per-class dense affinity heatmaps  (text anchor → patch similarity)',
                  fontsize=10, y=1.005)
    fig.tight_layout()
    _save(fig, out_path)


# ─── main ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',      type=Path, required=True)
    ap.add_argument('--manifest',  type=Path, required=True,
                    help='Stage-I val manifest, e.g. .../merged_val/manifest_<ds>_val.jsonl')
    ap.add_argument('--data-root', type=Path, required=True,
                    help='dataset root that resolves the manifest "img" field')
    ap.add_argument('--out-dir',   type=Path, required=True)
    ap.add_argument('--num-images',       type=int, default=12)
    ap.add_argument('--topk',             type=int, default=5)
    ap.add_argument('--retrieval-k',      type=int, default=6)
    ap.add_argument('--retrieval-scan',   type=int, default=80)
    ap.add_argument('--max-patches-tsne', type=int, default=10000)
    ap.add_argument('--max-patches-disk', type=int, default=4000)
    ap.add_argument('--seed',  type=int, default=42)
    ap.add_argument('--device', type=str, default='cuda')
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    m, ckpt = build_model(args.ckpt, args.device)
    class_names = ckpt['class_names']

    recs = iter_val_records(args.manifest, args.data_root,
                             n=max(args.num_images, args.retrieval_scan,
                                   args.max_patches_tsne // 100),
                             seed=args.seed)
    print(f'[data] loaded {len(recs)} val records  classes={class_names}')

    viz_word_predictions(m, recs, args.data_root, class_names,
                          args.out_dir/'1_word_predictions.png',
                          num_images=args.num_images, topk=args.topk,
                          device=args.device)

    viz_tsne_global(m, recs, args.data_root, class_names,
                     args.out_dir/'2_tsne_global.png',
                     max_patches=args.max_patches_tsne, device=args.device)

    viz_poincare_disk(m, recs, args.data_root, class_names,
                       args.out_dir/'3_poincare_disk.png',
                       max_patches=args.max_patches_disk, device=args.device)

    viz_topk_retrieval(m, recs, args.data_root, class_names,
                        args.out_dir/'4_topk_retrieval.png',
                        k=args.retrieval_k, max_imgs_scan=args.retrieval_scan,
                        device=args.device)

    viz_affinity_heatmaps(m, recs, args.data_root, class_names,
                           args.out_dir/'5_affinity_heatmaps.png',
                           num_images=min(6, args.num_images), device=args.device)

    print(f'\n[done] all figures in {args.out_dir}')


if __name__ == '__main__':
    main()
