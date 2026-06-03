"""Run a trained needle (3-class) ckpt on the ENTIRE test set, concatenate
all frames into one slow-playback side-by-side video.

Output: a single mp4 of the form
    [ Image | GT overlay | Pred overlay ]
played at a low fps so every frame is humanly inspectable.

Usage:
    python tools/test_video_needle_all.py \
        --ckpt /root/autodl-tmp/exp/needle/unimatch_v2_base_dinov2_small_r0.50_bs8_lr1e-5_ep80_cr490/best.pth \
        --config configs/needle_small.yaml \
        --test-id-path /root/autodl-tmp/data/autonomous_surgery/splits/unimatch_splits_needle_0.50_seed42/test.txt \
        --out /root/autodl-tmp/exp/needle/needle_test_all_fps2.mp4 \
        --fps 2 --alpha 0.5

Notes:
  • The test.txt entries are processed in the file's natural order, which
    typically groups frames by sequence.  Pass --sort-by-path to force a
    lexicographic sort, which keeps frames within each video contiguous.
  • Frame size = source resolution (no resize); each panel = source W,
    so the side-by-side has width 3 * src_W.
  • Per-frame HUD prints filename + per-frame mIoU + per-frame PA.
"""
from __future__ import annotations

import argparse, json, sys, time
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.semseg.dpt import DPT


# ---------------- colour helpers ----------------
def palette(n):
    cmap = plt.get_cmap('tab20', max(20, n))
    return (np.array([cmap(i)[:3] for i in range(n)]) * 255).astype(np.uint8)


def colorise_rgb(mask, pal, ignore=255):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(len(pal)):
        out[mask == c] = pal[c]
    out[mask == ignore] = 0
    return out


def overlay_bgr(img_bgr, mask, pal, alpha=0.5, ignore=255):
    color_bgr = cv2.cvtColor(colorise_rgb(mask, pal, ignore=ignore),
                              cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(img_bgr, 1 - alpha, color_bgr, alpha, 0)
    out[mask == ignore] = img_bgr[mask == ignore]
    return out


# ---------------- model io ----------------
def preprocess(img: Image.Image, crop: int):
    target = (crop // 14) * 14
    img_r = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img_r).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    return torch.from_numpy(norm).permute(2, 0, 1).float(), target


def build_model(cfg, device='cpu'):
    cfgs = {
        'small': dict(encoder_size='small', features=64,  out_channels=[48, 96, 192, 384]),
        'base':  dict(encoder_size='base',  features=128, out_channels=[96, 192, 384, 768]),
        'large': dict(encoder_size='large', features=256, out_channels=[256, 512, 1024, 1024]),
    }
    bb = cfg['backbone'].split('_')[-1]
    return DPT(**cfgs[bb], nclass=int(cfg['nclass'])).to(device)


def _strip_prefix(sd, prefixes=('module.', '_orig_mod.')):
    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    changed = True
        out[nk] = v
    return out


def load_ckpt(ck_path: Path, cfg, device, use_ema=True):
    """Load checkpoint with DDP + torch.compile prefix stripping
    + automatic visual_adapter / edge_seg_adapter detection."""
    model = build_model(cfg, 'cpu')
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    sd = ck.get('model_ema' if use_ema else 'model',
                ck.get('model', ck.get('model_ema', {})))
    sd = _strip_prefix(sd)

    has_va = any(k.startswith('visual_adapter.') or '.visual_adapter.' in k for k in sd)
    if has_va and hasattr(model, 'enable_visual_adapter'):
        down_key = next((k for k in sd if k.endswith('visual_adapter.blocks.0.down.weight')), None)
        red = 8
        if down_key is not None:
            hidden = int(sd[down_key].shape[0])
            if hidden > 0:
                red = max(1, int(model.backbone.embed_dim) // hidden)
        model.enable_visual_adapter(reduction=red, dropout=0.0)

    has_edge = any(k.endswith('edge_seg_adapter.gamma') or '.edge_seg_adapter.' in k for k in sd)
    if has_edge:
        from util.edge_enhance import EdgeSegResidualAdapter
        model.edge_seg_adapter = EdgeSegResidualAdapter(model.backbone.embed_dim)

    msg = model.load_state_dict(sd, strict=False)
    print(f'[load] VA={has_va}  edge={has_edge}  '
          f'missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}')
    return model.to(device).eval(), has_edge


@torch.no_grad()
def forward_predict(model, has_edge, x):
    if has_edge:
        from util.edge_enhance import rgb_edge_prior
        edge_in = rgb_edge_prior(x)
        logits = model(x, edge_prior=edge_in)
    else:
        logits = model(x)
    return logits


# ---------------- per-frame metric ----------------
def per_frame_metric(pred, gt, nclass, ignore=255):
    valid = gt != ignore
    if not valid.any():
        return float('nan'), float('nan')
    pa = float((pred[valid] == gt[valid]).mean())
    ious = []
    for c in range(nclass):
        gt_c = gt == c
        pr_c = pred == c
        if not gt_c.any():
            continue
        u = ((gt_c | pr_c) & valid).sum()
        i = (gt_c & pr_c & valid).sum()
        ious.append(float(i / u) if u > 0 else 0.0)
    return (float(np.mean(ious)) if ious else float('nan')), pa


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',         type=Path, required=True)
    ap.add_argument('--config',       type=Path, required=True)
    ap.add_argument('--test-id-path', type=Path, required=True)
    ap.add_argument('--out',          type=Path, required=True)
    ap.add_argument('--fps',          type=int,  default=2,
                    help='output video fps (lower = slower playback)')
    ap.add_argument('--alpha',        type=float, default=0.5,
                    help='overlay transparency')
    ap.add_argument('--device',       type=str, default='cuda')
    ap.add_argument('--no-ema',       action='store_true')
    ap.add_argument('--sort-by-path', action='store_true',
                    help='lexicographic sort of split entries (groups by sequence)')
    ap.add_argument('--max-frames',   type=int, default=None,
                    help='cap total number of frames (None = all)')
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- config / model ----
    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    nclass = int(cfg['nclass'])
    pal = palette(nclass)
    crop = int(cfg.get('crop_size', 490))
    data_root = cfg.get('data_root', '')

    model, has_edge = load_ckpt(args.ckpt, cfg, args.device, use_ema=not args.no_ema)

    # ---- read test list (tolerate tab OR whitespace separation) ----
    raw = [l.strip() for l in open(args.test_id_path).read().splitlines() if l.strip()]
    lines = []
    for ln in raw:
        parts = ln.split('\t') if '\t' in ln else ln.split()
        if len(parts) < 2:
            print(f'[warn] malformed split line, skipping: {ln!r}')
            continue
        lines.append((parts[0], parts[1]))
    if args.sort_by_path:
        lines.sort(key=lambda p: p[0])
    if args.max_frames:
        lines = lines[:args.max_frames]
    print(f'[plan] {len(lines)} frames at fps={args.fps}  '
          f'(playback duration ≈ {len(lines)/args.fps:.1f} s)')

    # ---- determine output size from first image ----
    sample_img = Image.open(f"{data_root}/{lines[0][0]}").convert('RGB')
    src_w, src_h = sample_img.size
    sbs_size = (src_w * 3, src_h)
    print(f'[size] per-panel = {src_w}x{src_h}  sbs = {sbs_size[0]}x{sbs_size[1]}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(args.out), fourcc, float(args.fps), sbs_size)
    if not vw.isOpened():
        raise RuntimeError(f'VideoWriter failed to open {args.out}')

    # ---- per-frame loop ----
    miou_acc, pa_acc, t_total = [], [], 0.0
    inter_cls = np.zeros(nclass, dtype=np.int64)
    union_cls = np.zeros(nclass, dtype=np.int64)
    correct_total, valid_total = 0, 0

    for fi, (img_rel, msk_rel) in enumerate(lines):
        img = Image.open(f"{data_root}/{img_rel}").convert('RGB')
        msk = Image.open(f"{data_root}/{msk_rel}")

        x, tgt = preprocess(img, crop)

        # ---- timed inference ----
        if args.device.startswith('cuda'): torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            logits = forward_predict(model, has_edge, x.unsqueeze(0).to(args.device))
        except torch.cuda.OutOfMemoryError:
            print(f'[oom] frame {fi}: falling back to CPU forward')
            torch.cuda.empty_cache()
            model_cpu = model.cpu()
            logits = forward_predict(model_cpu, has_edge, x.unsqueeze(0))
            model.to(args.device)
        if args.device.startswith('cuda'): torch.cuda.synchronize()
        t_total += time.perf_counter() - t0

        if logits.shape[-2:] != (src_h, src_w):
            logits = F.interpolate(logits, size=(src_h, src_w),
                                   mode='bilinear', align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

        # explicit free of GPU intermediate tensors per frame
        del logits, x
        if args.device.startswith('cuda') and (fi + 1) % 10 == 0:
            torch.cuda.empty_cache()
        gt   = np.array(msk.resize((src_w, src_h), Image.NEAREST))

        m, p = per_frame_metric(pred, gt, nclass)
        miou_acc.append(m); pa_acc.append(p)

        # confusion accumulators for clip-level mIoU
        valid = gt != 255
        valid_total   += int(valid.sum())
        correct_total += int((pred[valid] == gt[valid]).sum())
        for c in range(nclass):
            gt_c = (gt == c) & valid
            pr_c = (pred == c) & valid
            inter_cls[c] += int((gt_c & pr_c).sum())
            union_cls[c] += int((gt_c | pr_c).sum())

        # ---- write side-by-side frame ----
        img_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        sbs = np.concatenate([
            img_bgr,
            overlay_bgr(img_bgr, gt,   pal, alpha=args.alpha),
            overlay_bgr(img_bgr, pred, pal, alpha=args.alpha),
        ], axis=1)

        # column titles (top)
        for ci, t in enumerate(('Image', 'GT', 'Prediction')):
            cv2.putText(sbs, t, (10 + ci * src_w, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        # HUD bottom
        name = Path(img_rel).stem
        hud = f'#{fi:04d}  {name}  mIoU={m:.2f}  PA={p:.2f}'
        cv2.putText(sbs, hud, (10, src_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(sbs)

        if (fi + 1) % 50 == 0 or fi == len(lines) - 1:
            print(f'  frame {fi+1}/{len(lines)}  mIoU={m:.3f}  PA={p:.3f}')

    vw.release()

    # ---- clip-level summary ----
    miou_per_cls = [inter_cls[c] / union_cls[c]
                    for c in range(nclass) if union_cls[c] > 0]
    miou_clip = float(np.mean(miou_per_cls)) if miou_per_cls else float('nan')
    pa_clip   = float(correct_total / max(1, valid_total))
    fps_real  = len(lines) / t_total if t_total > 0 else float('nan')

    summary = {
        'n_frames':           len(lines),
        'output_video_fps':   args.fps,
        'src_resolution':     {'w': src_w, 'h': src_h},
        'inference_fps':      fps_real,
        'inference_ms_mean':  1000.0 * t_total / max(1, len(lines)),
        'clip_mIoU':          miou_clip,
        'clip_PA':            pa_clip,
        'per_frame_mIoU':     miou_acc,
        'per_frame_PA':       pa_acc,
        'ckpt':               str(args.ckpt),
        'config':             str(args.config),
    }
    rj = args.out.with_suffix('.json')
    with open(rj, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print('======= clip summary =======')
    print(f'frames        : {len(lines)}')
    print(f'inference fps : {fps_real:.2f}   (mean {1000.0*t_total/len(lines):.1f} ms / frame)')
    print(f'clip mIoU     : {miou_clip:.4f}')
    print(f'clip PA       : {pa_clip:.4f}')
    print(f'video out     : {args.out}')
    print(f'report json   : {rj}')


if __name__ == '__main__':
    main()
