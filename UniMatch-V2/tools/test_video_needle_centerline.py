"""Run a trained needle (3-class) ckpt on the test set on CPU, extract
geometric centerlines for the needle + thread classes, and emit a slow
side-by-side mp4 for inspection.

Panels per frame (left -> right):
    [ Image  |  Pred overlay  |  Centerline + head/tail + 10 samples ]

The centerline pipeline lives in tools/needle_thread_centerline.py and is
purely geometric (skeleton + spline + fragment merging by tangent /
curvature) -- no extra model forward.

Usage (CPU):
    python tools/test_video_needle_centerline.py \
        --ckpt /root/autodl-tmp/exp/needle/unimatch_v2_<TAG>/best.pth \
        --config configs/needle.yaml \
        --test-id-path /root/autodl-tmp/data/autonomous_surgery/splits/unimatch_splits_needle_0.50_seed42/test.txt \
        --out /root/autodl-tmp/exp/needle/needle_centerline_test.mp4 \
        --fps 2 --alpha 0.45 \
        --needle-id 1 --thread-id 2

CPU forward of DINOv2-base is slow (~0.5-3 s per 490 px frame depending
on cores). Use --max-frames N for quick sanity checks, or
--infer-size 350 to shrink the model input (mask is upsampled back).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.semseg.dpt import DPT  # noqa: E402
from tools.needle_thread_centerline import (  # noqa: E402
    draw_result,
    extract_needle_thread_geometry,
)


# ---------------- color helpers ----------------
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
def preprocess(img: Image.Image, target_side: int):
    # patch-multiple side for DINOv2 (patch_size=14)
    target = max(14, (target_side // 14) * 14)
    img_r = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img_r).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    return torch.from_numpy(norm).permute(2, 0, 1).float(), target


def build_model(cfg):
    cfgs = {
        'small': dict(encoder_size='small', features=64, out_channels=[48, 96, 192, 384]),
        'base':  dict(encoder_size='base',  features=128, out_channels=[96, 192, 384, 768]),
        'large': dict(encoder_size='large', features=256, out_channels=[256, 512, 1024, 1024]),
    }
    bb = cfg['backbone'].split('_')[-1]
    return DPT(**cfgs[bb], nclass=int(cfg['nclass']))


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


def load_ckpt(ck_path: Path, cfg, use_ema: bool = True):
    """Load checkpoint with DDP/torch.compile prefix stripping and auto
    detection of visual_adapter / edge_seg_adapter."""
    model = build_model(cfg)
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    sd = ck.get('model_ema' if use_ema else 'model',
                ck.get('model', ck.get('model_ema', {})))
    sd = _strip_prefix(sd)

    has_va = any(k.startswith('visual_adapter.') or '.visual_adapter.' in k
                 for k in sd)
    if has_va and hasattr(model, 'enable_visual_adapter'):
        down_key = next((k for k in sd
                          if k.endswith('visual_adapter.blocks.0.down.weight')),
                         None)
        red = 8
        if down_key is not None:
            hidden = int(sd[down_key].shape[0])
            if hidden > 0:
                red = max(1, int(model.backbone.embed_dim) // hidden)
        model.enable_visual_adapter(reduction=red, dropout=0.0)

    has_edge = any(k.endswith('edge_seg_adapter.gamma') or
                    '.edge_seg_adapter.' in k for k in sd)
    if has_edge:
        from util.edge_enhance import EdgeSegResidualAdapter
        model.edge_seg_adapter = EdgeSegResidualAdapter(model.backbone.embed_dim)

    msg = model.load_state_dict(sd, strict=False)
    print(f'[load] VA={has_va}  edge={has_edge}  '
          f'missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}')
    return model.eval(), has_edge


@torch.no_grad()
def forward_predict(model, has_edge, x):
    if has_edge:
        from util.edge_enhance import rgb_edge_prior
        edge_in = rgb_edge_prior(x)
        return model(x, edge_prior=edge_in)
    return model(x)


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


# ---------------- centerline-panel rendering ----------------
def _draw_template_fit(canvas, fit, color=(180, 255, 255), thick=1):
    """Render the fitted full circle (faint) so it's visible the arc
    template is being used, and the inferred center point."""
    if fit is None:
        return canvas
    if fit.get('model') == 'circle':
        cx, cy, r = int(round(fit['cx'])), int(round(fit['cy'])), int(round(fit['r']))
        if r > 1 and r < 5000:
            cv2.circle(canvas, (cx, cy), r, color, thick, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), 3, color, -1, cv2.LINE_AA)
    elif fit.get('model') == 'ellipse':
        cv2.ellipse(canvas,
                    (int(round(fit['cx'])), int(round(fit['cy']))),
                    (int(round(fit['a'])), int(round(fit['b']))),
                    float(fit['angle_deg']), 0, 360,
                    color, thick, cv2.LINE_AA)
    return canvas


def render_centerline_panel(img_bgr, pred, res, src_w, src_h):
    """Right panel: dark background + colored mask overlays + fitted
    template (faint) + centerline + head/tail + 10 samples.
    """
    dark = (img_bgr.astype(np.int32) * 0.45).clip(0, 255).astype(np.uint8)
    needle_mask = (pred == res['_needle_id']).astype(np.uint8)
    thread_mask = (pred == res['_thread_id']).astype(np.uint8)
    if needle_mask.any():
        m3 = np.zeros_like(dark)
        m3[needle_mask > 0] = (0, 80, 0)
        dark = cv2.addWeighted(dark, 1.0, m3, 0.6, 0)
    if thread_mask.any():
        m3 = np.zeros_like(dark)
        m3[thread_mask > 0] = (80, 40, 0)
        dark = cv2.addWeighted(dark, 1.0, m3, 0.6, 0)
    # show the FULL fitted circle as a hint that the template was used
    _draw_template_fit(dark, res.get('needle_template_fit'),
                       color=(200, 255, 255), thick=1)
    return draw_result(dark, res,
                        needle_color=(0, 255, 0),
                        thread_color=(255, 160, 0),
                        fragment_color=(120, 120, 120),
                        head_color=(0, 0, 255),
                        tail_color=(0, 255, 255),
                        sample_color=(255, 255, 255),
                        thickness=2,
                        sample_radius=4)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--test-id-path', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fps', type=int, default=2,
                    help='output video fps (low = slow inspection playback)')
    ap.add_argument('--alpha', type=float, default=0.5,
                    help='middle-panel mask overlay transparency')
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--sort-by-path', action='store_true')
    ap.add_argument('--max-frames', type=int, default=None)
    ap.add_argument('--show', action='store_true',
                    help='open a live cv2 window streaming each processed '
                         'frame; press q or ESC to abort early')
    ap.add_argument('--show-scale', type=float, default=0.0,
                    help='downscale factor for the cv2 display window only '
                         '(0 = auto-fit to --max-window-w / --max-window-h)')
    ap.add_argument('--max-window-w', type=int, default=1600,
                    help='max on-screen window width when --show-scale=0')
    ap.add_argument('--max-window-h', type=int, default=900,
                    help='max on-screen window height when --show-scale=0')
    ap.add_argument('--show-layout', type=str, default='auto',
                    choices=('auto', 'row', 'stack'),
                    help='auto picks row (3x1) for wide screens or stack '
                         '(1x3 vertical) when window aspect would force too '
                         'narrow a per-panel view')
    ap.add_argument('--show-wait', type=int, default=1,
                    help='cv2.waitKey ms (1 = stream as fast as possible; '
                         '0 = pause on every frame until a key is pressed)')
    ap.add_argument('--no-video', action='store_true',
                    help='skip writing the mp4 (useful when only previewing)')
    ap.add_argument('--backbone', type=str, default=None,
                    help='override cfg.backbone (e.g. dinov2_small / dinov2_base / '
                         'dinov2_large) when the ckpt was trained with a different '
                         'backbone than the config declares')

    # input size override -- crucial for CPU speed
    ap.add_argument('--infer-size', type=int, default=None,
                    help='if set, run the model at this side length '
                         '(rounded to a multiple of 14); mask is upsampled '
                         'back to source resolution before centerline ops')

    # centerline params (forwarded)
    ap.add_argument('--needle-id', type=int, default=1)
    ap.add_argument('--thread-id', type=int, default=2)
    ap.add_argument('--min-area', type=int, default=30)
    ap.add_argument('--needle-dist-thresh', type=float, default=60.0)
    ap.add_argument('--needle-angle-thresh', type=float, default=50.0)
    ap.add_argument('--thread-dist-thresh', type=float, default=120.0)
    ap.add_argument('--thread-angle-thresh', type=float, default=40.0)
    ap.add_argument('--needle-bridge-radius', type=int, default=0,
                    help='morphological close radius applied to the needle '
                         'mask BEFORE skeletonization; bridges intra-mask '
                         'gaps. Try 3-5 when needle masks are fragmented.')
    ap.add_argument('--thread-bridge-radius', type=int, default=0,
                    help='same as --needle-bridge-radius but for thread. '
                         'Try 5-10 when thread is broken across the frame; '
                         'this fuses fragments at the mask level instead of '
                         'relying on the angle/distance merge heuristic.')
    ap.add_argument('--needle-pca-fallback-area', type=int, default=0,
                    help='if needle component area < N pixels, skip skeleton '
                         'and use PCA major axis (more stable for blob-like '
                         'needles in close-up frames). Try 300-800.')
    ap.add_argument('--thread-pca-fallback-area', type=int, default=0)
    ap.add_argument('--n-sample', type=int, default=10)
    # ---- rigid-arc template correction (for fragmented needle masks) ----
    ap.add_argument('--needle-template', type=str, default='auto',
                    choices=['off', 'auto', 'circle', 'ellipse', 'force'],
                    help='fit a circle/ellipse template to the needle mask '
                         'to recover one coherent arc from fragmented masks. '
                         '"auto" picks template when the arc inlier ratio is '
                         'high; "force" always uses the template if a fit '
                         'exists; "off" disables.')
    ap.add_argument('--needle-template-pad-deg', type=float, default=3.0,
                    help='extend the fitted arc by this many degrees on '
                         'each side (helps when the tip is under-segmented)')
    ap.add_argument('--needle-template-inlier-thresh', type=float, default=3.0,
                    help='RANSAC inlier distance threshold (pixels)')
    ap.add_argument('--needle-template-min-inlier-ratio', type=float,
                    default=0.35,
                    help='reject fit if inlier ratio is below this')

    # CPU performance knobs
    ap.add_argument('--num-threads', type=int, default=None,
                    help='torch.set_num_threads (default: leave as is)')

    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.num_threads is not None:
        torch.set_num_threads(int(args.num_threads))
        print(f'[cpu] torch.num_threads = {torch.get_num_threads()}')

    # ---- config / model ----
    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    if args.backbone:
        print(f'[cfg] overriding backbone: {cfg.get("backbone")} -> {args.backbone}')
        cfg['backbone'] = args.backbone
    nclass = int(cfg['nclass'])
    pal = palette(nclass)
    crop = int(cfg.get('crop_size', 490))
    infer_side = int(args.infer_size) if args.infer_size else crop
    data_root = cfg.get('data_root', '')

    model, has_edge = load_ckpt(args.ckpt, cfg, use_ema=not args.no_ema)

    # ---- read test list ----
    raw = [l.strip() for l in open(args.test_id_path).read().splitlines()
           if l.strip()]
    lines = []
    n_single_col = 0
    for ln in raw:
        parts = ln.split('\t') if '\t' in ln else ln.split()
        if len(parts) < 2:
            n_single_col += 1
            # accept single-column lines (image only, no GT)
            lines.append((parts[0], None))
            continue
        lines.append((parts[0], parts[1]))
    if n_single_col > 0:
        print(f'[info] {n_single_col}/{len(raw)} lines have no mask path; '
              f'GT panel + mIoU will be SKIPPED for those frames.')
    if args.sort_by_path:
        lines.sort(key=lambda p: p[0])
    if args.max_frames:
        lines = lines[:args.max_frames]
    print(f'[plan] frames={len(lines)}  infer_side={infer_side}  fps={args.fps}  '
          f'(playback ~ {len(lines)/args.fps:.1f}s)')

    # ---- output size from first image ----
    if not lines:
        raise SystemExit(
            f'[fatal] no usable lines parsed from {args.test_id_path}\n'
            f'        raw lines read: {len(raw)}\n'
            f'        each line must be "image_path<TAB>mask_path" '
            f'(or whitespace-separated).\n'
            f'        if unlabeled.txt has only 1 column, this tool '
            f'(which needs GT for mIoU) cannot use it — try labeled.txt / '
            f'val.txt / test.txt instead, or use tools/needle_video_stream.py '
            f'for ungraphed streaming.')
    sample_img = Image.open(f"{data_root}/{lines[0][0]}").convert('RGB')
    src_w, src_h = sample_img.size
    sbs_size = (src_w * 3, src_h)
    print(f'[size] panel = {src_w}x{src_h}  sbs = {sbs_size[0]}x{sbs_size[1]}')

    vw = None
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(str(args.out), fourcc, float(args.fps), sbs_size)
        if not vw.isOpened():
            raise RuntimeError(f'VideoWriter failed to open {args.out}')

    # ---- on-screen layout & scaling ----
    # Choose row (3x1) or stack (1x3 vertical) based on the screen aspect we
    # can fill. Then auto-fit so the window stays within (max_w x max_h).
    def _pick_layout_and_scale():
        max_w = int(args.max_window_w)
        max_h = int(args.max_window_h)
        # row layout sbs = (3W, H+top_bar+bottom_bar)
        # stack layout sbs = (W, 3H+top_bar+bottom_bar)
        bar_top, bar_bot = 64, 44  # see HUD drawing below
        row_w, row_h = src_w * 3, src_h + bar_top + bar_bot
        stk_w, stk_h = src_w, src_h * 3 + bar_top + bar_bot
        row_fit = min(max_w / row_w, max_h / row_h)
        stk_fit = min(max_w / stk_w, max_h / stk_h)
        if args.show_layout == 'row':
            layout = 'row'
        elif args.show_layout == 'stack':
            layout = 'stack'
        else:
            # pick the layout that yields the LARGER per-panel pixel area
            # after auto-fit -- favours readability
            area_row = (row_w * row_fit) * (row_h * row_fit) / 3.0
            area_stk = (stk_w * stk_fit) * (stk_h * stk_fit) / 3.0
            layout = 'row' if area_row >= area_stk else 'stack'
        scale = row_fit if layout == 'row' else stk_fit
        if args.show_scale > 0:
            scale = float(args.show_scale)
        scale = min(1.0, max(0.1, scale))
        return layout, scale, bar_top, bar_bot

    layout, disp_scale, bar_top_h, bar_bot_h = _pick_layout_and_scale()

    win_name = 'needle-centerline   q/ESC=quit  p=pause'
    if args.show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        if layout == 'row':
            win_w = int((src_w * 3) * disp_scale)
            win_h = int((src_h + bar_top_h + bar_bot_h) * disp_scale)
        else:
            win_w = int(src_w * disp_scale)
            win_h = int((src_h * 3 + bar_top_h + bar_bot_h) * disp_scale)
        cv2.resizeWindow(win_name, win_w, win_h)
        try:
            cv2.moveWindow(win_name, 40, 40)
        except cv2.error:
            pass
        print(f'[show] layout={layout}  window={win_w}x{win_h}  '
              f'(scale={disp_scale:.2f})')

    miou_acc, pa_acc, t_total, t_geom_total = [], [], 0.0, 0.0
    inter_cls = np.zeros(nclass, dtype=np.int64)
    union_cls = np.zeros(nclass, dtype=np.int64)
    correct_total, valid_total = 0, 0

    centerline_kwargs = dict(
        min_area=args.min_area,
        needle_dist_thresh=args.needle_dist_thresh,
        needle_angle_thresh_deg=args.needle_angle_thresh,
        thread_dist_thresh=args.thread_dist_thresh,
        thread_angle_thresh_deg=args.thread_angle_thresh,
        needle_bridge_radius=args.needle_bridge_radius,
        thread_bridge_radius=args.thread_bridge_radius,
        needle_pca_fallback_area=args.needle_pca_fallback_area,
        thread_pca_fallback_area=args.thread_pca_fallback_area,
        n_sample=args.n_sample,
        needle_template=args.needle_template,
        needle_template_pad_deg=args.needle_template_pad_deg,
        needle_template_inlier_thresh=args.needle_template_inlier_thresh,
        needle_template_min_inlier_ratio=args.needle_template_min_inlier_ratio,
    )

    geom_log = []

    # ---- streaming display state ----
    aborted = False
    t_wall_prev = time.perf_counter()
    fps_inst_ema = None  # smoothed instantaneous fps

    for fi, (img_rel, msk_rel) in enumerate(lines):
        img = Image.open(f"{data_root}/{img_rel}").convert('RGB')
        msk = (Image.open(f"{data_root}/{msk_rel}") if msk_rel is not None
               else None)
        x, _ = preprocess(img, infer_side)

        # ---- forward (CPU) ----
        t0 = time.perf_counter()
        logits = forward_predict(model, has_edge, x.unsqueeze(0))
        t_total += time.perf_counter() - t0

        if logits.shape[-2:] != (src_h, src_w):
            logits = F.interpolate(logits, size=(src_h, src_w),
                                    mode='bilinear', align_corners=False)
        pred = logits.argmax(1)[0].numpy().astype(np.uint8)
        del logits, x

        if msk is not None:
            gt = np.array(msk.resize((src_w, src_h), Image.NEAREST))
            m, p = per_frame_metric(pred, gt, nclass)
            miou_acc.append(m)
            pa_acc.append(p)
            valid = gt != 255
            valid_total += int(valid.sum())
            correct_total += int((pred[valid] == gt[valid]).sum())
            for c in range(nclass):
                gt_c = (gt == c) & valid
                pr_c = (pred == c) & valid
                inter_cls[c] += int((gt_c & pr_c).sum())
                union_cls[c] += int((gt_c | pr_c).sum())
        else:
            gt = None
            m, p = float('nan'), float('nan')
            miou_acc.append(m); pa_acc.append(p)

        # ---- centerline ----
        t1 = time.perf_counter()
        needle_mask = (pred == args.needle_id).astype(np.uint8)
        thread_mask = (pred == args.thread_id).astype(np.uint8)
        res = extract_needle_thread_geometry(needle_mask, thread_mask,
                                              **centerline_kwargs)
        t_geom_total += time.perf_counter() - t1
        res['_needle_id'] = args.needle_id
        res['_thread_id'] = args.thread_id

        # ---- build per-panel images (panel titles burned in caption strip) ----
        img_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        pred_overlay = overlay_bgr(img_bgr, pred, pal, alpha=args.alpha)
        center_panel = render_centerline_panel(img_bgr, pred, res, src_w, src_h)
        panels = [(img_bgr, 'Image'),
                  (pred_overlay, 'Prediction'),
                  (center_panel, 'Centerline')]
        name = Path(img_rel).stem
        jd = res.get('junction_distance')
        jd_str = f'{jd:.1f}' if jd is not None else 'NA'
        has_head = res.get('needle_head') is not None
        has_tail = res.get('needle_tail') is not None
        has_thr = res.get('thread_centerline') is not None
        flag = ''
        if not has_head:
            flag += ' [no needle]'
        elif not has_thr:
            flag += ' [no thread]'
        elif not has_tail:
            flag += ' [no tail]'

        # ---- running clip-level metrics ----
        miou_run = float(np.mean([inter_cls[c] / union_cls[c]
                                  for c in range(nclass)
                                  if union_cls[c] > 0])) \
            if (union_cls > 0).any() else float('nan')
        pa_run = float(correct_total / max(1, valid_total))

        # ---- wall-clock fps (per-frame total: forward + geom + draw) ----
        t_wall_now = time.perf_counter()
        dt = max(1e-6, t_wall_now - t_wall_prev)
        t_wall_prev = t_wall_now
        fps_inst = 1.0 / dt
        # EMA smoothing for readable display
        fps_inst_ema = (fps_inst if fps_inst_ema is None
                        else 0.7 * fps_inst_ema + 0.3 * fps_inst)
        fps_mean = (fi + 1) / max(1e-6, (t_wall_now - (t_wall_prev - dt)
                                          + sum([dt]) * 0)) \
            if False else None  # placeholder removed below
        # cleaner mean fps: total elapsed since first frame
        # (use t_total + t_geom_total as proxy is wrong; use wall instead)
        # We track total wall time via fi+1 / cumulative dt:
        # (computed below from a separate accumulator)

        tf = res.get('needle_template_fit')
        if tf is not None:
            tmpl_str = (f'  arc[{tf["model"][:3]}'
                        f' ir={tf.get("inlier_ratio", 0):.2f}'
                        + (f' r={tf["r"]:.0f}' if tf['model'] == 'circle' else '')
                        + ']')
        else:
            tmpl_str = ''
        hud1 = (f'#{fi+1:04d}/{len(lines)}  {name}  '
                f'frame mIoU={m:.2f} PA={p:.2f}  gap={jd_str}{tmpl_str}{flag}')
        hud2 = (f'running mIoU={miou_run:.3f}  PA={pa_run:.3f}  '
                f'fps={fps_inst_ema:.2f} (inst {fps_inst:.2f})  '
                f'fwd={t_total/(fi+1)*1000:.0f}ms  '
                f'geom={t_geom_total/(fi+1)*1000:.0f}ms')

        # ---- compose layout (row or vertical stack) ----
        def _compose(panels, layout, top_bar_h, bot_bar_h):
            """Return BGR canvas: top HUD bar | panels | bottom HUD bar.

            Each panel gets a small caption strip overlaid in its corner so
            the layout works in both row and stack modes.
            """
            for arr, title in panels:
                # translucent dark caption pill at panel top-left
                tw, th = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.7, 2)[0]
                pad = 8
                x0, y0, x1, y1 = 8, 8, 8 + tw + 2 * pad, 8 + th + 2 * pad
                roi = arr[y0:y1, x0:x1]
                if roi.size:
                    arr[y0:y1, x0:x1] = (roi * 0.35).astype(np.uint8)
                cv2.putText(arr, title, (x0 + pad, y1 - pad),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                            2, cv2.LINE_AA)
            imgs = [p[0] for p in panels]
            if layout == 'row':
                body = np.concatenate(imgs, axis=1)
            else:
                body = np.concatenate(imgs, axis=0)
            W = body.shape[1]
            top = np.zeros((top_bar_h, W, 3), dtype=np.uint8)
            bot = np.zeros((bot_bar_h, W, 3), dtype=np.uint8)
            # subtle vertical gradient bars for polish
            top[:] = (24, 24, 24)
            bot[:] = (24, 24, 24)
            cv2.line(top, (0, top_bar_h - 1), (W, top_bar_h - 1),
                     (80, 80, 80), 1)
            cv2.line(bot, (0, 0), (W, 0), (80, 80, 80), 1)
            return np.concatenate([top, body, bot], axis=0)

        sbs = _compose(panels, layout, bar_top_h, bar_bot_h)

        # ---- HUD text (after composing) ----
        # 1st line top: big running metrics
        big_color = (0, 255, 255)
        cv2.putText(sbs, f'mIoU {miou_run:.3f}  PA {pa_run:.3f}',
                    (16, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, big_color, 2, cv2.LINE_AA)
        # 2nd line top: timing
        cv2.putText(sbs,
                    f'fps {fps_inst_ema:>5.2f} (inst {fps_inst:>5.2f})   '
                    f'fwd {t_total/(fi+1)*1000:>4.0f}ms   '
                    f'geom {t_geom_total/(fi+1)*1000:>4.0f}ms',
                    (16, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
                    cv2.LINE_AA)
        # bottom: per-frame
        cv2.putText(sbs, hud1, (16, sbs.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1,
                    cv2.LINE_AA)

        if vw is not None:
            # write video at native composed resolution (no on-screen scaling)
            if layout == 'row' and sbs.shape[:2][::-1] == sbs_size:
                vw.write(sbs)
            else:
                # composed shape may differ from sbs_size (stack layout or
                # added HUD bars); re-init VW on first frame if needed
                if not hasattr(_compose, '_vw_inited'):
                    vw.release()
                    sbs_size = (sbs.shape[1], sbs.shape[0])
                    vw = cv2.VideoWriter(str(args.out),
                                          cv2.VideoWriter_fourcc(*'mp4v'),
                                          float(args.fps), sbs_size)
                    _compose._vw_inited = True
                vw.write(sbs)

        if args.show:
            disp = sbs
            if disp_scale != 1.0:
                disp = cv2.resize(
                    sbs, None, fx=disp_scale, fy=disp_scale,
                    interpolation=cv2.INTER_AREA)
            cv2.imshow(win_name, disp)
            key = cv2.waitKey(int(args.show_wait)) & 0xFF
            if key in (ord('q'), 27):  # q or ESC
                print(f'[show] user aborted at frame {fi+1}')
                aborted = True
                break
            elif key == ord('p'):  # pause: wait for any key
                cv2.waitKey(0)

        # per-frame geometry record
        geom_log.append({
            'frame': fi,
            'name': name,
            'mIoU': None if np.isnan(m) else float(m),
            'PA': None if np.isnan(p) else float(p),
            'junction_distance': jd,
            'needle_head': (None if not has_head
                             else [float(x) for x in res['needle_head']]),
            'needle_tail': (None if not has_tail
                             else [float(x) for x in res['needle_tail']]),
            'needle_sample_points': (
                None if res.get('needle_sample_points') is None
                else res['needle_sample_points'].tolist()),
            'has_thread': has_thr,
        })

        if (fi + 1) % 25 == 0 or fi == len(lines) - 1:
            print(f'  frame {fi+1}/{len(lines)}  '
                  f'mIoU={m:.3f}  PA={p:.3f}  gap={jd_str}  '
                  f'fwd={t_total/(fi+1)*1000:.0f}ms  '
                  f'geom={t_geom_total/(fi+1)*1000:.0f}ms')

    if vw is not None:
        vw.release()
    if args.show:
        cv2.destroyAllWindows()

    # ---- clip summary ----
    miou_per_cls = [inter_cls[c] / union_cls[c]
                    for c in range(nclass) if union_cls[c] > 0]
    miou_clip = float(np.mean(miou_per_cls)) if miou_per_cls else float('nan')
    pa_clip = float(correct_total / max(1, valid_total))
    fps_real = len(lines) / t_total if t_total > 0 else float('nan')

    n_done = len(geom_log)
    summary = {
        'aborted_early': bool(aborted),
        'n_frames_processed': n_done,
        'n_frames_planned': len(lines),
        'n_frames': len(lines),
        'output_video_fps': args.fps,
        'src_resolution': {'w': src_w, 'h': src_h},
        'infer_side': infer_side,
        'inference_fps_cpu': fps_real,
        'inference_ms_mean': 1000.0 * t_total / max(1, len(lines)),
        'centerline_ms_mean': 1000.0 * t_geom_total / max(1, len(lines)),
        'clip_mIoU': miou_clip,
        'clip_PA': pa_clip,
        'per_frame_mIoU': miou_acc,
        'per_frame_PA': pa_acc,
        'ckpt': str(args.ckpt),
        'config': str(args.config),
        'centerline_params': centerline_kwargs,
        'frames': geom_log,
    }
    rj = args.out.with_suffix('.json')
    with open(rj, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda o: None)

    print()
    print('======= clip summary =======')
    print(f'frames        : {len(lines)}')
    print(f'CPU fwd fps   : {fps_real:.2f} '
          f'(mean {1000.0*t_total/len(lines):.0f} ms / frame)')
    print(f'centerline    : mean {1000.0*t_geom_total/len(lines):.0f} ms / frame')
    print(f'clip mIoU     : {miou_clip:.4f}')
    print(f'clip PA       : {pa_clip:.4f}')
    print(f'video out     : {args.out}')
    print(f'report json   : {rj}')


if __name__ == '__main__':
    main()
