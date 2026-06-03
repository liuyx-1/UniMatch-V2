"""Stream a raw video file through the needle/thread segmenter and the
geometric centerline pipeline; display the result live with cv2 and
(optionally) save a side-by-side mp4.

No ground-truth masks needed — this is the inference-only counterpart to
``test_video_needle_centerline.py``.

Usage (CPU, live display only, no mp4 written):

    python tools/needle_video_stream.py \
        --ckpt /root/autodl-tmp/exp/needle/unimatch_v2_<TAG>/best.pth \
        --config configs/needle.yaml \
        --backbone dinov2_small \
        --video /root/autodl-tmp/data/autonomous_surgery/private_data/original/march_whole/1.mp4 \
        --infer-size 350 --num-threads 8 \
        --show --show-scale 0.6 --no-video

GPU streaming with mp4 saved:

    python tools/needle_video_stream.py \
        --ckpt ... --config configs/needle.yaml --backbone dinov2_base \
        --video /path/to/clip.mp4 \
        --device cuda:0 \
        --out /tmp/clip_centerline.mp4 --fps 15 \
        --show --show-scale 0.7

Press q / ESC to abort early. p pauses (any key resumes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import cv2
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


# ============================================================
# checkpoint loading (mirrors test_video_needle_centerline)
# ============================================================
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


def _build_model(cfg):
    cfgs = {
        'small': dict(encoder_size='small', features=64, out_channels=[48, 96, 192, 384]),
        'base':  dict(encoder_size='base',  features=128, out_channels=[96, 192, 384, 768]),
        'large': dict(encoder_size='large', features=256, out_channels=[256, 512, 1024, 1024]),
    }
    bb = cfg['backbone'].split('_')[-1]
    return DPT(**cfgs[bb], nclass=int(cfg['nclass']))


def load_ckpt(ck_path: Path, cfg, use_ema: bool = True):
    model = _build_model(cfg)
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    sd = ck.get('model_ema' if use_ema else 'model',
                ck.get('model', ck.get('model_ema', {})))
    sd = _strip_prefix(sd)

    has_va = any(k.startswith('visual_adapter.') or '.visual_adapter.' in k for k in sd)
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


# ============================================================
# preprocess (cv2 BGR -> normalized tensor)
# ============================================================
def preprocess_frame_bgr(frame_bgr: np.ndarray, target_side: int):
    target = max(14, (target_side // 14) * 14)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (target, target), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    return torch.from_numpy(norm).permute(2, 0, 1).float(), target


@torch.no_grad()
def forward_predict(model, has_edge, x, device, amp_dtype=None):
    x = x.to(device, non_blocking=True)
    ctx = (torch.autocast(device_type=device.type, dtype=amp_dtype)
           if amp_dtype is not None and device.type == 'cuda'
           else torch.cuda.amp.autocast(enabled=False)
           if device.type == 'cuda' else
           torch.cpu.amp.autocast(enabled=False))
    with ctx:
        if has_edge:
            from util.edge_enhance import rgb_edge_prior
            edge_in = rgb_edge_prior(x)
            return model(x, edge_prior=edge_in).float()
        return model(x).float()


# ============================================================
# rendering
# ============================================================
def colorise_pred(pred, n_class, palette):
    h, w = pred.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(1, n_class):  # skip background
        out[pred == c] = palette[c]
    return out


def overlay_bgr(img_bgr, pred, palette, alpha=0.5):
    color_bgr = colorise_pred(pred, len(palette), palette)
    return cv2.addWeighted(img_bgr, 1 - alpha, color_bgr, alpha, 0)


def make_palette(n):
    # BGR palette: background black; needle bright green; thread orange-ish blue
    pal = np.zeros((max(n, 3), 3), dtype=np.uint8)
    pal[0] = (0, 0, 0)        # bg
    pal[1] = (0, 220, 80)     # needle (greenish)
    pal[2] = (255, 130, 0)    # thread (orange/blue mix)
    return pal


def render_centerline_panel(img_bgr, pred, res, needle_id, thread_id):
    dark = (img_bgr.astype(np.int32) * 0.45).clip(0, 255).astype(np.uint8)
    needle_m = (pred == needle_id).astype(np.uint8)
    thread_m = (pred == thread_id).astype(np.uint8)
    if needle_m.any():
        m3 = np.zeros_like(dark); m3[needle_m > 0] = (0, 80, 0)
        dark = cv2.addWeighted(dark, 1.0, m3, 0.6, 0)
    if thread_m.any():
        m3 = np.zeros_like(dark); m3[thread_m > 0] = (80, 40, 0)
        dark = cv2.addWeighted(dark, 1.0, m3, 0.6, 0)
    return draw_result(
        dark, res,
        needle_color=(0, 255, 0),
        thread_color=(255, 160, 0),
        fragment_color=(120, 120, 120),
        head_color=(0, 0, 255),
        tail_color=(0, 255, 255),
        sample_color=(255, 255, 255),
        thickness=2, sample_radius=4)


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--backbone', type=str, default=None,
                    help='override cfg.backbone (e.g. dinov2_small)')
    ap.add_argument('--video', type=Path, required=True,
                    help='path to raw mp4/mov/avi to stream')
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--no-ema', action='store_true')

    # output
    ap.add_argument('--out', type=Path, default=None,
                    help='optional mp4 output path; ignored if --no-video')
    ap.add_argument('--fps', type=float, default=None,
                    help='output mp4 fps; default = input video fps')
    ap.add_argument('--no-video', action='store_true',
                    help='skip writing mp4 (preview-only)')

    # streaming control
    ap.add_argument('--show', action='store_true',
                    help='open live cv2 preview window')
    ap.add_argument('--show-scale', type=float, default=1.0)
    ap.add_argument('--show-wait', type=int, default=1)
    ap.add_argument('--max-frames', type=int, default=None,
                    help='stop after N processed frames')
    ap.add_argument('--dump-every', type=int, default=0,
                    help='if >0, save every Nth side-by-side frame as PNG '
                         'next to --out (preview-while-running on headless '
                         'servers without cv2.imshow)')
    ap.add_argument('--mjpeg-port', type=int, default=0,
                    help='if >0, start a local MJPEG-over-HTTP server on '
                         'this port; open http://<host>:<port>/ in any '
                         'browser to watch a live feed. Works on headless '
                         'servers (autodl, etc) without X11.')
    ap.add_argument('--mjpeg-scale', type=float, default=0.5,
                    help='downscale factor for the MJPEG frame (server-side); '
                         'lower = faster delivery over slow networks')
    ap.add_argument('--mjpeg-quality', type=int, default=70,
                    help='JPEG quality for the MJPEG stream (1-100)')
    ap.add_argument('--frame-stride', type=int, default=1,
                    help='process every N-th frame; intermediate frames are '
                         'dropped (useful for fast preview on CPU)')

    # inference knobs
    ap.add_argument('--infer-size', type=int, default=350,
                    help='input side length passed to the model; '
                         'mask is upsampled back to source resolution.')
    ap.add_argument('--alpha', type=float, default=0.45,
                    help='middle-panel mask overlay transparency')
    ap.add_argument('--num-threads', type=int, default=None)
    # speed knobs (GPU)
    ap.add_argument('--amp', type=str, default='off',
                    choices=['off', 'fp16', 'bf16'],
                    help='autocast precision for GPU forward; fp16 ≈ 2x speedup '
                         'on Ampere+ (RTX 30/40, A100, etc), bf16 safer on H100/A100')
    ap.add_argument('--compile', dest='compile_mode', type=str, default='off',
                    choices=['off', 'default', 'reduce-overhead', 'max-autotune'],
                    help='torch.compile mode; "reduce-overhead" is safest. '
                         'First few frames will be slow (warm-up)')

    # centerline knobs
    ap.add_argument('--needle-id', type=int, default=1)
    ap.add_argument('--thread-id', type=int, default=2)
    ap.add_argument('--min-area', type=int, default=30)
    ap.add_argument('--needle-dist-thresh', type=float, default=60.0)
    ap.add_argument('--needle-angle-thresh', type=float, default=50.0)
    ap.add_argument('--thread-dist-thresh', type=float, default=120.0)
    ap.add_argument('--thread-angle-thresh', type=float, default=40.0)
    ap.add_argument('--thread-bridge-radius', type=int, default=0)
    ap.add_argument('--needle-bridge-radius', type=int, default=0)
    ap.add_argument('--thread-pca-fallback-area', type=int, default=0)
    ap.add_argument('--needle-pca-fallback-area', type=int, default=0)
    # false-positive filters: drop instrument/blob mis-segmentations on needle class
    ap.add_argument('--needle-max-area', type=int, default=0,
                    help='drop needle CCs larger than N pixels (instrument body filter); '
                         'typical real needle 200-3000 px at 1080p; try 4000-8000')
    ap.add_argument('--needle-min-aspect', type=float, default=0.0,
                    help='drop needle CCs with PCA major/minor below this; '
                         'real needles >4; try 2.5-3.0 to drop blobs')
    ap.add_argument('--needle-must-be-near-thread', action='store_true',
                    help='when thread is present, reject needle CCs whose nearest '
                         'point is > --needle-near-thread-dist pixels from any '
                         'thread polyline point')
    ap.add_argument('--needle-near-thread-dist', type=float, default=80.0)
    ap.add_argument('--thread-max-area', type=int, default=0)
    ap.add_argument('--thread-min-aspect', type=float, default=0.0)
    # ---- rigid arc template (needle-only) ----
    ap.add_argument('--needle-template', type=str, default='auto',
                    choices=['off', 'auto', 'circle', 'ellipse', 'force'],
                    help='rigid-arc template fitting mode. "force" replaces '
                         'the spline centerline with the arc fit, ignoring '
                         'fragments — strongest constraint. "auto" uses '
                         'whichever is better.')
    ap.add_argument('--needle-template-inlier-thresh', type=float, default=3.0,
                    help='RANSAC pixel threshold; lower = stricter fit')
    ap.add_argument('--needle-template-min-inlier-ratio', type=float, default=0.35,
                    help='minimum fraction of needle pixels that must fit '
                         'the arc; raise to 0.55-0.7 for stricter rigid prior')
    ap.add_argument('--needle-template-radius-min', type=float, default=0.0,
                    help='reject fits with radius < this (px); typical '
                         'surgical needle radius 5-50 px depending on zoom')
    ap.add_argument('--needle-template-radius-max', type=float, default=0.0,
                    help='reject fits with radius > this (px); blocks large '
                         'instrument silhouettes from being mistaken for arcs')
    ap.add_argument('--needle-template-arc-min-deg', type=float, default=0.0,
                    help='reject fits covering less than this arc span; real '
                         'needles cover 60-180 deg, smaller fits are usually '
                         'spurious 3-point fits on noise')
    ap.add_argument('--needle-template-arc-max-deg', type=float, default=0.0,
                    help='reject fits exceeding this arc; >270 deg ≈ full '
                         'circle (clearly not a needle)')
    ap.add_argument('--n-sample', type=int, default=10)

    args = ap.parse_args()

    if args.num_threads:
        torch.set_num_threads(int(args.num_threads))
        print(f'[cpu] torch.num_threads = {torch.get_num_threads()}')

    # ---- config / model ----
    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    if args.backbone:
        print(f'[cfg] override backbone: {cfg.get("backbone")} -> {args.backbone}')
        cfg['backbone'] = args.backbone
    nclass = int(cfg['nclass'])
    palette = make_palette(nclass)

    device = torch.device(args.device)
    model, has_edge = load_ckpt(args.ckpt, cfg, use_ema=not args.no_ema)
    model = model.to(device).eval()

    # ── speed compile / amp ──────────────────────────────────────────────
    amp_dtype = None
    if args.amp != 'off' and device.type == 'cuda':
        amp_dtype = torch.float16 if args.amp == 'fp16' else torch.bfloat16
        print(f'[amp] autocast={args.amp}')
    if args.compile_mode != 'off':
        try:
            model = torch.compile(model, mode=args.compile_mode)
            print(f'[compile] mode={args.compile_mode}  (first frames will be slow)')
        except Exception as e:
            print(f'[compile] failed: {e}; running uncompiled')

    # ---- video reader ----
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f'cannot open video: {args.video}')
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    total_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = float(args.fps) if args.fps else max(1.0, src_fps / max(1, args.frame_stride))
    print(f'[video] {args.video.name}  {src_w}x{src_h}  src_fps={src_fps:.1f}  '
          f'total={total_in}  stride={args.frame_stride}  out_fps={out_fps:.1f}')

    # ── new layout: row1 = [Image | Prediction], row2 = [Centerline 2x wide] ──
    sbs_w, sbs_h = src_w * 2, src_h * 2

    vw = None
    if not args.no_video and args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(str(args.out), fourcc, out_fps, (sbs_w, sbs_h))
        if not vw.isOpened():
            raise RuntimeError(f'VideoWriter failed to open {args.out}')

    win_name = 'needle stream (q/ESC abort, p pause)'
    if args.show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, int(sbs_w * args.show_scale),
                         int(sbs_h * args.show_scale))

    centerline_kwargs = dict(
        min_area=args.min_area,
        needle_bridge_radius=args.needle_bridge_radius,
        thread_bridge_radius=args.thread_bridge_radius,
        needle_pca_fallback_area=args.needle_pca_fallback_area,
        thread_pca_fallback_area=args.thread_pca_fallback_area,
        needle_dist_thresh=args.needle_dist_thresh,
        needle_angle_thresh_deg=args.needle_angle_thresh,
        thread_dist_thresh=args.thread_dist_thresh,
        thread_angle_thresh_deg=args.thread_angle_thresh,
        needle_max_area=args.needle_max_area,
        needle_min_aspect=args.needle_min_aspect,
        needle_must_be_near_thread=args.needle_must_be_near_thread,
        needle_near_thread_dist=args.needle_near_thread_dist,
        thread_max_area=args.thread_max_area,
        thread_min_aspect=args.thread_min_aspect,
        needle_template=args.needle_template,
        needle_template_inlier_thresh=args.needle_template_inlier_thresh,
        needle_template_min_inlier_ratio=args.needle_template_min_inlier_ratio,
        needle_template_radius_min=args.needle_template_radius_min,
        needle_template_radius_max=args.needle_template_radius_max,
        needle_template_arc_min_deg=args.needle_template_arc_min_deg,
        needle_template_arc_max_deg=args.needle_template_arc_max_deg,
        n_sample=args.n_sample,
    )

    geom_log = []
    in_idx = 0
    proc_idx = 0
    aborted = False
    t_total = 0.0
    t_geom_total = 0.0
    t_wall_prev = time.perf_counter()
    fps_inst_ema = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if in_idx % args.frame_stride != 0:
            in_idx += 1
            continue
        in_idx += 1

        # ---- forward ----
        x, _ = preprocess_frame_bgr(frame_bgr, args.infer_size)
        t0 = time.perf_counter()
        logits = forward_predict(model, has_edge, x.unsqueeze(0), device,
                                  amp_dtype=amp_dtype)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t_total += time.perf_counter() - t0

        if logits.shape[-2:] != (src_h, src_w):
            logits = F.interpolate(logits, size=(src_h, src_w),
                                    mode='bilinear', align_corners=False)
        pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        del logits, x

        # ---- centerline ----
        t1 = time.perf_counter()
        needle_mask = (pred == args.needle_id).astype(np.uint8)
        thread_mask = (pred == args.thread_id).astype(np.uint8)
        res = extract_needle_thread_geometry(needle_mask, thread_mask,
                                              **centerline_kwargs)
        t_geom_total += time.perf_counter() - t1

        # ---- panels (2-row layout) ----
        pred_overlay = overlay_bgr(frame_bgr, pred, palette, alpha=args.alpha)
        centerline_panel = render_centerline_panel(
            frame_bgr, pred, res, args.needle_id, args.thread_id)
        # row 1: [Image | Prediction]
        top_row = np.concatenate([frame_bgr, pred_overlay], axis=1)
        # row 2: centerline stretched to 2x width for prominence
        bottom_row = cv2.resize(centerline_panel, (src_w * 2, src_h),
                                  interpolation=cv2.INTER_LINEAR)
        sbs = np.concatenate([top_row, bottom_row], axis=0)

        # ---- HUD ----
        jd = res.get('junction_distance')
        jd_str = f'{jd:.1f}' if jd is not None else 'NA'
        has_head = res.get('needle_head') is not None
        has_thr = res.get('thread_centerline') is not None
        flag = ''
        if not has_head: flag += ' [no needle]'
        elif not has_thr: flag += ' [no thread]'

        t_wall_now = time.perf_counter()
        dt = max(1e-6, t_wall_now - t_wall_prev)
        t_wall_prev = t_wall_now
        fps_inst = 1.0 / dt
        fps_inst_ema = fps_inst if fps_inst_ema is None \
            else 0.7 * fps_inst_ema + 0.3 * fps_inst

        hud_top = (f'#{proc_idx+1:05d}  src#{in_idx}  '
                   f'fps={fps_inst_ema:.2f} (inst {fps_inst:.2f})  '
                   f'fwd={t_total/(proc_idx+1)*1000:.0f}ms  '
                   f'geom={t_geom_total/(proc_idx+1)*1000:.0f}ms')
        hud_bot = f'gap={jd_str}{flag}'

        # top HUD bar across entire canvas
        bar_h = 56
        cv2.rectangle(sbs, (0, 0), (sbs.shape[1], bar_h), (0, 0, 0), -1)
        cv2.putText(sbs, hud_top, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        # row-1 panel titles
        for ci, t in enumerate(('Image', 'Prediction')):
            cv2.putText(sbs, t, (10 + ci * src_w, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                        cv2.LINE_AA)
        # row-2 panel title (centerline; bigger since the panel is bigger)
        cv2.putText(sbs, 'Centerline (enlarged)', (10, src_h + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA)
        # bottom HUD at the very bottom of the canvas
        cv2.putText(sbs, hud_bot, (10, sbs.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                    cv2.LINE_AA)

        if vw is not None:
            vw.write(sbs)

        if args.dump_every > 0 and (proc_idx % args.dump_every) == 0:
            dump_dir = (args.out.parent if args.out is not None
                         else Path('/tmp')) / 'frames_preview'
            dump_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dump_dir / f'frame_{proc_idx:06d}.png'), sbs)

        if args.show:
            disp = sbs
            if args.show_scale != 1.0:
                disp = cv2.resize(sbs, None, fx=args.show_scale,
                                   fy=args.show_scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(win_name, disp)
            key = cv2.waitKey(int(args.show_wait)) & 0xFF
            if key in (ord('q'), 27):
                print(f'[show] user aborted at processed frame {proc_idx+1}')
                aborted = True
                break
            elif key == ord('p'):
                cv2.waitKey(0)

        geom_log.append({
            'proc_idx': proc_idx,
            'src_idx': in_idx - 1,
            'junction_distance': jd,
            'needle_head': (None if not has_head
                             else [float(x) for x in res['needle_head']]),
            'needle_tail': (None if res.get('needle_tail') is None
                             else [float(x) for x in res['needle_tail']]),
            'needle_sample_points': (
                None if res.get('needle_sample_points') is None
                else res['needle_sample_points'].tolist()),
            'has_thread': has_thr,
        })

        proc_idx += 1
        if args.max_frames and proc_idx >= args.max_frames:
            break
        if (proc_idx) % 50 == 0:
            print(f'  proc {proc_idx}  src {in_idx}/{total_in}  '
                  f'fps={fps_inst_ema:.2f}  gap={jd_str}')

    cap.release()
    if vw is not None: vw.release()
    if args.show: cv2.destroyAllWindows()

    # ---- summary json ----
    summary = {
        'video': str(args.video),
        'src_resolution': {'w': src_w, 'h': src_h},
        'src_fps': src_fps,
        'src_total_frames': total_in,
        'frame_stride': args.frame_stride,
        'n_processed': proc_idx,
        'aborted_early': aborted,
        'infer_size': args.infer_size,
        'output_video': str(args.out) if vw is not None else None,
        'output_fps': out_fps if vw is not None else None,
        'wall_fps_mean': proc_idx / max(1e-6, t_total + t_geom_total),
        'fwd_ms_mean': 1000.0 * t_total / max(1, proc_idx),
        'centerline_ms_mean': 1000.0 * t_geom_total / max(1, proc_idx),
        'centerline_params': centerline_kwargs,
        'frames': geom_log,
    }
    out_json = (args.out.with_suffix('.json') if args.out
                 else Path('/tmp') / (args.video.stem + '_stream.json'))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=lambda o: None)

    print()
    print('======= stream summary =======')
    print(f'processed     : {proc_idx} frames')
    print(f'fwd / geom    : {1000.0*t_total/max(1,proc_idx):.0f} / '
          f'{1000.0*t_geom_total/max(1,proc_idx):.0f} ms per frame')
    if vw is not None:
        print(f'video out     : {args.out}')
    print(f'json log      : {out_json}')


if __name__ == '__main__':
    main()
