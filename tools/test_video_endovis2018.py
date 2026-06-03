"""Pick a random sequence from the endovis2018 test split, run a trained
UniMatch-V2 ckpt on it, and save THREE artefacts:

  (1) <out>/clip_image.mp4    — original RGB frames as a video
  (2) <out>/clip_gt.mp4       — GT mask overlay video (semi-transparent)
  (3) <out>/clip_pred.mp4     — model-prediction overlay video
  (4) <out>/clip_sidebyside.mp4 — concat [image | GT | pred] in one video
  (5) <out>/clip_report.json  — per-frame + aggregate metrics, FPS, etc.

The "sequence" is detected automatically by grouping test-split entries
that share the same parent-of-frame folder (e.g. `seq_1/left_frames/...`).

Usage:
    python tools/test_video_endovis2018.py \\
        --ckpt /root/autodl-tmp/exp/endovis2018/unimatch_v2_<TAG>/best.pth \\
        --config configs/endovis2018.yaml \\
        --test-id-path /root/autodl-tmp/data/.../unimatch_splits_endovis2018_0.10_seed42/test.txt \\
        --out-dir /root/autodl-tmp/exp/endovis2018/unimatch_v2_<TAG>/video_demo \\
        --fps 8 --seed 42 [--use-ema] [--visual-adapter]
"""
from __future__ import annotations
import argparse, json, time, sys, os
from collections import defaultdict, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.semseg.dpt import DPT
from util.classes import CLASSES


# ----------------------------- colour utils -----------------------------
def palette(n):
    cmap = plt.get_cmap('tab20', max(20, n))
    return (np.array([cmap(i)[:3] for i in range(n)]) * 255).astype(np.uint8)

def colorise(mask, pal, ignore=255):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(len(pal)):
        out[mask == c] = pal[c]
    out[mask == ignore] = 0
    return out

def overlay(img_bgr, mask, pal, alpha=0.5, ignore=255):
    color_rgb = colorise(mask, pal, ignore=ignore)
    color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(img_bgr, 1 - alpha, color_bgr, alpha, 0)
    out[mask == ignore] = img_bgr[mask == ignore]
    return out


# ----------------------------- model io -----------------------------
def preprocess(img: Image.Image, crop=490):
    target = (crop // 14) * 14
    img_resized = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img_resized).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    return torch.from_numpy(norm).permute(2, 0, 1).float(), target

def build_model(cfg, device):
    cfgs = {
        'small': dict(encoder_size='small', features=64,  out_channels=[48,96,192,384]),
        'base':  dict(encoder_size='base',  features=128, out_channels=[96,192,384,768]),
        'large': dict(encoder_size='large', features=256, out_channels=[256,512,1024,1024]),
    }
    bb = cfg['backbone'].split('_')[-1]
    return DPT(**cfgs[bb], nclass=int(cfg['nclass'])).to(device)


# ----------------------------- sequence grouping -----------------------------
def group_sequences(lines: List[List[str]], data_root: str) -> Dict[str, List[Tuple[str,str]]]:
    """Group frames into sequences by trying common path layouts:
      1) <root>/<seq>/<frame>.png                → parent
      2) <root>/<seq>/left_frames/<frame>.png    → parent.parent
      3) <root>/images/<seq>/<frame>.png         → parent (still works)
    We pick the layout that yields >1 group; otherwise fall back to single-group."""
    candidates = []
    for level in (1, 2):
        g = defaultdict(list)
        for parts in lines:
            if len(parts) < 2: continue
            p = Path(parts[0])
            anc = p.parent if level == 1 else p.parent.parent
            g[str(anc)].append((parts[0], parts[1]))
        if len(g) > 1:
            candidates.append((level, g))

    if candidates:
        # Prefer the layout that yields the most balanced groups (smaller max-min ratio)
        _, best = min(candidates, key=lambda lg: max(len(v) for v in lg[1].values()))
    else:
        # Single sequence fallback
        best = defaultdict(list)
        for parts in lines:
            if len(parts) >= 2:
                best[str(Path(parts[0]).parent)].append((parts[0], parts[1]))

    for k in best: best[k].sort(key=lambda p: p[0])
    return dict(best)


# ----------------------------- ckpt loading (mirror test.py) -----------------------------
def _strip_prefix(sd, prefixes=('module.', '_orig_mod.')):
    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p): nk = nk[len(p):]; changed = True
        out[nk] = v
    return out


# ----------------------------- metric helpers -----------------------------
def per_frame_metrics(pred: np.ndarray, gt: np.ndarray, nclass: int, ignore=255):
    """Return (mIoU on present classes, pixel accuracy) for one frame."""
    valid = gt != ignore
    if not valid.any():
        return float('nan'), float('nan')
    pa = float((pred[valid] == gt[valid]).mean())

    ious = []
    for c in range(nclass):
        gt_c, pr_c = (gt == c), (pred == c)
        union = (gt_c | pr_c) & valid
        if not gt_c.any():
            continue   # skip absent classes for "miou_present"
        inter = (gt_c & pr_c & valid).sum()
        u = union.sum()
        ious.append(float(inter / u) if u > 0 else 0.0)
    miou = float(np.mean(ious)) if ious else float('nan')
    return miou, pa


# ----------------------------- video writer -----------------------------
def open_writer(path: Path, fps: int, size_wh: Tuple[int,int]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(path), fourcc, float(fps), size_wh)
    if not vw.isOpened():
        raise RuntimeError(f'VideoWriter failed to open {path} (size={size_wh}, fps={fps})')
    return vw


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',         type=Path, required=True)
    ap.add_argument('--config',       type=Path, required=True)
    ap.add_argument('--test-id-path', type=Path, required=True,
                    help='split file (test.txt or val.txt), one "img\\tmask" per line')
    ap.add_argument('--out-dir',      type=Path, required=True)
    ap.add_argument('--seed',         type=int, default=42)
    ap.add_argument('--fps',          type=int, default=8,
                    help='output video frame rate')
    ap.add_argument('--max-frames',   type=int, default=None,
                    help='cap chosen-sequence length (e.g. 60); None = full sequence')
    ap.add_argument('--device',       type=str, default='cuda')
    ap.add_argument('--use-ema',      action='store_true')
    ap.add_argument('--visual-adapter', action='store_true')
    ap.add_argument('--visual-adapter-reduction', type=int, default=8)
    ap.add_argument('--visual-adapter-dropout',   type=float, default=0.0)
    ap.add_argument('--alpha',        type=float, default=0.5,
                    help='overlay alpha for GT / pred videos')
    ap.add_argument('--side-by-side', action='store_true', default=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- config + model ---
    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    nclass = int(cfg['nclass'])
    class_names = CLASSES[cfg['dataset']]
    pal = palette(nclass)
    crop = int(cfg.get('crop_size', 490))
    data_root = cfg.get('data_root', '')

    # Build model on CPU first; enable VA / edge BEFORE moving to device
    model = build_model(cfg, 'cpu')

    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ck.get('model_ema' if args.use_ema else 'model',
                ck.get('model', ck.get('model_ema', {})))
    sd = _strip_prefix(sd)

    # Auto-detect VA / edge from ckpt keys
    has_va = args.visual_adapter or any(
        k.startswith('visual_adapter.') or '.visual_adapter.' in k for k in sd)
    has_edge_seg = any(k.endswith('edge_seg_adapter.gamma') or '.edge_seg_adapter.' in k
                       for k in sd)

    if has_va and hasattr(model, 'enable_visual_adapter'):
        # Infer reduction from down-projection shape
        va_reduction = args.visual_adapter_reduction
        down_key = next((k for k in sd if k.endswith('visual_adapter.blocks.0.down.weight')), None)
        if down_key is not None:
            hidden = int(sd[down_key].shape[0])
            if hidden > 0:
                va_reduction = max(1, int(model.backbone.embed_dim) // hidden)
        model.enable_visual_adapter(reduction=va_reduction,
                                    dropout=args.visual_adapter_dropout)
    if has_edge_seg:
        from util.edge_enhance import EdgeSegResidualAdapter
        model.edge_seg_adapter = EdgeSegResidualAdapter(model.backbone.embed_dim)

    msg = model.load_state_dict(sd, strict=False)
    aux_prefixes = ('affinity_side', 'boundary_head', 'cls_head')
    unexpected_core = [k for k in msg.unexpected_keys if not k.startswith(aux_prefixes)]
    print(f'[load] VA={has_va} edge={has_edge_seg}  '
          f'missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} '
          f'(non-aux={len(unexpected_core)})')
    if msg.missing_keys[:3]:
        print(f'       missing[:3]={msg.missing_keys[:3]}')

    # NOW move entire model (including newly added adapters) to device
    model = model.to(args.device).eval()

    # --- pick sequence ---
    lines = [l.strip().split('\t') for l in open(args.test_id_path).read().splitlines() if l.strip()]
    seqs = group_sequences(lines, data_root)
    if not seqs:
        raise RuntimeError(f'no sequences parsed from {args.test_id_path}')
    rng = np.random.RandomState(args.seed)
    chosen_key = sorted(seqs.keys())[rng.randint(len(seqs))]
    frames = seqs[chosen_key]
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f'[pick] sequence={chosen_key}  n_frames={len(frames)}')

    # --- determine output frame size ---
    sample_img = Image.open(f"{data_root}/{frames[0][0]}").convert('RGB')
    src_w, src_h = sample_img.size      # PIL: (W, H)
    # Resize all video frames to source resolution for human inspection
    out_size = (src_w, src_h)

    # --- writers ---
    vw_img  = open_writer(args.out_dir / 'clip_image.mp4',  args.fps, out_size)
    vw_gt   = open_writer(args.out_dir / 'clip_gt.mp4',     args.fps, out_size)
    vw_pred = open_writer(args.out_dir / 'clip_pred.mp4',   args.fps, out_size)
    vw_sbs  = None
    if args.side_by_side:
        sbs_size = (out_size[0] * 3, out_size[1])
        vw_sbs = open_writer(args.out_dir / 'clip_sidebyside.mp4', args.fps, sbs_size)

    # --- per-frame inference loop ---
    per_frame = []
    t_inf_total = 0.0
    inter_total = np.zeros(nclass, dtype=np.int64)
    union_total = np.zeros(nclass, dtype=np.int64)
    correct_total, valid_total = 0, 0

    for fi, (img_rel, msk_rel) in enumerate(frames):
        img = Image.open(f"{data_root}/{img_rel}").convert('RGB')
        msk = Image.open(f"{data_root}/{msk_rel}")
        x, tgt = preprocess(img, crop)
        gt_lo = np.array(msk.resize((tgt, tgt), Image.NEAREST))

        # ---- TIMED model forward only ----
        if args.device.startswith('cuda'): torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(args.device))
            if isinstance(logits, (tuple, list)): logits = logits[0]
        if args.device.startswith('cuda'): torch.cuda.synchronize()
        t_inf = time.perf_counter() - t0
        t_inf_total += t_inf

        # Upsample logits → src resolution for video output
        if logits.shape[-2:] != (src_h, src_w):
            logits_full = F.interpolate(logits, size=(src_h, src_w),
                                        mode='bilinear', align_corners=False)
        else:
            logits_full = logits
        pred_full = logits_full.argmax(1)[0].cpu().numpy().astype(np.uint8)

        # GT at source resolution for video & metrics
        gt_full = np.array(msk.resize((src_w, src_h), Image.NEAREST))

        miou_fr, pa_fr = per_frame_metrics(pred_full, gt_full, nclass)
        per_frame.append({'idx': fi, 'img': img_rel,
                          'mIoU_present': miou_fr, 'PA': pa_fr,
                          't_infer_ms': t_inf * 1000.0})

        # global confusion-style accumulators for aggregate mIoU
        valid = gt_full != 255
        valid_total   += int(valid.sum())
        correct_total += int((pred_full[valid] == gt_full[valid]).sum())
        for c in range(nclass):
            gt_c = (gt_full == c) & valid
            pr_c = (pred_full == c) & valid
            inter_total[c] += int((gt_c & pr_c).sum())
            union_total[c] += int((gt_c | pr_c).sum())

        # --- write video frames ---
        img_bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        vw_img.write(img_bgr)
        vw_gt.write(  overlay(img_bgr, gt_full,   pal, alpha=args.alpha))
        vw_pred.write(overlay(img_bgr, pred_full, pal, alpha=args.alpha))
        if vw_sbs is not None:
            sbs = np.concatenate([
                img_bgr,
                overlay(img_bgr, gt_full,   pal, alpha=args.alpha),
                overlay(img_bgr, pred_full, pal, alpha=args.alpha),
            ], axis=1)
            # text labels along top
            for ci, lbl in enumerate(('Image', 'GT', 'Prediction')):
                cv2.putText(sbs, lbl, (10 + ci * src_w, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
            # bottom-left HUD per frame
            txt = f'#{fi:03d}  mIoU={miou_fr:.3f}  PA={pa_fr:.3f}  {1000.0*t_inf:.1f} ms'
            cv2.putText(sbs, txt, (10, src_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2, cv2.LINE_AA)
            vw_sbs.write(sbs)

        if (fi + 1) % 20 == 0 or fi == len(frames) - 1:
            print(f'  frame {fi+1}/{len(frames)}  mIoU={miou_fr:.3f}  PA={pa_fr:.3f}  '
                  f'infer {1000*t_inf:.1f} ms')

    for vw in (vw_img, vw_gt, vw_pred):
        vw.release()
    if vw_sbs is not None: vw_sbs.release()

    # --- aggregate ---
    miou_per_cls = []
    for c in range(nclass):
        if union_total[c] > 0:
            miou_per_cls.append(inter_total[c] / union_total[c])
    miou_clip = float(np.mean(miou_per_cls)) if miou_per_cls else float('nan')
    pa_clip   = float(correct_total / max(1, valid_total))
    fps_real  = len(frames) / t_inf_total if t_inf_total > 0 else float('nan')

    report = {
        'sequence':          chosen_key,
        'n_frames':          len(frames),
        'crop_train':        crop,
        'src_resolution':    {'w': src_w, 'h': src_h},
        'video_fps_out':     args.fps,
        'inference_fps':     fps_real,
        'inference_ms_mean': 1000.0 * t_inf_total / max(1, len(frames)),
        'clip_mIoU_aggregate': miou_clip,
        'clip_PA_aggregate':   pa_clip,
        'class_names':       class_names,
        'per_frame':         per_frame,
        'ckpt':              str(args.ckpt),
    }
    out_json = args.out_dir / 'clip_report.json'
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print('========= clip summary =========')
    print(f'sequence       : {chosen_key}')
    print(f'frames         : {len(frames)}')
    print(f'inference FPS  : {fps_real:.2f}   (mean {1000*t_inf_total/len(frames):.1f} ms / frame)')
    print(f'aggregate mIoU : {miou_clip:.4f}')
    print(f'aggregate PA   : {pa_clip:.4f}')
    print(f'wrote → {args.out_dir}/clip_image.mp4, clip_gt.mp4, clip_pred.mp4, clip_sidebyside.mp4')
    print(f'wrote → {out_json}')


if __name__ == '__main__':
    main()
