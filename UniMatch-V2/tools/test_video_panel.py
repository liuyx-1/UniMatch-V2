"""Multi-variant side-by-side video panel.

For a chosen test-set clip (random sequence on endovis2018, or a random
N-frame slice on endoscapes_seg50), runs MULTIPLE trained ckpts and emits
ONE composite mp4 of the form:

    [Image | GT | pred_var1 | pred_var2 | ... | pred_varK]

Per-frame HUD prints fps & mIoU per variant; a JSON report holds aggregate
metrics and inference FPS for every variant.

Loading mirrors test.py: handles DDP `module.` AND torch.compile
`_orig_mod.` prefixes, auto-enables visual_adapter / edge_seg_adapter, and
rebuilds AffinitySideBranch (incl. edge_text_adapter) from the ckpt.

Usage:
    python tools/test_video_panel.py \\
        --run base:$EXP/endovis2018/unimatch_v2_base_r0.10_bs16_lr5e-5_cr518_va \\
        --run +text:$EXP/endovis2018/unimatch_v2_affinity_min_r0.10_bs16_lr5e-5_cr518_va_joint \\
        --config configs/endovis2018.yaml \\
        --test-id-path $SPLIT/test.txt \\
        --out clip_sidebyside.mp4 \\
        --seconds 20 --fps 8 --seed 42
"""
from __future__ import annotations
import argparse, json, time, sys
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
from util.edge_enhance import EdgeSegResidualAdapter, rgb_edge_prior, edge_to_tokens


# ============================================================ colour helpers
def palette(n):
    cmap = plt.get_cmap('tab20', max(20, n))
    return (np.array([cmap(i)[:3] for i in range(n)]) * 255).astype(np.uint8)

def colorise_rgb(mask, pal, ignore=255):
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for c in range(len(pal)): out[mask == c] = pal[c]
    out[mask == ignore] = 0
    return out

def overlay_bgr(img_bgr, mask, pal, alpha=0.5, ignore=255):
    color_bgr = cv2.cvtColor(colorise_rgb(mask, pal, ignore=ignore), cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(img_bgr, 1 - alpha, color_bgr, alpha, 0)
    out[mask == ignore] = img_bgr[mask == ignore]
    return out


# ============================================================ io
def preprocess(img: Image.Image, crop=490):
    target = (crop // 14) * 14
    img_r = img.resize((target, target), Image.BILINEAR)
    arr = np.asarray(img_r).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std
    return torch.from_numpy(norm).permute(2, 0, 1).float(), target

def build_model(cfg, device='cpu'):
    cfgs = {
        'small': dict(encoder_size='small', features=64,  out_channels=[48,96,192,384]),
        'base':  dict(encoder_size='base',  features=128, out_channels=[96,192,384,768]),
        'large': dict(encoder_size='large', features=256, out_channels=[256,512,1024,1024]),
    }
    bb = cfg['backbone'].split('_')[-1]
    return DPT(**cfgs[bb], nclass=int(cfg['nclass'])).to(device)


def _strip_prefix(sd, prefixes=('module.', '_orig_mod.')):
    out = OrderedDict()
    for k, v in sd.items():
        nk = k; changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p): nk = nk[len(p):]; changed = True
        out[nk] = v
    return out


def load_one_run(ck_path: Path, cfg, device='cuda', use_ema=True):
    """Returns (model, affinity_side_or_None, has_edge_seg)."""
    model = build_model(cfg, 'cpu')
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    sd = ck.get('model_ema' if use_ema else 'model',
                ck.get('model', ck.get('model_ema', {})))
    sd = _strip_prefix(sd)

    has_va = any(k.startswith('visual_adapter.') or '.visual_adapter.' in k for k in sd)
    has_edge_seg = any(k.endswith('edge_seg_adapter.gamma') or '.edge_seg_adapter.' in k for k in sd)

    if has_va and hasattr(model, 'enable_visual_adapter'):
        down_key = next((k for k in sd if k.endswith('visual_adapter.blocks.0.down.weight')), None)
        red = 8
        if down_key is not None:
            hidden = int(sd[down_key].shape[0])
            if hidden > 0:
                red = max(1, int(model.backbone.embed_dim) // hidden)
        model.enable_visual_adapter(reduction=red, dropout=0.0)
    if has_edge_seg:
        model.edge_seg_adapter = EdgeSegResidualAdapter(model.backbone.embed_dim)

    msg = model.load_state_dict(sd, strict=False)
    aux = ('affinity_side', 'boundary_head', 'cls_head')
    unexp_core = [k for k in msg.unexpected_keys if not k.startswith(aux)]
    print(f'    [load] VA={has_va} edge={has_edge_seg}  '
          f'missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)} '
          f'(non-aux={len(unexp_core)})')

    affinity_side = None
    if 'affinity_side' in ck:
        try:
            from util.affinity_side import AffinitySideBranch
            affinity_side = AffinitySideBranch(
                ck['affinity_side'],
                n_orig_classes=int(cfg['nclass']),
                class_names=None,
                dataset='unknown',
            )
            if ck['affinity_side'].get('edge_text_adapter') is not None:
                affinity_side.enable_edge_enhance()
            affinity_side.load_state_dict_from_save(ck['affinity_side'])
            affinity_side = affinity_side.to(device).eval()
            print('    [load] affinity_side reconstructed')
        except Exception as e:
            print(f'    [load] affinity_side failed: {e}')
            affinity_side = None

    model = model.to(device).eval()
    return model, affinity_side, has_edge_seg


@torch.no_grad()
def forward_predict(model, affinity_side, has_edge_seg, x, patch_size=14):
    edge_in = rgb_edge_prior(x) if has_edge_seg else None
    if affinity_side is not None:
        logits, _cls_tok, patches = model(x, return_cls=True, edge_prior=edge_in)
        ph = x.shape[-2] // patch_size; pw = x.shape[-1] // patch_size
        side = affinity_side(patches, logits.shape[-2], logits.shape[-1],
                             patch_hw=(ph, pw), edge_prior=edge_in)
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


# ============================================================ sequence picking
def group_sequences(lines):
    """Try parent / parent.parent grouping; pick layout giving >1 group."""
    cands = []
    for level in (1, 2):
        g = defaultdict(list)
        for parts in lines:
            if len(parts) < 2: continue
            p = Path(parts[0])
            anc = p.parent if level == 1 else p.parent.parent
            g[str(anc)].append((parts[0], parts[1]))
        if len(g) > 1: cands.append((level, g))
    if cands:
        _, best = min(cands, key=lambda lg: max(len(v) for v in lg[1].values()))
    else:
        best = defaultdict(list)
        for parts in lines:
            if len(parts) >= 2:
                best[str(Path(parts[0]).parent)].append((parts[0], parts[1]))
    for k in best: best[k].sort(key=lambda p: p[0])
    return dict(best)


def pick_clip(lines, n_frames, seed=42, random_clip=False):
    rng = np.random.RandomState(seed)
    if random_clip:
        # Treat all test frames as candidates; sample n_frames random ones (sorted).
        idx = rng.choice(len(lines), size=min(n_frames, len(lines)), replace=False)
        idx = sorted(idx.tolist())
        return 'random-clip', [(lines[i][0], lines[i][1]) for i in idx]
    seqs = group_sequences(lines)
    keys = sorted(seqs.keys())
    key = keys[rng.randint(len(keys))]
    return key, seqs[key][:n_frames]


# ============================================================ metric
def per_frame_metrics(pred, gt, nclass, ignore=255):
    valid = gt != ignore
    if not valid.any(): return float('nan'), float('nan')
    pa = float((pred[valid] == gt[valid]).mean())
    ious = []
    for c in range(nclass):
        gt_c, pr_c = (gt == c), (pred == c)
        if not gt_c.any(): continue
        u = ((gt_c | pr_c) & valid).sum()
        i = (gt_c & pr_c & valid).sum()
        ious.append(float(i / u) if u > 0 else 0.0)
    return (float(np.mean(ious)) if ious else float('nan')), pa


# ============================================================ main
def parse_run(s):
    if ':' not in s: raise argparse.ArgumentTypeError(f'expected LABEL:PATH, got {s!r}')
    lbl, path = s.split(':', 1)
    return lbl, Path(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run',          type=parse_run, action='append', required=True,
                    help='LABEL:CKPT_DIR — repeat to compare multiple variants')
    ap.add_argument('--config',       type=Path, required=True)
    ap.add_argument('--test-id-path', type=Path, required=True)
    ap.add_argument('--out',          type=Path, required=True,
                    help='output side-by-side mp4 path (single file)')
    ap.add_argument('--seconds',      type=float, default=20,
                    help='clip length in seconds (frames = seconds * fps)')
    ap.add_argument('--fps',          type=int,   default=8)
    ap.add_argument('--seed',         type=int,   default=42)
    ap.add_argument('--random-clip',  action='store_true',
                    help='sample N random frames from test set (use for endoscapes which '
                         'has no temporal sequences)')
    ap.add_argument('--alpha',        type=float, default=0.5)
    ap.add_argument('--device',       type=str,   default='cuda')
    ap.add_argument('--no-ema',       action='store_true')
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    nclass = int(cfg['nclass'])
    pal = palette(nclass)
    crop = int(cfg.get('crop_size', 518))
    data_root = cfg.get('data_root', '')
    use_ema = not args.no_ema

    # ---- pick clip ----
    lines = [l.strip().split('\t') for l in open(args.test_id_path).read().splitlines() if l.strip()]
    n_frames = int(args.seconds * args.fps)
    seq_key, frames = pick_clip(lines, n_frames, seed=args.seed, random_clip=args.random_clip)
    print(f'[pick] mode={"random" if args.random_clip else "sequence"}  '
          f'key={seq_key}  n_frames={len(frames)}  (~{len(frames)/args.fps:.1f}s)')

    # ---- load all frames once (raw + GT) ----
    sample_img = Image.open(f"{data_root}/{frames[0][0]}").convert('RGB')
    src_w, src_h = sample_img.size
    raws, gts, tensors = [], [], []
    for img_rel, msk_rel in frames:
        img = Image.open(f"{data_root}/{img_rel}").convert('RGB')
        msk = Image.open(f"{data_root}/{msk_rel}")
        x, _ = preprocess(img, crop)
        raws.append(np.asarray(img))
        gts.append(np.array(msk.resize((src_w, src_h), Image.NEAREST)))
        tensors.append(x)

    # ---- for each variant: load, predict, time ----
    preds_per_run = OrderedDict()
    timing  = OrderedDict()
    fpsr    = OrderedDict()
    miou_pf = OrderedDict()
    pa_pf   = OrderedDict()
    for lbl, run_dir in args.run:
        ck = run_dir / 'best.pth'
        print(f'[run] >>> {lbl}  ({run_dir.name})')
        if not ck.exists():
            print(f'    skip: missing {ck}'); continue
        model, aff, has_edge = load_one_run(ck, cfg, device=args.device, use_ema=use_ema)

        preds = []; t_total = 0.0; miou_list = []; pa_list = []
        for fi, x in enumerate(tensors):
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            t0 = time.perf_counter()
            logits = forward_predict(model, aff, has_edge, x.unsqueeze(0).to(args.device))
            if args.device.startswith('cuda'): torch.cuda.synchronize()
            t_total += time.perf_counter() - t0

            if logits.shape[-2:] != (src_h, src_w):
                logits = F.interpolate(logits, size=(src_h, src_w),
                                       mode='bilinear', align_corners=False)
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            preds.append(pred)
            m, p = per_frame_metrics(pred, gts[fi], nclass)
            miou_list.append(m); pa_list.append(p)

        preds_per_run[lbl] = preds
        timing[lbl]  = 1000.0 * t_total / len(tensors)
        fpsr[lbl]    = len(tensors) / t_total if t_total > 0 else float('nan')
        miou_pf[lbl] = miou_list
        pa_pf[lbl]   = pa_list
        print(f'    [done] fps={fpsr[lbl]:.1f}  mean={timing[lbl]:.1f} ms  '
              f'mIoU(mean)={np.nanmean(miou_list):.3f}  PA(mean)={np.nanmean(pa_list):.3f}')
        del model
        if aff is not None: del aff
        if args.device.startswith('cuda'): torch.cuda.empty_cache()

    if not preds_per_run:
        print('[error] no valid variants — nothing to write'); return

    # ---- assemble side-by-side video ----
    n_panels = 2 + len(preds_per_run)            # Image | GT | preds
    sbs_w = src_w * n_panels
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(args.out), fourcc, float(args.fps), (sbs_w, src_h))
    if not vw.isOpened():
        raise RuntimeError(f'VideoWriter failed: {args.out}')

    panel_titles = ['Image', 'GT'] + list(preds_per_run.keys())
    for fi in range(len(frames)):
        img_bgr = cv2.cvtColor(raws[fi], cv2.COLOR_RGB2BGR)
        gt_bgr  = overlay_bgr(img_bgr, gts[fi], pal, alpha=args.alpha)
        pred_bgrs = [overlay_bgr(img_bgr, preds_per_run[lbl][fi], pal, alpha=args.alpha)
                     for lbl in preds_per_run]
        sbs = np.concatenate([img_bgr, gt_bgr] + pred_bgrs, axis=1)

        # column titles (top)
        for ci, t in enumerate(panel_titles):
            cv2.putText(sbs, t, (10 + ci * src_w, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)

        # per-variant HUD at the bottom of each prediction panel
        for ci, lbl in enumerate(preds_per_run):
            x0 = (2 + ci) * src_w + 10
            txt = f'mIoU {miou_pf[lbl][fi]:.2f}  PA {pa_pf[lbl][fi]:.2f}  ' \
                  f'{timing[lbl]:.0f} ms'
            cv2.putText(sbs, txt, (x0, src_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # frame index global HUD bottom-left
        cv2.putText(sbs, f'#{fi:03d}  ({fi/args.fps:.1f}s)', (10, src_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        vw.write(sbs)
    vw.release()

    # ---- aggregate report ----
    report = {
        'mode':           'random-clip' if args.random_clip else 'sequence',
        'sequence_key':   seq_key,
        'n_frames':       len(frames),
        'video_fps_out':  args.fps,
        'src_resolution': {'w': src_w, 'h': src_h},
        'variants': {
            lbl: {
                'inference_fps':     fpsr[lbl],
                'inference_ms_mean': timing[lbl],
                'clip_mIoU_mean':    float(np.nanmean(miou_pf[lbl])),
                'clip_PA_mean':      float(np.nanmean(pa_pf[lbl])),
                'per_frame_mIoU':    miou_pf[lbl],
                'per_frame_PA':      pa_pf[lbl],
                'ckpt':              str(dict(args.run)[lbl] / 'best.pth'),
            } for lbl in preds_per_run
        },
    }
    rj = args.out.with_suffix('.json')
    with open(rj, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print()
    print('========= summary =========')
    for lbl in preds_per_run:
        print(f'  {lbl:>10s}   fps={fpsr[lbl]:6.1f}   '
              f'mIoU={np.nanmean(miou_pf[lbl]):.3f}   '
              f'PA={np.nanmean(pa_pf[lbl]):.3f}')
    print(f'wrote → {args.out}')
    print(f'wrote → {rj}')


if __name__ == '__main__':
    main()
