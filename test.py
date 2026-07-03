"""Evaluate a trained UniMatchV2 checkpoint on a held-out split.

Reports:
  - Per-class IoU, Dice, Precision, Recall, Pixel-accuracy
  - Per-class Average Precision (pixel-level binary AP from softmax)
  - Global Pixel Accuracy (PA)
  - Mean IoU / Dice / PA / AP (over present, non-excluded classes)
  - Frequency-weighted IoU (fwIoU)
  - "present"-only versions of mean metrics (skip classes with 0 GT pixels)

Usage:
    python test.py \
        --config configs/endoscapes_seg50.yaml \
        --checkpoint /data/exp/endo_v1_base/best.pth \
        --id-path /data/splits/.../test.txt \
        --use-ema \
        --save-csv /data/exp/endo_v1_base/test_metrics.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from util.classes import CLASSES
from util.bias_metrics import (
    ECEAccumulator, compute_bias_metrics, compute_train_pixel_freq,
    format_bias_report, bias_metrics_csv_rows,
)
from util.edge_enhance import EdgeSegResidualAdapter, rgb_edge_prior, edge_to_tokens


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     type=str, required=True)
    p.add_argument('--checkpoint', type=str, default=None,
                    help='best.pth / latest.pth produced by unimatch_v2.py')
    p.add_argument('--delta', type=str, default=None,
                    help='per-epoch delta ckpt (deltas/epNNN.pth); merged with '
                         '--frozen to reconstruct the full model')
    p.add_argument('--frozen', type=str, default=None,
                    help='frozen_base.pth for --delta (default: sibling of the '
                         'delta dir, i.e. <save_path>/frozen_base.pth)')
    p.add_argument('--id-path',    type=str, required=True,
                    help='split file (val.txt or test.txt) — image\\tmask per line')
    p.add_argument('--use-ema',    action='store_true',
                    help='evaluate model_ema state_dict instead of model state_dict')
    p.add_argument('--save-preds', type=str, default=None,
                    help='Directory to save per-frame predicted masks (PNG, value=class id). '
                         'Use with a sequence-filtered --id-path to dump one sequence.')
    p.add_argument('--show', action='store_true',
                    help='Streaming: live OpenCV window during inference (overlay + per-frame '
                         'mIoU/mDice/FPS). Needs a display; auto-skips on headless.')
    p.add_argument('--save-video', type=str, default=None,
                    help='Streaming: write the overlay frames to an mp4 during inference '
                         '(one pass, no post-processing). Works with unknown-GT (1-column) lists.')
    p.add_argument('--video-fps', type=float, default=12.0)
    p.add_argument('--eval-size', type=int, default=0,
                   help='cap inference long-side to this many px (0=full res); '
                        'predictions are upsampled back to original size. Use to avoid OOM.')
    p.add_argument('--save-csv',   type=str, default=None,
                    help='optional CSV path for per-class metrics')
    p.add_argument('--ignore-index', type=int, default=255)
    p.add_argument('--exclude-classes', type=int, nargs='*', default=[],
                    help='classes excluded from mean metrics (e.g. unlearnable rare classes)')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--ap', action='store_true',
                    help='also compute pixel-level per-class Average Precision (slower, ~2x memory)')
    p.add_argument('--train-id-path', type=str, default=None)
    p.add_argument('--train-freq-cache', type=str, default=None)
    p.add_argument('--ece', action='store_true')
    p.add_argument('--ece-bins', type=int, default=15)
    p.add_argument('--no-bias', action='store_true')
    p.add_argument('--affinity-warmstart', type=str, default=None,
                    help='Stage-1 affinity_<dataset>.pt for affinity / affinity_min checkpoints')
    p.add_argument('--visual-adapter', action='store_true',
                    help='force-enable DinoDPTAdapter; normally auto-detected from checkpoint')
    p.add_argument('--visual-adapter-reduction', type=int, default=8)
    p.add_argument('--visual-adapter-dropout', type=float, default=0.0)
    p.add_argument('--backbone', type=str, default=None,
                    help='override cfg.backbone (e.g. dinov2_small)')
    return p.parse_args()


# ----------------------------------------------------------------------
def build_model(cfg):
    """Build DPT matching the trainer config."""
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64,  'out_channels': [48,  96,  192,  384]},
        'base':  {'encoder_size': 'base',  'features': 128, 'out_channels': [96,  192, 384,  768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    size = cfg['backbone'].split('_')[-1]
    kwargs = {**model_configs[size], 'nclass': cfg['nclass']}
    return DPT(**kwargs)


def _strip_prefix(sd, prefixes=('module.', '_orig_mod.')):
    out = OrderedDict()
    for k, v in sd.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
        out[new_k] = v
    return out


def load_checkpoint(model, ckpt_path, use_ema=False, ck=None):
    if ck is None:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    key = 'model_ema' if use_ema else 'model'
    if key not in ck:
        # fall back to whichever is available
        key = 'model' if 'model' in ck else 'model_ema'
    sd = _strip_prefix(ck[key])
    msg = model.load_state_dict(sd, strict=False)
    print(f'[load] {key}  missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}')
    return ck.get('epoch', None), ck.get('mIoU', None), ck.get('mIoU_ema', None)


def _resolve_visual_adapter_cfg(model, ck, ck_model,
                                default_reduction=8, default_dropout=0.0):
    va_cfg = ck.get('visual_adapter') or {}
    va_reduction = int(va_cfg.get('reduction', default_reduction))
    va_dropout = float(va_cfg.get('dropout', default_dropout))
    is_moe = bool(va_cfg.get('moe', False)) or any(
        k.endswith('visual_adapter.blocks.0.router.0.weight')
        or '.visual_adapter.blocks.0.router.0.weight' in k
        for k in ck_model.keys())
    moe_edge_cond = bool(va_cfg.get('moe_edge_cond', False)) or any(
        k.endswith('visual_adapter.blocks.0.edge_proj.weight')
        or '.visual_adapter.blocks.0.edge_proj.weight' in k
        for k in ck_model.keys())
    moe_text_cond_dim = int(va_cfg.get('moe_text_cond_dim', 0))
    text_key = next((k for k in ck_model.keys()
                     if k.endswith('visual_adapter.blocks.0.text_proj.weight')), None)
    if text_key is not None:
        moe_text_cond_dim = int(ck_model[text_key].shape[1])
    down_key = next((k for k in ck_model.keys()
                     if k.endswith('visual_adapter.blocks.0.down.weight')), None)
    if down_key is not None:
        hidden = int(ck_model[down_key].shape[0])
        if hidden > 0:
            va_reduction = max(1, int(model.backbone.embed_dim) // hidden)
    return va_reduction, va_dropout, is_moe, moe_edge_cond, moe_text_cond_dim


def _make_moe_text_cond_fn(model, affinity_side):
    if affinity_side is None:
        return None
    if not (getattr(model, '_va_is_moe', False)
            and int(getattr(model, '_moe_text_cond_dim', 0)) > 0):
        return None

    def _fn(patches, patch_h, patch_w, edge_prior):
        with torch.no_grad():
            cond = affinity_side.moe_text_condition(
                patches.detach(), patch_hw=(patch_h, patch_w),
                edge_prior=edge_prior)
        expected = int(model._moe_text_cond_dim)
        if cond.shape[-1] != expected:
            raise RuntimeError(
                f"MOE_TEXT_COND_DIM={expected} but LC-PAM emits "
                f"{cond.shape[-1]} dims."
            )
        return cond
    return _fn


def _adapt_with_moe_text(model, affinity_side, features, patch_h, patch_w, edge_prior):
    text_cond = None
    fn = _make_moe_text_cond_fn(model, affinity_side)
    if fn is not None:
        text_cond = fn(features[-1], patch_h, patch_w, edge_prior)
    edge_cond = None
    if getattr(model, '_va_is_moe', False) and getattr(model, '_moe_edge_cond_dim', 0) > 0:
        if edge_prior is not None:
            edge_cond = edge_to_tokens(edge_prior.detach(), patch_h, patch_w)
            if edge_cond.dim() == 3 and edge_cond.shape[-1] != 1:
                edge_cond = edge_cond.mean(dim=-1, keepdim=True)
    return model.adapt_features(features, patch_h, patch_w,
                                text_cond=text_cond, edge_cond=edge_cond)


# ----------------------------------------------------------------------
@torch.no_grad()
def _id_to_stem(id_str):
    """Extract the image-file stem from a split-line id ('img_rel<TAB>mask_rel')."""
    first = id_str.split('\t')[0] if '\t' in id_str else id_str.split()[0]
    return os.path.splitext(os.path.basename(first))[0]


# --- streaming overlay helpers (used by --show / --save-video) -------------
_STREAM_PALETTE = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 128, 255), (128, 0, 255),
]
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _denorm_to_bgr(img_t):
    """Normalized CHW float tensor -> HxWx3 uint8 BGR image."""
    x = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    x = (x * _IMG_STD + _IMG_MEAN) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    import cv2
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def _overlay_pred_bgr(bgr, pred, alpha=0.5):
    import cv2
    out = bgr.copy()
    for v in np.unique(pred):
        if v == 0:
            continue
        region = pred == v
        color = _STREAM_PALETTE[(int(v) - 1) % len(_STREAM_PALETTE)]
        colored = np.zeros_like(out); colored[region] = color
        out[region] = cv2.addWeighted(out[region], 1 - alpha, colored[region], alpha, 0)
        cnts, _ = cv2.findContours(region.astype(np.uint8) * 255,
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    return out


def _frame_miou_dice(pred, gt, ignore):
    """Per-frame mIoU/mDice over classes present in (pred|gt); None if no valid GT."""
    valid = gt != ignore
    if valid.sum() == 0:
        return None, None
    p = pred[valid]; g = gt[valid]
    ious, dices = [], []
    for c in np.unique(np.concatenate([p, g])):
        pc = p == c; gc = g == c
        inter = np.logical_and(pc, gc).sum()
        union = np.logical_or(pc, gc).sum()
        denom = pc.sum() + gc.sum()
        if union > 0: ious.append(inter / union)
        if denom > 0: dices.append(2.0 * inter / denom)
    mi = float(np.mean(ious)) if ious else float('nan')
    md = float(np.mean(dices)) if dices else float('nan')
    return mi, md


def _draw_hud(img, lines):
    import cv2
    for i, t in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return img


# --- reusable model assembly + single-frame inference (for true streaming) ---
def build_inference_model(cfg, checkpoint, device, visual_adapter=True,
                          affinity_warmstart=None, use_ema=False,
                          visual_adapter_reduction=8, visual_adapter_dropout=0.0):
    """Assemble DPT (+ HPTA / edge / edge_refiner / cls_head / affinity_side) from a
    checkpoint, mirroring test.main(). Returns a dict ready for infer_pred().
    """
    from util.classes import display_names
    nclass = int(cfg['nclass'])
    class_names = display_names(cfg['dataset'], nclass)

    model = build_model(cfg)
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    ck.pop('optimizer', None)
    ck_model = ck.get('model_ema' if use_ema else 'model', ck.get('model', ck.get('model_ema', {})))

    has_edge_seg = any(k.endswith('edge_seg_adapter.gamma') or '.edge_seg_adapter.' in k
                       for k in ck_model.keys())
    has_visual_adapter = visual_adapter or any(
        k.startswith('visual_adapter.') or '.visual_adapter.' in k for k in ck_model.keys())
    if has_visual_adapter:
        va_reduction, va_dropout, is_moe, moe_edge_cond, moe_text_cond_dim = \
            _resolve_visual_adapter_cfg(
                model, ck, ck_model, visual_adapter_reduction,
                visual_adapter_dropout)
        model.enable_visual_adapter(
            reduction=va_reduction, dropout=va_dropout,
            moe=is_moe, moe_edge_cond=moe_edge_cond,
            moe_text_cond_dim=moe_text_cond_dim)
    if has_edge_seg:
        model.edge_seg_adapter = EdgeSegResidualAdapter(model.backbone.embed_dim)
    if any('edge_refiner.' in k for k in ck_model.keys()):
        from util.edge_enhance import MaskGuidedEdgeRefiner
        model.edge_refiner = MaskGuidedEdgeRefiner(mid=16, dropout=0.0)
    load_checkpoint(model, checkpoint, use_ema=use_ema, ck=ck)
    model = model.to(device).eval()

    affinity_side = None
    if affinity_warmstart or 'affinity_side' in ck:
        from util.affinity_side import AffinitySideBranch
        init_source = affinity_warmstart if affinity_warmstart else ck['affinity_side']
        affinity_side = AffinitySideBranch(init_source, n_orig_classes=nclass,
                                           class_names=class_names, dataset=cfg['dataset'])
        if 'affinity_side' in ck and ck['affinity_side'].get('edge_text_adapter') is not None:
            affinity_side.enable_edge_enhance()
        affinity_side = affinity_side.to(device).eval()
        if 'affinity_side' in ck:
            affinity_side.load_state_dict_from_save(ck['affinity_side'])
        if not ck.get('affinity_proj_unfrozen', False):
            affinity_side.freeze_proj()
    del ck
    patch_size = int(getattr(model.backbone, 'patch_size', 14))
    return {
        'model': model, 'affinity_side': affinity_side,
        'patch_size': patch_size, 'use_edge_enhance': has_edge_seg,
        'nclass': nclass, 'class_names': class_names,
    }


@torch.no_grad()
def infer_pred(bundle, img_chw, device):
    """Run inference on a single normalized CHW float tensor -> pred HxW uint8 (class ids)."""
    model = bundle['model']
    affinity_side = bundle['affinity_side']
    patch_size = bundle['patch_size']
    use_edge_enhance = bundle['use_edge_enhance']

    img = img_chw.unsqueeze(0).to(device)
    ori_h, ori_w = img.shape[-2:]
    new_h = max(patch_size, int(round(ori_h / patch_size)) * patch_size)
    new_w = max(patch_size, int(round(ori_w / patch_size)) * patch_size)
    img_in = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True) \
        if (new_h, new_w) != (ori_h, ori_w) else img

    edge_in = rgb_edge_prior(img_in) if use_edge_enhance else None
    if edge_in is not None and hasattr(model, 'edge_refiner'):
        edge_in = model.edge_refiner(img_in, edge_in)

    if affinity_side is not None:
        moe_text_cond_fn = _make_moe_text_cond_fn(model, affinity_side)
        logits, cls_tok, patches = model(
            img_in, return_cls=True, edge_prior=edge_in,
            moe_text_cond_fn=moe_text_cond_fn)
        ph, pw = img_in.shape[-2] // patch_size, img_in.shape[-1] // patch_size
        side = affinity_side(patches, logits.shape[-2], logits.shape[-1],
                             patch_hw=(ph, pw), edge_prior=edge_in)
        if getattr(affinity_side, 'affinity_metric', '') in ('hyperbolic_pathway', 'cma'):
            x_aware = getattr(affinity_side, 'last_x_aware', None)
            if x_aware is not None:
                if use_edge_enhance and hasattr(model, 'edge_seg_adapter'):
                    x_aware, _ = model.edge_seg_adapter(x_aware, edge_to_tokens(edge_in, ph, pw), ph, pw)
                feats, _, _, _ = model.encode(img_in, return_cls=False)
                feats = list(_adapt_with_moe_text(
                    model, affinity_side, feats, ph, pw, edge_in))
                feats[-1] = x_aware
                logits = model.decode(feats, ph, pw, comp_drop=False)
        logits = affinity_side.fuse(logits, side)
    else:
        logits = model(img_in, edge_prior=edge_in)

    if logits.shape[-2:] != (ori_h, ori_w):
        logits = F.interpolate(logits, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return pred


def collect_predictions(model, loader, nclass, ignore_index, device, collect_ap=False,
                         patch_size=16, cls_head=None, cls_min_pixels=32,
                         ece: ECEAccumulator = None, affinity_side=None,
                         use_edge_enhance=False, save_preds_dir=None,
                         show=False, save_video=None, video_fps=12.0, eval_size=0):
    """Returns (confusion_matrix [C,C], optional ap_scores dict).

    Validation images can have arbitrary HxW. We resize the input so both sides
    are multiples of ``patch_size`` (round-half-up), run inference, then
    interpolate the prediction back to the original size — mirrors the
    behaviour of ``supervised.evaluate`` in the official UniMatchV2 code.
    """
    cm = torch.zeros(nclass, nclass, dtype=torch.int64, device=device)
    pred_count = torch.zeros(nclass, dtype=torch.int64, device=device)
    ap_scores = {c: {'p': [], 't': []} for c in range(nclass)} if collect_ap else None
    # image-level multi-label accumulators (only filled when cls_head provided)
    cls_pred_all, cls_true_all = [], []

    model.eval()
    n_total = len(loader)
    is_cuda = (getattr(device, 'type', str(device)) == 'cuda') or str(device).startswith('cuda')
    timing_rows = []  # (stem, seconds, fps) per frame when saving preds
    stream_on = bool(save_preds_dir) or show or bool(save_video)
    stream_writer = None
    stream_aborted = False
    for _bi, batch in enumerate(loader):
        if _bi % 100 == 0:
            print(f'[eval] {_bi}/{n_total}', flush=True)
        img, mask, ids = batch
        if is_cuda:
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True).long()

        ori_h, ori_w = img.shape[-2:]
        # Optional long-side cap to bound attention memory (pred is upsampled
        # back to the original mask size below, so output resolution is intact).
        ref_h, ref_w = ori_h, ori_w
        if eval_size and max(ori_h, ori_w) > eval_size:
            scale = eval_size / float(max(ori_h, ori_w))
            ref_h, ref_w = int(round(ori_h * scale)), int(round(ori_w * scale))
        new_h = max(patch_size, int(round(ref_h / patch_size)) * patch_size)
        new_w = max(patch_size, int(round(ref_w / patch_size)) * patch_size)
        if (new_h, new_w) != (ori_h, ori_w):
            img_in = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)
        else:
            img_in = img

        edge_in = rgb_edge_prior(img_in) if use_edge_enhance else None
        if edge_in is not None and hasattr(model, 'edge_refiner'):
            edge_in = model.edge_refiner(img_in, edge_in)
        need_patches = cls_head is not None or affinity_side is not None
        if need_patches:
            moe_text_cond_fn = _make_moe_text_cond_fn(model, affinity_side)
            logits, cls_tok, patches = model(
                img_in, return_cls=True, edge_prior=edge_in,
                moe_text_cond_fn=moe_text_cond_fn)
            if affinity_side is not None:
                patch_h = img_in.shape[-2] // patch_size
                patch_w = img_in.shape[-1] // patch_size
                side = affinity_side(
                    patches,
                    logits.shape[-2],
                    logits.shape[-1],
                    patch_hw=(patch_h, patch_w),
                    edge_prior=edge_in,
                )
                # Hyperbolic pathway P2 OR Cross-Modal Adapter: re-decode DPT
                # head ONLY (no second backbone forward) with X_aware replacing
                # the last patches.
                if getattr(affinity_side, 'affinity_metric', '') in ('hyperbolic_pathway', 'cma'):
                    x_aware = getattr(affinity_side, 'last_x_aware', None)
                    if x_aware is not None:
                        ph = img_in.shape[-2] // patch_size
                        pw = img_in.shape[-1] // patch_size
                        if use_edge_enhance and hasattr(model, 'edge_seg_adapter'):
                            x_aware, _ = model.edge_seg_adapter(
                                x_aware, edge_to_tokens(edge_in, ph, pw), ph, pw)
                        feats, _, _, _ = model.encode(img_in, return_cls=False)
                        feats = list(_adapt_with_moe_text(
                            model, affinity_side, feats, ph, pw, edge_in))
                        feats[-1] = x_aware
                        logits = model.decode(feats, ph, pw, comp_drop=False)
                logits = affinity_side.fuse(logits, side)
        if cls_head is not None:
            feat = patches if getattr(cls_head, 'expects', 'cls') == 'patches' else cls_tok
            cls_logits = cls_head(feat)                       # [B, C]
            cls_prob   = torch.sigmoid(cls_logits)
            cls_pred_all.append(cls_prob.cpu().numpy())
            # build per-image GT presence from mask
            tgt = torch.zeros(mask.shape[0], nclass, device=device)
            safe = mask.clone(); safe[safe == ignore_index] = 0
            for c in range(nclass):
                tgt[:, c] = ((safe == c).flatten(1).sum(dim=1) >= cls_min_pixels).float()
            cls_true_all.append(tgt.cpu().numpy())
        if not need_patches:
            logits = model(img_in, edge_prior=edge_in)
        if logits.shape[-2:] != mask.shape[-2:]:
            logits = F.interpolate(logits, size=mask.shape[-2:],
                                    mode='bilinear', align_corners=True)
        prob = logits.softmax(dim=1)
        conf, pred = prob.max(dim=1)

        if stream_on:
            if is_cuda:
                torch.cuda.synchronize()
            bsz = pred.shape[0]
            dt = max(time.perf_counter() - _t0, 1e-9)
            per_frame_sec = dt / bsz
            per_frame_fps = 1.0 / per_frame_sec
            pred_np = pred.detach().cpu().numpy().astype(np.uint8)  # [B,H,W], value=class id
            mask_np = mask.detach().cpu().numpy()
            for b in range(pred_np.shape[0]):
                stem = _id_to_stem(ids[b])
                if save_preds_dir is not None:
                    Image.fromarray(pred_np[b]).save(os.path.join(save_preds_dir, stem + '.png'))
                    timing_rows.append((stem, round(per_frame_sec, 5), round(per_frame_fps, 3)))
                if show or save_video:
                    import cv2
                    bgr = _denorm_to_bgr(img[b])
                    vis = _overlay_pred_bgr(bgr, pred_np[b])
                    mi, md = _frame_miou_dice(pred_np[b], mask_np[b], ignore_index)
                    hud = [stem]
                    if mi is not None:
                        hud.append(f"mIoU {mi*100:5.1f}%   mDice {md*100:5.1f}%")
                    hud.append(f"FPS {per_frame_fps:5.2f}")
                    vis = _draw_hud(vis, hud)
                    if save_video:
                        if stream_writer is None:
                            os.makedirs(os.path.dirname(os.path.abspath(save_video)) or '.', exist_ok=True)
                            h, w = vis.shape[:2]
                            stream_writer = cv2.VideoWriter(
                                save_video, cv2.VideoWriter_fourcc(*'mp4v'),
                                float(video_fps), (w, h))
                        stream_writer.write(vis)
                    if show:
                        try:
                            cv2.imshow('UniMatch-V2 stream (ESC to stop)', vis)
                            if (cv2.waitKey(1) & 0xFF) == 27:
                                stream_aborted = True
                        except cv2.error:
                            show = False
        if stream_aborted:
            break

        valid = (mask != ignore_index)
        v_pred = pred[valid]
        v_mask = mask[valid]
        idx = v_mask * nclass + v_pred
        cm += torch.bincount(idx, minlength=nclass * nclass).view(nclass, nclass)
        pred_count += torch.bincount(v_pred, minlength=nclass)

        if ece is not None:
            v_conf = conf[valid].detach().cpu().numpy().astype(np.float32)
            v_correct = (v_pred == v_mask).detach().cpu().numpy().astype(np.uint8)
            v_pred_np = v_pred.detach().cpu().numpy().astype(np.int32)
            ece.update(v_conf, v_correct, v_pred_np)

        if collect_ap:
            for c in range(nclass):
                ap_scores[c]['p'].append(prob[:, c][valid].detach().cpu().numpy().astype(np.float32))
                ap_scores[c]['t'].append((v_mask == c).cpu().numpy().astype(np.uint8))
    if stream_writer is not None:
        stream_writer.release()
        print(f'[stream] wrote video -> {save_video}', flush=True)
    if show:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass

    if save_preds_dir is not None and timing_rows:
        with open(os.path.join(save_preds_dir, '_timing.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['stem', 'seconds', 'fps'])
            w.writerows(timing_rows)
        mean_fps = len(timing_rows) / sum(r[1] for r in timing_rows)
        print(f'[timing] {len(timing_rows)} frames, mean inference fps = {mean_fps:.2f} '
              f'(device={device})', flush=True)

    cls_pack = None
    if cls_head is not None and cls_pred_all:
        cls_pack = (np.concatenate(cls_pred_all, axis=0),
                    np.concatenate(cls_true_all, axis=0))
    return cm.cpu().numpy(), pred_count.cpu().numpy(), ap_scores, cls_pack


# ----------------------------------------------------------------------
def compute_metrics(cm: np.ndarray, exclude=None):
    """All metrics derivable from the confusion matrix."""
    exclude = set(exclude or [])
    nclass = cm.shape[0]
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    support = cm.sum(axis=1).astype(np.float64)  # # of GT pixels per class
    total = cm.sum().astype(np.float64)

    iou       = tp / np.maximum(tp + fp + fn, 1e-12)
    dice      = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall    = tp / np.maximum(tp + fn, 1e-12)
    cls_pa    = tp / np.maximum(support, 1e-12)

    pa = tp.sum() / np.maximum(total, 1e-12)             # global pixel accuracy

    keep   = np.array([c not in exclude for c in range(nclass)])
    present = support > 0
    val_all     = keep                                    # all classes (except excluded)
    val_present = keep & present                          # plus only classes present in GT

    def safe_mean(x, m):
        return float(x[m].mean()) if m.any() else 0.0

    miou_all       = safe_mean(iou,  val_all)
    miou_present   = safe_mean(iou,  val_present)
    mdice_all      = safe_mean(dice, val_all)
    mdice_present  = safe_mean(dice, val_present)
    mpa_all        = safe_mean(cls_pa, val_all)
    mpa_present    = safe_mean(cls_pa, val_present)

    # Frequency-weighted IoU
    freq = support / np.maximum(total, 1e-12)
    fwiou = float((freq[keep] * iou[keep]).sum())

    return {
        'per_class': {
            'iou': iou, 'dice': dice, 'precision': precision,
            'recall': recall, 'cls_pa': cls_pa, 'support': support,
        },
        'global': {
            'pa': float(pa),
            'miou_all': miou_all, 'miou_present': miou_present,
            'mdice_all': mdice_all, 'mdice_present': mdice_present,
            'mpa_all': mpa_all, 'mpa_present': mpa_present,
            'fwiou': fwiou,
        },
        'keep': keep, 'present': present,
    }


def compute_ap(ap_scores, present_mask):
    """Per-class binary AP from accumulated (prob, label) arrays."""
    from sklearn.metrics import average_precision_score
    ap = np.zeros(len(ap_scores), dtype=np.float64)
    for c, d in ap_scores.items():
        if not present_mask[c] or not d['p']:
            continue
        p = np.concatenate(d['p'])
        t = np.concatenate(d['t'])
        if t.sum() == 0 or t.sum() == t.size:
            continue
        ap[c] = average_precision_score(t, p)
    return ap


# ----------------------------------------------------------------------
def print_report(metrics, class_names, ap=None, ckpt_info=None):
    g = metrics['global']
    pc = metrics['per_class']
    keep, present = metrics['keep'], metrics['present']
    nclass = len(class_names)

    if ckpt_info:
        ep, miou_train, miou_train_ema = ckpt_info
        print(f"checkpoint epoch={ep}  recorded mIoU={miou_train}  EMA={miou_train_ema}")

    print('\n' + '=' * 100)
    print(f"{'cls':>3}  {'name':<26}  {'support':>10}  {'IoU%':>7}  {'Dice%':>7}  "
          f"{'Prec%':>7}  {'Rec%':>7}  {'PA%':>7}" + ('  ' + f"{'AP%':>7}" if ap is not None else ''))
    print('-' * 100)
    for c in range(nclass):
        flag = '' if keep[c] else ' [excl]'
        flag += '' if present[c] else ' [absent]'
        row = (f"{c:>3}  {class_names[c]:<26}  {int(pc['support'][c]):>10d}  "
               f"{pc['iou'][c]*100:>7.2f}  {pc['dice'][c]*100:>7.2f}  "
               f"{pc['precision'][c]*100:>7.2f}  {pc['recall'][c]*100:>7.2f}  "
               f"{pc['cls_pa'][c]*100:>7.2f}")
        if ap is not None:
            row += f"  {ap[c]*100:>7.2f}"
        print(row + flag)
    print('-' * 100)
    print(f"global pixel accuracy (PA):      {g['pa']*100:.2f}%")
    print(f"mean IoU (all kept):             {g['miou_all']*100:.2f}%")
    print(f"mean IoU (present-only):         {g['miou_present']*100:.2f}%")
    print(f"mean Dice (all kept):            {g['mdice_all']*100:.2f}%")
    print(f"mean Dice (present-only):        {g['mdice_present']*100:.2f}%")
    print(f"mean class PA (all kept):        {g['mpa_all']*100:.2f}%")
    print(f"mean class PA (present-only):    {g['mpa_present']*100:.2f}%")
    print(f"frequency-weighted IoU (fwIoU):  {g['fwiou']*100:.2f}%")
    if ap is not None:
        mask = keep & present
        mAP = ap[mask].mean() if mask.any() else 0.0
        print(f"mean AP (present-only):          {mAP*100:.2f}%")
    print('=' * 100 + '\n')


def save_csv(path, metrics, class_names, ap=None, bias=None):
    pc = metrics['per_class']
    g = metrics['global']
    keep, present = metrics['keep'], metrics['present']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['cls', 'name', 'support', 'iou', 'dice', 'precision', 'recall', 'cls_pa', 'kept', 'present']
        if ap is not None:
            header.append('ap')
        w.writerow(header)
        for c in range(len(class_names)):
            row = [c, class_names[c], int(pc['support'][c]),
                   float(pc['iou'][c]), float(pc['dice'][c]),
                   float(pc['precision'][c]), float(pc['recall'][c]),
                   float(pc['cls_pa'][c]), int(keep[c]), int(present[c])]
            if ap is not None:
                row.append(float(ap[c]))
            w.writerow(row)
        # summary row
        w.writerow([])
        w.writerow(['summary', 'pa', g['pa']])
        for k in ('miou_all', 'miou_present', 'mdice_all', 'mdice_present',
                  'mpa_all', 'mpa_present', 'fwiou'):
            w.writerow(['summary', k, g[k]])
        if ap is not None:
            mask = keep & present
            mAP = float(ap[mask].mean()) if mask.any() else 0.0
            w.writerow(['summary', 'mAP_present', mAP])
        if bias is not None:
            w.writerow([])
            for row in bias_metrics_csv_rows(bias):
                w.writerow(row)
    print(f"[csv] wrote {path}")


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    device = torch.device(args.device)

    nclass = int(cfg['nclass'])
    from util.classes import display_names as _disp
    class_names = _disp(cfg['dataset'], nclass)        # paper short names when available
    assert len(class_names) == nclass, \
        f"class_names length ({len(class_names)}) != nclass ({nclass})"

    if args.backbone:
        print(f'[cfg] override backbone {cfg.get("backbone")} -> {args.backbone}')
        cfg['backbone'] = args.backbone

    model = build_model(cfg)
    if args.delta:
        # Reconstruct a full checkpoint = frozen backbone + per-epoch delta.
        frozen_path = args.frozen or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(args.delta))), 'frozen_base.pth')
        fb = torch.load(frozen_path, map_location='cpu', weights_only=False)['backbone']
        dl = torch.load(args.delta, map_location='cpu', weights_only=False)
        ck = {k: v for k, v in dl.items() if k not in ('model', 'model_ema')}
        ck['model'] = {**fb, **dl['model']}
        ck['model_ema'] = {**fb, **dl.get('model_ema', dl['model'])}
        print(f'[delta] merged frozen={frozen_path} + delta={args.delta} '
              f'(epoch={dl.get("epoch")})')
    else:
        if not args.checkpoint:
            raise SystemExit('provide either --checkpoint or --delta')
        ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    # Optimizer state (AdamW moments ≈ 2× params) is never needed at test time.
    # Drop it immediately so it does not sit in host RAM beside the model copies.
    ck.pop('optimizer', None)
    ck_model = ck.get('model_ema' if args.use_ema else 'model', ck.get('model', ck.get('model_ema', {})))
    has_edge_seg = any(k.endswith('edge_seg_adapter.gamma') or '.edge_seg_adapter.' in k
                       for k in ck_model.keys())
    has_visual_adapter = args.visual_adapter or any(
        k.startswith('visual_adapter.') or '.visual_adapter.' in k
        for k in ck_model.keys()
    )
    if has_visual_adapter:
        va_reduction, va_dropout, is_moe, moe_edge_cond, moe_text_cond_dim = \
            _resolve_visual_adapter_cfg(
                model, ck, ck_model, args.visual_adapter_reduction,
                args.visual_adapter_dropout)
        model.enable_visual_adapter(
            reduction=va_reduction,
            dropout=va_dropout,
            moe=is_moe,
            moe_edge_cond=moe_edge_cond,
            moe_text_cond_dim=moe_text_cond_dim,
        )
        print(f'[visual-adapter] loaded '
              f'{"HPTA-MoE" if is_moe else "DinoDPTAdapter"} from checkpoint  '
              f'reduction={va_reduction}  dropout={va_dropout:.3f}  '
              f'edge_cond={moe_edge_cond}  text_dim={moe_text_cond_dim}')
    if has_edge_seg:
        model.edge_seg_adapter = EdgeSegResidualAdapter(model.backbone.embed_dim)
        print('[edge] loaded edge_seg_adapter from checkpoint')
    has_edge_refiner = any('edge_refiner.' in k for k in ck_model.keys())
    if has_edge_refiner:
        from util.edge_enhance import MaskGuidedEdgeRefiner
        model.edge_refiner = MaskGuidedEdgeRefiner(mid=16, dropout=0.0)
        print('[edge] loaded MaskGuidedEdgeRefiner from checkpoint')
    ep, ck_miou, ck_miou_ema = load_checkpoint(model, args.checkpoint,
                                               use_ema=args.use_ema, ck=ck)
    model = model.to(device).eval()

    # ---- Model parameter count (total / encoder / decoder / trainable) ----
    from util.utils import count_params
    _tot = count_params(model)
    _enc = count_params(model.backbone) if hasattr(model, 'backbone') else 0.0
    _dec = count_params(model.head) if hasattr(model, 'head') else 0.0
    _tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    _peft = sum(p.numel() for n, p in model.named_parameters()
                if ('backbone' not in n)) / 1e6
    print(f'[params] total={_tot:.2f}M  encoder={_enc:.2f}M  decoder={_dec:.2f}M  '
          f'trainable={_tr:.2f}M  non-backbone(PEFT+head)={_peft:.2f}M')

    # Optional aux cls head — loaded if present in the checkpoint
    cls_head = None
    try:
        if 'cls_head' in ck:
            from util.cls_head import build_cls_head
            from collections import OrderedDict as _OD
            cls_sd = ck['cls_head']
            cls_sd = _OD((k[len('module.'):] if k.startswith('module.') else k, v)
                          for k, v in cls_sd.items())
            embed_dim = int(model.backbone.embed_dim)
            cls_cfg = (cfg.get('cls') or {})
            cls_head = build_cls_head(embed_dim, nclass, cls_cfg).to(device).eval()
            missing, unexpected = cls_head.load_state_dict(cls_sd, strict=False)
            decoder_type = cls_cfg.get('decoder_type', 'ml_decoder')
            print(f'[cls-head] loaded  decoder={decoder_type}  '
                  f'missing={len(missing)}  unexpected={len(unexpected)}')
    except Exception as e:
        print(f'[cls-head] skipped: {e}')

    affinity_side = None
    if args.affinity_warmstart or 'affinity_side' in ck:
        from util.affinity_side import AffinitySideBranch
        init_source = args.affinity_warmstart if args.affinity_warmstart else ck['affinity_side']
        affinity_side = AffinitySideBranch(
            init_source,
            n_orig_classes=nclass,
            class_names=class_names,
            dataset=cfg['dataset'],
        )
        if 'affinity_side' in ck and ck['affinity_side'].get('edge_text_adapter') is not None:
            affinity_side.enable_edge_enhance()
        affinity_side = affinity_side.to(device).eval()
        # When --use-ema, prefer the EMA-smoothed affinity_side weights if
        # the checkpoint has them. Falls back to the main affinity_side.
        if 'affinity_side' in ck:
            ema_aff_state = ck.get('affinity_side_ema') if args.use_ema else None
            if ema_aff_state is not None:
                affinity_side.load_state_dict_from_save(ema_aff_state)
                print('[affinity] loaded EMA affinity_side state from checkpoint')
            else:
                affinity_side.load_state_dict_from_save(ck['affinity_side'])
                print('[affinity] loaded affinity_side state from checkpoint'
                      + ('  (no EMA copy in ckpt; falling back to main)'
                         if args.use_ema else ''))
        else:
            print('[affinity] checkpoint has no affinity_side state; using warmstart only')
        if not ck.get('affinity_proj_unfrozen', False):
            affinity_side.freeze_proj()
        print(f'[affinity] enabled source={args.affinity_warmstart or "checkpoint"}')

    # All sub-states have been consumed (model / cls_head / affinity_side /
    # visual_adapter cfg). Release the full host-RAM checkpoint before inference.
    del ck
    import gc; gc.collect()

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', id_path=args.id_path)
    loader = DataLoader(valset, batch_size=1, num_workers=2, pin_memory=True)

    ece = ECEAccumulator(n_bins=args.ece_bins) if (args.ece and not args.no_bias) else None

    if args.save_preds:
        os.makedirs(args.save_preds, exist_ok=True)
    cm, pred_count, ap_scores, cls_pack = collect_predictions(
        model, loader, nclass, args.ignore_index, device, collect_ap=args.ap,
        patch_size=int(getattr(model.backbone, 'patch_size', 14)),
        cls_head=cls_head,
        ece=ece,
        affinity_side=affinity_side,
        use_edge_enhance=has_edge_seg,
        save_preds_dir=args.save_preds,
        show=args.show,
        save_video=args.save_video,
        video_fps=args.video_fps,
        eval_size=args.eval_size,
    )
    metrics = compute_metrics(cm, exclude=args.exclude_classes)
    ap = compute_ap(ap_scores, metrics['present']) if args.ap else None

    print_report(metrics, class_names, ap=ap,
                  ckpt_info=(ep, ck_miou, ck_miou_ema))

    bias = None
    if not args.no_bias:
        if args.train_id_path:
            train_freq = compute_train_pixel_freq(
                args.train_id_path, cfg['data_root'], nclass,
                ignore_index=args.ignore_index, cache_path=args.train_freq_cache,
            )
            print(f"[bias] train pixel frequency from {args.train_id_path}")
        else:
            support = cm.sum(axis=1).astype(np.float64)
            train_freq = support / max(support.sum(), 1.0)
            print('[bias] WARNING: --train-id-path not given; using eval-set GT frequency as proxy.')
        pred_freq = pred_count.astype(np.float64) / max(pred_count.sum(), 1.0)
        bias = compute_bias_metrics(
            cm, train_freq=train_freq, pred_freq=pred_freq, ece=ece,
            exclude=args.exclude_classes,
        )
        print(format_bias_report(bias, class_names))

    # ---- Image-level multi-label classification metrics (if cls_head was loaded) ----
    cls_metrics = None
    if cls_pack is not None:
        from sklearn.metrics import average_precision_score, f1_score
        probs, tgts = cls_pack            # [N, C] each
        per_cls_ap = np.zeros(nclass)
        per_cls_f1 = np.zeros(nclass)
        for c in range(nclass):
            if tgts[:, c].sum() == 0 or tgts[:, c].sum() == tgts.shape[0]:
                continue
            per_cls_ap[c] = average_precision_score(tgts[:, c], probs[:, c])
            per_cls_f1[c] = f1_score(tgts[:, c], (probs[:, c] >= 0.5).astype(int),
                                       zero_division=0)
        print('\n' + '=' * 100)
        print('Image-level multi-label classification (from cls_head)')
        print('-' * 100)
        print(f"{'cls':>3}  {'name':<26}  {'#pos':>6}  {'AP%':>7}  {'F1%':>7}")
        for c in range(nclass):
            print(f'{c:>3}  {class_names[c]:<26}  {int(tgts[:, c].sum()):>6d}  '
                  f'{per_cls_ap[c]*100:>7.2f}  {per_cls_f1[c]*100:>7.2f}')
        mAP = per_cls_ap[per_cls_ap > 0].mean() if (per_cls_ap > 0).any() else 0.0
        mF1 = per_cls_f1[per_cls_ap > 0].mean() if (per_cls_ap > 0).any() else 0.0
        print('-' * 100)
        print(f'image-level mAP : {mAP*100:.2f}%   mF1 : {mF1*100:.2f}%')
        print('=' * 100 + '\n')
        cls_metrics = {'per_class_ap': per_cls_ap, 'per_class_f1': per_cls_f1,
                        'mAP': float(mAP), 'mF1': float(mF1)}

    if args.save_csv:
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        save_csv(args.save_csv, metrics, class_names, ap=ap, bias=bias)
        if cls_metrics is not None:
            cls_csv = str(args.save_csv).replace('.csv', '_cls.csv')
            with open(cls_csv, 'w', encoding='utf-8') as f:
                f.write('cls,name,n_pos,ap,f1\n')
                for c in range(nclass):
                    f.write(f'{c},{class_names[c]},{int(cls_pack[1][:,c].sum())},'
                            f'{cls_metrics["per_class_ap"][c]:.6f},'
                            f'{cls_metrics["per_class_f1"][c]:.6f}\n')
                f.write(f'summary,mAP,,{cls_metrics["mAP"]:.6f},\n')
                f.write(f'summary,mF1,,,{cls_metrics["mF1"]:.6f}\n')
            print(f'[csv] wrote {cls_csv}')


if __name__ == '__main__':
    main()
