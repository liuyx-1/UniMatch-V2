"""Per-dataset Stage 1 training.

For each of the 5 datasets independently:
  - read manifest_<ds>.jsonl (or filter merged manifest)
  - drop background class (orig id 0)
  - map remaining orig ids to compact new ids [0..C_d-1]
  - look up class templates via normalised name; fallback to "a {class}"
  - train AffinityClassifier on this dataset only
  - eval on manifest_<ds>_val.jsonl
  - save affinity_<ds>.pt

Output structure (per dataset):
    <out-dir>/<ds>/affinity_<ds>.pt        — model state + class list + remap
    <out-dir>/<ds>/per_class_metrics.csv   — val per-class P/R/F1/AP

Usage (AutoDL paths):
    python tools/train_affinity_per_ds.py \\
        --manifest-dir /root/autodl-tmp/data/autonomous_surgery/siglip_train \\
        --val-dir      /root/autodl-tmp/data/autonomous_surgery/siglip_train/merged_val \\
        --out-dir      /root/autodl-tmp/data/autonomous_surgery/siglip_train/affinity_per_ds \\
        --model google/siglip2-base-patch16-256 \\
        --vision-backbone dinov2 --dinov2-input-size 518 \\
        --epochs 40 --bs 16 --lr 1e-3 --topk 5 --eval-every 5

Default pooling is `soft_threshold` with a learnable per-class θ and hard mask
at eval time — this gives adaptive class selection (each image sees a different
subset of classes, with explicit candidate counts at inference). Revert to the
legacy top-K mean via `--affinity-pooling-mode topk`.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.affinity_cls import AffinityClassifier, compute_loss
from tools.class_templates import TEMPLATES


CLASS_NAMES = {
    'endoscapes_seg50': ['background-tissue', 'cystic_plate', 'calot_triangle',
                          'cystic_artery', 'cystic_duct', 'gallbladder', 'tool'],
    'cholecseg8k': ['background', 'abdominal_wall', 'liver', 'gastrointestinal_tract',
                     'fat', 'grasper', 'connective_tissue', 'blood', 'cystic_duct',
                     'l_hook_electrocautery', 'gallbladder', 'hepatic_vein', 'liver_ligament'],
    'endovis2018': ['background-tissue', 'instrument-shaft', 'instrument-clasper',
                     'instrument-wrist', 'kidney-parenchyma', 'covered-kidney',
                     'thread', 'clamps', 'suturing-needle', 'suction-instrument',
                     'small-intestine', 'ultrasound-probe'],
    'endovis2017_parts': ['background', 'shaft', 'wrist', 'clasper'],
    'endovis2017_type': ['background', 'Bipolar Forceps', 'Prograsp Forceps',
                          'Large Needle Driver', 'Vessel Sealer', 'Grasping Retractor',
                          'Monopolar Curved Scissors', 'Ultrasound Probe'],
    # Surgical multi-video set (needle/thread/clamps); mirrors util/classes.py.
    # Anchors come from tools/class_templates.py (needle/thread/clamps keys).
    'surgical_combined': ['background', 'needle', 'thread', 'clamps'],
}


def normalize_name(name: str) -> str:
    """Map original class name to TEMPLATES key form (snake_case lowercase)."""
    return name.lower().replace(' ', '_').replace('-', '_')


def get_templates_for_class(orig_name: str):
    key = normalize_name(orig_name)
    if key in TEMPLATES:
        return TEMPLATES[key]
    return [f'a photo of {orig_name.replace("_", " ").replace("-", " ")} in surgery']


class PerDsDataset(Dataset):
    def __init__(self, records, n_cls, orig_to_new, preprocess_fn):
        self.records = records
        self.n_cls = n_cls
        self.orig_to_new = orig_to_new
        self.preprocess_fn = preprocess_fn
        self.patch_grid = 37          # set by trainer for DINOv2-518; harmless otherwise
        self.return_patch_mask = False
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        rec = self.records[i]
        img = Image.open(rec['image']).convert('RGB')
        pixel = self.preprocess_fn(img)
        y = torch.zeros(self.n_cls, dtype=torch.float32)
        for c_orig in rec['classes']:
            nid = self.orig_to_new.get(int(c_orig))
            if nid is not None: y[nid] = 1.0
        if not self.return_patch_mask:
            return pixel, y
        # Patch-grid mask in new class id space (255 = ignore: bg or unmapped)
        try:
            mask_full = np.array(Image.open(rec['mask']))
            if mask_full.ndim == 3: mask_full = mask_full[..., 0]
            mask_p = np.array(Image.fromarray(mask_full).resize(
                (self.patch_grid, self.patch_grid), Image.NEAREST))
            mask_new = np.full((self.patch_grid, self.patch_grid), 255, dtype=np.int64)
            for c_orig, n_id in self.orig_to_new.items():
                mask_new[mask_p == int(c_orig)] = int(n_id)
            mask_t = torch.from_numpy(mask_new)
        except Exception:
            mask_t = torch.full((self.patch_grid, self.patch_grid), 255, dtype=torch.long)
        return pixel, y, mask_t


def collate(batch):
    if len(batch[0]) == 2:
        return torch.stack([b[0] for b in batch]), torch.stack([b[1] for b in batch])
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


def _ap(s, g):
    order = np.argsort(-s); g = g[order]
    if g.sum() == 0: return float('nan')
    tp = np.cumsum(g); prec = tp / np.arange(1, len(g)+1)
    return float((prec*g).sum()/g.sum())


def per_class_metrics(scores, gts, n_cls):
    """top-k = max(1, |gt|)."""
    tp = np.zeros(n_cls); fp = np.zeros(n_cls); fn = np.zeros(n_cls); n_pos = gts.sum(0)
    for i in range(gts.shape[0]):
        gt = set(int(c) for c in np.where(gts[i])[0])
        k = max(1, len(gt))
        pred = set(int(c) for c in np.argsort(-scores[i])[:k])
        for c in range(n_cls):
            if c in gt and c in pred: tp[c] += 1
            elif c not in gt and c in pred: fp[c] += 1
            elif c in gt and c not in pred: fn[c] += 1
    prec = tp / np.maximum(tp + fp, 1)
    rec  = tp / np.maximum(tp + fn, 1)
    f1   = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    ap   = np.array([_ap(scores[:, c], gts[:, c]) for c in range(n_cls)])
    return prec, rec, f1, ap, n_pos.astype(int)


def run_eval(m, val_records, new_names, n_cls, orig_to_new, bs, device, prefix='',
             patch_grid: int = 37):
    """Compute BOTH image-level multi-label classification metrics
    (precision / recall / F1 / AP) AND patch-level coarse-segmentation
    metrics (IoU / Dice / pixel accuracy) at the DINOv2 patch grid."""
    sub = PerDsDataset(val_records, n_cls, orig_to_new, m.preprocess)
    sub.patch_grid       = int(patch_grid)
    sub.return_patch_mask = True            # ← request patch GT for seg metrics
    ld = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=2, collate_fn=collate)
    m.eval()
    sc_all, y_all = [], []                  # image-level
    cm_patch = np.zeros((n_cls, n_cls), dtype=np.int64)  # patch-level confusion
    with torch.no_grad():
        for batch in ld:
            if len(batch) == 3:
                pix, y, mask_p = batch
            else:
                pix, y = batch
                mask_p = None
            out = m(pix.to(device))
            sc_all.append(out['cls'].cpu().numpy())
            y_all.append(y.numpy())

            # ── Patch-level confusion matrix ──
            if mask_p is not None and 'dense' in out:
                A_dense = out['dense']                          # [B, C', H_p, W_p]
                if A_dense.shape[-2:] != mask_p.shape[-2:]:
                    A_dense = F.interpolate(A_dense, size=mask_p.shape[-2:],
                                             mode='bilinear', align_corners=False)
                pred_p = A_dense.argmax(dim=1).cpu().numpy()    # [B, H_p, W_p]
                gt_p   = mask_p.numpy()                          # [B, H_p, W_p]
                valid  = (gt_p != 255)
                if valid.any():
                    gt = gt_p[valid].astype(np.int64).clip(0, n_cls - 1)
                    pr = pred_p[valid].astype(np.int64).clip(0, n_cls - 1)
                    cm_patch += np.bincount(gt * n_cls + pr,
                                              minlength=n_cls * n_cls
                                              ).reshape(n_cls, n_cls)

    # ── Image-level metrics ──
    sc = np.concatenate(sc_all, 0); ys = np.concatenate(y_all, 0)
    prec, rec, f1, ap, n_pos = per_class_metrics(sc, ys, n_cls)
    keep = n_pos > 0
    a_f1 = float(np.mean(f1[keep])) if keep.any() else float('nan')
    a_ap_vals = ap[keep & ~np.isnan(ap)]
    a_ap = float(np.mean(a_ap_vals)) if a_ap_vals.size else float('nan')

    # ── Patch-level metrics (coarse seg @ patch grid) ──
    tp_p  = np.diag(cm_patch).astype(np.float64)
    fp_p  = cm_patch.sum(axis=0) - tp_p
    fn_p  = cm_patch.sum(axis=1) - tp_p
    sup_p = cm_patch.sum(axis=1).astype(np.float64)
    iou_p  = tp_p / np.maximum(tp_p + fp_p + fn_p, 1e-12)
    dice_p = 2 * tp_p / np.maximum(2 * tp_p + fp_p + fn_p, 1e-12)
    present_p = sup_p > 0
    miou  = float(np.mean(iou_p[present_p]))  if present_p.any() else float('nan')
    mdice = float(np.mean(dice_p[present_p])) if present_p.any() else float('nan')
    pa    = float(tp_p.sum() / max(cm_patch.sum(), 1))

    # ── Joint print ──
    print(f'[eval{prefix}]  N={sc.shape[0]}  '
          f'CLS:  F1={a_f1:.3f}  mAP={a_ap:.3f}  ({int(keep.sum())}/{n_cls} present)  |  '
          f'PATCH:  mIoU={miou:.3f}  mDice={mdice:.3f}  PA={pa:.3f}  ({int(present_p.sum())}/{n_cls} present)')

    m.train()
    return {'prec': prec, 'rec': rec, 'f1': f1, 'ap': ap, 'n_pos': n_pos,
            'all_F1': a_f1, 'all_mAP': a_ap, 'N': sc.shape[0],
            # patch-level
            'patch_iou': iou_p, 'patch_dice': dice_p,
            'patch_support': sup_p, 'patch_present': present_p,
            'patch_mIoU': miou, 'patch_mDice': mdice, 'patch_PA': pa}


def _save_ckpt(m, args, ds, orig_names, orig_to_new, keep_orig_ids,
                ckpt_path, result, save_epoch):
    """Persist Stage-I checkpoint at `ckpt_path` (called for best/final)."""
    ckpt = {
        'visual_proj': m.visual_proj.state_dict(),
        'text_proj':   m.text_proj.state_dict(),
        'log_tau':     m.log_tau.detach().cpu(),
        'cls_bias':    m.cls_bias.detach().cpu(),
        'T_anchor':    m.T_anchor.detach().cpu(),
        'class_names': [orig_names[c] for c in keep_orig_ids],
        'orig_class_names': orig_names,
        'orig_to_new': orig_to_new,
        'model': args.model, 'd_latent': 128, 'topk': args.affinity_topk,
        'pooling_mode': m.pooling.mode,
        'threshold': m.pooling.theta.detach().cpu(),
        'threshold_init': m.pooling.threshold_init,
        'threshold_learnable': m.pooling.threshold_learnable,
        'threshold_gamma': m.pooling.gamma,
        'min_selected': m.pooling.min_selected,
        'hard_threshold_eval': m.pooling.use_hard_threshold_eval,
        'vision_backbone': args.vision_backbone,
        'dinov2_input_size': args.dinov2_input_size if args.vision_backbone == 'dinov2' else None,
        'dataset': ds,
        'freeze_vision': bool(args.freeze_vision),
        'freeze_text':   bool(args.freeze_text),
        'save_epoch':    int(save_epoch),
        'val_all_F1':       float(result['all_F1'])     if result else None,
        'val_all_mAP':      float(result['all_mAP'])    if result else None,
        'val_patch_mIoU':   float(result.get('patch_mIoU',  float('nan'))) if result else None,
        'val_patch_mDice':  float(result.get('patch_mDice', float('nan'))) if result else None,
        'val_patch_PA':     float(result.get('patch_PA',    float('nan'))) if result else None,
        'patch_loss_type':  str(args.patch_loss_type),
        'patch_dice_alpha': float(args.patch_dice_alpha),
        # ── Hyperbolic config + learned parameters
        'affinity_metric': m.affinity_metric,
        'hyperbolic_dim':  m.d_hyperbolic,
        'hyperbolic_eps':  m.hyperbolic_eps,
        'hyperbolic_distance_scale': m.hyperbolic_distance_scale,
        'hyperbolic_rho':       (m.rho.detach().cpu()       if m.affinity_metric == 'hyperbolic' else None),
        'hyperbolic_log_tau_h': (m.log_tau_h.detach().cpu() if m.affinity_metric == 'hyperbolic' else None),
        'hyperbolic_v_pre_ln':  (m.v_pre_ln.state_dict()    if m.affinity_metric in ('hyperbolic','hyperbolic_pathway') else None),
        'hyperbolic_t_pre_ln':  (m.t_pre_ln.state_dict()    if m.affinity_metric in ('hyperbolic','hyperbolic_pathway') else None),
        # ── hyp_pathway
        'hyp_pathway_state':    (m.hyp_pathway.state_dict_for_save() if m.affinity_metric == 'hyperbolic_pathway' else None),
        'hyp_pathway_layers':   (args.hyperbolic_pathway_layers       if m.affinity_metric == 'hyperbolic_pathway' else None),
        'hyp_pathway_alpha_init': (args.hyperbolic_pathway_alpha_init if m.affinity_metric == 'hyperbolic_pathway' else None),
        # ── CMA
        'cma_block_state':   (m.cma_block.state_dict_for_save() if m.affinity_metric == 'cma' else None),
        'cma_reduction':     (args.cma_reduction               if m.affinity_metric == 'cma' else None),
        'cma_share_dim':     (args.cma_share_dim               if m.affinity_metric == 'cma' else None),
        'cma_non_linearity': (args.cma_non_linearity           if m.affinity_metric == 'cma' else None),
    }
    if (not args.freeze_vision) and args.vision_backbone == 'dinov2' and m._dinov2 is not None:
        ckpt['dinov2_state_dict']   = {k: v.detach().cpu() for k, v in m._dinov2.state_dict().items()}
    if (not args.freeze_vision) and args.vision_backbone == 'siglip' and m.vision_model is not None:
        ckpt['siglip_vision_state'] = {k: v.detach().cpu() for k, v in m.vision_model.state_dict().items()}
    if not args.freeze_text:
        ckpt['siglip_text_state']   = {k: v.detach().cpu() for k, v in m.text_model.state_dict().items()}
    torch.save(ckpt, ckpt_path)


def train_one_ds(ds: str, args, device):
    orig_names = CLASS_NAMES[ds]
    # drop background (anything starting with 'background')
    keep_orig_ids = [c for c, n in enumerate(orig_names) if not n.lower().startswith('background')]
    orig_to_new = {c: i for i, c in enumerate(keep_orig_ids)}
    new_names = [orig_names[c] for c in keep_orig_ids]
    n_cls = len(new_names)
    print(f'\n========== {ds}  ({n_cls} classes after bg removal) ==========')

    # Load train manifest
    mpath = args.manifest_dir / f'manifest_{ds}.jsonl'
    if not mpath.exists():
        print(f'[skip] missing {mpath}'); return None
    records = [json.loads(l) for l in mpath.read_text(encoding='utf-8').splitlines() if l.strip()]
    if args.max_records > 0: records = records[:args.max_records]
    print(f'[train] {len(records)} records')

    # Val
    vpath = args.val_dir / f'manifest_{ds}_val.jsonl'
    if vpath.exists():
        val_records = [json.loads(l) for l in vpath.read_text(encoding='utf-8').splitlines() if l.strip()]
        print(f'[val] {len(val_records)} records from {vpath}')
    else:
        val_records = None
        print(f'[val] no {vpath}')

    # Build model with this dataset's class space
    m = AffinityClassifier(args.model, new_names, d_latent=128, topk=args.affinity_topk,
                            pooling_mode=args.affinity_pooling_mode,
                            threshold_init=args.affinity_threshold_init,
                            threshold_learnable=args.affinity_threshold_learnable,
                            threshold_gamma=args.affinity_threshold_gamma,
                            min_selected=args.affinity_min_selected,
                            hard_threshold_eval=args.affinity_hard_threshold_eval,
                            vision_backbone=args.vision_backbone,
                            dinov2_input_size=args.dinov2_input_size,
                            freeze_vision=args.freeze_vision,
                            freeze_text=args.freeze_text,
                            affinity_metric=args.affinity_metric,
                            hyperbolic_dim=args.hyperbolic_dim,
                            hyperbolic_curvature_init=args.hyperbolic_curvature_init,
                            hyperbolic_learn_curvature=args.hyperbolic_learn_curvature,
                            hyperbolic_temperature_init=args.hyperbolic_temperature_init,
                            hyperbolic_learn_temperature=args.hyperbolic_learn_temperature,
                            hyperbolic_eps=args.hyperbolic_eps,
                            hyperbolic_distance_scale=args.hyperbolic_distance_scale,
                            hyperbolic_pathway_layers=args.hyperbolic_pathway_layers,
                            hyperbolic_pathway_alpha_init=args.hyperbolic_pathway_alpha_init,
                            hyperbolic_pathway_light=args.hyperbolic_pathway_light,
                            hyperbolic_pathway_grad_checkpoint=args.hyperbolic_pathway_grad_checkpoint,
                            hyperbolic_pathway_proj_dropout=args.hyperbolic_pathway_proj_dropout,
                            hyperbolic_pathway_fuse_dropout=args.hyperbolic_pathway_fuse_dropout,
                            hyperbolic_pathway_fuse_layernorm=args.hyperbolic_pathway_fuse_layernorm,
                            hyperbolic_pathway_residual_mode=args.hyperbolic_pathway_residual_mode,
                            hyperbolic_pathway_tau_clamp_min=args.hyperbolic_pathway_tau_clamp_min,
                            hyperbolic_pathway_tau_clamp_max=args.hyperbolic_pathway_tau_clamp_max,
                            hyperbolic_pathway_distance_clamp_max=args.hyperbolic_pathway_distance_clamp_max,
                            cma_reduction=args.cma_reduction,
                            cma_share_dim=args.cma_share_dim,
                            cma_non_linearity=args.cma_non_linearity,
                            cma_adapter_dropout=args.cma_adapter_dropout,
                            cma_adapter_init_std=args.cma_adapter_init_std,
                            cma_fuse_dropout=args.cma_fuse_dropout,
                            cma_fuse_layernorm=args.cma_fuse_layernorm,
                            cma_tau_init=args.cma_tau_init,
                            cma_tau_clamp_min=args.cma_tau_clamp_min,
                            cma_tau_clamp_max=args.cma_tau_clamp_max,
                            ).to(device)
    print(f'[backbone] vision={args.vision_backbone} '
          f'({"frozen" if args.freeze_vision else f"trainable lr={args.lr_vision:g}"})  '
          f'text=SigLIP-2 ({"frozen" if args.freeze_text else f"trainable lr={args.lr_text:g}"})')
    # Always print pooling + patch-loss config — both affect Stage-I math
    print(f'[pooling] mode={m.pooling.mode}  '
          f'θ_init={m.pooling.threshold_init:g}  '
          f'θ_learnable={m.pooling.threshold_learnable}  '
          f'γ={m.pooling.gamma:g}  '
          f'topk_fallback={m.pooling.topk}  '
          f'min_selected={m.pooling.min_selected}  '
          f'hard_eval={m.pooling.use_hard_threshold_eval}')
    print(f'[patch_loss] type={args.patch_loss_type}  '
          f'λ_patch={args.lambda_patch:g}  '
          + (f'α_dice={args.patch_dice_alpha:g}' if args.patch_loss_type == 'bce_dice' else ''))

    # Template dict for this dataset's classes
    tpl = {n: get_templates_for_class(n) for n in new_names}
    m.init_T_from_templates(tpl)
    print(f'[init] trainable params = {sum(p.numel() for p in m.parameters() if p.requires_grad)}')

    # Class-balanced sampling: weight inversely to class freq
    ys_all = np.zeros((len(records), n_cls), dtype=np.float32)
    for i, r in enumerate(records):
        for c in r['classes']:
            nid = orig_to_new.get(int(c))
            if nid is not None: ys_all[i, nid] = 1.0
    freq = ys_all.sum(0) + 1.0
    rare_score_per_sample = (1.0 / np.sqrt(freq))[None, :] * ys_all
    sample_w = rare_score_per_sample.max(1) + 0.1  # never 0
    sampler = WeightedRandomSampler(sample_w.tolist(), num_samples=len(records), replacement=True)

    dset = PerDsDataset(records, n_cls, orig_to_new, m.preprocess)
    dset.patch_grid = int(args.patch_grid)
    dset.return_patch_mask = bool(args.lambda_patch > 0)
    loader = DataLoader(dset, batch_size=args.bs, sampler=sampler, num_workers=2, collate_fn=collate)

    neg = (len(records) - freq).clip(min=1.0)
    pos_weight = torch.from_numpy((neg / freq).astype(np.float32)).to(device)

    # Param groups: heads at args.lr, vision backbone at args.lr_vision (if unfrozen),
    # text encoder at args.lr_text (if unfrozen).
    head_names  = ('visual_proj', 'text_proj', 'log_tau', 'cls_bias')
    head_params, vis_params, txt_params = [], [], []
    for name, p in m.named_parameters():
        if not p.requires_grad: continue
        if name.startswith(('vision_model', '_dinov2')):
            vis_params.append(p)
        elif name.startswith('text_model'):
            txt_params.append(p)
        else:
            head_params.append(p)
    param_groups = [{'params': head_params, 'lr': args.lr}]
    if vis_params:
        param_groups.append({'params': vis_params, 'lr': args.lr_vision})
    if txt_params:
        param_groups.append({'params': txt_params, 'lr': args.lr_text})
    n_head = sum(p.numel() for p in head_params)
    n_vis  = sum(p.numel() for p in vis_params)
    n_txt  = sum(p.numel() for p in txt_params)
    print(f'[opt] head={n_head:,}  vision={n_vis:,}  text={n_txt:,}  '
          f'total trainable={n_head + n_vis + n_txt:,}')
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # ── Early stopping state ──────────────────────────────────────────
    es_patience       = int(args.early_stop_patience)
    es_min_epoch      = int(args.early_stop_min_epoch)
    es_metric         = str(args.early_stop_metric)
    es_min_delta      = float(args.early_stop_min_delta)
    best_score        = -float('inf') if es_metric != 'loss' else float('inf')
    best_epoch        = 0
    best_result       = None
    patience_counter  = 0
    out_ds            = args.out_dir / ds
    out_ds.mkdir(parents=True, exist_ok=True)
    best_ckpt_path    = out_ds / f'affinity_{ds}.pt'           # save best here

    use_patch_loss = bool(args.lambda_patch > 0)
    for ep in range(args.epochs):
        m.train()
        tot_total = tot_img = tot_patch = 0.0; nb = 0
        sel_sum = None
        sel_pixels = 0
        removed_total = 0
        removed_images = 0
        for batch in loader:
            if use_patch_loss:
                pix, y, mask_p = batch
                mask_p = mask_p.to(device)
            else:
                pix, y = batch
            pix = pix.to(device); y = y.to(device)
            out = m(pix)
            mask = out.get('selection_mask')
            if mask is not None:
                ms = mask.detach()
                if sel_sum is None:
                    sel_sum = torch.zeros(ms.shape[-1], device=ms.device)
                sel_sum += ms.sum(dim=(0, 1))
                sel_pixels += ms.shape[0] * ms.shape[1]
                cnt = out.get('selected_count')
                if cnt is not None:
                    removed_total += int((cnt < float(m.pooling.min_selected)).sum().item())
                    removed_images += int(cnt.shape[0])
            loss_img = compute_loss(out['cls'], y, pos_weight=pos_weight)
            loss = loss_img
            loss_patch_val = 0.0
            if use_patch_loss:
                A = out['dense']                          # [B, C_new, H_p, W_p]  (raw cos/tau + bias)
                B, C, H, W = A.shape
                if H != mask_p.shape[-2] or W != mask_p.shape[-1]:
                    # Mismatch: re-resize on the fly (forward grid != args.patch_grid)
                    mask_p_r = nn.functional.interpolate(
                        mask_p.unsqueeze(1).float(), size=(H, W),
                        mode='nearest').squeeze(1).long()
                else:
                    mask_p_r = mask_p
                valid = (mask_p_r != 255)                 # [B, H, W]
                y_one = nn.functional.one_hot(
                    mask_p_r.clamp(0, C - 1), num_classes=C
                ).permute(0, 3, 1, 2).float()              # [B, C, H, W]
                v = valid.unsqueeze(1).float()             # [B, 1, H, W] gate

                # ── BCE term (optionally with class-imbalance pos_weight) ──
                if args.patch_loss_type == 'bce_weighted':
                    pw = pos_weight.view(1, C, 1, 1)        # reuse image-level pos_weight
                    bce_pix = nn.functional.binary_cross_entropy_with_logits(
                        A, y_one, pos_weight=pw, reduction='none')
                else:
                    bce_pix = nn.functional.binary_cross_entropy_with_logits(
                        A, y_one, reduction='none')
                bce_pix = bce_pix * v
                denom_bce = v.sum().clamp(min=1.0) * C
                loss_bce = bce_pix.sum() / denom_bce

                # ── Dice term (only when patch_loss_type == 'bce_dice') ──
                if args.patch_loss_type == 'bce_dice':
                    p_prob = torch.sigmoid(A) * v                          # [B, C, H, W]
                    y_v    = y_one * v
                    inter  = (p_prob * y_v).flatten(2).sum(dim=2)          # [B, C]
                    psum   = p_prob.flatten(2).sum(dim=2)
                    ysum   = y_v.flatten(2).sum(dim=2)
                    dice_c = 1.0 - (2.0 * inter + 1.0) / (psum + ysum + 1.0)  # [B, C]
                    loss_dice = dice_c.mean()
                    alpha = float(args.patch_dice_alpha)
                    loss_patch = (1.0 - alpha) * loss_bce + alpha * loss_dice
                else:
                    loss_patch = loss_bce

                loss = loss + float(args.lambda_patch) * loss_patch
                loss_patch_val = float(loss_patch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_total += float(loss); tot_img += float(loss_img); tot_patch += loss_patch_val
            nb += 1
        msg = (f'  ep {ep+1:>3}  loss={tot_total/nb:.4f}  '
               f'cls={tot_img/nb:.4f}')
        if use_patch_loss:
            msg += f'  patch={tot_patch/nb:.4f}'
        msg += f'  tau={float(m.log_tau.exp()):.3f}'
        print(msg)
        if not args.quiet and sel_sum is not None and sel_pixels > 0 \
                and m.pooling.mode in ('soft_threshold', 'hard_threshold'):
            ratios = (sel_sum / float(sel_pixels)).detach().cpu().numpy()
            ratio_txt = ', '.join(f'{new_names[i]}={ratios[i]:.3f}' for i in range(len(new_names)))
            print(f'    selected_ratio: {ratio_txt}')
        m.init_T_from_templates(tpl)        # refresh anchors with updated text_proj

        # ── Periodic eval + early stopping check ──────────────────────
        if args.eval_every > 0 and (ep+1) % args.eval_every == 0 and val_records is not None:
            eval_res = run_eval(m, val_records, new_names, n_cls, orig_to_new,
                                 args.bs, device, patch_grid=int(args.patch_grid),
                                 prefix=f' @ep{ep+1}')

            # Pick the score to track
            if   es_metric == 'F1':         cur = eval_res['all_F1']
            elif es_metric == 'mAP':        cur = eval_res['all_mAP']
            elif es_metric == 'patch_mIoU': cur = eval_res['patch_mIoU']
            elif es_metric == 'patch_mDice':cur = eval_res['patch_mDice']
            else:                           cur = tot_total / nb       # training loss

            improved = ((cur > best_score + es_min_delta) if es_metric != 'loss'
                        else (cur < best_score - es_min_delta))
            if improved or best_result is None:
                best_score, best_epoch, best_result = cur, ep + 1, eval_res
                patience_counter = 0
                # Snapshot best ckpt (overwrite affinity_<ds>.pt)
                _save_ckpt(m, args, ds, orig_names, orig_to_new, keep_orig_ids,
                            best_ckpt_path, eval_res, best_epoch)
                print(f'  [best ↑]  {es_metric}={cur:.4f}  saved at ep {ep+1}')
            else:
                patience_counter += 1
                print(f'  [no impr] {es_metric}={cur:.4f}  best={best_score:.4f} '
                      f'@ep{best_epoch}  patience {patience_counter}/{es_patience}')

            if (es_patience > 0
                    and patience_counter >= es_patience
                    and (ep + 1) >= es_min_epoch):
                print(f'  [early-stop] no improvement for {es_patience} evals '
                      f'(min-epoch={es_min_epoch} reached) — stopping at ep {ep+1}')
                break

    # Final eval — but the ckpt we SAVE is the best one (already written by
    # the early-stopping branch).  We just report on the current state and
    # write per-class CSV against the BEST result.
    if val_records is not None:
        result_final = run_eval(m, val_records, new_names, n_cls, orig_to_new,
                                  args.bs, device, patch_grid=int(args.patch_grid),
                                  prefix=' FINAL')
    else:
        result_final = run_eval(m, records, new_names, n_cls, orig_to_new,
                                  args.bs, device, patch_grid=int(args.patch_grid),
                                  prefix=' FINAL-on-train')

    # Pick which result to persist in CSV / printout: best if available, else final
    result = best_result if best_result is not None else result_final
    print(f'\n[summary] best {es_metric}={best_score:.4f} @ep{best_epoch}  '
          f'(ckpt saved to {best_ckpt_path})')

    # Save (only re-save if we have no best, i.e. early-stopping never triggered the save)
    if best_result is None:
        _save_ckpt(m, args, ds, orig_names, orig_to_new, keep_orig_ids,
                    best_ckpt_path, result_final, args.epochs)

    # Per-class CSV (against the BEST result kept in `result`)
    with open(out_ds / 'per_class_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['orig_id', 'orig_name', 'n_pos_img', 'precision', 'recall', 'f1', 'ap',
                     'patch_support', 'patch_iou', 'patch_dice'])
        for orig_id in keep_orig_ids:
            nid = orig_to_new[orig_id]
            ap_v = result['ap'][nid]
            piou = result.get('patch_iou',  np.full(len(keep_orig_ids), np.nan))[nid]
            pdic = result.get('patch_dice', np.full(len(keep_orig_ids), np.nan))[nid]
            psup = int(result.get('patch_support', np.zeros(len(keep_orig_ids)))[nid])
            w.writerow([orig_id, orig_names[orig_id], int(result['n_pos'][nid]),
                         round(result['prec'][nid], 4), round(result['rec'][nid], 4),
                         round(result['f1'][nid], 4),
                         (round(ap_v, 4) if not np.isnan(ap_v) else ''),
                         psup,
                         (round(float(piou), 4) if not np.isnan(piou) else ''),
                         (round(float(pdic), 4) if not np.isnan(pdic) else '')])
    print(f'[save] {out_ds}/affinity_{ds}.pt + per_class_metrics.csv')

    # Verbose per-class print — CLS metrics | PATCH metrics
    print(f'\n=== {ds} per-class FINAL ===')
    print(f'  {"oid":>3}  {"orig name":<28}  '
          f'{"n_img":>5} {"prec":>5} {"rec":>5} {"F1":>5} {"AP":>5}   '
          f'{"n_pat":>6} {"IoU":>5} {"Dice":>5}')
    has_patch = 'patch_iou' in result
    for orig_id in keep_orig_ids:
        nid = orig_to_new[orig_id]
        ap_v = result['ap'][nid]
        ap_str  = f'{ap_v:.3f}' if not np.isnan(ap_v) else '  nan'
        if has_patch:
            piou = result['patch_iou'][nid]; pdic = result['patch_dice'][nid]
            psup = int(result['patch_support'][nid])
            iou_str = f'{piou:.3f}' if not np.isnan(piou) else '  nan'
            dic_str = f'{pdic:.3f}' if not np.isnan(pdic) else '  nan'
            ptail = f'  {psup:>6} {iou_str:>5} {dic_str:>5}'
        else:
            ptail = ''
        print(f'  {orig_id:>3}  {orig_names[orig_id]:<28}  '
              f'{int(result["n_pos"][nid]):>5}  '
              f'{result["prec"][nid]:>5.2f} {result["rec"][nid]:>5.2f} '
              f'{result["f1"][nid]:>5.2f} {ap_str:>5}{ptail}')
    if has_patch:
        print(f'  {"-"*3}  {"MEAN (present only)":<28}  {"":>5} {"":>5} {"":>5} '
              f'{result["all_F1"]:>5.2f} {result["all_mAP"]:>5.2f}   '
              f'{"":>6} {result["patch_mIoU"]:>5.3f} {result["patch_mDice"]:>5.3f}')

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest-dir', type=Path, required=True)
    ap.add_argument('--val-dir',      type=Path, default=None)
    ap.add_argument('--out-dir',      type=Path, required=True)
    ap.add_argument('--datasets', nargs='+', default=list(CLASS_NAMES.keys()))
    ap.add_argument('--model', type=str, default='google/siglip2-base-patch16-256')
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--topk', type=int, default=5)
    ap.add_argument('--affinity-pooling-mode', type=str, default='soft_threshold',
                    choices=['topk', 'soft_threshold', 'hard_threshold'],
                    help='default: soft_threshold (differentiable per-class threshold). '
                          'Use --affinity-pooling-mode topk to revert to the legacy mean of top-K patches.')
    ap.add_argument('--affinity-topk', type=int, default=None,
                    help='Top-k used by topk pooling and threshold fallback. Defaults to --topk.')
    ap.add_argument('--affinity-threshold-init', type=float, default=0.3)
    ap.add_argument('--affinity-threshold-learnable', dest='affinity_threshold_learnable',
                    action='store_true', default=True,
                    help='per-class threshold θ is learnable (default ON for soft_threshold)')
    ap.add_argument('--no-affinity-threshold-learnable', dest='affinity_threshold_learnable',
                    action='store_false',
                    help='keep θ fixed at --affinity-threshold-init')
    ap.add_argument('--affinity-threshold-gamma', type=float, default=0.1)
    ap.add_argument('--affinity-min-selected', type=int, default=1)
    ap.add_argument('--affinity-hard-threshold-eval', dest='affinity_hard_threshold_eval',
                    action='store_true', default=True,
                    help='use hard threshold masks during eval when mode=soft_threshold '
                          '(default ON: exposes explicit class candidate set at inference)')
    ap.add_argument('--no-affinity-hard-threshold-eval', dest='affinity_hard_threshold_eval',
                    action='store_false',
                    help='keep soft sigmoid mask at eval time too')
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--max-records', type=int, default=0)
    # ── Early stopping ─────────────────────────────────────────────────
    ap.add_argument('--early-stop-patience', type=int, default=3,
                    help='stop after N consecutive eval checkpoints with no '
                          'improvement on --early-stop-metric (0 disables)')
    ap.add_argument('--early-stop-min-epoch', type=int, default=15,
                    help='do not stop before this epoch (allows warm-up to finish)')
    ap.add_argument('--early-stop-metric', type=str, default='F1',
                    choices=['F1', 'mAP', 'loss', 'patch_mIoU', 'patch_mDice'],
                    help='metric tracked for early stopping (cls F1/mAP, train loss, or '
                          'patch-level mIoU/mDice on the patch grid)')
    ap.add_argument('--early-stop-min-delta', type=float, default=1e-3,
                    help='minimum metric delta to count as improvement')
    ap.add_argument('--quiet', action='store_true',
                    help='only print epoch progress, losses, and metric scores')
    ap.add_argument('--vision-backbone', type=str, default='dinov2',
                    choices=['siglip', 'dinov2'])
    ap.add_argument('--dinov2-input-size', type=int, default=518)
    # Backbone fine-tune controls — defaults UNFROZEN per user directive
    # NOTE: when `--affinity-metric cma`, we auto-set `freeze_vision=True` unless
    # the user explicitly passes `--no-freeze-vision`. This matches the CMA
    # paper's PEFT protocol (CLIP backbone frozen, only adapters trained).
    ap.add_argument('--freeze-vision', dest='freeze_vision',
                    action='store_true', default=None,
                    help='freeze DINOv2 / SigLIP-vision (default: unfrozen for cosine/hyperbolic, '
                          'auto-frozen for cma)')
    ap.add_argument('--no-freeze-vision', dest='freeze_vision', action='store_false',
                    help='explicit override: keep visual backbone TRAINABLE even in cma mode')
    ap.add_argument('--freeze-text',   action='store_true',
                    help='freeze SigLIP text encoder (default: unfrozen, trainable with --lr-text)')
    ap.add_argument('--lr-vision', type=float, default=1e-5,
                    help='lr for the visual backbone when unfrozen (kept small to avoid overfit)')
    ap.add_argument('--lr-text',   type=float, default=1e-5,
                    help='lr for the SigLIP text encoder when unfrozen')
    # Patch-level supervision (dense BCE on the affinity field)
    # ── Hyperbolic affinity ─────────────────────────────────────────────
    ap.add_argument('--affinity-metric', type=str, default='cosine',
                    choices=['cosine', 'hyperbolic', 'hyperbolic_pathway', 'cma'],
                    help='cosine | hyperbolic (logit-level) | hyperbolic_pathway | '
                          'cma (Cross-Modal Adapter; PR 2025)')
    # ── Cross-Modal Adapter (used when --affinity-metric cma) ───────────
    ap.add_argument('--cma-reduction',         type=int,   default=8,
                    help='r — bottleneck dim of the cross-modal adapter (paper: 8)')
    ap.add_argument('--cma-share-dim',         type=int,   default=32,
                    help='d_s — shared up-projection dim between vision/text (paper: 16-32)')
    ap.add_argument('--cma-non-linearity',     type=str,   default='quick_gelu',
                    choices=['quick_gelu','gelu','gelu_new','relu','silu'])
    ap.add_argument('--cma-adapter-dropout',   type=float, default=0.1)
    ap.add_argument('--cma-adapter-init-std',  type=float, default=0.01,
                    help='N(0, std) init for adapter weights (paper recommends 0.01)')
    ap.add_argument('--cma-fuse-dropout',      type=float, default=0.1)
    ap.add_argument('--cma-no-fuse-layernorm', dest='cma_fuse_layernorm',
                    action='store_false', default=True)
    ap.add_argument('--cma-tau-init',          type=float, default=0.07)
    ap.add_argument('--cma-tau-clamp-min',     type=float, default=0.01)
    ap.add_argument('--cma-tau-clamp-max',     type=float, default=2.0)
    ap.add_argument('--hyperbolic-pathway-layers', type=int, default=2,
                    help='HypMLP refinement depth; 0 = skip HypMLP (lighter)')
    ap.add_argument('--hyperbolic-pathway-alpha-init', type=float, default=-3.0,
                    help='alpha_bias initial value for residual gate (σ ≈ 0.05 at -3)')
    ap.add_argument('--hyperbolic-pathway-light', action='store_true',
                    help='use Euclidean up-projection (no Möbius matvec) — ~20%% memory savings')
    ap.add_argument('--hyperbolic-pathway-grad-checkpoint', action='store_true',
                    help='wrap fusion block in torch.utils.checkpoint — ~50%% memory, 30%% slower')
    # ── Stability / regularisation ─────────────────────────────────────
    ap.add_argument('--hyperbolic-pathway-residual-mode', type=str, default='rezero',
                    choices=['rezero','sigmoid_gate'],
                    help='rezero (γ scalar, init 0 — strictly identity at start) | sigmoid_gate (legacy)')
    ap.add_argument('--hyperbolic-pathway-proj-dropout',       type=float, default=0.1)
    ap.add_argument('--hyperbolic-pathway-fuse-dropout',       type=float, default=0.1)
    ap.add_argument('--hyperbolic-pathway-no-fuse-layernorm',
                    dest='hyperbolic_pathway_fuse_layernorm',
                    action='store_false', default=True,
                    help='disable LayerNorm on the fused branch (not recommended)')
    ap.add_argument('--hyperbolic-pathway-tau-clamp-min',      type=float, default=0.05)
    ap.add_argument('--hyperbolic-pathway-tau-clamp-max',      type=float, default=2.0)
    ap.add_argument('--hyperbolic-pathway-distance-clamp-max', type=float, default=20.0)
    ap.add_argument('--hyperbolic-dim', type=int, default=None,
                    help='hyperbolic latent dim; defaults to --d-latent (128)')
    ap.add_argument('--hyperbolic-curvature-init',  type=float, default=1.0)
    ap.add_argument('--hyperbolic-learn-curvature', action='store_true', default=True)
    ap.add_argument('--no-hyperbolic-learn-curvature',
                    dest='hyperbolic_learn_curvature', action='store_false')
    ap.add_argument('--hyperbolic-temperature-init',  type=float, default=0.1)
    ap.add_argument('--hyperbolic-learn-temperature', action='store_true', default=True)
    ap.add_argument('--no-hyperbolic-learn-temperature',
                    dest='hyperbolic_learn_temperature', action='store_false')
    ap.add_argument('--hyperbolic-eps',            type=float, default=1e-5)
    ap.add_argument('--hyperbolic-distance-scale', type=float, default=1.0)

    ap.add_argument('--lambda-patch', type=float, default=1.0,
                    help='weight of the patch-level loss (0 to disable)')
    ap.add_argument('--patch-grid',   type=int, default=37,
                    help='patch-grid side used to downsample GT mask (37 for DINOv2@518)')
    ap.add_argument('--patch-loss-type', type=str, default='bce_dice',
                    choices=['bce', 'bce_weighted', 'bce_dice'],
                    help='bce (legacy, uniform) | bce_weighted (with class freq pos_weight) | '
                          'bce_dice (BCE + Dice, recommended for class-imbalanced surgical data)')
    ap.add_argument('--patch-dice-alpha', type=float, default=0.5,
                    help='Dice weight in bce_dice combo: L = (1-α)·BCE + α·Dice')
    args = ap.parse_args()
    if args.affinity_topk is None:
        args.affinity_topk = args.topk

    # ── Auto-PEFT defaults for cross-modal adapter (CMA, PR 2025) ─────
    # The paper trains ONLY the adapter; vision (and optionally text)
    # backbones stay frozen.  Replicate that protocol by default unless
    # the user explicitly opts out.
    if args.affinity_metric == 'cma':
        if args.freeze_vision is None:
            args.freeze_vision = True
            print('[cma-peft] --affinity-metric cma → auto-enabling --freeze-vision '
                  '(use --no-freeze-vision to override).')
        else:
            print(f'[cma-peft] --affinity-metric cma + user freeze_vision={args.freeze_vision}')
    else:
        # Legacy metrics keep their original UNFROZEN default
        if args.freeze_vision is None:
            args.freeze_vision = False

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    summary = []
    for ds in args.datasets:
        if ds not in CLASS_NAMES:
            print(f'[skip] unknown dataset {ds}'); continue
        r = train_one_ds(ds, args, device)
        if r is not None:
            summary.append((ds, r['N'], r['all_F1'], r['all_mAP']))

    print('\n' + '='*60)
    print('PER-DATASET FINAL SUMMARY')
    print('='*60)
    print(f'  {"dataset":<22} {"N_val":>5}  {"all F1":>6} {"all mAP":>7}')
    for ds, n, f1, ap_v in summary:
        f1s = f'{f1:.3f}' if not np.isnan(f1) else '  n/a'
        aps = f'{ap_v:.3f}' if not np.isnan(ap_v) else '  n/a'
        print(f'  {ds:<22} {n:>5}  {f1s:>6} {aps:>7}')


if __name__ == '__main__':
    main()
