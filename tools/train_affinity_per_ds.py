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

Usage:
    python tools/train_affinity_per_ds.py \\
        --manifest-dir /data/pretrained/siglip_train \\
        --val-dir      /data/pretrained/siglip_train/merged_val \\
        --out-dir      /data/pretrained/siglip_train/affinity_per_ds \\
        --model google/siglip2-base-patch16-256 \\
        --vision-backbone dinov2 --dinov2-input-size 518 \\
        --epochs 40 --bs 16 --lr 1e-3 --topk 5 --eval-every 5
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


def run_eval(m, val_records, new_names, n_cls, orig_to_new, bs, device, prefix=''):
    sub = PerDsDataset(val_records, n_cls, orig_to_new, m.preprocess)
    ld = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=2, collate_fn=collate)
    m.eval()
    sc_all, y_all = [], []
    with torch.no_grad():
        for pix, y in ld:
            out = m(pix.to(device))
            sc_all.append(out['cls'].cpu().numpy()); y_all.append(y.numpy())
    sc = np.concatenate(sc_all, 0); ys = np.concatenate(y_all, 0)
    prec, rec, f1, ap, n_pos = per_class_metrics(sc, ys, n_cls)
    # exclude n_pos=0 classes from averages
    keep = n_pos > 0
    a_f1 = float(np.mean(f1[keep])) if keep.any() else float('nan')
    a_ap_vals = ap[keep & ~np.isnan(ap)]
    a_ap = float(np.mean(a_ap_vals)) if a_ap_vals.size else float('nan')
    print(f'[eval{prefix}]  N={sc.shape[0]}  cls={int(keep.sum())}/{n_cls}  '
          f'all F1={a_f1:.3f}  mAP={a_ap:.3f}')
    m.train()
    return {'prec': prec, 'rec': rec, 'f1': f1, 'ap': ap, 'n_pos': n_pos,
             'all_F1': a_f1, 'all_mAP': a_ap, 'N': sc.shape[0]}


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
    m = AffinityClassifier(args.model, new_names, d_latent=128, topk=args.topk,
                            vision_backbone=args.vision_backbone,
                            dinov2_input_size=args.dinov2_input_size,
                            freeze_vision=args.freeze_vision,
                            freeze_text=args.freeze_text,
                            ).to(device)
    print(f'[backbone] vision={args.vision_backbone} '
          f'({"frozen" if args.freeze_vision else f"trainable lr={args.lr_vision:g}"})  '
          f'text=SigLIP-2 ({"frozen" if args.freeze_text else f"trainable lr={args.lr_text:g}"})')

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
    print(f'[opt] head params={n_head:,}  vision params={n_vis:,}  text params={n_txt:,}  '
          f'total trainable={n_head + n_vis + n_txt:,}')
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    use_patch_loss = bool(args.lambda_patch > 0)
    for ep in range(args.epochs):
        m.train()
        tot_total = tot_img = tot_patch = 0.0; nb = 0
        for batch in loader:
            if use_patch_loss:
                pix, y, mask_p = batch
                mask_p = mask_p.to(device)
            else:
                pix, y = batch
            pix = pix.to(device); y = y.to(device)
            out = m(pix)
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
                bce_pix = nn.functional.binary_cross_entropy_with_logits(
                    A, y_one, reduction='none')
                bce_pix = bce_pix * valid.unsqueeze(1).float()
                denom = valid.float().sum().clamp(min=1.0) * C
                loss_patch = bce_pix.sum() / denom
                loss = loss + float(args.lambda_patch) * loss_patch
                loss_patch_val = float(loss_patch)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_total += float(loss); tot_img += float(loss_img); tot_patch += loss_patch_val
            nb += 1
        msg = (f'  ep {ep+1:>2}  loss={tot_total/nb:.4f}  '
               f'cls={tot_img/nb:.4f}')
        if use_patch_loss:
            msg += f'  patch={tot_patch/nb:.4f}'
        msg += (f'  tau={float(m.log_tau.exp()):.3f}  '
                f'bias min/max={float(m.cls_bias.min()):.2f}/{float(m.cls_bias.max()):.2f}')
        print(msg)
        m.init_T_from_templates(tpl)        # refresh anchors with updated text_proj
        if args.eval_every > 0 and (ep+1) % args.eval_every == 0 and val_records is not None:
            run_eval(m, val_records, new_names, n_cls, orig_to_new,
                      args.bs, device, prefix=f' @ep{ep+1}')

    # Final eval
    if val_records is not None:
        result = run_eval(m, val_records, new_names, n_cls, orig_to_new,
                           args.bs, device, prefix=' FINAL')
    else:
        result = run_eval(m, records, new_names, n_cls, orig_to_new,
                           args.bs, device, prefix=' FINAL-on-train')

    # Save
    out_ds = args.out_dir / ds
    out_ds.mkdir(parents=True, exist_ok=True)
    ckpt = {
        'visual_proj': m.visual_proj.state_dict(),
        'text_proj':   m.text_proj.state_dict(),
        'log_tau':     m.log_tau.detach().cpu(),
        'cls_bias':    m.cls_bias.detach().cpu(),
        'T_anchor':    m.T_anchor.detach().cpu(),
        'class_names': new_names,
        'orig_class_names': orig_names,
        'orig_to_new': orig_to_new,
        'model': args.model, 'd_latent': 128, 'topk': args.topk,
        'vision_backbone': args.vision_backbone,
        'dinov2_input_size': args.dinov2_input_size if args.vision_backbone == 'dinov2' else None,
        'dataset': ds,
        'freeze_vision': bool(args.freeze_vision),
        'freeze_text':   bool(args.freeze_text),
    }
    # Persist trained encoder weights if they were unfrozen (so Stage 2 can load them)
    if (not args.freeze_vision) and args.vision_backbone == 'dinov2' and m._dinov2 is not None:
        ckpt['dinov2_state_dict']    = {k: v.detach().cpu() for k, v in m._dinov2.state_dict().items()}
    if (not args.freeze_vision) and args.vision_backbone == 'siglip' and m.vision_model is not None:
        ckpt['siglip_vision_state']  = {k: v.detach().cpu() for k, v in m.vision_model.state_dict().items()}
    if not args.freeze_text:
        ckpt['siglip_text_state']    = {k: v.detach().cpu() for k, v in m.text_model.state_dict().items()}
    torch.save(ckpt, out_ds / f'affinity_{ds}.pt')

    # Per-class CSV
    with open(out_ds / 'per_class_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['orig_id', 'orig_name', 'n_pos_eval', 'precision', 'recall', 'f1', 'ap'])
        for orig_id in keep_orig_ids:
            nid = orig_to_new[orig_id]
            ap_v = result['ap'][nid]
            w.writerow([orig_id, orig_names[orig_id], int(result['n_pos'][nid]),
                         round(result['prec'][nid], 4), round(result['rec'][nid], 4),
                         round(result['f1'][nid], 4),
                         (round(ap_v, 4) if not np.isnan(ap_v) else '')])
    print(f'[save] {out_ds}/affinity_{ds}.pt + per_class_metrics.csv')

    # Verbose per-class print
    print(f'\n=== {ds} per-class FINAL ===')
    print(f'  {"oid":>3}  {"orig name":<28}  {"n+":>4}  '
          f'{"prec":>5} {"rec":>5} {"F1":>5} {"AP":>5}')
    for orig_id in keep_orig_ids:
        nid = orig_to_new[orig_id]
        ap_v = result['ap'][nid]
        ap_str = f'{ap_v:.3f}' if not np.isnan(ap_v) else '  nan'
        print(f'  {orig_id:>3}  {orig_names[orig_id]:<28}  '
              f'{int(result["n_pos"][nid]):>4}  '
              f'{result["prec"][nid]:>5.2f} {result["rec"][nid]:>5.2f} '
              f'{result["f1"][nid]:>5.2f} {ap_str:>5}')

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
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--max-records', type=int, default=0)
    ap.add_argument('--vision-backbone', type=str, default='dinov2',
                    choices=['siglip', 'dinov2'])
    ap.add_argument('--dinov2-input-size', type=int, default=518)
    # Backbone fine-tune controls — defaults UNFROZEN per user directive
    ap.add_argument('--freeze-vision', action='store_true',
                    help='freeze DINOv2 / SigLIP-vision (default: unfrozen, trainable with --lr-vision)')
    ap.add_argument('--freeze-text',   action='store_true',
                    help='freeze SigLIP text encoder (default: unfrozen, trainable with --lr-text)')
    ap.add_argument('--lr-vision', type=float, default=1e-5,
                    help='lr for the visual backbone when unfrozen (kept small to avoid overfit)')
    ap.add_argument('--lr-text',   type=float, default=1e-5,
                    help='lr for the SigLIP text encoder when unfrozen')
    # Patch-level supervision (dense BCE on the affinity field)
    ap.add_argument('--lambda-patch', type=float, default=1.0,
                    help='weight of the patch-level BCE loss (0 to disable)')
    ap.add_argument('--patch-grid',   type=int, default=37,
                    help='patch-grid side used to downsample GT mask (37 for DINOv2@518)')
    args = ap.parse_args()
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
