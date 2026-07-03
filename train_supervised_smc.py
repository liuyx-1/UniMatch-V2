"""Fully-supervised (NO EMA) trainer fusing the 3 improvement modules
+ Self-Masked Consistency (SMC).

Modules (all on by default):
  * HPTA   — frozen-DINOv2 visual adapter (model.enable_visual_adapter)
  * LC-PAM — text/affinity side branch (util.affinity_side.AffinitySideBranch),
             jointly trained from dataset class names
  * EDGE   — RGB-edge residual adapter + Mask-Guided Edge Refiner (MGER)

Consistency WITHOUT EMA — Self-Masked Consistency (SMC):
  reference  p_full = sg[softmax(f(x))]            (model's OWN clean-view pred)
  current    p_mask = softmax(f(x ⊙ M))            (masked-view pred, with grad)
  gate       g = 1[H(p_full) < H(p_mask)]          (entropy gate, reused from TCR)
  L_smc = Σ (1-M)·g·|p_mask - p_full|_1 / Σ (1-M)·g   (loss only on masked pixels)

Total loss (per labeled image):
  L = CE(fuse(DPT(x), LC-PAM), Y)
      + α_aff·BCE(aff_cls, y_multi)
      + w_b·boundary(edge_logits, Y) + w_r·refined_edge(MGER, Y)
      + [epoch≥warmup] λ_smc·L_smc

Single-GPU (no DDP, no SyncBN, no EMA). Run directly:
  python train_supervised_smc.py --config configs/endovis2018_scene.yaml \
      --labeled-id-path .../labeled.txt --val-id-path .../val.txt \
      --test-id-path .../test.txt --save-path .../exp/<run> \
      --epochs 150 --batch-size 8 --lr 5e-5 --crop-size 490
"""
from __future__ import annotations
import argparse, logging, math, os, time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate, ResolutionBatchSampler
from util.classes import CLASSES, display_names
from util.utils import count_params, init_log, AverageMeter
from util.affinity_side import AffinitySideBranch
from util.edge_enhance import (EdgeSegResidualAdapter, MaskGuidedEdgeRefiner,
                               rgb_edge_prior, refined_edge_loss, edge_to_tokens)
from util.boundary_loss import boundary_loss
from util.temporal import entropy_gated_consistency
from util.combined_loss import CombinedCETversky
from util.instance_aware_loss import instance_aware_loss


# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser('No-EMA supervised + 3 modules + Self-Masked Consistency')
    p.add_argument('--config', required=True)
    p.add_argument('--labeled-id-path', required=True)
    p.add_argument('--val-id-path', default=None)
    p.add_argument('--test-id-path', default=None)
    p.add_argument('--save-path', required=True)
    # overrides
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--crop-size', type=int, default=None)
    p.add_argument('--grad-clip', type=float, default=1.0,
                   help='max global grad-norm for clipping (0 disables). Prevents the '
                        'loss-spike -> NaN -> all-background collapse seen mid-training.')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-workers', type=int, default=4)
    # module toggles (all ON by default)
    p.add_argument('--no-visual-adapter', dest='visual_adapter', action='store_false', default=True)
    p.add_argument('--visual-adapter-reduction', type=int, default=8)
    p.add_argument('--visual-adapter-dropout', type=float, default=0.0)
    p.add_argument('--no-lcpam', dest='lcpam', action='store_false', default=True)
    p.add_argument('--affinity-warmstart', default=None,
                   help='optional LC-PAM warmstart ckpt; else init from class names')
    p.add_argument('--require-affinity-warmstart', action='store_true',
                   help='fail fast when LC-PAM is enabled without a warmstart checkpoint')
    p.add_argument('--affinity-aux-weight', type=float, default=0.1)
    p.add_argument('--affinity-aux-warmup', type=int, default=5)
    p.add_argument('--affinity-aux-min-pixels', type=int, default=32,
                   help='min mask pixels for a class to count as present in the '
                        'image-level LC-PAM BCE target (aligns with the cls-head min_pixels)')
    p.add_argument('--no-match-prior-scale', action='store_true',
                   help='disable moment-matching the LC-PAM prior to the DPT logit '
                        'scale before fusion (ablation; default is matched)')
    p.add_argument('--no-edge', dest='edge', action='store_false', default=True)
    p.add_argument('--edge-boundary-weight', type=float, default=0.05)
    p.add_argument('--edge-boundary-pos-weight', type=float, default=8.0,
                   help='BCE positive weight for the seg-side aux boundary head; '
                        'boundaries are sparse (~5-10%%) so 4-8 prevents collapse')
    p.add_argument('--edge-refiner-weight', type=float, default=0.1)
    p.add_argument('--edge-refiner-thickness', type=int, default=3)
    p.add_argument('--edge-refiner-pos-weight', type=float, default=8.0)
    p.add_argument('--edge-dropout', type=float, default=0.0)
    p.add_argument('--edge-gate-dropout', type=float, default=0.0,
                   help='dropout on edge-gated token features; 0 disables hidden stochastic gating')
    p.add_argument('--edge-refiner-dropout', type=float, default=0.05,
                   help='dropout inside MGER; set 0 for fully deterministic edge refiner')
    p.add_argument('--edge-ignore-dilation', type=int, default=1,
                   help='pixels around ignore_index removed from edge/boundary targets')
    # Self-Masked Consistency
    p.add_argument('--no-smc', dest='smc', action='store_false', default=True)
    p.add_argument('--smc-weight', type=float, default=0.1)
    p.add_argument('--smc-ratio', type=float, default=0.6, help='fraction of patches masked')
    p.add_argument('--smc-block', type=int, default=3, help='mask block size in patches (×patch_size px)')
    p.add_argument('--smc-warmup', type=int, default=5)
    # main segmentation loss (paper: CE + Tversky)
    p.add_argument('--seg-loss', choices=['ce_tversky', 'ce'], default='ce_tversky',
                   help='ce_tversky = Surg-SegFormer paper loss (default); ce = plain CrossEntropy')
    p.add_argument('--tversky-alpha', type=float, default=0.7, help='Tversky FP weight (paper 0.7)')
    p.add_argument('--tversky-beta', type=float, default=0.3, help='Tversky FN weight (paper 0.3)')
    p.add_argument('--alpha-weight', type=float, default=0.5,
                   help='L = alpha_weight*Tversky + (1-alpha_weight)*CE (paper 0.5)')
    # instance-aware multi-class loss (Kundu et al. 2026 style; labeled-only,
    # this trainer has no unlabeled branch). Part of the proposed method
    # (default ON); --no-instance-loss is for ablation only.
    p.add_argument('--instance-loss', dest='instance_loss', action='store_true', default=True,
                   help='instance-aware (one-vs-rest + 2D connected-component) loss on '
                        'top of the main seg loss; targets long-tail/small-instrument '
                        'classes. Labeled-only (this trainer is fully-supervised). Default ON.')
    p.add_argument('--no-instance-loss', dest='instance_loss', action='store_false',
                   help='disable the instance-aware loss (ablation only).')
    p.add_argument('--instance-loss-weight', type=float, default=0.2)
    p.add_argument('--instance-loss-min-size', type=int, default=16)
    # eval
    p.add_argument('--eval-interval', type=int, default=1)
    p.add_argument('--eval-size', type=int, default=0)
    p.add_argument('--eval-batch-size', type=int, default=1,
                   help='val/test eval batch size; >1 batches same-resolution frames '
                        '(resolution-bucketed) to speed up validation on multi-video sets')
    p.add_argument('--exclude-classes', type=int, nargs='*', default=None,
                   help='classes dropped from present-only means (default: [0]=background '
                        'for endovis2017_* to match the paper, else none). Reporting/selection '
                        'only — never affects the loss.')
    # checkpoint/resume
    p.add_argument('--resume', default=None,
                   help='checkpoint path to resume; default auto-loads save-path/latest.pth when present')
    p.add_argument('--no-resume', action='store_true',
                   help='disable automatic resume from save-path/latest.pth')
    return p.parse_args()


def _unwrap(m):
    return m.module if hasattr(m, 'module') else m


def _move_optimizer_state(optimizer, device):
    for state in optimizer.state.values():
        for k, v in list(state.items()):
            if torch.is_tensor(v):
                state[k] = v.to(device)


def make_block_mask(B, H, W, patch, block, ratio, device):
    """Patch-aligned block mask, [B,1,H,W], 1 = keep, 0 = masked."""
    if H % patch != 0 or W % patch != 0:
        raise ValueError(f'SMC expects patch-aligned inputs, got HxW={H}x{W}, patch={patch}')
    ratio = min(max(float(ratio), 0.0), 1.0)
    gh, gw = H // patch, W // patch
    bph = max(1, int(block))
    nbh, nbw = math.ceil(gh / bph), math.ceil(gw / bph)
    keep = (torch.rand(B, 1, nbh, nbw, device=device) >= ratio).float()
    m = F.interpolate(keep, size=(gh, gw), mode='nearest')
    m = F.interpolate(m, size=(H, W), mode='nearest')
    return m


def round_to_multiple(x, multiple):
    return max(int(multiple), int(x / multiple + 0.5) * int(multiple))


# ----------------------------------------------------------------------
def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.Loader)
    if args.epochs:     cfg['epochs'] = int(args.epochs)
    if args.batch_size: cfg['batch_size'] = int(args.batch_size)
    if args.lr:         cfg['lr'] = float(args.lr)
    if args.crop_size:  cfg['crop_size'] = int(args.crop_size)

    logger = init_log('global', logging.INFO); logger.propagate = 0
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device(args.device)
    cudnn.enabled = True; cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    nclass = int(cfg['nclass'])
    dataset = cfg['dataset']
    class_names = CLASSES[dataset]
    assert len(class_names) == nclass, (len(class_names), nclass)
    disp_names = display_names(dataset, nclass)        # paper short names for printing

    # ---- model + 3 modules ----------------------------------------
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64,  'out_channels': [48, 96, 192, 384]},
        'base':  {'encoder_size': 'base',  'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    size = cfg['backbone'].split('_')[-1]
    model = DPT(**{**model_configs[size], 'nclass': nclass})
    ckpt_path = cfg.get('backbone_checkpoint') or f'./pretrained/{cfg["backbone"]}.pth'
    sd = torch.load(ckpt_path, map_location='cpu')
    if isinstance(sd, dict) and 'model' in sd: sd = sd['model']
    if isinstance(sd, dict) and 'state_dict' in sd: sd = sd['state_dict']
    model.backbone.load_state_dict(sd)

    use_va, use_lcpam, use_edge, use_smc = (args.visual_adapter, args.lcpam,
                                            args.edge, args.smc)
    if use_lcpam and args.require_affinity_warmstart and not args.affinity_warmstart:
        raise ValueError('--require-affinity-warmstart was set, but --affinity-warmstart is empty')
    if use_lcpam and not args.affinity_warmstart:
        logger.warning('LC-PAM has no warmstart checkpoint; anchors will be initialized '
                       'from class-name prompts and trained jointly. This is weaker than '
                       'a validated warmstart and may require cached/network SigLIP weights.')
    if use_va:
        model.enable_visual_adapter(reduction=args.visual_adapter_reduction,
                                    dropout=args.visual_adapter_dropout)
        model.lock_backbone()                      # HPTA = frozen-DINO PEFT
    if use_edge:
        model.edge_seg_adapter = EdgeSegResidualAdapter(model.backbone.embed_dim,
                                                        dropout=float(args.edge_dropout),
                                                        gate_dropout=float(args.edge_gate_dropout))
        model.edge_refiner = MaskGuidedEdgeRefiner(mid=16,
                                                   dropout=float(args.edge_refiner_dropout))
    model = model.to(device)
    embed_dim = int(model.backbone.embed_dim)
    patch_size = int(getattr(model.backbone, 'patch_size', 14))
    if int(cfg['crop_size']) % patch_size != 0:
        raise ValueError(
            f"crop_size={cfg['crop_size']} must be a multiple of DINOv2 patch_size={patch_size}. "
            f"Use e.g. {round_to_multiple(int(cfg['crop_size']), patch_size)}.")

    affinity_side = None
    if use_lcpam:
        affinity_side = AffinitySideBranch(
            args.affinity_warmstart, n_orig_classes=nclass,
            class_names=class_names, dataset=dataset,
            match_prior_scale=not args.no_match_prior_scale)
        if use_edge:
            affinity_side.enable_edge_enhance(dropout=float(args.edge_dropout),
                                              gate_dropout=float(args.edge_gate_dropout))
        affinity_side.unfreeze_proj()              # joint text training
        affinity_side = affinity_side.to(device)

    # ---- optimizer (no EMA) ---------------------------------------
    lr, lr_multi = cfg['lr'], cfg['lr_multi']
    groups = []
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    if bb:
        groups.append({'params': bb, 'lr': lr})
    groups.append({'params': [p for n, p in model.named_parameters()
                              if 'backbone' not in n and p.requires_grad],
                   'lr': lr * lr_multi})
    if affinity_side is not None:
        groups.append({'params': [p for p in affinity_side.parameters() if p.requires_grad],
                       'lr': lr * lr_multi})
    optimizer = AdamW(groups, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    base_lrs = [g['lr'] for g in optimizer.param_groups]   # for poly schedule

    logger.info('modules: HPTA=%s  LC-PAM=%s  EDGE/MGER=%s  SMC=%s' %
                (use_va, use_lcpam, use_edge, use_smc))
    total_params = count_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if affinity_side is not None:
        total_params += sum(p.numel() for p in affinity_side.parameters()) / 1e6
        trainable_params += sum(p.numel() for p in affinity_side.parameters()
                                if p.requires_grad)
    logger.info('Total %.1fM  trainable %.1fM' %
                (total_params, trainable_params / 1e6))

    ignore_index = int(cfg.get('criterion', {}).get('kwargs', {}).get('ignore_index', 255))
    if args.seg_loss == 'ce_tversky':
        criterion = CombinedCETversky(
            num_classes=nclass, alpha_weight=args.alpha_weight,
            tversky_alpha=args.tversky_alpha, tversky_beta=args.tversky_beta,
            ignore_index=ignore_index).to(device)
        logger.info('seg loss: paper CE+Tversky  alpha_weight=%.2f  tversky(a=%.2f,b=%.2f)'
                    % (args.alpha_weight, args.tversky_alpha, args.tversky_beta))
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device)
        logger.info('seg loss: plain CrossEntropy (ignore_index=%d)' % ignore_index)

    # ---- data (labeled only; SMC reuses the same images) ----------
    trainset = SemiDataset(dataset, cfg['data_root'], 'train_l',
                           cfg['crop_size'], args.labeled_id_path)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True,
                             pin_memory=True, num_workers=args.num_workers, drop_last=True)
    valset = SemiDataset(dataset, cfg['data_root'], 'val', id_path=args.val_id_path)
    _ebs = max(1, int(args.eval_batch_size))
    valloader = DataLoader(valset, batch_sampler=ResolutionBatchSampler(valset, _ebs),
                           pin_memory=True, num_workers=2)
    testloader = None
    if args.test_id_path and os.path.isfile(args.test_id_path) \
            and os.path.realpath(args.test_id_path) != os.path.realpath(args.val_id_path or ''):
        testset = SemiDataset(dataset, cfg['data_root'], 'val', id_path=args.test_id_path)
        testloader = DataLoader(testset, batch_sampler=ResolutionBatchSampler(testset, _ebs),
                                pin_memory=True, num_workers=2)

    eval_mode = 'sliding_window' if dataset == 'cityscapes' else 'original'
    total_iters = len(trainloader) * cfg['epochs']
    previous_best, best_epoch = 0.0, 0
    start_epoch = 0

    # present-only reporting: drop absent classes; for endovis2017_* also drop
    # background (class 0) to match the paper's 7-instrument mIoU convention.
    if args.exclude_classes is None:
        exclude_classes = [0] if dataset.startswith('endovis2017') else []
    else:
        exclude_classes = list(args.exclude_classes)
    logger.info('present-only means exclude classes: %s', exclude_classes or 'none')

    resume_path = args.resume
    if resume_path is None and not args.no_resume:
        auto_resume = os.path.join(args.save_path, 'latest.pth')
        if os.path.isfile(auto_resume):
            resume_path = auto_resume
    if resume_path:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f'resume checkpoint not found: {resume_path}')
        ckpt = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        if affinity_side is not None:
            if ckpt.get('affinity_side') is not None:
                _unwrap(affinity_side).load_state_dict_from_save(ckpt['affinity_side'])
            else:
                logger.warning('resume checkpoint has no affinity_side state; LC-PAM keeps current init')
        if ckpt.get('optimizer') is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            _move_optimizer_state(optimizer, device)
        else:
            logger.warning('resume checkpoint has no optimizer state; optimizer restarts at configured LR')
        previous_best = float(ckpt.get('previous_best', 0.0))
        best_epoch = int(ckpt.get('best_epoch', ckpt.get('epoch', -1)))
        start_epoch = int(ckpt.get('epoch', -1)) + 1
        logger.info('resumed from %s: start_epoch=%d/%d previous_best=%.2f @ep%d' %
                    (resume_path, start_epoch + 1, cfg['epochs'], previous_best,
                     best_epoch + 1 if best_epoch >= 0 else 0))
    if start_epoch >= cfg['epochs']:
        logger.warning('checkpoint epoch already reaches configured epochs=%d; nothing to train',
                       cfg['epochs'])
        return

    def _edge(img, with_grad):
        if not use_edge:
            return None
        raw = rgb_edge_prior(img)
        ref = getattr(_unwrap(model), 'edge_refiner', None)
        if ref is None:
            return raw
        if with_grad:
            return ref(img, raw)
        with torch.no_grad():
            return ref(img, raw)

    def _fuse_with_affinity(img, logits, patches, edge_prior):
        if affinity_side is None:
            return logits, None
        ph, pw = img.shape[-2] // patch_size, img.shape[-1] // patch_size
        side = affinity_side(patches, logits.shape[-2], logits.shape[-1],
                             patch_hw=(ph, pw), edge_prior=edge_prior)

        # Warmstarted CMA / hyperbolic-pathway checkpoints produce an X-aware
        # patch stream. Decode it before logit fusion so train/eval matches
        # the standalone test.py inference path.
        if getattr(_unwrap(affinity_side), 'affinity_metric', '') in ('hyperbolic_pathway', 'cma'):
            x_aware = getattr(_unwrap(affinity_side), 'last_x_aware', None)
            if x_aware is not None:
                if use_edge and edge_prior is not None and hasattr(_unwrap(model), 'edge_seg_adapter'):
                    x_aware, edge_boundary_logits = _unwrap(model).edge_seg_adapter(
                        x_aware, edge_to_tokens(edge_prior, ph, pw), ph, pw)
                    _unwrap(model).last_edge_boundary_logits = edge_boundary_logits
                feats, _, _, _ = model.encode(img, return_cls=False)
                feats = list(model.adapt_features(feats, ph, pw))
                feats[-1] = x_aware
                logits = model.decode(feats, ph, pw, comp_drop=False)
        return affinity_side.fuse(logits, side), side

    def _forward_full(img, edge_prior=None, return_side=False):
        logits, _, patches = model(img, return_cls=True, edge_prior=edge_prior)
        logits, side = _fuse_with_affinity(img, logits, patches, edge_prior)
        if return_side:
            return logits, side, patches
        return logits

    def _eval_forward(img):
        edge_eval = _edge(img, with_grad=False)
        return _forward_full(img, edge_prior=edge_eval, return_side=False)

    # ---- train ----------------------------------------------------
    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        if affinity_side is not None:
            affinity_side.train()
        mtr = {k: AverageMeter() for k in ('loss', 'x', 'aff', 'edge', 'smc', 'inst')}
        it = trainloader
        if tqdm is not None:
            it = tqdm(trainloader, desc='Epoch %d/%d' % (epoch + 1, cfg['epochs']),
                      dynamic_ncols=True, mininterval=1.0)

        for i, (img, mask) in enumerate(it):
            img, mask = img.to(device), mask.to(device)
            B, _, H, W = img.shape

            # ---- clean forward (supervised) ----
            edge_x = _edge(img, with_grad=True)
            pred, side, patches = _forward_full(img, edge_prior=edge_x, return_side=True)
            edge_logits = getattr(_unwrap(model), 'last_edge_boundary_logits', None)

            aff_cls_logits = None
            if side is not None:
                aff_cls_logits = side['cls_logits']

            loss_x = criterion(pred, mask)
            loss = loss_x

            # ---- Instance-aware multi-class loss (Kundu et al. 2026 style) ----
            # One-vs-rest + 2D connected-component decomposition, uniform
            # class/instance averaging. Additive on top of loss_x, using clean
            # GT (this trainer has no unlabeled branch / pseudo-labels).
            loss_inst = torch.zeros((), device=device)
            if getattr(args, 'instance_loss', False):
                inst_class_ids = [c for c in range(nclass) if c not in exclude_classes]
                loss_inst = instance_aware_loss(
                    pred, mask, inst_class_ids, ignore_index=ignore_index,
                    min_component_size=int(args.instance_loss_min_size),
                    tversky_alpha=float(args.tversky_alpha),
                    tversky_beta=float(args.tversky_beta),
                    alpha_weight=float(args.alpha_weight))
                loss = loss + float(args.instance_loss_weight) * loss_inst

            # ---- LC-PAM image-level aux BCE ----
            loss_aff = torch.zeros((), device=device)
            if aff_cls_logits is not None:
                aw = max(1, args.affinity_aux_warmup)
                w_aff = (args.affinity_aux_weight * 0.5 *
                         (1 - math.cos(math.pi * min(1.0, epoch / aw))))
                idx_o2n = _unwrap(affinity_side).idx_orig_to_new.tolist()
                y = torch.zeros(B, aff_cls_logits.shape[-1], device=device)
                min_px = max(1, int(args.affinity_aux_min_pixels))
                for c in range(nclass):
                    j = idx_o2n[c]
                    if j < 0:
                        continue
                    y[:, j] = ((mask == c).flatten(1).sum(1) >= min_px).float()
                loss_aff = F.binary_cross_entropy_with_logits(aff_cls_logits, y)
                if w_aff > 0:
                    loss = loss + w_aff * loss_aff

            # ---- EDGE boundary + MGER supervision (labeled GT) ----
            loss_edge = torch.zeros((), device=device)
            if use_edge and edge_logits is not None:
                loss_edge = args.edge_boundary_weight * boundary_loss(
                    edge_logits, mask, nclass, ignore_index=ignore_index,
                    pos_weight=args.edge_boundary_pos_weight, dice_weight=0.5,
                    ignore_dilation=args.edge_ignore_dilation)
                if edge_x is not None and edge_x.requires_grad:
                    loss_edge = loss_edge + args.edge_refiner_weight * refined_edge_loss(
                        edge_x, mask, nclass, thickness=args.edge_refiner_thickness,
                        ignore_index=ignore_index,
                        pos_weight=args.edge_refiner_pos_weight, dice_weight=0.5,
                        ignore_dilation=args.edge_ignore_dilation)
                loss = loss + loss_edge

            # ---- Self-Masked Consistency (no EMA) ----
            loss_smc = torch.zeros((), device=device)
            if use_smc and epoch >= args.smc_warmup:
                M = make_block_mask(B, H, W, patch_size, args.smc_block, args.smc_ratio, device)
                masked = img * M
                # EDGE for the masked view: do NOT re-run Sobel on the masked
                # composite (its hard 0-block borders create spurious "grid"
                # edges that the edge pathway latches onto). Instead reuse the
                # CLEAN refined edge (detached) and zero it inside masked blocks
                # -> no artifacts, and occluded regions carry no edge cue either.
                edge_m = (edge_x.detach() * M) if (use_edge and edge_x is not None) else None
                pred_m = _forward_full(masked, edge_prior=edge_m, return_side=False)
                p_mask = pred_m.softmax(1)
                p_full = pred.detach().softmax(1)
                ig = (1.0 - M[:, 0]) * (mask != ignore_index).float()  # masked & valid
                loss_smc = entropy_gated_consistency(p_mask, p_full, ignore_mask=ig)
                loss = loss + args.smc_weight * loss_smc

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g['params']],
                    max_norm=float(args.grad_clip))
            optimizer.step()

            # poly LR (5% floor), per-group base preserved
            cur = epoch * len(trainloader) + i
            scale = max(0.05, (1 - cur / max(total_iters, 1)) ** 0.9)
            for gi, g in enumerate(optimizer.param_groups):
                g['lr'] = base_lrs[gi] * scale

            mtr['loss'].update(loss.item()); mtr['x'].update(loss_x.item())
            mtr['aff'].update(float(loss_aff)); mtr['edge'].update(float(loss_edge))
            mtr['smc'].update(float(loss_smc)); mtr['inst'].update(float(loss_inst))
            if tqdm is not None and hasattr(it, 'set_postfix'):
                it.set_postfix(loss='%.3f' % mtr['loss'].avg, x='%.3f' % mtr['x'].avg,
                               smc='%.3f' % mtr['smc'].avg, aff='%.3f' % mtr['aff'].avg,
                               edge='%.3f' % mtr['edge'].avg, inst='%.3f' % mtr['inst'].avg,
                               lr='%.2e' % optimizer.param_groups[-1]['lr'])

        # ---- eval ----
        do_eval = (epoch % max(1, args.eval_interval) == 0) or (epoch == cfg['epochs'] - 1)
        if not do_eval:
            continue
        model.eval()
        if affinity_side is not None:
            affinity_side.eval()
        (mIoU, iou_c, allAcc, acc_c, mAcc, dice_c, mDice,
         mIoU_p, mDice_p, mAcc_p, present) = evaluate(
            model, valloader, eval_mode, cfg, multiplier=patch_size,
            return_acc=True, max_size=int(args.eval_size), device=device,
            forward_fn=_eval_forward, ignore_index=ignore_index,
            return_present=True, exclude_classes=exclude_classes)
        tag = 'val(copied)'
        if testloader is not None:
            (tmIoU, tiou, tallAcc, tacc, tmAcc, tdice, tmDice,
             tmIoU_p, tmDice_p, tmAcc_p, _) = evaluate(
                model, testloader, eval_mode, cfg, multiplier=patch_size,
                return_acc=True, max_size=int(args.eval_size), device=device,
                forward_fn=_eval_forward, ignore_index=ignore_index,
                return_present=True, exclude_classes=exclude_classes)
            tag = 'test'
        else:
            tmIoU_p, tmDice_p, tallAcc = mIoU_p, mDice_p, allAcc

        logger.info('[eval] ep=%d/%d  [val] mIoU(pres)=%.2f mDice(pres)=%.2f PixAcc=%.2f '
                    '| mIoU(all)=%.2f mAcc(pres)=%.2f  [%s] mIoU(pres)=%.2f mDice(pres)=%.2f PixAcc=%.2f'
                    % (epoch + 1, cfg['epochs'], mIoU_p, mDice_p, allAcc, mIoU, mAcc_p,
                       tag, tmIoU_p, tmDice_p, tallAcc))
        for c in range(nclass):
            flag = '' if present[c] else ' [absent]'
            flag += '' if (c not in exclude_classes) else ' [excl]'
            logger.info('    cls %d %-10s IoU=%.2f Dice=%.2f PixAcc=%.2f%s'
                        % (c, disp_names[c], iou_c[c], dice_c[c], acc_c[c], flag))

        # select best by present-only val mIoU (the reported headline metric)
        is_best = mIoU_p >= previous_best
        if is_best:
            previous_best, best_epoch = mIoU_p, epoch
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'previous_best': previous_best,
                'best_epoch': best_epoch, 'cfg_dataset': dataset, 'nclass': nclass,
                'config': cfg, 'args': vars(args),
                'edge': {'enabled': use_edge,
                         'dropout': args.edge_dropout,
                         'gate_dropout': args.edge_gate_dropout,
                         'refiner_dropout': args.edge_refiner_dropout,
                         'ignore_dilation': args.edge_ignore_dilation}}
        if affinity_side is not None:
            ckpt['affinity_side'] = _unwrap(affinity_side).state_dict_for_save()
        if use_va:
            ckpt['visual_adapter'] = {'enabled': True,
                                      'reduction': args.visual_adapter_reduction,
                                      'dropout': args.visual_adapter_dropout}
        torch.save(ckpt, os.path.join(args.save_path, 'latest.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(args.save_path, 'best.pth'))
        logger.info('best mIoU=%.2f @ep%d' % (previous_best, best_epoch + 1))


if __name__ == '__main__':
    main()
