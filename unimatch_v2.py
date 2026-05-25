import argparse
from collections import deque
from copy import deepcopy
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

# Surgical long-tail extensions (only loaded when enabled)
from util.debias import PriorEstimator, adjust_logits
from util.boundary_loss import BoundaryAuxHead, boundary_loss
from util.boundary_loss_manifold import BoundaryAuxHeadTangent, boundary_loss_with_tangent
from util.bias_metrics import compute_train_pixel_freq, split_head_body_tail
from util.cls_head import (build_cls_head, build_cls_loss,
                             cls_target_from_mask, cls_loss as _bce_cls_loss)
import torch.distributed as dist


parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--val-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
# Variant toggles
parser.add_argument('--debias',   dest='debias_enabled',   action='store_true', default=None)
parser.add_argument('--boundary', dest='boundary_enabled', action='store_true', default=None)
parser.add_argument('--cls-head', dest='cls_enabled',      action='store_true', default=None)
parser.add_argument('--tangent', dest='tangent_enabled', action='store_true', default=None)
parser.add_argument('--tangent-weight', type=float, default=None)
# Stage-1 affinity prior side branch
parser.add_argument('--affinity-warmstart', type=str, default=None,
                    help='Path to Stage 1 affinity_<ds>.pt. Enables the side branch.')
parser.add_argument('--affinity-aux-weight', type=float, default=0.5,
                    help='Final weight of the image-level cls BCE aux loss from affinity head.')
parser.add_argument('--affinity-freeze-warmup', type=int, default=15,
                    help='Min epochs to keep affinity proj frozen before plateau-triggered unfreeze.')
parser.add_argument('--affinity-plateau-window', type=int, default=5,
                    help='Number of consecutive val epochs used for plateau detection.')
parser.add_argument('--affinity-plateau-eps', type=float, default=0.001,
                    help='mIoU plateau threshold (fractional, 0.001 = 0.1%).')
parser.add_argument('--affinity-unfreeze-lr-mult', type=float, default=1.0,
                    help='Multiplier on backbone lr for unfrozen affinity proj group.')
parser.add_argument('--affinity-aux-warmup', type=int, default=5,
                    help='Epochs to ramp affinity-aux-weight from 0 to its target value.')
# Hyperparameter overrides
parser.add_argument('--batch-size', type=int,   default=None)
parser.add_argument('--lr',         type=float, default=None)
parser.add_argument('--epochs',     type=int,   default=None)
parser.add_argument('--crop-size',  type=int,   default=None)


def main():
    args = parser.parse_args()
    # Anomaly detection ONLY when AFFINITY_DEBUG=1; cheap-ish so 1 epoch is enough to locate
    import os as _os
    if _os.environ.get('AFFINITY_DEBUG', '') == '1':
        torch.autograd.set_detect_anomaly(True)
        print('[debug] autograd anomaly detection ON')

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    # --- CLI overrides ---
    if args.debias_enabled is not None:
        cfg.setdefault('debias', {})['enabled'] = bool(args.debias_enabled)
    if args.boundary_enabled is not None:
        cfg.setdefault('boundary', {})['enabled'] = bool(args.boundary_enabled)
    if args.cls_enabled is not None:
        cfg.setdefault('cls', {})['enabled'] = bool(args.cls_enabled)
    if args.tangent_enabled is not None:
        cfg.setdefault('boundary', {})['tangent_enabled'] = bool(args.tangent_enabled)
    if args.tangent_weight is not None:
        cfg.setdefault('boundary', {})['tangent_weight'] = float(args.tangent_weight)
    if args.batch_size is not None: cfg['batch_size'] = int(args.batch_size)
    if args.lr         is not None: cfg['lr']         = float(args.lr)
    if args.epochs     is not None: cfg['epochs']     = int(args.epochs)
    if args.crop_size  is not None: cfg['crop_size']  = int(args.crop_size)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    ckpt_path = cfg.get('backbone_checkpoint') or f'./pretrained/{cfg["backbone"]}.pth'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    model.backbone.load_state_dict(state_dict)
    # If Stage 1 trained the visual backbone, prefer its weights as a warm-start.
    # NOTE: model_ema is built via deepcopy(model) AFTER this block, so the
    # EMA teacher automatically inherits the Stage 1 backbone weights too.
    if args.affinity_warmstart:
        try:
            _aff = torch.load(args.affinity_warmstart, map_location='cpu', weights_only=False)
            if 'dinov2_state_dict' in _aff:
                missing, unexpected = model.backbone.load_state_dict(_aff['dinov2_state_dict'], strict=False)
                if rank == 0:
                    logger.info('[affinity] backbone warm-started from Stage 1 ckpt  '
                                'missing=%d  unexpected=%d  '
                                '(model_ema will inherit via deepcopy below)'
                                % (len(missing), len(unexpected)))
            elif rank == 0:
                logger.info('[affinity] Stage 1 ckpt has no dinov2_state_dict '
                            '(was --freeze-vision); using default pretrained backbone.')
        except Exception as e:
            if rank == 0:
                logger.warning('[affinity] could not load dinov2_state_dict from %s: %s'
                                % (args.affinity_warmstart, e))
        
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none', ignore_index=255).cuda(local_rank)

    trainset_u = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    )
    trainset_l = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val', id_path=args.val_id_path
    )
    
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    )
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    )
    
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    )
    
    # --- Debias module (long-tail logit adjustment + EMA prior) ---
    debias_cfg = cfg.get('debias') or {}
    use_debias = bool(debias_cfg.get('enabled', False))
    if use_debias:
        init_arg = debias_cfg.get('init_from_labeled', False)
        init_path = args.labeled_id_path if init_arg is True else (init_arg if isinstance(init_arg, str) else None)
        prior = PriorEstimator(num_classes=cfg['nclass'],
                                decay=float(debias_cfg.get('running_decay', 0.99)),
                                init_from_labeled=init_path,
                                data_root=cfg['data_root']).cuda(local_rank)
        debias_tau_target  = float(debias_cfg.get('tau', 1.0))
        debias_tau_warmup  = int(debias_cfg.get('tau_warmup_epochs', 0))
        prior_conf_thresh  = float(debias_cfg.get('prior_conf_thresh', cfg.get('conf_thresh', 0.95)))
        debias_sync_ddp    = bool(debias_cfg.get('sync_ddp', True))
        if rank == 0:
            logger.info('[debias] enabled  tau=%.2f (warmup=%d ep)  decay=%.3f' %
                        (debias_tau_target, debias_tau_warmup,
                          float(debias_cfg.get('running_decay', 0.99))))
    else:
        prior = None

    # --- Cls aux head (image-level multi-label, on backbone CLS token) ---
    cls_cfg = cfg.get('cls') or {}
    use_cls = bool(cls_cfg.get('enabled', False))
    if use_cls:
        embed_dim = int(model.module.backbone.embed_dim)
        cls_head = build_cls_head(embed_dim, cfg['nclass'], cls_cfg).cuda(local_rank)
        cls_head = torch.nn.parallel.DistributedDataParallel(
            cls_head, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)
        optimizer.add_param_group({'params': cls_head.parameters(),
                                     'lr': cfg['lr'] * cfg['lr_multi']})
        cls_loss_fn  = build_cls_loss(cls_cfg)             # None when loss_type='bce'
        cls_alpha    = float(cls_cfg.get('alpha', 0.1))
        cls_min_pixels = int(cls_cfg.get('min_pixels', 32))
        cls_pw_tau   = float(cls_cfg.get('pos_weight_tau', 0.0))
        cls_detach   = bool(cls_cfg.get('detach_backbone', True))
        cls_warmup   = int(cls_cfg.get('warmup_epochs', 5))
        cls_expects  = getattr(cls_head.module, 'expects', 'cls')
        if rank == 0:
            logger.info(
                '[cls] enabled  decoder=%s  loss=%s  alpha=%.3f  warmup=%d ep  '
                'detach_backbone=%s  expects=%s' %
                (cls_cfg.get('decoder_type', 'ml_decoder'),
                  cls_cfg.get('loss_type',    'asl'),
                  cls_alpha, cls_warmup, cls_detach, cls_expects))
    else:
        cls_head = None

    # --- Boundary aux head ---
    boundary_cfg = cfg.get('boundary') or {}
    use_boundary = bool(boundary_cfg.get('enabled', False))
    if bool(boundary_cfg.get('tangent_enabled', False)) and not use_boundary:
        raise ValueError('--tangent / boundary.tangent_enabled requires --boundary / boundary.enabled')
    use_tangent = use_boundary and bool(boundary_cfg.get('tangent_enabled', False))
    tangent_weight = float(boundary_cfg.get('tangent_weight', 0.1)) if use_tangent else 0.0
    tangent_detach_seg = bool(boundary_cfg.get('tangent_detach_seg', True))
    if use_boundary:
        head_cls = BoundaryAuxHeadTangent if use_tangent else BoundaryAuxHead
        boundary_pos_weight  = float(boundary_cfg.get('pos_weight', 20.0))
        boundary_head = head_cls(cfg['nclass'],
                                  pos_weight=boundary_pos_weight).cuda(local_rank)
        boundary_head = torch.nn.parallel.DistributedDataParallel(
            boundary_head, device_ids=[local_rank], output_device=local_rank)
        optimizer.add_param_group({'params': boundary_head.parameters(),
                                     'lr': cfg['lr'] * cfg['lr_multi']})
        # pos_weight 现在作为 head 的 buffer (boundary_head.module.pos_weight),
        # 在每次 boundary_loss 调用时直接传引用,不再每步重建标量 tensor。
        boundary_alpha       = float(boundary_cfg.get('alpha', 0.1))
        boundary_dice_weight = float(boundary_cfg.get('dice_weight', 0.5))
        boundary_warmup      = int(boundary_cfg.get('warmup_epochs', 3))
        boundary_seam_thick  = int(boundary_cfg.get('cutmix_edge_thickness', 2))
        if rank == 0:
            logger.info('[boundary] enabled  alpha=%.3f  pos_weight=%.1f  '
                        'dice_weight=%.2f  warmup=%d ep  seam_thick=%d%s' %
                        (boundary_alpha, boundary_pos_weight,
                          boundary_dice_weight, boundary_warmup, boundary_seam_thick,
                          f'  +tangent (w={tangent_weight:.3f}, detach_seg={tangent_detach_seg})'
                          if use_tangent else ''))
    else:
        boundary_head = None

    # --- Stage-1 Affinity Prior side branch ---
    use_affinity = bool(args.affinity_warmstart)
    affinity_side = None
    affinity_proj_param_group_idx = None      # set when proj is unfrozen (added param group)
    miou_window = deque(maxlen=int(args.affinity_plateau_window))
    affinity_pos_weight = None
    affinity_class_freq = None
    if use_affinity:
        from util.affinity_side import AffinitySideBranch
        affinity_side = AffinitySideBranch(args.affinity_warmstart,
                                            n_orig_classes=cfg['nclass']).cuda(local_rank)
        affinity_side = torch.nn.parallel.DistributedDataParallel(
            affinity_side, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, broadcast_buffers=False,
        )
        # gate_conv is always trainable — add to optimizer immediately
        optimizer.add_param_group({'params': affinity_side.module.gate_conv.parameters(),
                                     'lr': cfg['lr'] * cfg['lr_multi']})
        if rank == 0:
            logger.info(
                '[affinity] side branch enabled  ckpt=%s  freeze_warmup=%d ep  '
                'plateau_window=%d  plateau_eps=%.4f  aux_weight_target=%.2f  '
                'aux_warmup=%d ep' % (
                    args.affinity_warmstart,
                    int(args.affinity_freeze_warmup),
                    int(args.affinity_plateau_window),
                    float(args.affinity_plateau_eps),
                    float(args.affinity_aux_weight),
                    int(args.affinity_aux_warmup)))

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1

    hbt_split = {'head': [], 'body': [], 'tail': []}
    if rank == 0:
        try:
            tf = compute_train_pixel_freq(
                args.labeled_id_path, cfg['data_root'], cfg['nclass'],
                ignore_index=255,
                cache_path=os.path.join(args.save_path, 'train_pixel_freq.json'),
            )
            hbt_split = split_head_body_tail(tf, exclude=None, n_groups=3)
            logger.info('[bias-log] head=%s  body=%s  tail=%s' %
                        (hbt_split['head'], hbt_split['body'], hbt_split['tail']))
        except Exception as e:
            logger.warning('[bias-log] failed to derive H/B/T split: %s' % e)
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        if use_boundary and 'boundary_head' in checkpoint:
            boundary_head.load_state_dict(checkpoint['boundary_head'])
        if use_debias and 'debias_prior' in checkpoint:
            prior.load_state_dict(checkpoint['debias_prior'])
        if use_cls and 'cls_head' in checkpoint:
            cls_head.load_state_dict(checkpoint['cls_head'])
        if affinity_side is not None and 'affinity_side' in checkpoint:
            affinity_side.module.load_state_dict_from_save(checkpoint['affinity_side'])
            if checkpoint.get('affinity_proj_unfrozen', False):
                # restore the extra param group
                proj_params = affinity_side.module.proj_parameters()
                optimizer.add_param_group({'params': proj_params,
                                             'lr': cfg['lr'] * float(args.affinity_unfreeze_lr_mult)})
                affinity_proj_param_group_idx = len(optimizer.param_groups) - 1
            for v in checkpoint.get('affinity_miou_window', []):
                miou_window.append(float(v))

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_b = AverageMeter()
        total_loss_tan = AverageMeter()
        total_loss_c = AverageMeter()
        total_loss_aff = AverageMeter()
        total_gate     = AverageMeter()
        total_mask_ratio = AverageMeter()
        counter_device = torch.device('cuda', local_rank)
        pseudo_kept_per_class = torch.zeros(cfg['nclass'], dtype=torch.long, device=counter_device)
        pseudo_total_per_class = torch.zeros(cfg['nclass'], dtype=torch.long, device=counter_device)

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        
        model.train()

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            
            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                if use_debias:
                    if debias_tau_warmup > 0 and epoch < debias_tau_warmup:
                        tau_eff = debias_tau_target * (epoch + i / max(len(trainloader_u), 1)) / max(debias_tau_warmup, 1)
                    else:
                        tau_eff = debias_tau_target
                    pred_u_w_adj = adjust_logits(pred_u_w, prior.get_log_prior(), tau=tau_eff)
                    conf_u_w = pred_u_w_adj.softmax(dim=1).max(dim=1)[0]
                    mask_u_w = pred_u_w_adj.argmax(dim=1)
                    raw_conf  = pred_u_w.softmax(dim=1).max(dim=1)[0]
                    raw_pseudo = pred_u_w.argmax(dim=1)
                    valid_for_prior = (raw_conf >= prior_conf_thresh) & (ignore_mask != 255)
                    prior.update_from_pseudo(raw_pseudo, valid_for_prior, sync_ddp=debias_sync_ddp)
                else:
                    conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                    mask_u_w = pred_u_w.argmax(dim=1)
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            # When affinity_side OR use_cls is on, request patches from DPT's backbone.
            need_patches = (use_cls or affinity_side is not None)
            if need_patches:
                pred_x, cls_tok_x, patches_x = model(img_x, return_cls=True)
            else:
                pred_x = model(img_x)

            if need_patches:
                pred_u, _, patches_u = model(torch.cat((img_u_s1, img_u_s2)),
                                              comp_drop=True, return_cls=True)
                pred_u_s1, pred_u_s2 = pred_u.chunk(2)
                patches_u1, patches_u2 = patches_u.chunk(2)
            else:
                pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)),
                                              comp_drop=True).chunk(2)

            # ---- Affinity side branch: fuse prior into supervised + unlabeled student logits.
            # pred_u_w (teacher) is INTENTIONALLY NOT fused, to keep pseudo-label generation clean.
            affinity_cls_logits_x = None
            if affinity_side is not None:
                H_x, W_x = pred_x.shape[-2:]
                side_x = affinity_side(patches_x, H_x, W_x)
                pred_x = affinity_side.module.fuse(pred_x, side_x)
                affinity_cls_logits_x = side_x['cls_logits']            # for aux BCE

                H_u, W_u = pred_u_s1.shape[-2:]
                side_u1 = affinity_side(patches_u1, H_u, W_u)
                side_u2 = affinity_side(patches_u2, H_u, W_u)
                pred_u_s1 = affinity_side.module.fuse(pred_u_s1, side_u1)
                pred_u_s2 = affinity_side.module.fuse(pred_u_s2, side_u2)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            gate1 = ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255)).float()
            denom1 = (ignore_mask_cutmixed1 != 255).float().sum().clamp(min=1.0)
            loss_u_s1 = (loss_u_s1 * gate1).sum() / denom1

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            gate2 = ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255)).float()
            denom2 = (ignore_mask_cutmixed2 != 255).float().sum().clamp(min=1.0)
            loss_u_s2 = (loss_u_s2 * gate2).sum() / denom2
            
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0

            loss = (loss_x + loss_u_s) / 2.0

            loss_c = torch.zeros((), device=loss.device)
            if use_cls and epoch >= cls_warmup:
                # Pick feature source: 'cls' (MLP head) or 'patches' (ML-Decoder)
                feat = patches_x if cls_expects == 'patches' else cls_tok_x
                if cls_detach:
                    feat = feat.detach()
                cls_logits_x = cls_head(feat)
                cls_tgt_x = cls_target_from_mask(mask_x, cfg['nclass'],
                                                   min_pixels=cls_min_pixels)
                if cls_loss_fn is not None:                # ASL or any module-style loss
                    loss_c = cls_loss_fn(cls_logits_x, cls_tgt_x)
                else:                                       # BCE (optionally prior-weighted)
                    pw = None
                    if use_debias and cls_pw_tau > 0.0:
                        p  = prior.get_prior().clamp(min=1e-4).to(cls_logits_x.device)
                        pw = (1.0 / p) ** cls_pw_tau
                        pw = pw / pw.mean()
                    loss_c = _bce_cls_loss(cls_logits_x, cls_tgt_x, pos_weight=pw)
                loss = loss + cls_alpha * loss_c

            # ---- Affinity image-level cls BCE aux loss (cosine warmup) ----
            loss_aff_cls = torch.zeros((), device=loss.device)
            if affinity_cls_logits_x is not None:
                aux_w_target = float(args.affinity_aux_weight)
                aux_warm = max(1, int(args.affinity_aux_warmup))
                if epoch < aux_warm:
                    cos_t = epoch / aux_warm                # epoch 0 -> 0, epoch aux_warm -> 1
                    aux_w = aux_w_target * 0.5 * (1.0 - torch.cos(torch.tensor(cos_t * 3.141592653589793))).item()
                else:
                    aux_w = aux_w_target
                if aux_w > 0:
                    # Build multi-label target from mask_x, restricted to non-bg orig classes
                    n_orig = cfg['nclass']
                    idx_o2n = affinity_side.module.idx_orig_to_new.tolist()  # [n_orig]
                    # cls target in NEW (non-bg) space
                    new_cls = affinity_cls_logits_x.shape[-1]
                    y_multi = torch.zeros(mask_x.shape[0], new_cls,
                                           device=mask_x.device, dtype=torch.float32)
                    for c_orig in range(n_orig):
                        new_id = idx_o2n[c_orig]
                        if new_id < 0: continue
                        # any image where mask == c_orig in significant area
                        # vectorised: True if sum(mask==c_orig) > min_pix
                        present = ((mask_x == c_orig).flatten(1).sum(dim=1) >= 16)
                        y_multi[:, new_id] = present.float()
                    loss_aff_cls = nn.functional.binary_cross_entropy_with_logits(
                        affinity_cls_logits_x, y_multi)
                    loss = loss + aux_w * loss_aff_cls

            loss_b = torch.zeros((), device=loss.device)
            loss_tan = torch.zeros((), device=loss.device)
            if use_boundary and epoch >= boundary_warmup:
                valid_u1  = (conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255)
                valid_u2  = (conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255)
                if use_tangent:
                    bh = boundary_head.module if hasattr(boundary_head, 'module') else boundary_head
                    b_pred_x, feat_x = bh.forward_with_features(pred_x)
                    b_pred_u1, feat_u1 = bh.forward_with_features(pred_u_s1)
                    b_pred_u2, feat_u2 = bh.forward_with_features(pred_u_s2)
                    loss_b_x, loss_tan_x = boundary_loss_with_tangent(
                        b_pred_x, feat_x, pred_x, mask_x, cfg['nclass'],
                        pos_weight=boundary_head.module.pos_weight,   # buffer, no per-step realloc
                        dice_weight=boundary_dice_weight,
                        tan_detach_seg=tangent_detach_seg,
                    )
                    loss_b_u1, loss_tan_u1 = boundary_loss_with_tangent(
                        b_pred_u1, feat_u1, pred_u_s1, mask_u_w_cutmixed1, cfg['nclass'],
                        pos_weight=boundary_head.module.pos_weight,   # buffer, no per-step realloc
                        valid_mask=valid_u1,
                        cutmix_box=cutmix_box1,
                        cutmix_edge_thickness=boundary_seam_thick,
                        dice_weight=boundary_dice_weight,
                        tan_detach_seg=tangent_detach_seg,
                    )
                    loss_b_u2, loss_tan_u2 = boundary_loss_with_tangent(
                        b_pred_u2, feat_u2, pred_u_s2, mask_u_w_cutmixed2, cfg['nclass'],
                        pos_weight=boundary_head.module.pos_weight,   # buffer, no per-step realloc
                        valid_mask=valid_u2,
                        cutmix_box=cutmix_box2,
                        cutmix_edge_thickness=boundary_seam_thick,
                        dice_weight=boundary_dice_weight,
                        tan_detach_seg=tangent_detach_seg,
                    )
                    loss_tan = (loss_tan_x + (loss_tan_u1 + loss_tan_u2) / 2.0) / 2.0
                else:
                    b_pred_x = boundary_head(pred_x)
                    loss_b_x = boundary_loss(b_pred_x, mask_x, cfg['nclass'],
                                              pos_weight=boundary_head.module.pos_weight,   # buffer, no per-step realloc
                                              dice_weight=boundary_dice_weight)
                    b_pred_u1 = boundary_head(pred_u_s1)
                    loss_b_u1 = boundary_loss(b_pred_u1, mask_u_w_cutmixed1, cfg['nclass'],
                                                pos_weight=boundary_head.module.pos_weight,   # buffer, no per-step realloc
                                                valid_mask=valid_u1,
                                                cutmix_box=cutmix_box1,
                                                cutmix_edge_thickness=boundary_seam_thick,
                                                dice_weight=boundary_dice_weight)
                    b_pred_u2 = boundary_head(pred_u_s2)
                    loss_b_u2 = boundary_loss(b_pred_u2, mask_u_w_cutmixed2, cfg['nclass'],
                                                pos_weight=boundary_head.module.pos_weight,   # buffer, no per-step realloc
                                                valid_mask=valid_u2,
                                                cutmix_box=cutmix_box2,
                                                cutmix_edge_thickness=boundary_seam_thick,
                                                dice_weight=boundary_dice_weight)
                loss_b = (loss_b_x + (loss_b_u1 + loss_b_u2) / 2.0) / 2.0
                loss = loss + boundary_alpha * loss_b
                if use_tangent:
                    loss = loss + tangent_weight * loss_tan

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            if use_boundary: total_loss_b.update(loss_b.item())
            if use_tangent:  total_loss_tan.update(loss_tan.item())
            if use_cls:      total_loss_c.update(loss_c.item())
            if affinity_side is not None:
                total_loss_aff.update(float(loss_aff_cls))
                with torch.no_grad():
                    # side_x gate mean (after non-bg mask)
                    g = side_x['gate']
                    total_gate.update(float(g.mean()))
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            with torch.no_grad():
                valid_pix = (ignore_mask != 255)
                kept_pix = (conf_u_w >= cfg['conf_thresh']) & valid_pix
                flat_total = mask_u_w[valid_pix].reshape(-1).long()
                flat_kept = mask_u_w[kept_pix].reshape(-1).long()
                if flat_total.numel() > 0:
                    pseudo_total_per_class += torch.bincount(flat_total, minlength=cfg['nclass'])
                if flat_kept.numel() > 0:
                    pseudo_kept_per_class += torch.bincount(flat_kept, minlength=cfg['nclass'])

            iters = epoch * len(trainloader_u) + i
            progress = min(iters / max(total_iters, 1), 1.0)
            lr = cfg['lr'] * max(0.05, (1.0 - progress) ** 0.9)   # 5% floor: 防 resume 过 epochs 时 LR 死亡
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            for gi, g in enumerate(optimizer.param_groups[2:], start=2):
                if (affinity_proj_param_group_idx is not None
                        and gi == affinity_proj_param_group_idx):
                    g["lr"] = lr * float(args.affinity_unfreeze_lr_mult)
                else:
                    g["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            # In-place EMA: equivalent to UniMatch-V2 official formulation but
            # avoids two temporary tensors per parameter (saves ~30% EMA memory).
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    param_ema.mul_(ema_ratio).add_(param.detach(), alpha=1.0 - ema_ratio)
                for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                    if buffer_ema.dtype.is_floating_point:
                        buffer_ema.mul_(ema_ratio).add_(buffer.detach(), alpha=1.0 - ema_ratio)
                    else:
                        buffer_ema.copy_(buffer.detach())   # int / long buffers don't EMA
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                if use_tangent:
                    writer.add_scalar('train/loss_tan', loss_tan.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                msg = ('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, '
                       'Loss s: {:.3f}').format(i, optimizer.param_groups[0]['lr'],
                                                  total_loss.avg, total_loss_x.avg, total_loss_s.avg)
                if use_boundary:
                    msg += ', Loss b: {:.3f}'.format(total_loss_b.avg)
                if use_tangent:
                    msg += ', Loss tan: {:.4f}'.format(total_loss_tan.avg)
                if use_cls:
                    msg += ', Loss c: {:.3f}'.format(total_loss_c.avg)
                if affinity_side is not None:
                    msg += ', Loss aff: {:.3f}, gate: {:.3f}'.format(
                        total_loss_aff.avg, total_gate.avg)
                msg += ', Mask ratio: {:.3f}'.format(total_mask_ratio.avg)
                logger.info(msg)
        
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(pseudo_kept_per_class, op=dist.ReduceOp.SUM)
            dist.all_reduce(pseudo_total_per_class, op=dist.ReduceOp.SUM)
        kept_np = pseudo_kept_per_class.detach().cpu().numpy()
        total_np = pseudo_total_per_class.detach().cpu().numpy()
        keep_ratio_per_class = kept_np.astype('float64') / total_np.astype('float64').clip(min=1.0)

        def _grp_keep_ratio(idxs):
            if not idxs:
                return 0.0
            k = float(kept_np[idxs].sum())
            t = float(total_np[idxs].sum())
            return k / t if t > 0 else 0.0

        keep_head = _grp_keep_ratio(hbt_split.get('head', []))
        keep_body = _grp_keep_ratio(hbt_split.get('body', []))
        keep_tail = _grp_keep_ratio(hbt_split.get('tail', []))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=int(getattr(model.module, 'patch_size', 14)) if hasattr(model, 'module') else 14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, multiplier=int(getattr(model.module, 'patch_size', 14)) if hasattr(model, 'module') else 14)
        
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx]))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)

            # ---- CSV per-epoch log (one row per epoch, easy for matplotlib) ----
            csv_path = os.path.join(args.save_path, 'train_log.csv')
            cnames = CLASSES[cfg['dataset']]
            new_file = not os.path.exists(csv_path)
            with open(csv_path, 'a', encoding='utf-8') as f:
                if new_file:
                    header = ['epoch', 'lr', 'loss_total', 'loss_x', 'loss_s',
                              'loss_b', 'loss_tan', 'loss_c', 'loss_aff', 'gate_mean',
                              'affinity_proj_unfrozen',
                              'mask_ratio', 'mIoU', 'mIoU_ema']
                    header += [f'iou_{c}' for c in cnames] + [f'iou_{c}_ema' for c in cnames]
                    header += ['keep_ratio_head', 'keep_ratio_body', 'keep_ratio_tail']
                    header += [f'keep_ratio_{c}' for c in cnames]
                    header += [f'pseudo_kept_{c}' for c in cnames]
                    header += [f'pseudo_total_{c}' for c in cnames]
                    f.write(','.join(header) + '\n')
                row = [epoch, optimizer.param_groups[0]['lr'],
                       total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                       (total_loss_b.avg if use_boundary else 0.0),
                       (total_loss_tan.avg if use_tangent else 0.0),
                       (total_loss_c.avg if use_cls      else 0.0),
                       (total_loss_aff.avg if affinity_side is not None else 0.0),
                       (total_gate.avg     if affinity_side is not None else 0.0),
                       (1 if affinity_side is not None and affinity_side.module._proj_unfrozen else 0),
                       total_mask_ratio.avg, mIoU, mIoU_ema]
                row += list(iou_class) + list(iou_class_ema)
                row += [keep_head, keep_body, keep_tail]
                row += [float(x) for x in keep_ratio_per_class]
                row += [int(x) for x in kept_np]
                row += [int(x) for x in total_np]
                f.write(','.join('%.6f' % v if isinstance(v, float) else str(v) for v in row) + '\n')

            writer.add_scalar('train/keep_ratio_head', keep_head, epoch)
            writer.add_scalar('train/keep_ratio_body', keep_body, epoch)
            writer.add_scalar('train/keep_ratio_tail', keep_tail, epoch)
            logger.info('***** Pseudo-label keep ratio ***** head=%.3f  body=%.3f  tail=%.3f' %
                        (keep_head, keep_body, keep_tail))

        # ---- Affinity proj: plateau-triggered unfreeze ----
        if (affinity_side is not None
                and not affinity_side.module._proj_unfrozen
                and epoch >= int(args.affinity_freeze_warmup)):
            miou_window.append(float(mIoU))
            if (len(miou_window) == miou_window.maxlen
                    and (max(miou_window) - min(miou_window)) < float(args.affinity_plateau_eps) * 100.0):
                # mIoU is reported as percentage (e.g. 65.32) so eps is also in % units
                affinity_side.module.unfreeze_proj()
                unfreeze_lr = cfg['lr'] * float(args.affinity_unfreeze_lr_mult)
                proj_params = affinity_side.module.proj_parameters()
                optimizer.add_param_group({'params': proj_params, 'lr': unfreeze_lr})
                affinity_proj_param_group_idx = len(optimizer.param_groups) - 1
                if rank == 0:
                    logger.info('[affinity] mIoU plateaued at ep %d (window=%s) → '
                                'unfreezing proj with lr=%.6f' %
                                (epoch, [round(x, 2) for x in miou_window], unfreeze_lr))

        is_best = mIoU >= previous_best

        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            if use_boundary:
                checkpoint['boundary_head'] = boundary_head.state_dict()
            if use_debias:
                checkpoint['debias_prior'] = prior.state_dict()
            if use_cls:
                checkpoint['cls_head'] = cls_head.state_dict()
            if affinity_side is not None:
                checkpoint['affinity_side'] = affinity_side.module.state_dict_for_save()
                checkpoint['affinity_proj_unfrozen'] = affinity_side.module._proj_unfrozen
                checkpoint['affinity_miou_window'] = list(miou_window)
            tmp_path = os.path.join(args.save_path, 'latest.pth.tmp')
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
