import argparse
from collections import deque
from copy import deepcopy
import logging
import os
import pprint
import time

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

from util.boundary_loss import boundary_loss
from util.bias_metrics import compute_train_pixel_freq, split_head_body_tail
from util.cls_head import (build_cls_head, build_cls_loss,
                             cls_target_from_mask, cls_loss as _bce_cls_loss)
from util.edge_enhance import (EdgeSegResidualAdapter, rgb_edge_prior, edge_to_tokens,
                                edge_consistency_loss,
                                MaskGuidedEdgeRefiner, refined_edge_loss)
# Temporal Consistency Regularization (DA-VSN style). Optional — guard import.
try:
    from util.temporal import entropy_gated_consistency as _entropy_gated_consistency
except ImportError:
    _entropy_gated_consistency = None
import torch.distributed as dist


def _format_seconds(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f'{hours:d}h{minutes:02d}m'
    return f'{minutes:d}m{seconds:02d}s'


parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--val-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
# Variant toggles
# NOTE: The legacy standalone --boundary / --tangent auxiliary branch has
#       been removed. The edge-enhance branch may still predict an edge
#       boundary logit and supervise it through boundary_loss().
parser.add_argument('--cls-head', dest='cls_enabled',      action='store_true', default=None)
# ---- Temporal Consistency Regularization (DA-VSN, ACCEL two-stream + entropy gate) ----
parser.add_argument('--temporal-consistency', dest='temporal_consistency',
                    action='store_true', default=False,
                    help='enable DA-VSN-style entropy-gated consistency from the '
                         'CutMix-aligned EMA weak teacher to each strong student '
                         'view')
parser.add_argument('--temporal-weight', type=float, default=0.1,
                    help='weight λ_tcr for the temporal consistency loss')
parser.add_argument('--temporal-warmup', type=int, default=2,
                    help='epochs to wait before enabling the temporal consistency loss')
parser.add_argument('--temporal-original', action='store_true', default=False,
                    help='restore the v1 TCR: symmetric L1 between p_u_s1 and '
                         'p_u_s2 (no CutMix alignment, no teacher reference). '
                         'Kept for legacy / ablation reproducibility — '
                         'normally inferior to the default teacher-vs-student form.')
# ---- TS-MDR (Text-Semantic Morphology Dynamic Router) ----
parser.add_argument('--tsmdr-enabled', dest='tsmdr_enabled',
                    action='store_true', default=False,
                    help='enable Text-Semantic Morphology Dynamic Router')
parser.add_argument('--tsmdr-weight', type=float, default=0.1,
                    help='λ_tsmdr for the TSMDR consistency loss')
parser.add_argument('--tsmdr-warmup', type=int, default=10,
                    help='epochs to wait before enabling TSMDR (match TCR default)')
parser.add_argument('--tsmdr-route-ent', type=float, default=0.01,
                    help='λ_route entropy regulariser on the routing distribution')
parser.add_argument('--tsmdr-radius', type=int, default=3,
                    help='local directional matcher radius (R)')
parser.add_argument('--tsmdr-use-edge', action='store_true', default=False,
                    help='also apply boundary consistency inside TSMDR')
parser.add_argument('--tsmdr-shape-only', action='store_true', default=False,
                    help='use the pure-geometry ShapeMorphologyRouter '
                         '(NO text encoder, NO class-name dependency)')
# Stage-1 affinity prior side branch
parser.add_argument('--affinity-warmstart', type=str, default=None,
                    help='Optional text-affinity warmstart checkpoint. affinity_min can also initialize from dataset class names when --joint-text-stage is set.')
parser.add_argument('--affinity-aux-weight', type=float, default=0.5,
                    help='Final weight of the image-level cls BCE aux loss from affinity head.')
# ── Backbone freeze policy during segmentation training ────────────────
# Default (None): when AFFINITY_WARMSTART points to a CMA-trained ckpt, the
# backbone is AUTO-LOCKED (pure PEFT, matches that checkpoint); otherwise the YAML's
# `lock_backbone` decides.  Pass `--lock-backbone-stage2` to force lock on any
# metric, or `--no-lock-backbone-stage2` to disable the CMA auto-lock.
parser.add_argument('--lock-backbone-stage2', dest='lock_backbone_stage2',
                    action='store_true', default=None,
                    help='lock DINOv2 backbone during segmentation training (auto-True when '
                          'affinity ckpt is CMA-trained, follows YAML otherwise)')
parser.add_argument('--no-lock-backbone-stage2', dest='lock_backbone_stage2',
                    action='store_false',
                    help='explicitly KEEP backbone trainable, overriding CMA auto-lock')
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
parser.add_argument('--joint-text-stage', action='store_true',
                    help='train the affinity/text side branch jointly from the main segmentation run start; '
                         'initializes from dataset class names when --affinity-warmstart is omitted')
parser.add_argument('--joint-text-lr-mult', type=float, default=0.25,
                    help='initial LR multiplier for unfrozen affinity/text parameters during joint text training')
parser.add_argument('--joint-text-warmup', type=int, default=5,
                    help='epochs to keep the joint affinity/text LR small before normalizing it')
parser.add_argument('--visual-adapter', action='store_true',
                    help='enable HOM-lite DinoDPTAdapter between frozen DINOv2 and DPT head')
parser.add_argument('--visual-adapter-reduction', type=int, default=8,
                    help='channel reduction ratio for DinoDPTAdapter')
parser.add_argument('--visual-adapter-dropout', type=float, default=0.0,
                    help='dropout inside DinoDPTAdapter')
# RGB edge prior residual enhancement.
parser.add_argument('--edge-enhance', dest='edge_enhance', action='store_true', default=None,
                    help='enable non-shared RGB-edge residual adapters for text latent and DPT features')
parser.add_argument('--no-edge-enhance', dest='edge_enhance', action='store_false',
                    help='disable RGB-edge residual enhancement')
parser.add_argument('--edge-boundary-weight', type=float, default=0.05,
                    help='weight for GT/pseudo-mask supervision of the edge boundary head')
parser.add_argument('--edge-consistency-weight', type=float, default=0.01,
                    help='weight for weak consistency between edge boundary head and RGB edge prior')
parser.add_argument('--edge-warmup', type=int, default=0,
                    help='epochs to wait before applying edge-enhance losses')
parser.add_argument('--edge-dropout', type=float, default=0.0,
                    help='dropout inside edge residual adapters')
parser.add_argument('--edge-refiner', action='store_true', default=False,
                    help='enable Mask-Guided Edge Refiner: a lightweight 2D CNN '
                         'that purifies the raw Sobel edge into a clean '
                         'semantic-boundary map before it gates DINO/text '
                         'features. Strongly recommended with --edge-enhance.')
parser.add_argument('--edge-refiner-weight', type=float, default=0.1,
                    help='weight for the MGER mask-boundary supervision loss')
parser.add_argument('--edge-refiner-dropout', type=float, default=0.05)
parser.add_argument('--edge-refiner-thickness', type=int, default=3,
                    help='GT boundary dilation width used as MGER target')
parser.add_argument('--edge-refiner-pos-weight', type=float, default=8.0,
                    help='BCE positive-class weight in refined_edge_loss; '
                         'use 3-5 for binary tasks, 6-10 for multi-class')
parser.add_argument('--edge-refiner-checkpoint', action='store_true',
                    help='gradient-checkpoint the MGER forward on the labeled '
                         'branch; trades ~3%% wallclock for ~250 MB activations. '
                         'Enable when edge+text OOMs at the desired BS.')
parser.add_argument('--ema-decay-cap', type=float, default=0.996,
                    help='EMA decay upper bound (warmup formula '
                         '"min(1-1/(iter+1), cap)"). Default 0.996. '
                         'Lower values (e.g. 0.99) track student faster; '
                         'going below 0.95 hurts pseudo-label stability.')
parser.add_argument('--backbone', type=str, default=None,
                    help='override cfg.backbone (e.g. dinov2_small / dinov2_base / '
                         'dinov2_large). Use with --backbone-checkpoint to point '
                         'at the matching pretrained weights file.')
parser.add_argument('--backbone-checkpoint', type=str, default=None,
                    help='override cfg.backbone_checkpoint path; pair with --backbone.')
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
    # Standalone boundary aux head is permanently disabled. Edge-enhance has
    # its own boundary-logit supervision and is controlled separately.
    cfg.setdefault('boundary', {})['enabled'] = False
    cfg.setdefault('boundary', {})['tangent_enabled'] = False
    if args.cls_enabled is not None:
        cfg.setdefault('cls', {})['enabled'] = bool(args.cls_enabled)
    if args.batch_size is not None: cfg['batch_size'] = int(args.batch_size)
    if args.lr         is not None: cfg['lr']         = float(args.lr)
    if args.epochs     is not None: cfg['epochs']     = int(args.epochs)
    if args.crop_size  is not None: cfg['crop_size']  = int(args.crop_size)
    use_edge_enhance = bool(args.edge_enhance)
    # static_graph (forced True when use_edge_enhance) cannot tolerate a
    # warmup-induced topology change in the loss graph. Fail fast.
    if use_edge_enhance and bool(getattr(args, 'edge_refiner', False)) \
            and int(getattr(args, 'edge_warmup', 0)) > 0:
        raise ValueError(
            f"EDGE_WARMUP must be 0 when EDGE_REFINER=1 (DDP static_graph "
            f"requires constant backward graph; got edge_warmup="
            f"{int(args.edge_warmup)}).")
    use_visual_adapter = bool(args.visual_adapter)

    # ── Peek affinity ckpt ONCE: drives both static_graph and lock_backbone ──
    _aff_metric_in_ckpt = None
    if args.affinity_warmstart:
        try:
            _peek_ck = torch.load(args.affinity_warmstart, map_location='cpu',
                                   weights_only=False)
            _aff_metric_in_ckpt = _peek_ck.get('affinity_metric')
            del _peek_ck
        except Exception:
            pass

    # ── Resolve lock_backbone for segmentation training ──────────────────
    # Priority:  explicit CLI > CMA auto-lock > YAML `lock_backbone`
    if args.lock_backbone_stage2 is not None:
        cfg['lock_backbone'] = bool(args.lock_backbone_stage2)
        _lock_reason = f'CLI --{"" if args.lock_backbone_stage2 else "no-"}lock-backbone-stage2'
    elif use_visual_adapter:
        cfg['lock_backbone'] = True
        _lock_reason = 'auto: --visual-adapter uses frozen-DINO PEFT protocol'
    elif _aff_metric_in_ckpt == 'cma':
        cfg['lock_backbone'] = True
        _lock_reason = 'auto: CMA ckpt detected (pure PEFT protocol)'
    else:
        _lock_reason = f"YAML lock_backbone={cfg.get('lock_backbone', False)}"

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        logger.info('[lock_backbone] %s  (final=%s, ckpt_metric=%s)' %
                    (_lock_reason, cfg.get('lock_backbone', False),
                     _aff_metric_in_ckpt or 'n/a'))
        
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
    # CLI override for backbone size / weights (keeps single yaml usable across sizes)
    if args.backbone:
        cfg['backbone'] = args.backbone
    if args.backbone_checkpoint:
        cfg['backbone_checkpoint'] = args.backbone_checkpoint
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    ckpt_path = cfg.get('backbone_checkpoint') or f'./pretrained/{cfg["backbone"]}.pth'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    model.backbone.load_state_dict(state_dict)
    # If a legacy affinity checkpoint trained the visual backbone, prefer its weights as a warm-start.
    # NOTE: model_ema is built via deepcopy(model) AFTER this block, so the
    # EMA teacher automatically inherits those backbone weights too.
    if args.affinity_warmstart:
        try:
            _aff = torch.load(args.affinity_warmstart, map_location='cpu', weights_only=False)
            if 'dinov2_state_dict' in _aff:
                missing, unexpected = model.backbone.load_state_dict(_aff['dinov2_state_dict'], strict=False)
                if rank == 0:
                    logger.info('[affinity] backbone warm-started from affinity ckpt  '
                                'missing=%d  unexpected=%d  '
                                '(model_ema will inherit via deepcopy below)'
                                % (len(missing), len(unexpected)))
            elif rank == 0:
                logger.info('[affinity] affinity ckpt has no dinov2_state_dict '
                            '(was --freeze-vision); using default pretrained backbone.')
        except Exception as e:
            if rank == 0:
                logger.warning('[affinity] could not load dinov2_state_dict from %s: %s'
                                % (args.affinity_warmstart, e))
        
    if use_visual_adapter:
        model.enable_visual_adapter(
            reduction=int(args.visual_adapter_reduction),
            dropout=float(args.visual_adapter_dropout),
        )
        if rank == 0:
            logger.info('[visual-adapter] enabled  type=DinoDPTAdapter-HOM-lite  '
                        'reduction=%d  dropout=%.3f'
                        % (int(args.visual_adapter_reduction),
                           float(args.visual_adapter_dropout)))

    if cfg['lock_backbone']:
        model.lock_backbone()
    if use_edge_enhance:
        model.edge_seg_adapter = EdgeSegResidualAdapter(
            model.backbone.embed_dim, dropout=float(args.edge_dropout))
        if bool(args.edge_refiner):
            model.edge_refiner = MaskGuidedEdgeRefiner(
                mid=16, dropout=float(args.edge_refiner_dropout))
        if rank == 0:
            logger.info('[edge] enabled  boundary_w=%.4f  consistency_w=%.4f  '
                        'refiner=%s  refiner_w=%.4f  warmup=%d'
                        % (float(args.edge_boundary_weight),
                           float(args.edge_consistency_weight),
                           bool(args.edge_refiner),
                           float(args.edge_refiner_weight),
                           int(args.edge_warmup)))
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        trainable_params = sum(p.numel() for p in model.parameters()
                               if p.requires_grad) / 1e6
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M'.format(count_params(model.head)))
        logger.info('Trainable params: {:.1f}M\n'.format(trainable_params))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # static_graph=True is required when a sub-module's parameters are touched
    # by more than one autograd-engine hook per iter, causing DDP's default
    # reducer to raise "marked ready twice".  This happens for:
    #   (a) affinity_side with hyperbolic_pathway / cma feature-level injection,
    #       which re-decodes the DPT backbone with x_aware.
    #   (b) edge_seg_adapter, whose boundary sub-head produces BOTH the
    #       edge-aware feature residual AND an auxiliary boundary logit,
    #       so its boundary.* parameters appear in two loss paths per iter.
    _need_static_graph = (
        (_aff_metric_in_ckpt in ('hyperbolic_pathway', 'cma'))
        or bool(use_edge_enhance)
    )

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=(not _need_static_graph),
        static_graph=_need_static_graph,
    )
    if rank == 0:
        logger.info('[ddp] main model: find_unused_parameters=%s  static_graph=%s' %
                    (not _need_static_graph, _need_static_graph))
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False

    # ---- TS-MDR init (once, before the training loop) -----------------
    tsmdr_pack = None
    if getattr(args, 'tsmdr_enabled', False):
        try:
            from util.classes import CLASSES as _CLS
            from util.temporal_tsmdr import (
                TextMorphologyPrior, MaskMorphologyExtractor,
                TextSemanticMorphologyRouter, ShapeMorphologyRouter,
                TSMDRConsistencyLoss)
            _class_names = _CLS[cfg['dataset']]
            # Reuse LC-PAM SigLIP2 text encoder if available, else fallback.
            _encode_text = None
            _txt = TextMorphologyPrior(_class_names,
                                       encode_text=_encode_text).cuda(local_rank)
            _ext = MaskMorphologyExtractor(soft_threshold=0.5).cuda(local_rank)
            if bool(getattr(args, 'tsmdr_shape_only', False)):
                _router = ShapeMorphologyRouter(
                    lambda_temp_base=0.1, lambda_edge_base=0.1
                ).cuda(local_rank)
                if rank == 0:
                    logger.info('[tsmdr] router = ShapeMorphologyRouter (no text, no class names)')
            else:
                _router = TextSemanticMorphologyRouter(
                    q_dim=6, hidden=32, dropout=0.1,
                    lambda_temp_base=0.1, lambda_edge_base=0.1
                ).cuda(local_rank)
            _tsmdr_loss = TSMDRConsistencyLoss(
                num_classes=cfg['nclass'],
                radius=int(args.tsmdr_radius), n_directions=8,
                use_feat=False, use_mask=True,
                use_edge=bool(args.tsmdr_use_edge), use_con=False,
                lambda_m=0.2, lambda_e=0.1).cuda(local_rank)
            optimizer.add_param_group({
                'params': list(_router.parameters())
                          + list(_tsmdr_loss.parameters()),
                'lr': cfg['lr'] * 10.0,
            })
            tsmdr_pack = dict(txt=_txt, ext=_ext,
                              router=_router, loss=_tsmdr_loss)
            if rank == 0:
                logger.info('[tsmdr] enabled  radius=%d  use_edge=%s  warmup=%d  '
                            'λ_tsmdr=%.3f  λ_route_ent=%.3f',
                            args.tsmdr_radius, args.tsmdr_use_edge,
                            args.tsmdr_warmup, args.tsmdr_weight,
                            args.tsmdr_route_ent)
        except Exception as _e:
            if rank == 0:
                logger.warning('[tsmdr] init failed: %s — disabled', _e)
            tsmdr_pack = None

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

    # CutMix seam suppression thickness used by the edge-boundary auxiliary
    # loss. This is not the removed standalone boundary head.
    boundary_cfg = cfg.get('boundary') or {}
    boundary_seam_thick = int(boundary_cfg.get('cutmix_edge_thickness', 2))

    # --- Text-affinity prior side branch ---
    use_affinity = bool(args.affinity_warmstart) or bool(args.joint_text_stage)
    affinity_side = None
    affinity_proj_param_group_idx = None      # set when proj is unfrozen (added param group)
    affinity_joint_param_group_idx = None
    miou_window = deque(maxlen=int(args.affinity_plateau_window))
    affinity_pos_weight = None
    affinity_class_freq = None
    if use_affinity:
        from util.affinity_side import AffinitySideBranch
        affinity_side = AffinitySideBranch(
            args.affinity_warmstart,
            n_orig_classes=cfg['nclass'],
            class_names=CLASSES[cfg['dataset']],
            dataset=cfg['dataset'],
        )
        if use_edge_enhance:
            affinity_side.enable_edge_enhance(dropout=float(args.edge_dropout))
        affinity_side = affinity_side.cuda(local_rank)
        # static_graph=True lets DDP tolerate parameters being touched
        # by more than one autograd path per iter — required because the
        # hyperbolic_pathway block's outputs flow into BOTH (a) the affinity
        # side branch's own R/G fusion and (b) the DPT decoder via X_aware
        # re-decode, so its params (e.g. alpha_bias) fire twice in backward.
        _aff_metric_now = getattr(affinity_side, 'affinity_metric', 'cosine')
        _aff_need_static = _aff_metric_now in ('hyperbolic_pathway', 'cma')
        affinity_side = torch.nn.parallel.DistributedDataParallel(
            affinity_side, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=(not _aff_need_static),
            broadcast_buffers=False,
            static_graph=_aff_need_static,
        )
        if rank == 0:
            logger.info('[ddp] affinity_side: find_unused_parameters=%s  static_graph=%s' %
                        (not _aff_need_static, _aff_need_static))
        # ── EMA copy of affinity_side for fair --use-ema testing ──────────
        # Without this, test.py loads model_ema (slow-moving) but uses the
        # LATEST affinity_side weights → modality mismatch when LC-PAM is
        # unfrozen.  Synced in the same EMA loop as model below.
        affinity_side_ema = deepcopy(affinity_side)
        affinity_side_ema.eval()
        for p in affinity_side_ema.parameters():
            p.requires_grad = False
    else:
        affinity_side_ema = None
        # gate_conv is always trainable — add to optimizer immediately
        optimizer.add_param_group({'params': affinity_side.module.gate_conv.parameters(),
                                     'lr': cfg['lr'] * cfg['lr_multi']})
        if use_edge_enhance:
            optimizer.add_param_group({'params': affinity_side.module.edge_parameters(),
                                         'lr': cfg['lr'] * cfg['lr_multi']})
        if args.joint_text_stage:
            affinity_side.module.unfreeze_proj()
            optimizer.add_param_group({
                'params': affinity_side.module.proj_parameters(),
                'lr': cfg['lr'] * float(args.joint_text_lr_mult),
            })
            affinity_joint_param_group_idx = len(optimizer.param_groups) - 1
        if rank == 0:
            logger.info(
                '[affinity] side branch enabled  ckpt=%s  freeze_warmup=%d ep  '
                'plateau_window=%d  plateau_eps=%.4f  aux_weight_target=%.2f  '
                'aux_warmup=%d ep  joint_text=%s' % (
                    args.affinity_warmstart or '<joint-init-from-class-names>',
                    int(args.affinity_freeze_warmup),
                    int(args.affinity_plateau_window),
                    float(args.affinity_plateau_eps),
                    float(args.affinity_aux_weight),
                    int(args.affinity_aux_warmup),
                    bool(args.joint_text_stage)))

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    epoch_time = AverageMeter()

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
        if use_edge_enhance:
            msg = model.load_state_dict(checkpoint['model'], strict=False)
            msg_ema = model_ema.load_state_dict(checkpoint['model_ema'], strict=False)
            if rank == 0 and (msg.missing_keys or msg.unexpected_keys or
                              msg_ema.missing_keys or msg_ema.unexpected_keys):
                logger.warning('[resume] edge-enhance non-strict load: '
                               'model missing=%d unexpected=%d; '
                               'ema missing=%d unexpected=%d'
                               % (len(msg.missing_keys), len(msg.unexpected_keys),
                                  len(msg_ema.missing_keys), len(msg_ema.unexpected_keys)))
        else:
            model.load_state_dict(checkpoint['model'])
            model_ema.load_state_dict(checkpoint['model_ema'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']
        if use_cls and 'cls_head' in checkpoint:
            cls_head.load_state_dict(checkpoint['cls_head'])
        if affinity_side is not None and 'affinity_side' in checkpoint:
            affinity_side.module.load_state_dict_from_save(checkpoint['affinity_side'])
            # restore EMA copy too (falls back to main copy if checkpoint
            # predates the EMA-aff feature)
            if affinity_side_ema is not None:
                ema_state = checkpoint.get('affinity_side_ema',
                                            checkpoint['affinity_side'])
                affinity_side_ema.module.load_state_dict_from_save(ema_state)
            if checkpoint.get(
                    'affinity_proj_unfrozen',
                    checkpoint['affinity_side'].get('proj_unfrozen', False)):
                # Restore the extra param group before loading optimizer state.
                if affinity_joint_param_group_idx is None:
                    proj_params = affinity_side.module.proj_parameters()
                    optimizer.add_param_group({'params': proj_params,
                                                 'lr': cfg['lr'] * float(args.affinity_unfreeze_lr_mult)})
                    affinity_proj_param_group_idx = len(optimizer.param_groups) - 1
            for v in checkpoint.get('affinity_miou_window', []):
                miou_window.append(float(v))
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError as e:
            if rank == 0:
                logger.warning('[resume] skipped optimizer state from %s: %s. '
                               'Continuing with freshly initialized optimizer state.'
                               % (os.path.join(args.save_path, 'latest.pth'), e))

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        epoch_start = time.time()

        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_c = AverageMeter()
        total_loss_aff = AverageMeter()
        total_loss_edge = AverageMeter()
        total_loss_tcr  = AverageMeter()
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
            
            # ── Edge prior; optionally refined by MGER (training-mode pass
            #    so the refiner gets gradients from the seg path + its own
            #    GT-boundary loss).  For the EMA teacher pass we use the
            #    same refiner snapshot (no_grad context already covers it).
            _mod = model.module if hasattr(model, 'module') else model
            _refiner = getattr(_mod, 'edge_refiner', None) if use_edge_enhance else None

            def _edge(img, with_grad=True):
                if not use_edge_enhance:
                    return None
                raw = rgb_edge_prior(img)
                if _refiner is None:
                    return raw
                if with_grad:
                    # gradient checkpointing: re-compute MGER forward in
                    # backward instead of caching its activations.  Adds
                    # ~3% wallclock, saves ~250 MB at CROP=518 BS=16.
                    if bool(getattr(args, 'edge_refiner_checkpoint', False)):
                        return torch.utils.checkpoint.checkpoint(
                            _refiner, img, raw, use_reentrant=False)
                    return _refiner(img, raw)
                with torch.no_grad():
                    return _refiner(img, raw)

            with torch.no_grad():
                edge_u_w = _edge(img_u_w, with_grad=False)
                pred_u_w = model_ema(img_u_w, edge_prior=edge_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            # ── MGER gradient policy: ONLY labeled branch backprops through
            #    the refiner (cheap, and pseudo-boundaries are too noisy to
            #    supervise the refiner anyway).  Unlabeled branches still
            #    USE the refined edge for gating but in no_grad mode, so the
            #    autograd graph stays small.  Saves ~2/3 of MGER activation
            #    memory; BS=16 fits again at CROP=518.
            edge_x = _edge(img_x, with_grad=True) if use_edge_enhance else None
            edge_u_s1 = _edge(img_u_s1, with_grad=False) if use_edge_enhance else None
            edge_u_s2 = _edge(img_u_s2, with_grad=False) if use_edge_enhance else None
            
            # When affinity_side OR use_cls is on, request patches from DPT's backbone.
            need_patches = (use_cls or affinity_side is not None)
            if need_patches:
                pred_x, cls_tok_x, patches_x = model(img_x, return_cls=True,
                                                       edge_prior=edge_x)
            else:
                pred_x = model(img_x, edge_prior=edge_x)
            edge_boundary_x = getattr(model.module, 'last_edge_boundary_logits', None) if use_edge_enhance else None

            if need_patches:
                edge_u_cat = torch.cat((edge_u_s1, edge_u_s2)) if use_edge_enhance else None
                pred_u, _, patches_u = model(torch.cat((img_u_s1, img_u_s2)),
                                              comp_drop=True, return_cls=True,
                                              edge_prior=edge_u_cat)
                pred_u_s1, pred_u_s2 = pred_u.chunk(2)
                patches_u1, patches_u2 = patches_u.chunk(2)
            else:
                edge_u_cat = torch.cat((edge_u_s1, edge_u_s2)) if use_edge_enhance else None
                pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)),
                                              comp_drop=True, edge_prior=edge_u_cat).chunk(2)
            edge_boundary_u = getattr(model.module, 'last_edge_boundary_logits', None) if use_edge_enhance else None
            if edge_boundary_u is not None:
                edge_boundary_u1, edge_boundary_u2 = edge_boundary_u.chunk(2)
            else:
                edge_boundary_u1, edge_boundary_u2 = None, None

            # ---- Affinity side branch: fuse prior into supervised + unlabeled student logits.
            # pred_u_w (teacher) is INTENTIONALLY NOT fused, to keep pseudo-label generation clean.
            affinity_cls_logits_x = None
            if affinity_side is not None:
                # Both hyperbolic_pathway and cma write `last_x_aware` and
                # expect the trainer to re-decode DPT with X_aware replacing patches.
                _hyp_pathway_active = (getattr(affinity_side.module, 'affinity_metric', '')
                                        in ('hyperbolic_pathway', 'cma'))

                # ── Labeled stream ─────────────────────────────────────────────
                H_x, W_x = pred_x.shape[-2:]
                side_x = affinity_side(patches_x, H_x, W_x, edge_prior=edge_x)
                if _hyp_pathway_active:
                    # Single-pass re-decode: reuse the intermediate features we
                    # already captured via the first model(...) call.  This costs
                    # only ONE DPT head + interpolate (no second backbone forward),
                    # so memory and time impact is small.
                    mod = model.module if hasattr(model, 'module') else model
                    x_aware = getattr(affinity_side.module, 'last_x_aware', None)
                    if x_aware is not None:
                        ph_x = img_x.shape[-2] // 14
                        pw_x = img_x.shape[-1] // 14
                        if use_edge_enhance and hasattr(mod, 'edge_seg_adapter'):
                            x_aware, edge_boundary_x = mod.edge_seg_adapter(
                                x_aware, edge_to_tokens(edge_x, ph_x, pw_x), ph_x, pw_x)
                        # `patches_x` came from features[-1] of the first forward;
                        # we don't have features[0..2] cached, so encode JUST the
                        # backbone again for the shallow features. The encode-only
                        # call is the same cost as before but we *avoid* a second
                        # comp_drop'd DPT head + bilinear. Trade-off is acceptable.
                        feats_x, _, _, _ = mod.encode(img_x, return_cls=False)
                        feats_x = list(mod.adapt_features(feats_x, ph_x, pw_x))
                        feats_x[-1] = x_aware
                        pred_x = mod.decode(feats_x, ph_x, pw_x, comp_drop=False)
                pred_x = affinity_side.module.fuse(pred_x, side_x)
                affinity_cls_logits_x = side_x['cls_logits']

                # ── Unlabeled strong views ─────────────────────────────────────
                H_u, W_u = pred_u_s1.shape[-2:]
                side_u1 = affinity_side(patches_u1, H_u, W_u, edge_prior=edge_u_s1)
                xa1 = getattr(affinity_side.module, 'last_x_aware', None)
                side_u2 = affinity_side(patches_u2, H_u, W_u, edge_prior=edge_u_s2)
                xa2 = getattr(affinity_side.module, 'last_x_aware', None)
                if _hyp_pathway_active and xa1 is not None and xa2 is not None:
                    mod = model.module if hasattr(model, 'module') else model
                    img_u_cat = torch.cat((img_u_s1, img_u_s2))
                    x_aware_cat = torch.cat((xa1, xa2), dim=0)
                    ph_u = img_u_cat.shape[-2] // 14
                    pw_u = img_u_cat.shape[-1] // 14
                    if use_edge_enhance and hasattr(mod, 'edge_seg_adapter'):
                        x_aware_cat, edge_boundary_u = mod.edge_seg_adapter(
                            x_aware_cat, edge_to_tokens(edge_u_cat, ph_u, pw_u), ph_u, pw_u)
                        edge_boundary_u1, edge_boundary_u2 = edge_boundary_u.chunk(2)
                    feats_u, _, _, _ = mod.encode(img_u_cat, return_cls=False)
                    feats_u = list(mod.adapt_features(feats_u, ph_u, pw_u))
                    feats_u[-1] = x_aware_cat
                    pred_u = mod.decode(feats_u, ph_u, pw_u, comp_drop=True)
                    pred_u_s1, pred_u_s2 = pred_u.chunk(2)
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

            # ------------------------------------------------------------
            # Temporal Consistency Regularization (DA-VSN style).
            #
            # CORRECTED FORMULATION (v2):
            #   kf  = EMA teacher's WEAK-view soft probability, CutMix-mixed
            #         in the same way as the corresponding strong view, so it
            #         is pixel-aligned with the underlying composited image.
            #   cf  = student's STRONG-view soft probability.
            # The entropy gate then picks pixels where the teacher's weak-
            # view prediction is more confident than the student's strong-
            # view one, and pulls the student toward the teacher there.
            #
            # The old "p_u_s1 vs p_u_s2" symmetric form was conceptually
            # wrong because the two strong views use INDEPENDENT CutMix
            # boxes — at many pixel positions they cover different source
            # images, making "consistency" between them ill-defined.
            # ------------------------------------------------------------
            loss_tcr = torch.zeros((), device=loss.device)
            if getattr(args, 'temporal_consistency', False) \
                    and epoch >= int(getattr(args, 'temporal_warmup', 2)) \
                    and _entropy_gated_consistency is not None:
                # ---- v1 (original symmetric s1↔s2, no CutMix alignment) ----
                if bool(getattr(args, 'temporal_original', False)):
                    p_u_s1 = pred_u_s1.softmax(dim=1)
                    p_u_s2 = pred_u_s2.softmax(dim=1)
                    loss_tcr_a = _entropy_gated_consistency(p_u_s1, p_u_s2)
                    loss_tcr_b = _entropy_gated_consistency(p_u_s2, p_u_s1)
                    loss_tcr = 0.5 * (loss_tcr_a + loss_tcr_b)
                    loss = loss + float(getattr(args, 'temporal_weight', 0.1)) * loss_tcr
                    continue_with_tsmdr = True  # placeholder; original path falls through
                    if False: pass   # short-circuit the v2 block below
                # ---- v2 (fixed: teacher weak vs student strong) ----
                elif True:   # branch kept for indentation parity
                    # 1) teacher's weak-view soft probability (already detached
                    #    upstream via model_ema(...).detach() on pred_u_w)
                    prob_u_w = pred_u_w.softmax(dim=1)               # [B, C, H, W]

                    # 2) CutMix-mix the soft probability with the SAME boxes
                    #    used to produce the strong views, so kf and cf
                    #    align pixel-for-pixel on the composited image.
                    prob_u_w_cm1 = prob_u_w.clone()
                    prob_u_w_cm2 = prob_u_w.clone()
                    em1 = cutmix_box1.unsqueeze(1).expand_as(prob_u_w) == 1
                    em2 = cutmix_box2.unsqueeze(1).expand_as(prob_u_w) == 1
                    prob_u_w_cm1[em1] = prob_u_w.flip(0)[em1]
                    prob_u_w_cm2[em2] = prob_u_w.flip(0)[em2]

                    # 3) Valid-pixel ignore mask (drop padding etc.)
                    valid1 = (ignore_mask_cutmixed1 != 255).float()
                    valid2 = (ignore_mask_cutmixed2 != 255).float()

                    p_u_s1 = pred_u_s1.softmax(dim=1)
                    p_u_s2 = pred_u_s2.softmax(dim=1)

                    # 4) Entropy-gated L1 with teacher's weak prob as kf.
                    loss_tcr_1 = _entropy_gated_consistency(
                        p_u_s1, prob_u_w_cm1, ignore_mask=valid1)
                    loss_tcr_2 = _entropy_gated_consistency(
                        p_u_s2, prob_u_w_cm2, ignore_mask=valid2)
                    loss_tcr = 0.5 * (loss_tcr_1 + loss_tcr_2)
                    # NOTE: `loss` above is already (loss_x + loss_u_s) / 2, so the
                    # implicit weights on loss_x and loss_u_s are each 0.5. Multiply
                    # TCR by 0.5 too so that `temporal_weight` is comparable to those.
                    loss = loss + 0.5 * float(getattr(args, 'temporal_weight', 0.1)) * loss_tcr

            # ------------------------------------------------------------
            # TS-MDR — Text-Semantic Morphology Dynamic Router
            # Reuses prob_u_w_cm1/cm2 (CutMix-aligned EMA teacher) as P_{t+1},
            # and pred_u_s1/s2 softmax as P_t. The text morphology prior is
            # cached once at init; the MLP router emits per-class λ_temp.
            # ------------------------------------------------------------
            loss_tsmdr = torch.zeros((), device=loss.device)
            if getattr(args, 'tsmdr_enabled', False) \
                    and epoch >= int(getattr(args, 'tsmdr_warmup', 10)) \
                    and tsmdr_pack is not None:
                _txt, _ext, _router, _tsmdr_loss = (
                    tsmdr_pack['txt'], tsmdr_pack['ext'],
                    tsmdr_pack['router'], tsmdr_pack['loss'])
                # β schedule: 0.8 → 0.3 over training
                _router.set_beta(max(0.3, 0.8 - 0.5 * epoch / cfg['epochs']))

                with torch.no_grad():
                    s_c   = _txt()
                    p_s1d = pred_u_s1.softmax(dim=1).detach()
                    p_s2d = pred_u_s2.softmax(dim=1).detach()
                    q_t1  = _ext(p_s1d, prev_prob=prob_u_w_cm1.detach())
                    q_t2  = _ext(p_s2d, prev_prob=prob_u_w_cm2.detach())
                    q_t   = 0.5 * (q_t1 + q_t2)
                route = _router(s_c, q_t, bg_mask=_txt.bg_mask)

                losses1 = _tsmdr_loss(
                    feat_t=None, feat_tp1=None,
                    prob_t=pred_u_s1.softmax(dim=1),
                    prob_tp1=prob_u_w_cm1,
                    lam_temp=route['lam_temp'], lam_edge=route['lam_edge'],
                    boundary_t=None, boundary_tp1=None,
                    valid_mask=(ignore_mask_cutmixed1 != 255).float())
                losses2 = _tsmdr_loss(
                    feat_t=None, feat_tp1=None,
                    prob_t=pred_u_s2.softmax(dim=1),
                    prob_tp1=prob_u_w_cm2,
                    lam_temp=route['lam_temp'], lam_edge=route['lam_edge'],
                    boundary_t=None, boundary_tp1=None,
                    valid_mask=(ignore_mask_cutmixed2 != 255).float())
                loss_tsmdr = 0.5 * (losses1['loss_total'] + losses2['loss_total'])
                loss = loss + 0.5 * float(args.tsmdr_weight) * loss_tsmdr
                loss = loss + float(args.tsmdr_route_ent) * route.get(
                    'loss_route_ent', torch.zeros((), device=loss.device))

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
                else:
                    loss_c = _bce_cls_loss(cls_logits_x, cls_tgt_x)
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

            valid_u1  = (conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255)
            valid_u2  = (conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255)

            loss_edge = torch.zeros((), device=loss.device)
            if (use_edge_enhance and epoch >= int(args.edge_warmup)
                    and edge_boundary_x is not None):
                edge_mask_loss_x = boundary_loss(
                    edge_boundary_x, mask_x, cfg['nclass'],
                    pos_weight=1.0, dice_weight=0.5)
                edge_mask_loss_u1 = boundary_loss(
                    edge_boundary_u1, mask_u_w_cutmixed1, cfg['nclass'],
                    pos_weight=1.0, valid_mask=valid_u1,
                    cutmix_box=cutmix_box1,
                    cutmix_edge_thickness=boundary_seam_thick,
                    dice_weight=0.5) if edge_boundary_u1 is not None else torch.zeros_like(loss)
                edge_mask_loss_u2 = boundary_loss(
                    edge_boundary_u2, mask_u_w_cutmixed2, cfg['nclass'],
                    pos_weight=1.0, valid_mask=valid_u2,
                    cutmix_box=cutmix_box2,
                    cutmix_edge_thickness=boundary_seam_thick,
                    dice_weight=0.5) if edge_boundary_u2 is not None else torch.zeros_like(loss)
                edge_mask_loss = (edge_mask_loss_x + (edge_mask_loss_u1 + edge_mask_loss_u2) / 2.0) / 2.0

                edge_cons_x = edge_consistency_loss(edge_boundary_x, edge_x)
                edge_cons_u1 = edge_consistency_loss(
                    edge_boundary_u1, edge_u_s1, valid_mask=valid_u1) if edge_boundary_u1 is not None else torch.zeros_like(loss)
                edge_cons_u2 = edge_consistency_loss(
                    edge_boundary_u2, edge_u_s2, valid_mask=valid_u2) if edge_boundary_u2 is not None else torch.zeros_like(loss)
                edge_cons = (edge_cons_x + (edge_cons_u1 + edge_cons_u2) / 2.0) / 2.0
                loss_edge = (float(args.edge_boundary_weight) * edge_mask_loss
                             + float(args.edge_consistency_weight) * edge_cons)
                # ---- MGER mask-boundary supervision (only when refiner is on)
                if bool(args.edge_refiner) and edge_x is not None and edge_x.requires_grad:
                    # MGER is supervised ONLY on the labeled branch: pseudo
                    # boundaries are too noisy to train an edge refiner with,
                    # and adding the u branch doubles memory for marginal
                    # gain.  The refiner still inferences on u_s1/u_s2 in
                    # no_grad mode so gating quality there is unchanged.
                    loss_refiner = refined_edge_loss(
                        edge_x, mask_x, cfg['nclass'],
                        thickness=int(args.edge_refiner_thickness),
                        pos_weight=float(args.edge_refiner_pos_weight),
                        dice_weight=0.5)
                    loss_edge = loss_edge + float(args.edge_refiner_weight) * loss_refiner
                loss = loss + loss_edge

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            if getattr(args, 'temporal_consistency', False):
                total_loss_tcr.update(loss_tcr.item())
            if use_cls:      total_loss_c.update(loss_c.item())
            if use_edge_enhance: total_loss_edge.update(loss_edge.item())
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
                if (affinity_joint_param_group_idx is not None
                        and gi == affinity_joint_param_group_idx):
                    warm = max(1, int(args.joint_text_warmup))
                    warm_scale = min(1.0, (epoch + i / max(len(trainloader_u), 1)) / warm)
                    target_mult = float(args.affinity_unfreeze_lr_mult)
                    mult = float(args.joint_text_lr_mult) + (
                        target_mult - float(args.joint_text_lr_mult)) * warm_scale
                    g["lr"] = lr * mult
                elif (affinity_proj_param_group_idx is not None
                        and gi == affinity_proj_param_group_idx):
                    g["lr"] = lr * float(args.affinity_unfreeze_lr_mult)
                else:
                    g["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), float(args.ema_decay_cap))
            
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
                # ---- affinity_side EMA sync (LC-PAM) ----
                # Only meaningful once proj is unfrozen; before that, params
                # are unchanged so EMA == main copy and the loop is cheap.
                if affinity_side is not None and affinity_side_ema is not None:
                    for p_main, p_ema in zip(affinity_side.parameters(),
                                              affinity_side_ema.parameters()):
                        p_ema.mul_(ema_ratio).add_(p_main.detach(),
                                                    alpha=1.0 - ema_ratio)
                    for b_main, b_ema in zip(affinity_side.buffers(),
                                              affinity_side_ema.buffers()):
                        if b_ema.dtype.is_floating_point:
                            b_ema.mul_(ema_ratio).add_(b_main.detach(),
                                                        alpha=1.0 - ema_ratio)
                        else:
                            b_ema.copy_(b_main.detach())
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                if getattr(args, 'temporal_consistency', False):
                    writer.add_scalar('train/loss_tcr', loss_tcr.item(), iters)
                if use_edge_enhance:
                    writer.add_scalar('train/loss_edge', loss_edge.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

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

        epoch_time.update(time.time() - epoch_start)
        remaining_epochs = max(0, cfg['epochs'] - epoch - 1)
        eta = epoch_time.avg * remaining_epochs

        if rank == 0:
            logger.info('[eval] epoch=%d/%d mode=%s mIoU=%.2f EMA=%.2f '
                        'best=%.2f bestEMA=%.2f time=%s ETA=%s'
                        % (epoch + 1, cfg['epochs'], eval_mode, mIoU, mIoU_ema,
                           max(mIoU, previous_best),
                           max(mIoU_ema, previous_best_ema),
                           _format_seconds(epoch_time.val),
                           _format_seconds(eta)))
            
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
                              'loss_tcr', 'loss_c', 'loss_aff', 'loss_edge', 'gate_mean',
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
                       (total_loss_tcr.avg if getattr(args, 'temporal_consistency', False) else 0.0),
                       (total_loss_c.avg if use_cls      else 0.0),
                       (total_loss_aff.avg if affinity_side is not None else 0.0),
                       (total_loss_edge.avg if use_edge_enhance else 0.0),
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

        # ---- Affinity proj: plateau-triggered unfreeze ----
        if (affinity_side is not None
                and not args.joint_text_stage
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
            if use_visual_adapter:
                checkpoint['visual_adapter'] = {
                    'enabled': True,
                    'reduction': int(args.visual_adapter_reduction),
                    'dropout': float(args.visual_adapter_dropout),
                }
            if use_cls:
                checkpoint['cls_head'] = cls_head.state_dict()
            if affinity_side is not None:
                checkpoint['affinity_side'] = affinity_side.module.state_dict_for_save()
                checkpoint['affinity_proj_unfrozen'] = affinity_side.module._proj_unfrozen
                checkpoint['affinity_miou_window'] = list(miou_window)
                # EMA-smoothed affinity_side weights for fair --use-ema test
                if affinity_side_ema is not None:
                    checkpoint['affinity_side_ema'] = \
                        affinity_side_ema.module.state_dict_for_save()
            tmp_path = os.path.join(args.save_path, 'latest.pth.tmp')
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
