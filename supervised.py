import argparse
import logging
import glob
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.semseg.dpt import DPT
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from PIL import Image as _PILImage


class ResolutionBatchSampler(torch.utils.data.Sampler):
    """Batch eval indices so every batch holds ONE native resolution, enabling
    eval batch_size>1 on multi-video sets that mix frame sizes (default collation
    fails otherwise). No resizing -> the present-only mIoU protocol is unchanged.
    DDP-aware: each rank takes a disjoint, unpadded shard so the all_reduce in
    evaluate() counts every image exactly once."""

    def __init__(self, dataset, batch_size):
        self.batch_size = max(1, int(batch_size))
        if dist.is_available() and dist.is_initialized():
            num_replicas, rank = dist.get_world_size(), dist.get_rank()
        else:
            num_replicas, rank = 1, 0
        sizes = []
        for line in dataset.ids:
            rel = (line.split('\t') if '\t' in line else line.split(' '))[0]
            with _PILImage.open(os.path.join(dataset.root, rel)) as im:
                sizes.append(im.size)                      # (W, H)
        shard = list(range(len(dataset.ids)))[rank::num_replicas]
        buckets = {}
        for i in shard:
            buckets.setdefault(sizes[i], []).append(i)
        self.batches = []
        for _, group in sorted(buckets.items()):
            for j in range(0, len(group), self.batch_size):
                self.batches.append(group[j:j + self.batch_size])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


parser = argparse.ArgumentParser(description='Fully-Supervised Training in Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--pretrained-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--val-id-path', type=str, default=None)
parser.add_argument('--test-id-path', type=str, default=None)
parser.add_argument('--batch-size', type=int, default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--epochs', type=int, default=None)
parser.add_argument('--crop-size', type=int, default=None)
parser.add_argument('--visual-adapter', action='store_true')
parser.add_argument('--visual-adapter-reduction', type=int, default=8)
parser.add_argument('--visual-adapter-dropout', type=float, default=0.0)
parser.add_argument('--eval-size', type=int, default=0)
parser.add_argument('--eval-batch-size', type=int, default=1)


def evaluate(model, loader, mode, cfg, multiplier=None, return_acc=False, max_size=0,
             device=None, forward_fn=None, ignore_index=None,
             return_present=False, exclude_classes=None, desc=None):
    """Evaluate segmentation metrics over ``loader``.

    Returns ``(mIoU, iou_class)`` by default. When ``return_acc=True`` it also
    returns pixel-accuracy metrics:
        (mIoU, iou_class, allAcc, acc_class, mAcc, dice_class, mDice)
    where
        allAcc    : global pixel accuracy  (Σ correct / Σ valid pixels)  [%]
        acc_class : per-class pixel accuracy (recall = TP/GT)            [%]
        mAcc      : mean of per-class pixel accuracy                     [%]

    These "all-class" means average over every class id, so classes ABSENT from
    the eval GT (support == 0) contribute a constant IoU/Dice/Acc of 0 and drag
    the mean down. When ``return_present=True`` (requires ``return_acc``) the
    tuple is EXTENDED with present-only means that skip absent classes and any
    ``exclude_classes`` (e.g. background), matching the headline number reported
    by ``test.py`` and the literature:
        (..., mDice, miou_present, mdice_present, macc_present, present)
    where ``present`` is a bool array (len nclass), True where GT support>0.
    Metrics never feed the loss — this only changes reporting / model selection.
    """
    model.eval()
    if ignore_index is None:
        ignore_index = int(cfg.get('criterion', {}).get('kwargs', {}).get('ignore_index', 255))
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
    forward = forward_fn if forward_fn is not None else model
    assert mode in ['original', 'center_crop', 'sliding_window']
    # fp16 autocast on CUDA: ~2x faster eval, half the VRAM, no accuracy change
    # for argmax metrics (matches test.py's default fp16 inference).
    amp_enabled = (device.type == 'cuda')

    def _fwd(x):
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            return forward(x)

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    iterator = loader
    if tqdm is not None and rank == 0:
        iterator = tqdm(loader, desc=(desc or 'eval'), total=len(loader),
                        dynamic_ncols=True, leave=False, mininterval=0.5)

    with torch.no_grad():
        for img, mask, id in iterator:
            
            img = img.to(device, non_blocking=True)
                
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, int(cfg['nclass']), h, w, device=device)
                
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = _fwd(img[:, :, row: row + grid, col: col + grid])
                        final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)
                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)
                    
                pred = final
            
            else:
                assert mode == 'original'
                
                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    if multiplier == 512:
                        new_h, new_w = 512, 512
                    else:
                        # Optional cap on the longer side: bounds the O(N^2)
                        # attention cost / memory of full-resolution eval frames
                        # (e.g. endovis 1280x1024). Predictions are upsampled back
                        # to (ori_h, ori_w) below, so scoring stays at GT resolution.
                        sh, sw = float(ori_h), float(ori_w)
                        if max_size and max(sh, sw) > max_size:
                            scale = float(max_size) / max(sh, sw)
                            sh, sw = sh * scale, sw * scale
                        new_h = max(multiplier, int(sh / multiplier + 0.5) * multiplier)
                        new_w = max(multiplier, int(sw / multiplier + 0.5) * multiplier)
                    img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)

                pred = _fwd(img)

                if multiplier is not None:
                    pred = pred.float()
                    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)
            
            pred = pred.argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'],
                                     int(ignore_index))

            reduced_intersection = torch.from_numpy(intersection).to(device)
            reduced_union = torch.from_numpy(union).to(device)
            reduced_target = torch.from_numpy(target).to(device)

            # all_reduce only under an initialized process group; single-process
            # (non-DDP) callers like train_supervised_smc.py skip it.
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(reduced_intersection)
                dist.all_reduce(reduced_union)
                dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())
            target_meter.update(reduced_target.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    if return_acc:
        acc_class = intersection_meter.sum / (target_meter.sum + 1e-10) * 100.0
        mAcc = np.mean(acc_class)
        allAcc = intersection_meter.sum.sum() / (target_meter.sum.sum() + 1e-10) * 100.0
        # Dice (F1) per class: 2·TP / (2·TP + FP + FN) = 2·inter / (inter + union)
        dice_class = 2.0 * intersection_meter.sum \
            / (intersection_meter.sum + union_meter.sum + 1e-10) * 100.0
        mDice = np.mean(dice_class)
        if return_present:
            # present = class actually appears in the eval GT (support > 0);
            # keep = not in exclude_classes (e.g. background). Both masks are a
            # fixed property of the eval set, independent of predictions.
            support = target_meter.sum
            present = support > 0
            exclude = set(int(c) for c in (exclude_classes or []))
            keep = np.array([c not in exclude for c in range(int(cfg['nclass']))])
            sel = present & keep
            _mean = lambda a: float(np.mean(a[sel])) if sel.any() else 0.0
            miou_present = _mean(iou_class)
            mdice_present = _mean(dice_class)
            macc_present = _mean(acc_class)
            return (mIOU, iou_class, allAcc, acc_class, mAcc, dice_class, mDice,
                    miou_present, mdice_present, macc_present, present)
        return mIOU, iou_class, allAcc, acc_class, mAcc, dice_class, mDice

    return mIOU, iou_class


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    # --- CLI overrides (mirror unimatch_v2.py so the same env recipe works) ---
    if args.lr        is not None: cfg['lr']        = float(args.lr)
    if args.epochs    is not None: cfg['epochs']    = int(args.epochs)
    if args.crop_size is not None: cfg['crop_size'] = int(args.crop_size)
    if args.batch_size is not None:
        # explicit BS is taken literally (no doubling)
        cfg['batch_size'] = int(args.batch_size)
    else:
        # official convention: supervised uses 2x batch (no unlabeled stream)
        cfg['batch_size'] *= 2

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
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    model.backbone.load_state_dict(state_dict)

    # Optional HPTA visual adapter (frozen-DINO PEFT) — matches the *_va runs so
    # the fully-supervised baseline is architecturally comparable.
    if args.visual_adapter:
        model.enable_visual_adapter(reduction=int(args.visual_adapter_reduction),
                                    dropout=float(args.visual_adapter_dropout))
        cfg['lock_backbone'] = True
        if rank == 0:
            logger.info('[visual-adapter] enabled  reduction=%d  dropout=%.3f'
                        % (int(args.visual_adapter_reduction), float(args.visual_adapter_dropout)))

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
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    )
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    # Per-epoch upsampling of tiny labeled sets (keeps iters/epoch reasonable).
    # Surgical datasets are not in this table → default to no upsampling.
    n_upsampled = {
        'pascal': 3000,
        'cityscapes': 3000,
        'ade20k': 6000,
        'coco': 30000
    }
    trainset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path,
        nsample=n_upsampled.get(cfg['dataset'], None)
    )
    valset = SemiDataset(
        cfg['dataset'], cfg['data_root'], 'val', id_path=args.val_id_path
    )
    testset = None
    if args.test_id_path and os.path.isfile(args.test_id_path) \
            and os.path.realpath(args.test_id_path) != os.path.realpath(args.val_id_path or ''):
        testset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', id_path=args.test_id_path)

    eval_bs = max(1, int(args.eval_batch_size))
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler
    )

    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=eval_bs, pin_memory=True, num_workers=4, drop_last=False, sampler=valsampler
    )
    testloader = None
    if testset is not None:
        testsampler = torch.utils.data.distributed.DistributedSampler(testset)
        testloader = DataLoader(
            testset, batch_size=eval_bs, pin_memory=True, num_workers=4, drop_last=False, sampler=testsampler)
    
    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = criterion(pred, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)
            
            if (i % (len(trainloader) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
        
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        _mult = int(getattr(model.module, 'patch_size', 14)) if hasattr(model, 'module') else 14
        _esz = int(args.eval_size)
        mIoU, iou_class, allAcc, acc_class, mAcc, dice_class, mDice = evaluate(
            model, valloader, eval_mode, cfg, multiplier=_mult, return_acc=True, max_size=_esz)

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)

        # Test split: only on a new best epoch or the final epoch (keeps it fast).
        have_test = False
        if testloader is not None and (is_best or epoch == cfg['epochs'] - 1):
            tmIoU, tiou_class, tallAcc, tacc_class, tmAcc, tdice_class, tmDice = evaluate(
                model, testloader, eval_mode, cfg, multiplier=_mult, return_acc=True, max_size=_esz)
            have_test = True

        if rank == 0:
            logger.info('[val ] mIoU=%.2f  mDice=%.2f  PixAcc=%.2f  mAcc=%.2f  (best=%.2f)'
                        % (mIoU, mDice, allAcc, mAcc, previous_best))
            if have_test:
                logger.info('[test] mIoU=%.2f  mDice=%.2f  PixAcc=%.2f  mAcc=%.2f'
                            % (tmIoU, tmDice, tallAcc, tmAcc))
            for cls_idx in range(cfg['nclass']):
                logger.info('    cls %d %-22s  IoU=%.2f  Dice=%.2f  PixAcc=%.2f'
                            % (cls_idx, CLASSES[cfg['dataset']][cls_idx],
                               iou_class[cls_idx], dice_class[cls_idx], acc_class[cls_idx]))

            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mDice', mDice, epoch)
            writer.add_scalar('eval/PixAcc', allAcc, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
            if have_test:
                writer.add_scalar('test/mIoU', tmIoU, epoch)
                writer.add_scalar('test/mDice', tmDice, epoch)
                writer.add_scalar('test/PixAcc', tallAcc, epoch)

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                # best.pth is only used for eval → drop optimizer state.
                best_ckpt = {k: v for k, v in checkpoint.items() if k != 'optimizer'}
                torch.save(best_ckpt, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
