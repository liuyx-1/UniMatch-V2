"""Plain fully-supervised DINOv2+DPT segmentation baseline.

No EMA, no DDP, no HPTA/visual adapter, no LC-PAM/text prior, no EDGE/MGER,
no SMC, no auxiliary heads. This script is intended as the "zero-improvement"
anchor for ablation studies.
"""
from __future__ import annotations

import argparse
import logging
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
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
from util.combined_loss import CombinedCETversky
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import AverageMeter, count_params, init_log


def parse_args():
    p = argparse.ArgumentParser('Plain fully-supervised DINOv2+DPT baseline')
    p.add_argument('--config', required=True)
    p.add_argument('--labeled-id-path', required=True)
    p.add_argument('--val-id-path', default=None)
    p.add_argument('--test-id-path', default=None)
    p.add_argument('--save-path', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--crop-size', type=int, default=None)
    p.add_argument('--eval-interval', type=int, default=1)
    p.add_argument('--eval-size', type=int, default=0)
    p.add_argument('--eval-batch-size', type=int, default=1,
                   help='val/test batch size; >1 batches same-resolution frames (bucketed)')
    p.add_argument('--grad-clip', type=float, default=1.0,
                   help='max global grad-norm (0 disables); guards against NaN collapse')
    p.add_argument('--exclude-classes', type=int, nargs='*', default=None,
                   help='classes dropped from present-only means (default: [0]=background '
                        'for endovis2017_* to match the paper, else none). Reporting/selection '
                        'only — never affects the loss.')
    p.add_argument('--seg-loss', choices=['config', 'ce', 'ohem', 'ce_tversky'],
                   default='config',
                   help='config follows cfg.criterion; ce is the pure baseline default when cfg is CELoss')
    p.add_argument('--tversky-alpha', type=float, default=0.7)
    p.add_argument('--tversky-beta', type=float, default=0.3)
    p.add_argument('--alpha-weight', type=float, default=0.5)
    return p.parse_args()


def round_to_multiple(x, multiple):
    return max(int(multiple), int(x / multiple + 0.5) * int(multiple))


def build_model(cfg):
    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
    }
    size = cfg['backbone'].split('_')[-1]
    model = DPT(**{**model_configs[size], 'nclass': int(cfg['nclass'])})

    ckpt_path = cfg.get('backbone_checkpoint') or f'./pretrained/{cfg["backbone"]}.pth'
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict) and 'model' in state:
        state = state['model']
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.backbone.load_state_dict(state)

    if bool(cfg.get('lock_backbone', False)):
        model.lock_backbone()
    return model


def build_criterion(args, cfg, nclass, ignore_index, device):
    name = args.seg_loss
    if name == 'config':
        name = str(cfg.get('criterion', {}).get('name', 'CELoss')).lower()
        if name == 'celoss':
            name = 'ce'

    if name == 'ce':
        return nn.CrossEntropyLoss(ignore_index=ignore_index).to(device), 'CE'
    if name == 'ohem':
        kwargs = cfg.get('criterion', {}).get('kwargs', {})
        return ProbOhemCrossEntropy2d(**kwargs).to(device), 'OHEM'
    if name == 'ce_tversky':
        return CombinedCETversky(
            num_classes=nclass,
            alpha_weight=args.alpha_weight,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            ignore_index=ignore_index,
        ).to(device), 'CE+Tversky'
    raise ValueError(f'unsupported seg loss: {args.seg_loss}')


def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.Loader)
    if args.epochs is not None:
        cfg['epochs'] = int(args.epochs)
    if args.batch_size is not None:
        cfg['batch_size'] = int(args.batch_size)
    if args.lr is not None:
        cfg['lr'] = float(args.lr)
    if args.crop_size is not None:
        cfg['crop_size'] = int(args.crop_size)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    os.makedirs(args.save_path, exist_ok=True)

    device = torch.device(args.device)
    if device.type == 'cuda':
        cudnn.enabled = True
        cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dataset = cfg['dataset']
    nclass = int(cfg['nclass'])
    class_names = CLASSES[dataset]
    assert len(class_names) == nclass, (len(class_names), nclass)
    disp_names = display_names(dataset, nclass)

    model = build_model(cfg).to(device)
    patch_size = int(getattr(model.backbone, 'patch_size', 14))
    if int(cfg['crop_size']) % patch_size != 0:
        raise ValueError(
            f"crop_size={cfg['crop_size']} must be a multiple of patch_size={patch_size}. "
            f"Use e.g. {round_to_multiple(int(cfg['crop_size']), patch_size)}.")

    ignore_index = int(cfg.get('criterion', {}).get('kwargs', {}).get('ignore_index', 255))
    criterion, criterion_name = build_criterion(args, cfg, nclass, ignore_index, device)

    lr = float(cfg['lr'])
    lr_multi = float(cfg.get('lr_multi', 10.0))
    bb_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if 'backbone' not in n and p.requires_grad]
    param_groups = []
    if bb_params:
        param_groups.append({'params': bb_params, 'lr': lr})
    param_groups.append({'params': head_params, 'lr': lr * lr_multi})
    optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    base_lrs = [group['lr'] for group in optimizer.param_groups]

    trainset = SemiDataset(dataset, cfg['data_root'], 'train_l',
                           cfg['crop_size'], args.labeled_id_path)
    trainloader = DataLoader(
        trainset, batch_size=int(cfg['batch_size']), shuffle=True,
        pin_memory=(device.type == 'cuda'), num_workers=args.num_workers,
        drop_last=True)

    valset = SemiDataset(dataset, cfg['data_root'], 'val', id_path=args.val_id_path)
    _ebs = max(1, int(args.eval_batch_size))
    valloader = DataLoader(
        valset, batch_sampler=ResolutionBatchSampler(valset, _ebs),
        pin_memory=(device.type == 'cuda'), num_workers=2)

    testloader = None
    if args.test_id_path and os.path.isfile(args.test_id_path) \
            and os.path.realpath(args.test_id_path) != os.path.realpath(args.val_id_path or ''):
        testset = SemiDataset(dataset, cfg['data_root'], 'val', id_path=args.test_id_path)
        testloader = DataLoader(
            testset, batch_sampler=ResolutionBatchSampler(testset, _ebs),
            pin_memory=(device.type == 'cuda'), num_workers=2)

    logger.info('plain supervised baseline: dataset=%s nclass=%d loss=%s', dataset, nclass, criterion_name)
    logger.info('Total %.1fM  trainable %.1fM',
                count_params(model),
                sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6)

    total_iters = max(1, len(trainloader) * int(cfg['epochs']))
    eval_mode = 'sliding_window' if dataset == 'cityscapes' else 'original'
    previous_best, best_epoch = 0.0, 0

    # present-only reporting: drop absent classes; endovis2017_* also drops bg (cls 0).
    if args.exclude_classes is None:
        exclude_classes = [0] if dataset.startswith('endovis2017') else []
    else:
        exclude_classes = list(args.exclude_classes)
    logger.info('present-only means exclude classes: %s', exclude_classes or 'none')

    for epoch in range(int(cfg['epochs'])):
        model.train()
        loss_meter = AverageMeter()
        iterator = trainloader
        if tqdm is not None:
            iterator = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{cfg["epochs"]}',
                            dynamic_ncols=True, mininterval=1.0)

        for i, (img, mask) in enumerate(iterator):
            img = img.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

            logits = model(img)
            loss = criterion(logits, mask)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in optimizer.param_groups for p in g['params']],
                    max_norm=float(args.grad_clip))
            optimizer.step()

            cur = epoch * len(trainloader) + i
            scale = max(0.05, (1.0 - cur / total_iters) ** 0.9)
            for gi, group in enumerate(optimizer.param_groups):
                group['lr'] = base_lrs[gi] * scale

            loss_meter.update(float(loss.detach()))
            if tqdm is not None and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix(loss=f'{loss_meter.avg:.3f}',
                                     lr=f'{optimizer.param_groups[-1]["lr"]:.2e}')

        do_eval = (epoch % max(1, int(args.eval_interval)) == 0) or (epoch == int(cfg['epochs']) - 1)
        if not do_eval:
            continue

        (mIoU, iou_c, allAcc, acc_c, mAcc, dice_c, mDice,
         mIoU_p, mDice_p, mAcc_p, present) = evaluate(
            model, valloader, eval_mode, cfg, multiplier=patch_size,
            return_acc=True, max_size=int(args.eval_size), device=device,
            return_present=True, exclude_classes=exclude_classes)

        tag = 'val(copied)'
        if testloader is not None:
            (tmIoU, tiou_c, tallAcc, tacc_c, tmAcc, tdice_c, tmDice,
             tmIoU_p, tmDice_p, tmAcc_p, _) = evaluate(
                model, testloader, eval_mode, cfg, multiplier=patch_size,
                return_acc=True, max_size=int(args.eval_size), device=device,
                return_present=True, exclude_classes=exclude_classes)
            tag = 'test'
        else:
            tmIoU_p, tmDice_p, tallAcc = mIoU_p, mDice_p, allAcc

        logger.info('[eval] ep=%d/%d [val] mIoU(pres)=%.2f mDice(pres)=%.2f PixAcc=%.2f '
                    '| mIoU(all)=%.2f mAcc(pres)=%.2f [%s] mIoU(pres)=%.2f mDice(pres)=%.2f PixAcc=%.2f',
                    epoch + 1, cfg['epochs'], mIoU_p, mDice_p, allAcc, mIoU, mAcc_p,
                    tag, tmIoU_p, tmDice_p, tallAcc)
        for c in range(nclass):
            flag = '' if present[c] else ' [absent]'
            flag += '' if (c not in exclude_classes) else ' [excl]'
            logger.info('    cls %d %-10s IoU=%.2f Dice=%.2f PixAcc=%.2f%s',
                        c, disp_names[c], iou_c[c], dice_c[c], acc_c[c], flag)

        ckpt = {
            'model': model.state_dict(),
            'epoch': epoch,
            'previous_best': max(mIoU_p, previous_best),
            'cfg_dataset': dataset,
            'nclass': nclass,
            'baseline': 'plain_supervised_dpt',
        }
        torch.save(ckpt, os.path.join(args.save_path, 'latest.pth'))
        # select best by present-only val mIoU (the reported headline metric)
        if mIoU_p >= previous_best:
            previous_best, best_epoch = mIoU_p, epoch
            torch.save(ckpt, os.path.join(args.save_path, 'best.pth'))
        logger.info('best mIoU=%.2f @ep%d', previous_best, best_epoch + 1)


if __name__ == '__main__':
    main()
