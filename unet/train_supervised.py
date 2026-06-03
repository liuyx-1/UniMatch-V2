"""Fully-supervised SMP UNet baseline on the same surgical datasets used by
UniMatch-V2 (endoscapes_seg50 / cholecseg8k / endovis2017_* / endovis2018 /
needle).  Reuses the project's SemiDataset + transform pipeline so the
preprocessing and augmentation are byte-identical to the UniMatch labeled
branch.

Default label_rate = 1.0  (use ALL training images with their GT masks).
For comparison runs at lower rates (0.10 / 0.20 / ...), passing
--label-rate <r> uses only that split's labeled.txt — the unlabeled
portion is discarded (this gives the "supervised lower-bound" baseline).

Usage (single GPU):

    python unet/train_supervised.py \
        --config configs/endoscapes_seg50.yaml \
        --encoder resnet50 \
        --arch Unet \
        --label-rate 1.0 \
        --epochs 150 --batch-size 16 --crop 504 --lr 1e-4 \
        --save-dir /root/autodl-tmp/exp/endoscapes_seg50/smp_unet_r1.0_resnet50

CPU testing (after training):

    python unet/train_supervised.py --test-only \
        --config configs/endoscapes_seg50.yaml --device cpu \
        --save-dir <same path as training>
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# add project root to path so we can import dataset.* and util.*
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# SMP package shipped under unet/ in this repo
sys.path.insert(0, str(_ROOT / 'unet'))
import segmentation_models_pytorch as smp  # noqa: E402

from dataset.semi import SemiDataset  # noqa: E402
from util.classes import CLASSES  # noqa: E402


# ----------------------------------------------------------------------
# args
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True,
                    help='dataset yaml (configs/<name>.yaml) — same one as UniMatch')
    p.add_argument('--splits-root', type=str,
                    default='/root/autodl-tmp/data/autonomous_surgery/splits',
                    help='parent of unimatch_splits_<dataset>_<rate>_seed42 dirs')
    p.add_argument('--label-rate', type=float, default=1.0,
                    help='1.0 = use all training images (labeled+unlabeled treated '
                         'as labeled); 0.10/0.20/... = use only that rate split\'s '
                         'labeled.txt (supervised lower-bound).')
    p.add_argument('--seed', type=int, default=42)

    # model
    p.add_argument('--arch', type=str, default='Unet',
                    choices=['Unet', 'UnetPlusPlus', 'MAnet', 'Linknet', 'FPN',
                             'PSPNet', 'PAN', 'DeepLabV3', 'DeepLabV3Plus',
                             'Segformer', 'UPerNet'])
    p.add_argument('--encoder', type=str, default='resnet50',
                    help='any SMP-supported encoder (resnet18/34/50/101, '
                         'efficientnet-b0..b7, mit_b0..b5, timm-resnest50d, ...)')
    p.add_argument('--encoder-weights', type=str, default='imagenet',
                    help='pretrained weights id ("imagenet", "ssl", "swsl", None)')

    # train
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--crop', type=int, default=504)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--save-dir', type=str, required=True)

    # loss
    p.add_argument('--loss', type=str, default='ce_dice',
                    choices=['ce', 'ce_dice', 'dice'],
                    help='ce_dice = 0.5 CE + 0.5 Dice (multiclass)')
    p.add_argument('--ignore-index', type=int, default=255)

    # modes
    p.add_argument('--test-only', action='store_true',
                    help='skip training; load <save-dir>/best.pth and evaluate '
                         'on the test split (falls back to val if test missing)')
    p.add_argument('--resume', action='store_true',
                    help='resume training from <save-dir>/latest.pth if present')
    return p.parse_args()


# ----------------------------------------------------------------------
# build the labeled-id list  (label_rate handling)
# ----------------------------------------------------------------------
def build_train_id_path(args, dataset_name: str) -> str:
    """For label_rate=1.0 we concatenate labeled.txt + unlabeled.txt of the
    smallest split present, since both sides have the same image-mask
    structure (only the SSL pipeline treats unlabeled.txt as unlabeled).
    For label_rate<1.0 we use that split's labeled.txt as-is.
    """
    if abs(args.label_rate - 1.0) < 1e-6:
        # find any split dir (rate doesn't matter — labeled+unlabeled spans the whole train set)
        candidates = sorted(Path(args.splits_root).glob(
            f'unimatch_splits_{dataset_name}_*_seed{args.seed}'))
        if not candidates:
            raise FileNotFoundError(
                f'no splits found under {args.splits_root} for {dataset_name}')
        src = candidates[0]
        lab = (src / 'labeled.txt').read_text().splitlines()
        unl = (src / 'unlabeled.txt').read_text().splitlines() \
            if (src / 'unlabeled.txt').exists() else []
        all_lines = [l for l in (lab + unl) if l.strip()]
        # dedup preserving order
        seen, dedup = set(), []
        for l in all_lines:
            if l not in seen:
                seen.add(l); dedup.append(l)
        out = Path(args.save_dir) / 'train_full.txt'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('\n'.join(dedup) + '\n')
        logging.info('[split] label_rate=1.0  union(labeled+unlabeled)=%d  '
                     'from %s  -> %s', len(dedup), src.name, out)
        return str(out)
    else:
        rate_str = f'{args.label_rate:.2f}'
        src = Path(args.splits_root) / \
              f'unimatch_splits_{dataset_name}_{rate_str}_seed{args.seed}'
        lab = src / 'labeled.txt'
        if not lab.exists():
            raise FileNotFoundError(f'no labeled.txt at {lab}')
        logging.info('[split] label_rate=%s  using %s', rate_str, lab)
        return str(lab)


def val_test_paths(args, dataset_name: str):
    """Pick a split dir for val.txt / test.txt — any rate works since these
    are dataset-wide, not rate-dependent."""
    candidates = sorted(Path(args.splits_root).glob(
        f'unimatch_splits_{dataset_name}_*_seed{args.seed}'))
    src = candidates[0]
    val_p = src / 'val.txt'
    test_p = src / 'test.txt'
    if not test_p.exists():
        test_p = val_p
    return str(val_p), str(test_p)


# ----------------------------------------------------------------------
# model factory
# ----------------------------------------------------------------------
def build_model(args, nclass: int) -> nn.Module:
    return smp.create_model(
        args.arch,
        encoder_name=args.encoder,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        classes=nclass,
    )


# ----------------------------------------------------------------------
# loss
# ----------------------------------------------------------------------
def build_loss(args, nclass: int):
    ce = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    if args.loss == 'ce':
        return lambda logits, target: ce(logits, target)
    dice = smp.losses.DiceLoss(mode='multiclass',
                                ignore_index=args.ignore_index,
                                from_logits=True)
    if args.loss == 'dice':
        return lambda logits, target: dice(logits, target)
    return lambda logits, target: 0.5 * ce(logits, target) + 0.5 * dice(logits, target)


# ----------------------------------------------------------------------
# eval
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, nclass: int, device: torch.device, ignore_index: int):
    model.eval()
    cm = torch.zeros(nclass, nclass, dtype=torch.int64)
    for batch in loader:
        img, mask, _ = batch
        img = img.to(device, non_blocking=True)
        mask = mask.long()
        ori_h, ori_w = img.shape[-2:]
        # pad to be divisible by 32 (UNet decoder constraint)
        pad_h = (32 - ori_h % 32) % 32
        pad_w = (32 - ori_w % 32) % 32
        img_in = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        logits = model(img_in)
        logits = logits[..., :ori_h, :ori_w]
        pred = logits.argmax(1).cpu()
        valid = mask != ignore_index
        v_pred = pred[valid]
        v_mask = mask[valid]
        idx = v_mask * nclass + v_pred
        cm += torch.bincount(idx, minlength=nclass * nclass).view(nclass, nclass)
    cm = cm.numpy()
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1e-12)
    support = cm.sum(axis=1)
    present = support > 0
    miou = float(iou[present].mean()) if present.any() else 0.0
    pa = float(tp.sum() / max(1, cm.sum()))
    return miou, pa, iou, support


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                         format='%(asctime)s [%(levelname)s] %(message)s',
                         datefmt='%H:%M:%S')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    cfg = yaml.load(open(args.config), Loader=yaml.Loader)
    dataset_name = cfg['dataset']
    data_root = cfg['data_root']
    nclass = int(cfg['nclass'])
    class_names = CLASSES.get(dataset_name, [f'cls{i}' for i in range(nclass)])

    device = torch.device(args.device)

    # ---- splits ----
    train_id_path = build_train_id_path(args, dataset_name)
    val_id_path, test_id_path = val_test_paths(args, dataset_name)

    # ---- datasets (reuse UniMatch's labeled-branch transforms) ----
    trainset = SemiDataset(dataset_name, data_root, 'train_l',
                            size=args.crop, id_path=train_id_path,
                            nsample=None)
    valset = SemiDataset(dataset_name, data_root, 'val',
                          id_path=val_id_path)
    testset = SemiDataset(dataset_name, data_root, 'val',
                           id_path=test_id_path)
    logging.info('train=%d  val=%d  test=%d  nclass=%d  classes=%s',
                  len(trainset), len(valset), len(testset),
                  nclass, class_names)

    train_loader = DataLoader(trainset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.workers,
                               pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=1, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ---- model / opt / loss ----
    model = build_model(args, nclass).to(device)
    loss_fn = build_loss(args, nclass)

    # SMP encoders are usually loaded with ImageNet-pretrained weights →
    # decoder learns from scratch.  Give decoder 10x LR (matches UniMatch's
    # lr_multi=10 convention).
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if not n.startswith('encoder.')]
    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': args.lr},
        {'params': decoder_params, 'lr': args.lr * 10.0},
    ], weight_decay=args.weight_decay)

    # cosine LR with 5% floor
    base_lrs = [g['lr'] for g in optimizer.param_groups]

    def set_lr(progress):
        scale = max(0.05, 0.5 * (1 + np.cos(np.pi * progress)))
        for g, base in zip(optimizer.param_groups, base_lrs):
            g['lr'] = base * scale

    # ---- resume / test-only ----
    start_epoch = 0
    best_miou = 0.0
    best_epoch = 0
    history = []
    latest_p = Path(args.save_dir) / 'latest.pth'
    best_p = Path(args.save_dir) / 'best.pth'

    if args.test_only:
        if not best_p.exists():
            raise FileNotFoundError(f'{best_p} not found — train first.')
        ck = torch.load(best_p, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        model = model.to(device)
        miou, pa, iou, support = evaluate(model, test_loader, nclass, device,
                                           args.ignore_index)
        rows = ['cls,name,support,iou']
        for c in range(nclass):
            rows.append(f'{c},{class_names[c]},{int(support[c])},{iou[c]:.6f}')
        rows += ['', f'summary,mIoU,{miou:.6f}', f'summary,PA,{pa:.6f}']
        csv_p = Path(args.save_dir) / 'test_metrics.csv'
        csv_p.write_text('\n'.join(rows) + '\n')
        logging.info('TEST  mIoU=%.4f  PA=%.4f  -> %s', miou, pa, csv_p)
        return

    if args.resume and latest_p.exists():
        ck = torch.load(latest_p, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = int(ck['epoch']) + 1
        best_miou = float(ck.get('best_miou', 0.0))
        best_epoch = int(ck.get('best_epoch', 0))
        history = ck.get('history', [])
        logging.info('resumed from epoch %d  best_mIoU=%.4f',
                      start_epoch, best_miou)

    # ---- train ----
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        loss_sum, n_batches = 0.0, 0
        for it, batch in enumerate(train_loader):
            img, mask = batch
            img = img.to(device, non_blocking=True)
            mask = mask.long().to(device, non_blocking=True)
            progress = (epoch + it / len(train_loader)) / args.epochs
            set_lr(progress)
            optimizer.zero_grad()
            logits = model(img)
            if logits.shape[-2:] != mask.shape[-2:]:
                logits = F.interpolate(logits, size=mask.shape[-2:],
                                        mode='bilinear', align_corners=False)
            loss = loss_fn(logits, mask)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        train_loss = loss_sum / max(1, n_batches)

        miou, pa, _iou, _sup = evaluate(model, val_loader, nclass, device,
                                          args.ignore_index)
        dt = time.time() - t0
        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                         'val_mIoU': miou, 'val_PA': pa, 'sec': dt})
        is_best = miou > best_miou
        if is_best:
            best_miou = miou
            best_epoch = epoch + 1
        logging.info('ep %3d/%d  loss=%.4f  val mIoU=%.4f  PA=%.4f  '
                      'best=%.4f@ep%d  %.0fs',
                      epoch + 1, args.epochs, train_loss, miou, pa,
                      best_miou, best_epoch, dt)

        ck = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'best_miou': best_miou,
            'best_epoch': best_epoch,
            'history': history,
        }
        torch.save(ck, latest_p)
        if is_best:
            torch.save(ck, best_p)

        (Path(args.save_dir) / 'history.json').write_text(
            json.dumps(history, indent=2))

    logging.info('training done.  best mIoU=%.4f at ep%d', best_miou, best_epoch)

    # ---- final test pass ----
    if best_p.exists():
        ck = torch.load(best_p, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
    miou, pa, iou, support = evaluate(model, test_loader, nclass, device,
                                       args.ignore_index)
    rows = ['cls,name,support,iou']
    for c in range(nclass):
        rows.append(f'{c},{class_names[c]},{int(support[c])},{iou[c]:.6f}')
    rows += ['', f'summary,mIoU,{miou:.6f}', f'summary,PA,{pa:.6f}']
    csv_p = Path(args.save_dir) / 'test_metrics.csv'
    csv_p.write_text('\n'.join(rows) + '\n')
    logging.info('TEST  mIoU=%.4f  PA=%.4f  -> %s', miou, pa, csv_p)


if __name__ == '__main__':
    main()
