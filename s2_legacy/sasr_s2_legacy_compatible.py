import argparse
from copy import deepcopy
import json
import logging
import math
import os
import pprint
import random

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset
from model.sasr import refined_edge_loss
from model.semseg.sasr import SASR
from util.classes import CLASSES
from util.dist_helper import setup_distributed
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import AverageMeter, count_params, init_log, intersectionAndUnion


parser = argparse.ArgumentParser(description='SASR S2 based on UniMatch V2')
parser.add_argument('--config', required=True)
parser.add_argument('--labeled-id-path', required=True)
parser.add_argument('--unlabeled-id-path', required=True)
parser.add_argument('--val-id-path', required=True)
parser.add_argument('--test-id-path')
parser.add_argument('--save-path', required=True)
parser.add_argument('--stage1-checkpoint')
parser.add_argument('--backbone-checkpoint')
parser.add_argument('--batch-size', type=int)
parser.add_argument('--crop-size', type=int)
parser.add_argument('--epochs', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--patience', type=int, default=15)
parser.add_argument('--verify-init-only', action='store_true')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', type=int)


MODEL_CONFIGS = {
    'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'giant': {'encoder_size': 'giant', 'features': 384,
              'out_channels': [1536, 1536, 1536, 1536]},
}


def load_config(args):
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    for key in ('batch_size', 'crop_size', 'epochs', 'lr'):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.backbone_checkpoint:
        config['backbone_checkpoint'] = args.backbone_checkpoint
    if args.stage1_checkpoint:
        config['sasr']['stage1_checkpoint'] = args.stage1_checkpoint
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, config, multiplier=14):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    with torch.no_grad():
        for image, mask, _ in loader:
            image = image.cuda(non_blocking=True)
            original_size = image.shape[-2:]
            resized = tuple(int(size / multiplier + 0.5) * multiplier for size in original_size)
            image = F.interpolate(image, resized, mode='bilinear', align_corners=True)
            logits = model(image).logits
            logits = F.interpolate(logits, original_size, mode='bilinear', align_corners=True)
            prediction = logits.argmax(dim=1)
            intersection, union, target = intersectionAndUnion(
                prediction.cpu().numpy(), mask.numpy(), config['nclass'], 255
            )
            intersection = torch.from_numpy(intersection).cuda()
            union = torch.from_numpy(union).cuda()
            target = torch.from_numpy(target).cuda()
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)
            intersection_meter.update(intersection.cpu().numpy())
            union_meter.update(union.cpu().numpy())
            target_meter.update(target.cpu().numpy())

    class_iou = intersection_meter.sum / (union_meter.sum + 1.0e-10) * 100.0
    present = target_meter.sum > 0
    return float(np.mean(class_iou[present])), class_iou, present, float(np.mean(class_iou))


def checkpoint_state(path, key):
    checkpoint = torch.load(path, map_location='cpu')
    # Current completed S1 checkpoints store student weights under `model`;
    # old SASR S2 checkpoints stored EMA weights under `model_ema`.
    state = checkpoint.get(key)
    if state is None and key == 'model_ema':
        state = checkpoint.get('model')
    if state is None:
        raise KeyError(f'checkpoint {path} has no {key} or compatible model key')
    return {name.removeprefix('module.'): value for name, value in state.items()}


def build_model(config):
    sasr_config = config['sasr']
    size = config['backbone'].split('_')[-1]
    model = SASR(
        anchor_checkpoint=sasr_config['anchor_checkpoint'],
        adapter_reduction=sasr_config['adapter_reduction'],
        adapter_dropout=sasr_config['adapter_dropout'],
        edge_dropout=sasr_config['edge_dropout'],
        nclass=config['nclass'],
        **MODEL_CONFIGS[size],
    )
    backbone_state = torch.load(config['backbone_checkpoint'], map_location='cpu')
    model.backbone.load_state_dict(backbone_state)

    state = checkpoint_state(sasr_config['stage1_checkpoint'], sasr_config['stage1_weights'])
    missing, unexpected = model.load_state_dict(state, strict=False)
    new_roots = ('text_prior.', 'edge_refiner.', 'hpta.')
    invalid_missing = [name for name in missing if not name.startswith(new_roots)]
    if invalid_missing or unexpected:
        raise RuntimeError('S1 checkpoint is incompatible: missing=%s unexpected=%s' %
                           (invalid_missing, unexpected))
    model.set_gradient_checkpointing(sasr_config['gradient_checkpointing'])
    return model


def build_criterion(config, device):
    name = config['criterion']['name']
    kwargs = config['criterion']['kwargs']
    if name == 'CELoss':
        return nn.CrossEntropyLoss(**kwargs).to(device)
    if name == 'OHEM':
        return ProbOhemCrossEntropy2d(**kwargs).to(device)
    raise NotImplementedError('%s criterion is not implemented' % name)


def build_optimizer(model, config):
    sasr_config = config['sasr']
    return AdamW(
        (
            {'params': model.head.parameters(), 'lr_scale': sasr_config['dpt_lr_mult']},
            {'params': list(model.hpta.parameters()) + list(model.edge_refiner.parameters()),
             'lr_scale': sasr_config['adapter_lr_mult']},
            {'params': model.text_prior.parameters(), 'lr_scale': sasr_config['text_lr_start_mult'],
             'text_group': True},
        ),
        lr=config['lr'],
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )


def cutmix(value, box):
    selection = box.bool()
    if value.ndim == 4:
        selection = selection.unsqueeze(1).expand_as(value)
    result = value.clone()
    result[selection] = value.flip(0)[selection]
    return result


def image_class_targets(mask, nclass):
    return torch.stack(
        [(mask == class_id).flatten(1).any(dim=1) for class_id in range(nclass)],
        dim=1,
    ).float()


@torch.no_grad()
def update_ema(model, model_ema, decay):
    for source, target in zip(model.parameters(), model_ema.parameters()):
        target.mul_(decay).add_(source, alpha=1.0 - decay)
    for source, target in zip(model.buffers(), model_ema.buffers()):
        target.copy_(source)


def learning_rate(iteration, total_iterations, iterations_per_epoch, config):
    warmup = config['sasr']['warmup_epochs'] * iterations_per_epoch
    if iteration < warmup:
        return config['lr'] * (iteration + 1) / max(warmup, 1)
    progress = (iteration - warmup) / max(total_iterations - warmup, 1)
    ratio = max(config['sasr']['min_lr_ratio'], (1.0 - progress) ** 0.9)
    return config['lr'] * ratio


def update_learning_rates(optimizer, base_lr, progress, config):
    sasr_config = config['sasr']
    text_scale = sasr_config['text_lr_start_mult'] + (
        sasr_config['text_lr_mult'] - sasr_config['text_lr_start_mult']
    ) * min(1.0, progress / max(sasr_config['warmup_epochs'], 1))
    for group in optimizer.param_groups:
        scale = text_scale if group.get('text_group') else group['lr_scale']
        group['lr'] = base_lr * scale


def masked_unsupervised_loss(criterion, logits, pseudo_mask, confidence, ignore_mask, threshold):
    pixel_loss = criterion(logits, pseudo_mask)
    selected = confidence.ge(threshold) & ignore_mask.ne(255)
    return (pixel_loss * selected).sum() / ignore_mask.ne(255).sum().clamp_min(1)


def make_loaders(config, args):
    labeled = SemiDataset(
        config['dataset'], config['data_root'], 'train_l', config['crop_size'],
        args.labeled_id_path,
    )
    unlabeled = SemiDataset(
        config['dataset'], config['data_root'], 'train_u', config['crop_size'],
        args.unlabeled_id_path,
    )
    labeled = SemiDataset(
        config['dataset'], config['data_root'], 'train_l', config['crop_size'],
        args.labeled_id_path, nsample=len(unlabeled.ids),
    )
    validation = SemiDataset(
        config['dataset'], config['data_root'], 'val', id_path=args.val_id_path,
    )
    labeled_sampler = torch.utils.data.distributed.DistributedSampler(labeled)
    unlabeled_sampler = torch.utils.data.distributed.DistributedSampler(unlabeled)
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation, shuffle=False)
    loader_options = dict(
        batch_size=config['batch_size'], num_workers=config.get('num_workers', 4),
        pin_memory=True, drop_last=True,
    )
    labeled_loader = DataLoader(labeled, sampler=labeled_sampler, **loader_options)
    unlabeled_loader = DataLoader(unlabeled, sampler=unlabeled_sampler, **loader_options)
    validation_loader = DataLoader(
        validation, batch_size=config.get('eval_batch_size', 1),
        sampler=validation_sampler, num_workers=config.get('eval_num_workers', 2),
        pin_memory=True,
    )
    test_loader = None
    if args.test_id_path:
        test = SemiDataset(config['dataset'], config['data_root'], 'val', id_path=args.test_id_path)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test, shuffle=False)
        test_loader = DataLoader(
            test, batch_size=config.get('eval_batch_size', 1), sampler=test_sampler,
            num_workers=config.get('eval_num_workers', 2), pin_memory=True,
        )
    return labeled_loader, unlabeled_loader, validation_loader, test_loader


def main():
    args = parser.parse_args()
    config = load_config(args)
    seed_everything(args.seed)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    rank, world_size = setup_distributed(port=args.port)
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda', local_rank)

    if rank == 0:
        os.makedirs(args.save_path, exist_ok=True)
        writer = SummaryWriter(args.save_path)
        logger.info('%s\n', pprint.pformat({**config, **vars(args), 'ngpus': world_size}))

    cudnn.enabled = True
    cudnn.benchmark = True
    model = build_model(config).to(device)
    if args.verify_init_only:
        if rank == 0:
            logger.info('INIT_VERIFIED')
        return
    model_ema = deepcopy(model).to(device).eval()
    for parameter in model_ema.parameters():
        parameter.requires_grad = False
    optimizer = build_optimizer(model, config)
    if world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP enabled
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=True,
    )
    criterion_labeled = build_criterion(config, device)
    criterion_unlabeled = nn.CrossEntropyLoss(reduction='none').to(device)
    labeled_loader, unlabeled_loader, validation_loader, test_loader = make_loaders(config, args)
    total_iterations = len(unlabeled_loader) * config['epochs']
    best_ema = -float('inf')
    best_epoch = -1

    if rank == 0:
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        logger.info('Total params: %.1fM, trainable: %.1fM\n',
                    count_params(model.module), trainable / 1.0e6)

    for epoch in range(config['epochs']):
        model.train()
        labeled_loader.sampler.set_epoch(epoch)
        unlabeled_loader.sampler.set_epoch(epoch)
        loss_meter = AverageMeter()

        for index, ((image_x, mask_x), unlabeled_batch) in enumerate(zip(labeled_loader, unlabeled_loader)):
            image_u_w, image_u_s1, image_u_s2, ignore_mask, box1, box2 = unlabeled_batch
            image_x = image_x.to(device, non_blocking=True)
            mask_x = mask_x.to(device, non_blocking=True)
            image_u_w = image_u_w.to(device, non_blocking=True)
            image_u_s1 = image_u_s1.to(device, non_blocking=True)
            image_u_s2 = image_u_s2.to(device, non_blocking=True)
            ignore_mask = ignore_mask.to(device, non_blocking=True)
            box1 = box1.to(device, non_blocking=True)
            box2 = box2.to(device, non_blocking=True)

            with torch.no_grad():
                weak_logits = model_ema(
                    image_u_w, apply_text_prior=False, compute_edge=False
                ).logits
                weak_probabilities = weak_logits.softmax(dim=1)
                confidence, pseudo_mask = weak_probabilities.max(dim=1)

            strong1 = cutmix(image_u_s1, box1)
            strong2 = cutmix(image_u_s2, box2)
            pseudo1, pseudo2 = cutmix(pseudo_mask, box1), cutmix(pseudo_mask, box2)
            confidence1, confidence2 = cutmix(confidence, box1), cutmix(confidence, box2)
            ignore1, ignore2 = cutmix(ignore_mask, box1), cutmix(ignore_mask, box2)

            output_x = model(image_x)
            output_u = model(
                torch.cat((strong1, strong2)),
                comp_drop=True,
            )
            logits_u1, logits_u2 = output_u.logits.chunk(2)
            refined_u1, refined_u2 = output_u.refined_edge.chunk(2)

            labeled_loss = criterion_labeled(output_x.logits, mask_x)
            unlabeled_loss = 0.5 * (
                masked_unsupervised_loss(
                    criterion_unlabeled, logits_u1, pseudo1, confidence1, ignore1,
                    config['conf_thresh'],
                )
                + masked_unsupervised_loss(
                    criterion_unlabeled, logits_u2, pseudo2, confidence2, ignore2,
                    config['conf_thresh'],
                )
            )
            text_loss = nn.functional.binary_cross_entropy_with_logits(
                output_x.class_logits,
                image_class_targets(mask_x, config['nclass']).index_select(
                    1, model.module.text_prior.class_ids
                ),
            )
            edge_loss = 0.5 * (
                refined_edge_loss(output_x.refined_edge, mask_x)
                + 0.5 * (
                    refined_edge_loss(
                        refined_u1, pseudo1,
                        valid_mask=confidence1.ge(config['conf_thresh']) & ignore1.ne(255),
                    )
                    + refined_edge_loss(
                        refined_u2, pseudo2,
                        valid_mask=confidence2.ge(config['conf_thresh']) & ignore2.ne(255),
                    )
                )
            )

            progress = epoch + index / len(unlabeled_loader)
            ramp = min(1.0, progress / max(config['sasr']['warmup_epochs'], 1))
            loss = 0.5 * (labeled_loss + unlabeled_loss)
            loss = loss + ramp * (
                config['sasr']['text_loss_weight'] * text_loss
                + config['sasr']['edge_loss_weight'] * edge_loss
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['sasr']['grad_clip'])
            optimizer.step()

            iteration = epoch * len(unlabeled_loader) + index
            decay = min(1.0 - 1.0 / (iteration + 1), config['sasr']['ema_decay'])
            update_ema(model.module, model_ema, decay)
            base_lr = learning_rate(
                iteration, total_iterations, len(unlabeled_loader), config
            )
            update_learning_rates(optimizer, base_lr, progress, config)
            loss_meter.update(loss.item())
            if rank == 0:
                writer.add_scalar('train/loss', loss.item(), iteration)
                writer.add_scalar('train/labeled_loss', labeled_loss.item(), iteration)
                writer.add_scalar('train/unlabeled_loss', unlabeled_loss.item(), iteration)
                writer.add_scalar('train/text_loss', text_loss.item(), iteration)
                writer.add_scalar('train/edge_loss', edge_loss.item(), iteration)

        validation_miou, class_iou, val_present, val_miou_all = evaluate(model_ema, validation_loader, config)
        is_best = validation_miou > best_ema
        best_ema = max(best_ema, validation_miou)
        if is_best:
            best_epoch = epoch
        if rank == 0:
            logger.info('Epoch %d: loss %.4f, val EMA present-only mIoU %.2f, all-class %.2f',
                        epoch, loss_meter.avg, validation_miou, val_miou_all)
            for class_id, class_name in enumerate(CLASSES[config['dataset']]):
                logger.info('Class [%d %s] IoU %.2f', class_id, class_name, class_iou[class_id])
            checkpoint = {
                'model_ema': model_ema.state_dict(),
                'epoch': epoch,
                'previous_best_ema': best_ema,
                'config': config,
                'seed': args.seed,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest_ema.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best_ema.pth'))
            writer.add_scalar('eval/mIoU_ema', validation_miou, epoch)
            with open(os.path.join(args.save_path, 'metrics.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps({
                    'epoch': epoch + 1, 'seed': args.seed, 'train_loss': loss_meter.avg,
                    'val_mIoU_present_only': validation_miou, 'val_mIoU_all_class': val_miou_all,
                    'val_present_class_ids': np.flatnonzero(val_present).tolist(),
                    'val_IoU_class': class_iou.tolist(), 'best_epoch': best_epoch + 1,
                    'is_best': is_best, 'metric_protocol': 'GT_present_only',
                }) + '\n')

        if is_best and test_loader is not None:
            test_miou, _, test_present, test_miou_all = evaluate(model_ema, test_loader, config)
            if rank == 0:
                logger.info('Test EMA present-only mIoU %.2f, all-class %.2f', test_miou, test_miou_all)
        if args.patience > 0 and epoch - best_epoch >= args.patience:
            if rank == 0:
                with open(os.path.join(args.save_path, 'EARLY_STOPPED'), 'w', encoding='utf-8') as f:
                    f.write(f'epoch={epoch + 1} best_epoch={best_epoch + 1} patience={args.patience}\n')
                logger.info('EARLY_STOPPED epoch=%d best_epoch=%d patience=%d', epoch + 1, best_epoch + 1, args.patience)
            break
    if rank == 0:
        with open(os.path.join(args.save_path, 'DONE'), 'w', encoding='utf-8') as f:
            f.write('completed\n')


if __name__ == '__main__':
    main()
