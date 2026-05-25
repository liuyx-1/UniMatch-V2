"""PyTorch Dataset + augmentation for SigLIP-2 contrastive fine-tuning.

Reads JSONL manifest produced by tools/build_siglip_manifest.py and yields
(image_tensor, class_id_list, dataset_name) per item, with:

  * Level-1 sampling weight handled by the caller via WeightedRandomSampler
  * Level-2 image-level augmentation:
        - mild aug for normal images
        - STRONG aug for images flagged has_rare_class
          (geometric: rotation/shear/elastic + stronger color jitter)

Mask is geometrically synced with the image (only used to RE-derive the
class list after geometric crop/rotation, since some regions may be cut
out by aggressive crop and the original class list becomes wrong).

NOTE: we do NOT do region-level Copy-Paste here (Layer-3 omitted by design).
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Albumentations is the cleanest way to keep image+mask geometrically synced
# during rotation / shear / elastic transforms. Falls back to torchvision-only
# transforms (image-side only) if albumentations is not installed.
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALBU = True
except ImportError:
    _HAS_ALBU = False


# ImageNet stats — SigLIP-2 uses the same normalisation as CLIP/DINOv2 family
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------
def _build_transforms(input_size: int = 256, strong: bool = False):
    """Construct Albumentations transform pipeline.

    strong=False: mild aug (suitable for any image)
    strong=True:  + geometric distortion + harder colour jitter
                   (used for images containing rare classes)
    """
    if not _HAS_ALBU:
        # Minimal torchvision-only fallback (no geometric mask-sync)
        return None

    # Albumentations 2.x API: value/mask_value -> fill/fill_mask;
    # RandomResizedCrop uses size=(h, w) instead of height/width
    common = [
        A.LongestMaxSize(max_size=input_size),
        A.PadIfNeeded(min_height=input_size, min_width=input_size,
                       border_mode=0, fill=0, fill_mask=255),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.3 if not strong else 0.5,
                       contrast=0.3   if not strong else 0.5,
                       saturation=0.3 if not strong else 0.5,
                       hue=0.1        if not strong else 0.2,
                       p=0.8),
        A.GaussianBlur(blur_limit=(3, 7),
                        p=0.3 if not strong else 0.7),
    ]
    geometric_strong = [
        A.Rotate(limit=15, border_mode=0, fill=0, fill_mask=255, p=0.7),
        A.Affine(shear=(-10, 10), border_mode=0, fill=0, fill_mask=255, p=0.4),
        A.ElasticTransform(alpha=50, sigma=20,
                            border_mode=0, fill=0, fill_mask=255, p=0.4),
    ]
    crop_resize = [
        A.RandomResizedCrop(size=(input_size, input_size),
                              scale=(0.6, 1.0) if strong else (0.8, 1.0),
                              ratio=(0.85, 1.18)),
    ]
    finish = [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    if strong:
        pipeline = common[:1] + geometric_strong + common[1:] + crop_resize + finish
    else:
        pipeline = common + crop_resize + finish
    return A.Compose(pipeline)


# ----------------------------------------------------------------------
class SigLIPManifestDataset(Dataset):
    """Yields (image, class_ids, dataset_name).

    Args:
        manifest_path: path to manifest_<dataset>.jsonl OR manifest_all.jsonl
        input_size:    SigLIP-2 input resolution (256 for base/patch16/256, 512 for patch16/512)
        min_pixels_after_aug: classes still in the augmented mask >= this count
                              are kept; smaller ones are dropped (geometric crop
                              may chop out small regions).
    """

    def __init__(self, manifest_path: str | Path,
                 input_size: int = 256,
                 min_pixels_after_aug: int = 16,
                 strong_aug_p: float = 1.0,
                 strong_aug_for_rare: bool = True):
        super().__init__()
        self.records: List[Dict] = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    self.records.append(json.loads(ln))
        self.input_size = input_size
        self.min_pixels_after_aug = int(min_pixels_after_aug)
        self.strong_for_rare = bool(strong_aug_for_rare)
        self.strong_aug_p = float(strong_aug_p)

        # Build two transform pipelines once
        self.tf_mild   = _build_transforms(input_size, strong=False)
        self.tf_strong = _build_transforms(input_size, strong=True)

        if not _HAS_ALBU:
            raise RuntimeError(
                'albumentations is required for synced image+mask augmentation. '
                'Install with: pip install albumentations')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = np.array(Image.open(rec['image']).convert('RGB'))
        mask  = np.array(Image.open(rec['mask']))
        if mask.ndim == 3:
            mask = mask[..., 0]

        # Pick transform: strong if this record is rare AND coin flip passes
        use_strong = (self.strong_for_rare
                       and rec.get('has_rare_class', False)
                       and random.random() < self.strong_aug_p)
        transform = self.tf_strong if use_strong else self.tf_mild

        try:
            out = transform(image=image, mask=mask)
        except Exception as e:
            # Fallback to mild aug if strong aug fails on edge cases
            out = self.tf_mild(image=image, mask=mask)
        img_tensor = out['image']                       # tensor [3, H, W]
        mask_aug   = out['mask'].numpy() if hasattr(out['mask'], 'numpy') else np.array(out['mask'])

        # Re-derive class list AFTER augmentation (some regions may have been
        # cropped out by aggressive crop / shear).
        bincnt = np.bincount(mask_aug.ravel().astype(np.int64), minlength=256)
        present_ids = [c for c in rec['classes']
                        if int(bincnt[c]) >= self.min_pixels_after_aug]
        if not present_ids:
            # Aug accidentally removed all classes — fall back to original list
            present_ids = list(rec['classes'])

        return {
            'image':       img_tensor,
            'class_ids':   torch.tensor(present_ids, dtype=torch.long),
            'dataset':     rec['dataset'],
            'has_rare':    bool(rec.get('has_rare_class', False)),
            'used_strong': bool(use_strong),
        }


def collate_variable_classes(batch: List[Dict]):
    """Collate fn that handles variable-length class_ids per sample by
    packing them into a flat tensor + per-sample lengths."""
    images   = torch.stack([b['image'] for b in batch])               # [B, 3, H, W]
    datasets = [b['dataset']      for b in batch]
    has_rare = [b['has_rare']     for b in batch]
    used_str = [b['used_strong']  for b in batch]
    class_ids = [b['class_ids']   for b in batch]                     # list of [n_i]
    lengths   = torch.tensor([len(c) for c in class_ids], dtype=torch.long)
    flat_ids  = torch.cat(class_ids, dim=0) if class_ids else torch.tensor([], dtype=torch.long)
    return {
        'images':       images,
        'class_ids':    flat_ids,         # flat per-sample-class id tensor
        'lengths':      lengths,          # how many classes per sample
        'datasets':     datasets,         # list[str]
        'has_rare':     has_rare,         # list[bool]
        'used_strong':  used_str,
    }


# ----------------------------------------------------------------------
def make_weighted_sampler(weights_path: str | Path):
    """Helper to construct a WeightedRandomSampler from the JSON weights
    file produced by build_siglip_manifest.py."""
    from torch.utils.data import WeightedRandomSampler
    weights = json.loads(Path(weights_path).read_text(encoding='utf-8'))
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                  num_samples=len(weights), replacement=True)


if __name__ == '__main__':
    # Smoke test: load one record from the merged manifest and print shape
    import sys, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest', type=str)
    args = ap.parse_args()
    ds = SigLIPManifestDataset(args.manifest, input_size=256)
    print(f'records: {len(ds)}')
    item = ds[0]
    print('image:', item['image'].shape, item['image'].dtype)
    print('class_ids:', item['class_ids'].tolist())
    print('dataset:', item['dataset'], 'rare:', item['has_rare'],
          'used_strong:', item['used_strong'])
