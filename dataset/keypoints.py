"""Keypoint-heatmap dataset for the learnable detector.

Reads the manifest from tools/export_keypoint_labels.py (one sample per image-view)
and produces (image, target_heatmaps[K,hs,hs], channel_weight[K]).
Gaussian peaks per keypoint; channels with no keypoint get weight 0 (ignored in
the loss). Single-peak channels (needle) and multi-peak (claw_tip) are handled
identically (peaks are just summed/maxed into the channel map).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

_NORM = T.Compose([T.ToTensor(),
                   T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class KeypointDataset(Dataset):
    def __init__(self, manifest, data_root, num_channels, size=490, stride=4,
                 sigma=2.0, augment=False):
        self.samples = json.loads(Path(manifest).read_text(encoding="utf-8"))
        self.root = Path(data_root)
        self.K = num_channels
        self.size = size
        self.stride = stride
        self.hs = size // stride
        self.sigma = sigma
        # photometric-only augmentation (keypoint coords unaffected) for regularization
        self.jitter = T.ColorJitter(0.2, 0.2, 0.2, 0.05) if augment else None

    def __len__(self):
        return len(self.samples)

    def _gauss(self, hm, cx, cy):
        s = self.sigma; r = int(3 * s)
        x0, x1 = max(0, int(cx - r)), min(self.hs, int(cx + r + 1))
        y0, y1 = max(0, int(cy - r)), min(self.hs, int(cy + r + 1))
        if x0 >= x1 or y0 >= y1:
            return
        xs = np.arange(x0, x1); ys = np.arange(y0, y1)
        g = np.outer(np.exp(-((ys - cy) ** 2) / (2 * s * s)),
                     np.exp(-((xs - cx) ** 2) / (2 * s * s)))
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(self.root / s["image"]).convert("RGB")
        W, H = img.size
        img_r = img.resize((self.size, self.size), Image.BILINEAR)
        if self.jitter is not None:
            img_r = self.jitter(img_r)          # color only; keypoints unchanged
        sx = self.size / W / self.stride
        sy = self.size / H / self.stride
        target = np.zeros((self.K, self.hs, self.hs), np.float32)
        weight = np.zeros((self.K,), np.float32)
        for kp in s["keypoints"]:
            ch = int(kp["channel"])
            if ch >= self.K:
                continue
            hx, hy = kp["x"] * sx, kp["y"] * sy
            if 0 <= hx < self.hs and 0 <= hy < self.hs:
                self._gauss(target[ch], hx, hy)
                weight[ch] = 1.0
        return _NORM(img_r), torch.from_numpy(target), torch.from_numpy(weight)
