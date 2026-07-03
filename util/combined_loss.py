"""Surg-SegFormer main segmentation loss: alpha_weight*Tversky + (1-alpha_weight)*CE.

Faithful to the paper repo (surg_segformer/losses/combined.py):
    tversky   = (tp + eps) / (tp + alpha*fp + beta*fn + eps)       # per class
    L_tversky = 1 - tversky.mean()
    L         = alpha_weight * L_tversky + (1 - alpha_weight) * CE
Defaults: tversky_alpha=0.7, tversky_beta=0.3, alpha_weight=0.5, eps=1e-7.

Added vs. the paper: `ignore_index` (255) handling — ignored pixels are excluded
from BOTH the CE and the Tversky tp/fp/fn sums. For datasets without ignore
pixels (e.g. EndoVis2018 12-class) this is numerically identical to the paper.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedCETversky(nn.Module):
    def __init__(self, num_classes: int, alpha_weight: float = 0.5,
                 tversky_alpha: float = 0.7, tversky_beta: float = 0.3,
                 ignore_index: int = 255, ce_weight=None, eps: float = 1e-7):
        super().__init__()
        self.nc = int(num_classes)
        self.aw = float(alpha_weight)
        self.ta = float(tversky_alpha)
        self.tb = float(tversky_beta)
        self.ignore = int(ignore_index)
        self.eps = float(eps)
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore, weight=ce_weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = (target != self.ignore)                          # [B,H,W]
        if not valid.any():
            return logits.sum() * 0.0

        # CE (handles ignore_index natively)
        ce = self.ce(logits, target)

        # Tversky (softmax probs vs one-hot, ignored pixels masked out)
        prob = logits.softmax(dim=1)                              # [B,C,H,W]
        t = target.clone()
        t[~valid] = 0
        onehot = F.one_hot(t.long(), self.nc).permute(0, 3, 1, 2).to(prob.dtype)
        vm = valid.unsqueeze(1).to(prob.dtype)                    # [B,1,H,W]
        prob = prob * vm
        onehot = onehot * vm

        dims = (0, 2, 3)
        tp = (prob * onehot).sum(dims)
        fp = (prob * (1.0 - onehot)).sum(dims)
        fn = ((1.0 - prob) * onehot).sum(dims)
        tversky = (tp + self.eps) / (tp + self.ta * fp + self.tb * fn + self.eps)
        l_tversky = 1.0 - tversky.mean()

        return self.aw * l_tversky + (1.0 - self.aw) * ce
