"""Image-level multi-label classification auxiliary head.

Supports two head architectures and two loss functions, plus warmup +
backbone-detach to prevent the aux head from poisoning the main task
when classes are extremely long-tailed.

  Head architectures:
    * MLP head        — 2-layer MLP on the DINOv2 CLS token
    * ML-Decoder      — per-class learnable query tokens cross-attend to
                         patch features (Ridnik et al., WACV 2023).
                         Each class has its own feature channel, so rare
                         classes do not compete with common ones for CLS
                         capacity.

  Loss functions:
    * BCE             — standard binary cross-entropy with optional
                         long-tail-aware ``pos_weight`` from EMA prior.
    * AsymmetricLoss  — different focusing for positives vs. negatives,
                         with hard-negative probability shifting (clip).
                         Ben-Baruch et al., ICCV 2021. Robust to extreme
                         class imbalance — beats Focal-BCE on long-tail
                         multi-label benchmarks.

  Operational guards (all togglable from config):
    * detach_backbone   — stop the cls-head gradient from updating the
                           backbone. Pure no-op if the aux head fails
                           to learn the tail class.
    * warmup_epochs     — skip the cls loss for the first N epochs so
                           the segmentation head + EMA prior stabilise
                           before introducing a noisy aux objective.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Heads
# ----------------------------------------------------------------------
class ClsHeadMLP(nn.Module):
    """2-layer MLP on the DINOv2 CLS token. [B, D] -> [B, C] logits."""

    def __init__(self, embed_dim: int, nclass: int,
                 hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, nclass),
        )
        self.expects = 'cls'        # consumed by trainer to pick feature

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.body(cls_token)


class MLDecoder(nn.Module):
    """Per-class query tokens cross-attend to patch features.

    Simplified single-block ML-Decoder (Ridnik WACV 2023). For each class
    a learnable query attends to the patch token sequence, producing one
    class-specific pooled feature. A shared `Linear(D, 1)` then outputs
    one logit per class.
    """
    def __init__(self, embed_dim: int, nclass: int,
                 num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(nclass, embed_dim) * 0.02)
        # Pre-LN for cross-attention (q and k/v normalised separately)
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        # Pre-LN for the FFN block (explicit, not folded into Sequential)
        self.ln_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.cls_fc = nn.Linear(embed_dim, 1)
        self.expects = 'patches'

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: [B, N, D]
        B = patches.shape[0]
        q = self.queries.unsqueeze(0).expand(B, -1, -1)        # [B, C, D]
        kv = self.ln_kv(patches)                                # 单次 LN,避免重复
        attn_out, _ = self.cross_attn(self.ln_q(q), kv, kv)
        h = q + attn_out                                        # residual 1
        h = h + self.ffn(self.ln_ffn(h))                        # residual 2 (Pre-LN 显式)
        return self.cls_fc(h).squeeze(-1)                       # [B, C]


def build_cls_head(embed_dim: int, nclass: int, cfg: dict) -> nn.Module:
    decoder_type = str(cfg.get('decoder_type', 'ml_decoder')).lower()
    if decoder_type == 'mlp':
        return ClsHeadMLP(embed_dim, nclass,
                           hidden=int(cfg.get('hidden', 512)),
                           dropout=float(cfg.get('dropout', 0.1)))
    elif decoder_type == 'ml_decoder':
        return MLDecoder(embed_dim, nclass,
                          num_heads=int(cfg.get('num_heads', 8)),
                          dropout=float(cfg.get('dropout', 0.1)))
    raise ValueError(f'unknown cls.decoder_type: {decoder_type}')


# ----------------------------------------------------------------------
# Losses
# ----------------------------------------------------------------------
class AsymmetricLoss(nn.Module):
    """Asymmetric multi-label loss for long-tail classification.

    Ben-Baruch et al., ICCV 2021.

    Reference defaults: gamma_neg=4, gamma_pos=0, clip=0.05 — strong
    focusing on negatives + hard-negative probability shifting; positive
    samples (which are RARE for tail classes) keep flat gradient.

    Args:
        gamma_neg: focusing param for the NEGATIVE term  (down-weights easy negatives).
        gamma_pos: focusing param for the POSITIVE term  (0 keeps full pos gradient).
        clip:      probability shift for negatives (m in the paper); 0 disables.
        eps:       numerical floor for log.
        disable_torch_grad_focal: minor optim flag from the original repo.
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8,
                 disable_torch_grad_focal: bool = True):
        super().__init__()
        self.gamma_neg = float(gamma_neg)
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = float(eps)
        self.disable_torch_grad_focal = bool(disable_torch_grad_focal)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1 - xs_pos
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = target * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - target) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal:
                with torch.no_grad():
                    pt0 = xs_pos * target
                    pt1 = xs_neg * (1 - target)
                    pt  = pt0 + pt1
                    one_sided_gamma = self.gamma_pos * target + self.gamma_neg * (1 - target)
                    one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            else:
                pt0 = xs_pos * target
                pt1 = xs_neg * (1 - target)
                pt  = pt0 + pt1
                one_sided_gamma = self.gamma_pos * target + self.gamma_neg * (1 - target)
                one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_w
        return -loss.mean()


def build_cls_loss(cfg: dict) -> nn.Module:
    loss_type = str(cfg.get('loss_type', 'asl')).lower()
    if loss_type == 'asl':
        return AsymmetricLoss(
            gamma_neg=float(cfg.get('asl_gamma_neg', 4.0)),
            gamma_pos=float(cfg.get('asl_gamma_pos', 0.0)),
            clip=float(cfg.get('asl_clip', 0.05)),
        )
    if loss_type == 'bce':
        # BCE handled inline in trainer (needs pos_weight from EMA prior)
        return None
    raise ValueError(f'unknown cls.loss_type: {loss_type}')


# ----------------------------------------------------------------------
# Target builder
# ----------------------------------------------------------------------
@torch.no_grad()
def cls_target_from_mask(mask: torch.Tensor, nclass: int,
                          min_pixels: int = 32,
                          ignore_index: int = 255) -> torch.Tensor:
    """[B, H, W] mask -> [B, C] {0,1} target. Class c present if >= min_pixels.

    Vectorised via one-hot + sum (no gather): single CUDA kernel launch
    instead of C per-class scans.
    """
    safe = mask.clone()
    safe[safe == ignore_index] = 0
    # [B, H, W] -> [B, H, W, C] -> [B, C, H, W]
    oh = F.one_hot(safe.long(), num_classes=nclass).permute(0, 3, 1, 2)
    cnt = oh.flatten(2).sum(dim=2)                  # [B, C]
    return (cnt >= min_pixels).float()


# Legacy BCE entry point (kept for backward compat with old ckpts / tests)
def cls_loss(logits: torch.Tensor, target: torch.Tensor,
              pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction='mean')


# Back-compat alias — old test.py expects this name
ClsHead = ClsHeadMLP
