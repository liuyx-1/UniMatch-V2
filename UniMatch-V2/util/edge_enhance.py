"""RGB-edge guided residual adapters + Mask-Guided Edge Refiner (MGER).

This module hosts the surgical-scene edge pathway:

  rgb_edge_prior(img)          : Sobel-based raw RGB edge in [0,1]
  MaskGuidedEdgeRefiner(img,e) : lightweight 2D CNN that purifies the raw
                                  RGB edge toward semantic mask boundaries.
                                  Supervised by GT mask boundary on labeled
                                  samples via refined_edge_loss().
  refined_edge_loss(...)       : BCE + Dice between MGER output and GT
                                  boundary (reuses boundary_target_from_mask).
  EdgeSegResidualAdapter       : residual edge-gating of DINO patch features
                                  before DPT decode + boundary aux head.
  EdgeTextResidualAdapter      : same gating mechanism for the LC-PAM /
                                  affinity text branch's visual tokens.
  edge_consistency_loss        : (legacy) BCE+Dice between boundary head
                                  output and raw RGB edge.  Kept for
                                  backward compat; recommended to disable
                                  (weight=0) when MGER is on, because the
                                  refiner now provides the cleaner target.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ----------------------------------------------------------------------
# Raw Sobel edge prior  (now with a small noise-floor to suppress flicker)
# ----------------------------------------------------------------------
def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()
    return kx, ky


@torch.no_grad()
def rgb_edge_prior(img: torch.Tensor,
                   low_q: float = 0.30,
                   high_q: float = 0.95) -> torch.Tensor:
    """Detached soft RGB edge prior in [0, 1] from ImageNet-normalized RGB.

    Args:
        img    : [B, 3, H, W] ImageNet-normalized.
        low_q  : per-image quantile clipped to 0 (noise floor). Larger
                 means more aggressive denoising; 0 disables.
        high_q : per-image quantile mapped to 1.
    """
    if img.ndim != 4 or img.shape[1] != 3:
        raise ValueError(f'expected normalized RGB [B,3,H,W], got {tuple(img.shape)}')
    mean = img.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = img.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    rgb = (img.detach() * std + mean).clamp(0.0, 1.0)
    gray = (0.2989 * rgb[:, 0:1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3])
    kx, ky = _sobel_kernels(gray.device, gray.dtype)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    mag = torch.sqrt(gx.square() + gy.square() + 1e-12)

    flat = mag.flatten(1)
    hi = flat.quantile(float(high_q), dim=1).clamp(min=1e-6).view(-1, 1, 1, 1)
    if low_q > 0:
        lo = flat.quantile(float(low_q), dim=1).view(-1, 1, 1, 1)
        out = ((mag - lo) / (hi - lo).clamp(min=1e-6)).clamp(0.0, 1.0)
    else:
        out = (mag / hi).clamp(0.0, 1.0)
    return out.detach()


def edge_to_tokens(edge: torch.Tensor, patch_h: int, patch_w: int) -> torch.Tensor:
    """Resize [B,1,H,W] edge map to patch grid and flatten to [B,N,1]."""
    if edge.ndim != 4 or edge.shape[1] != 1:
        raise ValueError(f'expected edge [B,1,H,W], got {tuple(edge.shape)}')
    e = F.interpolate(edge.float(), size=(int(patch_h), int(patch_w)),
                      mode='bilinear', align_corners=False)
    return e.flatten(2).transpose(1, 2).contiguous()


# ----------------------------------------------------------------------
# NEW: Mask-Guided Edge Refiner (MGER)
# ----------------------------------------------------------------------
class MaskGuidedEdgeRefiner(nn.Module):
    """Lightweight 2D CNN that purifies the Sobel RGB-edge map toward
    semantic mask boundaries.

    Architecture (depth-separable, ~1k params at mid=16):
        cat(img, edge) -> 3x3 conv -> GN -> GELU
                      -> 3x3 depthwise -> 1x1 pointwise -> GN -> GELU
                      -> Dropout2d -> 3x3 conv -> sigmoid

    Output is zero-initialized (last conv weight = 0, bias = -1.5) so at
    epoch 0 the refined edge is nearly uniform (~ sigmoid(-1.5) = 0.18),
    matching typical mask-boundary density. The model then learns to
    SELECTIVELY light up only the true semantic boundaries.

    Training supervision: BCE + Dice on labeled samples via
    refined_edge_loss(). Inference: pure forward, no GT needed.
    """

    def __init__(self, mid: int = 16, dropout: float = 0.05):
        super().__init__()
        # input = 3 (img) + 1 (edge) channels
        self.in_conv = nn.Conv2d(4, mid, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(4, mid), mid)
        self.dw = nn.Conv2d(mid, mid, kernel_size=3, padding=1,
                            groups=mid, bias=False)
        self.gn_dw = nn.GroupNorm(min(4, mid), mid)             # ← added: norm after DW
        self.pw = nn.Conv2d(mid, mid, kernel_size=1, bias=False)
        self.gn2 = nn.GroupNorm(min(4, mid), mid)
        # Use 1D dropout (channel-wise drop is too aggressive on a 1-out-channel head)
        self.drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.out = nn.Conv2d(mid, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, -1.5)

    def forward(self, img: torch.Tensor, rgb_edge: torch.Tensor) -> torch.Tensor:
        """Args:
            img      : [B,3,H,W]  ImageNet-normalized RGB.
            rgb_edge : [B,1,H,W]  Sobel-based prior from rgb_edge_prior().
        Returns:
            refined  : [B,1,H,W]  sigmoid in [0, 1].
        """
        if rgb_edge.shape[-2:] != img.shape[-2:]:
            rgb_edge = F.interpolate(rgb_edge, size=img.shape[-2:],
                                      mode='bilinear', align_corners=False)
        x = torch.cat([img, rgb_edge.to(dtype=img.dtype)], dim=1)
        h = F.gelu(self.gn1(self.in_conv(x)))             # in_conv → GN → GELU
        h = F.gelu(self.gn_dw(self.dw(h)))                # DW → GN → GELU  (was missing)
        h = F.gelu(self.gn2(self.pw(h)))                  # PW → GN → GELU
        h = self.drop(h)
        return torch.sigmoid(self.out(h))


# ----------------------------------------------------------------------
# NEW: supervision against GT mask boundary
# ----------------------------------------------------------------------
def _gt_boundary_target(mask: torch.Tensor, num_classes: int,
                         ignore_index: int = 255,
                         thickness: int = 3) -> torch.Tensor:
    """Soft binary boundary target derived from a class-id mask.

    Pixels are 1 if their class differs from any 4-neighbor, then dilated
    by ``thickness`` pixels. ``ignore_index`` pixels contribute 0 and are
    NOT spuriously labeled as boundaries.
    """
    if mask.ndim != 3:
        raise ValueError(f'expected mask [B,H,W], got {tuple(mask.shape)}')
    valid = (mask != ignore_index).unsqueeze(1).float()
    # encode each class as a one-hot channel, max-pool diff against 4-neighbors
    # (cheaper than full Sobel-per-class)
    safe = torch.where(mask == ignore_index, torch.zeros_like(mask), mask)
    m = safe.unsqueeze(1).float()
    pad = F.pad(m, (1, 1, 1, 1), mode='replicate')
    diff = ((pad[:, :, 1:-1, 2:] != pad[:, :, 1:-1, 1:-1]).float()
            + (pad[:, :, 1:-1, :-2] != pad[:, :, 1:-1, 1:-1]).float()
            + (pad[:, :, 2:, 1:-1] != pad[:, :, 1:-1, 1:-1]).float()
            + (pad[:, :, :-2, 1:-1] != pad[:, :, 1:-1, 1:-1]).float())
    edge = (diff > 0).float()
    if thickness > 1:
        k = thickness if thickness % 2 == 1 else thickness + 1
        edge = F.max_pool2d(edge, k, stride=1, padding=k // 2)
    return (edge * valid).clamp(0.0, 1.0)


def refined_edge_loss(refined: torch.Tensor,
                       mask: torch.Tensor,
                       num_classes: int,
                       ignore_index: int = 255,
                       thickness: int = 3,
                       dice_weight: float = 0.5,
                       pos_weight: float = 8.0,
                       valid_mask: Optional[torch.Tensor] = None,
                       eps: float = 1.0) -> torch.Tensor:
    """BCE + Dice between MGER output and a GT mask boundary.

    Args:
        refined     : [B,1,H,W] in [0,1] (output of MaskGuidedEdgeRefiner).
        mask        : [B,H,W] long class ids; ignore_index is excluded.
        num_classes : number of seg classes.
        thickness   : GT boundary dilation width (odd).
        pos_weight  : positive-class weighting in BCE (boundaries are
                       very sparse so a value of 5-10 stabilizes early
                       training).
        valid_mask  : optional [B,H,W] {0,1} additional gate (e.g. exclude
                       cutmix seams or low-conf pseudo regions).
    """
    target = _gt_boundary_target(mask, num_classes, ignore_index, thickness)
    if refined.shape[-2:] != target.shape[-2:]:
        refined = F.interpolate(refined, size=target.shape[-2:],
                                 mode='bilinear', align_corners=False)
    p = refined.clamp(1e-5, 1 - 1e-5)
    gate = (mask != ignore_index).unsqueeze(1).float().to(p.dtype)
    if valid_mask is not None:
        v = valid_mask.float().to(p.dtype)
        if v.ndim == 2:                                  # [H,W]
            v = v.unsqueeze(0).unsqueeze(0)
        elif v.ndim == 3:                                # [B,H,W]
            v = v.unsqueeze(1)
        elif v.ndim == 4 and v.shape[1] != 1:            # [B,H,W,1] or wrong
            raise ValueError(
                f'refined_edge_loss valid_mask has shape {tuple(valid_mask.shape)} '
                f'(expected [B,H,W], [B,1,H,W], or [H,W])')
        gate = gate * v

    bce = -(pos_weight * target * torch.log(p)
            + (1.0 - target) * torch.log(1.0 - p))
    loss_bce = (bce * gate).sum() / gate.sum().clamp(min=1.0)
    if dice_weight <= 0:
        return loss_bce
    pg = p * gate
    tg = target * gate
    inter = (pg * tg).flatten(1).sum(dim=1)
    psum = pg.flatten(1).sum(dim=1)
    tsum = tg.flatten(1).sum(dim=1)
    dice = (1.0 - (2.0 * inter + eps) / (psum + tsum + eps)).mean()
    return (1.0 - dice_weight) * loss_bce + dice_weight * dice


# ----------------------------------------------------------------------
# Residual edge adapters (now with gate dropout + soft modulation +
# negative bias init for the aux boundary head)
# ----------------------------------------------------------------------
class EdgeTextResidualAdapter(nn.Module):
    """Residual edge adapter in image-text latent space.

        Z_out = Z + gamma * MLP(LN(Z) * (0.5 + 0.5 * edge))
        gate_drop applied to the modulated input before MLP.
    """

    def __init__(self, dim: int, dropout: float = 0.0,
                 gate_dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
        )
        self.gate_drop = (nn.Dropout(float(gate_dropout))
                          if gate_dropout > 0 else nn.Identity())
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor, edge_tokens: torch.Tensor) -> torch.Tensor:
        if edge_tokens is None:
            return z
        if edge_tokens.shape[:2] != z.shape[:2] or edge_tokens.shape[-1] != 1:
            raise RuntimeError(
                f'edge token shape {tuple(edge_tokens.shape)} does not match latent {tuple(z.shape)}')
        e = edge_tokens.to(dtype=z.dtype).clamp(0.0, 1.0)
        # soft modulation: gate stays alive at >= 0.5 even when edge=0
        mod = 0.5 + 0.5 * e
        return z + self.gamma * self.proj(self.gate_drop(self.norm(z) * mod))


class EdgeSegResidualAdapter(nn.Module):
    """Residual edge adapter before the DPT segmentation head.

    Also predicts a patch-grid boundary logit (auxiliary head). The aux
    head's last bias is initialized to -2 so sigmoid starts at ~ 0.12,
    matching the sparse boundary prior and preventing the BCE term from
    spiking in the first warm-up epochs.
    """

    def __init__(self, dim: int, dropout: float = 0.0,
                 gate_dropout: float = 0.05):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
        )
        self.gate_drop = (nn.Dropout(float(gate_dropout))
                          if gate_dropout > 0 else nn.Identity())
        self.gamma = nn.Parameter(torch.zeros(1))
        self.boundary = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, max(32, dim // 8)),
            nn.GELU(),
            nn.Linear(max(32, dim // 8), 1),
        )
        nn.init.constant_(self.boundary[-1].bias, -2.0)

    def forward(self, feat: torch.Tensor, edge_tokens: torch.Tensor,
                patch_h: int, patch_w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_tokens is None:
            adapted = feat
        else:
            if edge_tokens.shape[:2] != feat.shape[:2] or edge_tokens.shape[-1] != 1:
                raise RuntimeError(
                    f'edge token shape {tuple(edge_tokens.shape)} does not match feature {tuple(feat.shape)}')
            e = edge_tokens.to(dtype=feat.dtype).clamp(0.0, 1.0)
            mod = 0.5 + 0.5 * e
            adapted = feat + self.gamma * self.proj(
                self.gate_drop(self.norm(feat) * mod))
        logits = self.boundary(adapted).transpose(1, 2).reshape(
            feat.shape[0], 1, int(patch_h), int(patch_w))
        return adapted, logits


# ----------------------------------------------------------------------
# Legacy consistency loss vs RAW RGB edge (kept for backward compat).
# Recommended to set its weight to 0 when MGER + refined_edge_loss is on.
# ----------------------------------------------------------------------
def edge_consistency_loss(boundary_logits: torch.Tensor, rgb_edge: torch.Tensor,
                          valid_mask: Optional[torch.Tensor] = None,
                          dice_weight: float = 0.5,
                          eps: float = 1.0) -> torch.Tensor:
    """Soft BCE+Dice consistency between predicted boundary and RGB edge prior."""
    if boundary_logits.shape[-2:] != rgb_edge.shape[-2:]:
        boundary_logits = F.interpolate(boundary_logits, size=rgb_edge.shape[-2:],
                                         mode='bilinear', align_corners=False)
    logits = boundary_logits.squeeze(1)
    target = rgb_edge.squeeze(1).to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
    gate = torch.ones_like(target)
    if valid_mask is not None:
        gate = gate * valid_mask.to(device=logits.device, dtype=logits.dtype)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    loss_bce = (bce * gate).sum() / gate.sum().clamp(min=1.0)
    if dice_weight <= 0.0:
        return loss_bce

    p = torch.sigmoid(logits) * gate
    t = target * gate
    inter = (p * t).flatten(1).sum(dim=1)
    psum = p.flatten(1).sum(dim=1)
    tsum = t.flatten(1).sum(dim=1)
    loss_dice = (1.0 - (2.0 * inter + eps) / (psum + tsum + eps)).mean()
    return (1.0 - dice_weight) * loss_bce + dice_weight * loss_dice
