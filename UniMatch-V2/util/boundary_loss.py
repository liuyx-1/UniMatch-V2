"""MA-Canny style boundary auxiliary loss for semantic segmentation.

Architecture: tiny 3x3 conv head that turns DPT seg-logits into a single-
channel boundary logit; supervised against a Sobel-derived boundary
target built from the GT (or pseudo) mask.

Loss: BCE + Dice combination.
  L_b = (1 - w) * BCE_with_logits(logits, target, pos_weight)
        + w     * Dice(prob, target)
where w = `dice_weight` (default 0.5). Dice helps for the sparse, line-like
positive set typical of class boundaries.

CutMix edge suppression:
  When `cutmix_box` is supplied (the [B, H, W] {0,1} rectangle pasted by
  the strong-augmentation pipeline), pixels lying within `cutmix_edge_thickness`
  of the rectangle's outline are excluded from the loss. Otherwise the Sobel
  of the cutmixed mask fires on the GT/pseudo seam, creating a spurious
  boundary target that the head would otherwise try to fit.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# Sobel kernels + GT-side boundary builder
# ----------------------------------------------------------------------
def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1., 0., 1.],
                       [-2., 0., 2.],
                       [-1., 0., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = kx.transpose(-1, -2).contiguous()
    return kx, ky


@torch.no_grad()
def boundary_target_from_mask(mask: torch.Tensor, num_classes: int,
                                ignore_index: int = 255) -> torch.Tensor:
    """Return a {0,1}-valued boundary target of shape [B, H, W]."""
    valid = (mask != ignore_index).float()
    safe = mask.clone()
    safe[mask == ignore_index] = 0
    oh = F.one_hot(safe.long(), num_classes).permute(0, 3, 1, 2).float()
    B, C, H, W = oh.shape
    kx, ky = _sobel_kernels(oh.device, oh.dtype)
    x = oh.reshape(B * C, 1, H, W)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = (gx ** 2 + gy ** 2).reshape(B, C, H, W).sum(dim=1).sqrt()
    return (mag > 0).float() * valid


# ----------------------------------------------------------------------
# CutMix edge mask (morphological gradient of the box)
# ----------------------------------------------------------------------
@torch.no_grad()
def cutmix_edge_mask(cutmix_box: torch.Tensor, thickness: int = 2) -> torch.Tensor:
    """Pixels within `thickness` of the rectangle's outline.

    Args:
        cutmix_box: [B, H, W] in {0, 1}.
    Returns:
        [B, H, W] in {0, 1}, where 1 marks "near the seam".
    """
    x = cutmix_box.unsqueeze(1).float()
    D = x
    E_inv = 1.0 - x
    k = max(1, int(thickness))
    for _ in range(k):
        D     = F.max_pool2d(D,     3, stride=1, padding=1)
        E_inv = F.max_pool2d(E_inv, 3, stride=1, padding=1)
    E = 1.0 - E_inv
    edge = (D - E).clamp(min=0.0, max=1.0)
    return edge.squeeze(1)


# ----------------------------------------------------------------------
# Head
# ----------------------------------------------------------------------
class BoundaryAuxHead(nn.Module):
    """3x3 conv -> GN -> GELU -> 1x1 conv -> 1-channel boundary logits.

    Also stores ``pos_weight`` as a registered buffer so the BCE positive
    weight does not need to be re-instantiated as a scalar tensor every step.
    Set / override at construction time via the ``pos_weight`` arg, or
    re-assign ``self.pos_weight.fill_(new_value)`` at runtime.
    """

    def __init__(self, in_channels: int, hidden: int = 32,
                 pos_weight: float = 20.0):
        super().__init__()
        groups = max(1, hidden // 4)
        while hidden % groups != 0 and groups > 1:
            groups -= 1
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # registered as buffer: follows .cuda() / .to(dtype) automatically,
        # ships in state_dict, no per-step tensor allocation.
        self.register_buffer('pos_weight',
                              torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def boundary_loss(boundary_logits: torch.Tensor, mask: torch.Tensor,
                    num_classes: int, ignore_index: int = 255,
                    pos_weight: float = 20.0,
                    valid_mask: Optional[torch.Tensor] = None,
                    cutmix_box: Optional[torch.Tensor] = None,
                    cutmix_edge_thickness: int = 2,
                    dice_weight: float = 0.5,
                    dice_eps: float = 1.0) -> torch.Tensor:
    """BCE + Dice on the Sobel boundary target, with CutMix-seam suppression.

    Args:
        boundary_logits:        [B, 1, H, W]
        mask:                    [B, H, W] long (GT or pseudo, possibly cutmixed)
        valid_mask:              optional [B, H, W] bool, additional gating
                                  (e.g. high-confidence pseudo mask).
        cutmix_box:              optional [B, H, W] in {0,1}; pixels near the
                                  outline are EXCLUDED from the loss.
        cutmix_edge_thickness:   how many pixels around the seam to drop.
        dice_weight:             Dice contribution; 0 = pure BCE, 1 = pure Dice,
                                  0.5 = recommended combo.
    """
    target = boundary_target_from_mask(mask, num_classes, ignore_index)
    if boundary_logits.shape[-2:] != target.shape[-2:]:
        boundary_logits = F.interpolate(boundary_logits, size=target.shape[-2:],
                                         mode='bilinear', align_corners=False)
    logits = boundary_logits.squeeze(1)

    # Per-pixel gate
    gate = (mask != ignore_index).float()
    if valid_mask is not None:
        gate = gate * valid_mask.float()
    if cutmix_box is not None:
        seam = cutmix_edge_mask(cutmix_box, thickness=cutmix_edge_thickness)
        gate = gate * (1.0 - seam)

    # BCE term — accept either a float (legacy) or a pre-built tensor (preferred,
    # from BoundaryAuxHead.pos_weight buffer); avoids per-step tensor allocation.
    # NOTE: when pos_weight is a buffer shared across multiple forward calls
    # (labelled + 2 unlabelled streams), passing the same tensor to BCE saves it
    # for backward each time; any incidental in-place op on the buffer between
    # forward and backward triggers a version-counter error. Force a .clone() per
    # call to give each BCE its own saved snapshot.
    if torch.is_tensor(pos_weight):
        pw = pos_weight.to(device=logits.device, dtype=logits.dtype).clone()
    else:
        pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw,
                                               reduction='none')
    denom = gate.sum().clamp(min=1.0)
    loss_bce = (bce * gate).sum() / denom

    if dice_weight <= 0.0:
        return loss_bce

    # Dice term — operates on probs, gated
    p = torch.sigmoid(logits) * gate
    t = target * gate
    inter = (p * t).flatten(1).sum(dim=1)
    psum  = p.flatten(1).sum(dim=1)
    tsum  = t.flatten(1).sum(dim=1)
    dice  = 1.0 - (2.0 * inter + dice_eps) / (psum + tsum + dice_eps)
    loss_dice = dice.mean()

    if dice_weight >= 1.0:
        return loss_dice
    return (1.0 - dice_weight) * loss_bce + dice_weight * loss_dice
