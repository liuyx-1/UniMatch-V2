"""Tangent-space alignment between two pixel-wise feature streams.

Geometric setup: each feature map F : Omega -> R^D embeds the image plane
as a 2-D submanifold of R^D. At pixel p the *structure tensor*
    g(p) = J_F(p)^T J_F(p)  in R^{2x2}
is the pullback metric — it captures the principal directions and
anisotropy of feature change in image-plane coordinates, *independent of
the ambient dimension D*. Two streams agree on local geometry iff their
normalized structure tensors agree, regardless of channel count.

L_tan defined here:
    L_tan = mean_p || g_a(p)/||g_a(p)|| - g_b(p)/||g_b(p)|| ||_F^2
with valid-pixel gating on degenerate (uniform) regions where g is near 0.

This is genuine differential-geometric consistency, not edge BCE; it
applies even with no labels and is dimension-agnostic, so the semantic
logits (D = nclass) and the boundary head's hidden feature map (D = 32)
can be compared directly.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(-1, -2).contiguous()


def _sobel(feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel Sobel gradients. feat: [B,D,H,W] -> (gx, gy) each same shape."""
    B, D, H, W = feat.shape
    kx = _SOBEL_X.to(device=feat.device, dtype=feat.dtype)
    ky = _SOBEL_Y.to(device=feat.device, dtype=feat.dtype)
    x = feat.reshape(B * D, 1, H, W)
    gx = F.conv2d(x, kx, padding=1).reshape(B, D, H, W)
    gy = F.conv2d(x, ky, padding=1).reshape(B, D, H, W)
    return gx, gy


def structure_tensor(feat: torch.Tensor) -> torch.Tensor:
    """Pullback metric g(p) = J^T J packed as (g_xx, g_xy, g_yy).

    Args:
        feat: [B, D, H, W]
    Returns:
        g_compact: [B, 3, H, W] with channels = (g_xx, g_xy, g_yy)
    """
    gx, gy = _sobel(feat)
    g_xx = (gx * gx).sum(dim=1)
    g_yy = (gy * gy).sum(dim=1)
    g_xy = (gx * gy).sum(dim=1)
    return torch.stack([g_xx, g_xy, g_yy], dim=1)


def _frob_norm2(g3: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius norm of a packed 2x2 symmetric matrix.

    For g = [[xx, xy], [xy, yy]], ||g||_F^2 = xx^2 + 2*xy^2 + yy^2.
    Args:
        g3: [B, 3, H, W] in (xx, xy, yy) order.
    Returns:
        [B, H, W]
    """
    xx, xy, yy = g3[:, 0], g3[:, 1], g3[:, 2]
    return xx * xx + 2.0 * xy * xy + yy * yy


def tangent_alignment_loss(feat_a: torch.Tensor, feat_b: torch.Tensor,
                            valid_mask: torch.Tensor | None = None,
                            eps: float = 1e-6,
                            detach_a: bool = False,
                            min_norm: float = 1e-4,
                            ) -> torch.Tensor:
    """Scale-invariant agreement between the two streams' structure tensors.

    Args:
        feat_a, feat_b: [B, D_a, H, W] and [B, D_b, H, W]. Channel counts
            may differ; only the 2-D image-plane structure is compared.
        valid_mask:    optional [B, H, W] bool gating (e.g. high-conf pseudo
            mask + non-ignore).
        detach_a:      if True, treat feat_a as the (non-differentiable)
            reference — gradients flow only through feat_b. Useful when the
            semantic stream is the "teacher" and the boundary stream should
            be the one adapting.
        min_norm:      pixels where either ||g||_F < min_norm are dropped
            (uniform regions; ill-defined orientation).

    Returns:
        scalar loss; 0 if no valid pixel remains.
    """
    if detach_a:
        feat_a = feat_a.detach()

    if feat_a.shape[-2:] != feat_b.shape[-2:]:
        feat_b = F.interpolate(feat_b, size=feat_a.shape[-2:],
                                mode='bilinear', align_corners=False)

    g_a = structure_tensor(feat_a)
    g_b = structure_tensor(feat_b)

    n_a = _frob_norm2(g_a).sqrt()      # [B, H, W]
    n_b = _frob_norm2(g_b).sqrt()

    inv_a = 1.0 / n_a.clamp(min=eps)
    inv_b = 1.0 / n_b.clamp(min=eps)

    g_a_n = g_a * inv_a.unsqueeze(1)
    g_b_n = g_b * inv_b.unsqueeze(1)
    diff = g_a_n - g_b_n               # [B, 3, H, W]

    pix_loss = (diff[:, 0] ** 2
                + 2.0 * diff[:, 1] ** 2
                + diff[:, 2] ** 2)      # [B, H, W]

    gate = (n_a > min_norm) & (n_b > min_norm)
    if valid_mask is not None:
        gate = gate & valid_mask.bool()
    # Match pix_loss dtype so AMP (fp16) flow not broken; otherwise stays fp32.
    gate = gate.to(pix_loss.dtype)
    denom = gate.sum().clamp(min=1.0)
    return (pix_loss * gate).sum() / denom


def principal_direction(g3: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Top eigenvector of the 2x2 structure tensor at every pixel.

    Closed-form 2x2 eigendecomposition — no batched SVD needed.
    Returns:
        unit_vec: [B, 2, H, W] with channels = (v_h, v_w), the dominant
        direction (sign arbitrary).
    """
    xx, xy, yy = g3[:, 0], g3[:, 1], g3[:, 2]
    tr = xx + yy
    disc = ((xx - yy) ** 2 + 4.0 * xy * xy).clamp(min=0.0).sqrt()
    lam1 = 0.5 * (tr + disc)
    # eigenvector for lam1: ( xy , lam1 - xx )  or  ( lam1 - yy , xy )
    v1 = xy
    v2 = lam1 - xx
    # fall back when xy ~ 0 (axis-aligned)
    fallback = xy.abs() < eps
    v1 = torch.where(fallback, lam1 - yy, v1)
    v2 = torch.where(fallback, xy,        v2)
    norm = (v1 * v1 + v2 * v2).sqrt().clamp(min=eps)
    return torch.stack([v1 / norm, v2 / norm], dim=1)
