"""Temporal Consistency Regularization (TCR) — borrowed from DA-VSN
(Guan et al., ICCV 2021) and adapted to UniMatch-V2's SSL setting.

Two ideas are integrated:

(1) Teacher-student aligned two-stream:
    The current view `cf` is the student's strong-augmentation prediction.
    The reference view `kf` is the EMA teacher's weak-view probability,
    CutMix-composited with the same box as `cf`, so both tensors describe
    the same pixels before the loss is applied.

(2) Entropy-driven propagation gating (the most novel piece):
    Instead of a global confidence threshold (e.g. conf > 0.95), the loss
    is enabled per-pixel only where the kf-side prediction is MORE
    confident (lower entropy) than the cf-side prediction. This avoids
    the head/tail asymmetry of fixed thresholds and adds zero
    hyperparameters.

The module is designed to:
  • drop in directly between two student-predicted probability maps,
  • require no optical-flow preprocessing (when used on dual-aug views),
  • optionally accept a precomputed flow field for true video pairs,
  • return a single scalar loss that can be λ-weighted and summed.

Usage (SSL teacher-student mode, recommended drop-in):

    from util.temporal import entropy_gated_consistency

    p_student = torch.softmax(logits_u_s1, dim=1)    # [B, C, H, W]
    p_teacher = prob_u_w_cutmixed1                   # [B, C, H, W], detached
    loss_tcr = entropy_gated_consistency(p_student, p_teacher)
    loss_total = ... + lambda_tcr * loss_tcr

Usage (video pair mode, with precomputed sampling flow in cf coordinates):

    loss_tcr = entropy_gated_consistency(p_cf, p_kf, flow_kf_to_cf=flow)
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- entropy
def normalized_entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Per-pixel normalised Shannon entropy.

    Args:
        probs:  softmax output, shape [B, C, H, W], sums to 1 along dim=1.
        eps:    numerical floor inside log.

    Returns:
        H of shape [B, H, W] with values in [0, 1].
        H = 0  →  one class dominates (deterministic prediction).
        H = 1  →  uniform distribution (maximally uncertain).
    """
    if probs.ndim != 4:
        raise ValueError(f'expected probs [B,C,H,W], got {tuple(probs.shape)}')
    C = probs.size(1)
    log_c = math.log2(float(C)) if C > 1 else 1.0
    self_info = -probs * torch.log2(probs.clamp_min(eps))
    return self_info.sum(dim=1) / log_c                  # [B, H, W]


# ---------------------------------------------------------------- flow warp
def warp_probs(probs: torch.Tensor,
               flow_kf_to_cf: torch.Tensor) -> torch.Tensor:
    """Backward-sample a probability map from `kf` coordinates to `cf`
    coordinates using a dense sampling flow field.

    Implements the standard image-warp pattern:
        warped(cf_y, cf_x) = probs(cf_y + flow_y, cf_x + flow_x)

    Args:
        probs:           [B, C, H, W] — probabilities at kf coordinates.
        flow_kf_to_cf:   [B, 2, H, W] — dense sampling offset at each cf
                         pixel, in pixel units;
                         channel 0 = dx, channel 1 = dy.

    Returns:
        probs warped to cf coordinates, same shape as `probs`.
    """
    if probs.ndim != 4:
        raise ValueError(f'expected probs [B,C,H,W], got {tuple(probs.shape)}')
    if flow_kf_to_cf.ndim != 4 or flow_kf_to_cf.shape[1] != 2:
        raise ValueError(f'expected flow [B,2,H,W], got {tuple(flow_kf_to_cf.shape)}')
    if flow_kf_to_cf.shape[0] != probs.shape[0] or flow_kf_to_cf.shape[-2:] != probs.shape[-2:]:
        raise ValueError(
            f'flow shape {tuple(flow_kf_to_cf.shape)} is not aligned with probs {tuple(probs.shape)}')
    B, _, H, W = probs.shape
    device = probs.device
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=probs.dtype),
        torch.arange(W, device=device, dtype=probs.dtype),
        indexing='ij',
    )
    grid_x = xx.unsqueeze(0).expand(B, -1, -1) + flow_kf_to_cf[:, 0]
    grid_y = yy.unsqueeze(0).expand(B, -1, -1) + flow_kf_to_cf[:, 1]
    # normalise to [-1, 1] for grid_sample
    grid_x = 2.0 * grid_x / max(W - 1, 1) - 1.0
    grid_y = 2.0 * grid_y / max(H - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)         # [B, H, W, 2]
    return F.grid_sample(probs, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=True)


# ---------------------------------------------------------------- main loss
def entropy_gated_consistency(p_cf: torch.Tensor,
                              p_kf: torch.Tensor,
                              flow_kf_to_cf: torch.Tensor | None = None,
                              ignore_mask: torch.Tensor | None = None,
                              reduction: str = 'mean') -> torch.Tensor:
    """Entropy-gated L1 consistency between two predictions.

    For every pixel position, the loss is enabled ONLY where the kf-side
    (possibly flow-warped) prediction is more confident than the cf-side
    prediction.  No global confidence threshold; no hyperparameter.

    The kf side is detached (acts as soft teacher, no gradient back to it).

    Args:
        p_cf:           softmax output of the cf view, [B, C, H, W].
        p_kf:           softmax output of the kf view, [B, C, H, W].
                        Detached internally — gradients flow to p_cf only.
        flow_kf_to_cf:  optional dense flow [B, 2, H, W] in pixel units.
                        If None, kf and cf are assumed already aligned
                        (e.g. dual-augmentation views of the SAME image
                        without geometric augmentation differences).
        ignore_mask:    optional [B, H, W] bool/float mask, 1 = include in loss.
        reduction:      'mean' | 'sum' | 'none'.

    Returns:
        Scalar loss (or [B, H, W] map if reduction='none').
    """
    if p_cf.ndim != 4 or p_kf.ndim != 4:
        raise ValueError(f'expected p_cf/p_kf [B,C,H,W], got {tuple(p_cf.shape)} and {tuple(p_kf.shape)}')
    if p_cf.shape != p_kf.shape:
        raise ValueError(f'p_cf shape {tuple(p_cf.shape)} must match p_kf shape {tuple(p_kf.shape)}')
    p_kf = p_kf.detach()
    if flow_kf_to_cf is not None:
        p_kf = warp_probs(p_kf, flow_kf_to_cf)

    H_cf = normalized_entropy(p_cf)                      # [B, H, W]
    H_kf = normalized_entropy(p_kf)                      # [B, H, W]
    gate = (H_kf < H_cf).float()                         # 1 where kf wins

    if ignore_mask is not None:
        if ignore_mask.ndim == 4 and ignore_mask.shape[1] == 1:
            ignore_mask = ignore_mask.squeeze(1)
        if ignore_mask.shape != gate.shape:
            raise ValueError(
                f'ignore_mask shape {tuple(ignore_mask.shape)} must match gate shape {tuple(gate.shape)}')
        gate = gate * ignore_mask.to(device=gate.device, dtype=gate.dtype)

    diff = (p_cf - p_kf).abs().sum(dim=1)                # [B, H, W]   L1 over class
    weighted = diff * gate

    if reduction == 'none':
        return weighted
    denom = gate.sum().clamp_min(1.0)
    if reduction == 'sum':
        return weighted.sum()
    return weighted.sum() / denom


# ---------------------------------------------------------------- diagnostics
@torch.no_grad()
def consistency_diagnostics(p_cf: torch.Tensor,
                            p_kf: torch.Tensor,
                            flow_kf_to_cf: torch.Tensor | None = None,
                            ) -> dict:
    """Return scalar diagnostics for logging.

    Keys:
      gate_ratio       fraction of pixels where kf wins (0..1)
      mean_ent_cf      mean cf entropy
      mean_ent_kf      mean (warped) kf entropy
      mean_l1_gated    mean L1 over gated pixels (the actual loss)
    """
    p_kf = p_kf.detach()
    if flow_kf_to_cf is not None:
        p_kf = warp_probs(p_kf, flow_kf_to_cf)
    H_cf = normalized_entropy(p_cf)
    H_kf = normalized_entropy(p_kf)
    gate = (H_kf < H_cf).float()
    diff = (p_cf - p_kf).abs().sum(dim=1)
    denom = gate.sum().clamp_min(1.0)
    return {
        'gate_ratio':    gate.mean().item(),
        'mean_ent_cf':   H_cf.mean().item(),
        'mean_ent_kf':   H_kf.mean().item(),
        'mean_l1_gated': (diff * gate).sum().item() / denom.item(),
    }
