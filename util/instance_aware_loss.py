"""Instance-aware multi-class segmentation loss.

Adapts Kundu et al. 2026 ("Instance Awareness of Multi-class Semantic
Segmentation Loss Functions") to our 2D surgical-scene setting:

  * One-vs-rest class decomposition: each foreground class is treated as an
    independent binary problem.
  * 2D connected-component decomposition (``scipy.ndimage.label``) per class,
    Blob-loss-style domain (image minus OTHER instances of the same class),
    instead of the paper's 3D Voronoi tessellation (CC loss) -- cheaper and
    equivalent in spirit for 2D frames.
  * Uniform averaging over (class, instance): every instance contributes the
    same gradient regardless of its pixel count or its class's frequency,
    which is what repurposes an instance-imbalance loss into also fixing
    class imbalance (Sec 2.4 of the paper).
  * A minimum-component-size filter and an optional confidence gate let the
    SAME loss run on pseudo-labels for unlabeled images (the paper only
    considers clean ground truth): both are needed to keep noisy / fragmented
    pseudo-components from being treated as real instances.

This is an *additional* auxiliary term, combined additively with the existing
per-pixel segmentation loss (CE / CombinedCETversky) exactly as Eq. (6) of the
paper combines global + instance losses, rather than replacing it:

    L_total = L_seg (unchanged) + w_instance * L_instance

Cost note: connected-component labeling has no native batched GPU op, so this
loop moves each [H,W] binary mask to CPU/numpy once per (image, class). This
is only run on the labeled branch (cheap) and, optionally, on the confidence-
gated pseudo-label of unlabeled strong views after a warmup.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

_STRUCT_8CONN = np.ones((3, 3), dtype=np.int64)


def _binary_ce_tversky(p_c: torch.Tensor, y_c: torch.Tensor, domain: torch.Tensor,
                        tversky_alpha: float, tversky_beta: float,
                        alpha_weight: float, eps: float) -> torch.Tensor:
    """BCE + Tversky between a single-class prob map and its binary target,
    restricted to ``domain`` (Blob-loss-style instance domain)."""
    d = domain
    denom = d.sum().clamp(min=1.0)
    p_clamp = p_c.clamp(eps, 1.0 - eps)
    ce = -(y_c * torch.log(p_clamp) + (1.0 - y_c) * torch.log(1.0 - p_clamp))
    loss_ce = (ce * d).sum() / denom

    tp = (p_c * y_c * d).sum()
    fp = (p_c * (1.0 - y_c) * d).sum()
    fn = ((1.0 - p_c) * y_c * d).sum()
    tversky = (tp + eps) / (tp + tversky_alpha * fp + tversky_beta * fn + eps)
    loss_tversky = 1.0 - tversky
    return alpha_weight * loss_tversky + (1.0 - alpha_weight) * loss_ce


def instance_aware_loss(logits: torch.Tensor,
                         mask: torch.Tensor,
                         class_ids: List[int],
                         ignore_index: int = 255,
                         confidence: Optional[torch.Tensor] = None,
                         conf_thresh: float = 0.95,
                         min_component_size: int = 16,
                         tversky_alpha: float = 0.7,
                         tversky_beta: float = 0.3,
                         alpha_weight: float = 0.5,
                         eps: float = 1e-7) -> torch.Tensor:
    """One-vs-rest, connected-component instance-aware loss.

    Args:
        logits      : [B, C, H, W] fused segmentation logits (post LC-PAM /
                      EDGE fusion -- this loss is agnostic to how the logits
                      were produced).
        mask        : [B, H, W] long class ids (ground truth OR hard
                      pseudo-label). ``ignore_index`` pixels are excluded.
        class_ids   : foreground class indices to decompose (background
                      already excluded by the caller, e.g. via LC-PAM's
                      ``non_bg_mask`` / the dataset's exclude-classes list).
        confidence  : optional [B, H, W] per-pixel confidence in [0,1]. When
                      given (pseudo-label case), pixels below
                      ``conf_thresh`` are treated as ignore BEFORE computing
                      connected components, so unreliable regions cannot
                      spawn spurious "instances".
        min_component_size: components smaller than this (in pixels) are
                      dropped -- filters pseudo-label noise blobs; on clean
                      ground truth it mainly discards annotation speckle.

    Returns:
        Scalar loss, uniformly averaged over (class, instance). Returns 0 if
        no valid instance is found in the batch (e.g. all pixels ignored).
    """
    if logits.dim() != 4:
        raise ValueError(f'expected logits [B,C,H,W], got {tuple(logits.shape)}')
    B, C, H, W = logits.shape
    device = logits.device
    probs = logits.softmax(dim=1)

    valid = (mask != ignore_index)
    if confidence is not None:
        valid = valid & (confidence >= conf_thresh)

    mask_np = mask.detach().cpu().numpy()
    valid_np = valid.detach().cpu().numpy()

    per_class_losses = []
    for c in class_ids:
        per_instance_losses = []
        for b in range(B):
            binary_gt_np = (mask_np[b] == c) & valid_np[b]
            if binary_gt_np.sum() < 1:
                continue
            labeled_np, num = ndimage.label(binary_gt_np, structure=_STRUCT_8CONN)
            if num == 0:
                continue
            y_c = (mask[b] == c).float()
            p_c = probs[b, c]
            valid_b = valid[b]
            for k in range(1, num + 1):
                comp_np = (labeled_np == k)
                if comp_np.sum() < min_component_size:
                    continue
                comp = torch.from_numpy(comp_np).to(device=device)
                # Blob-loss-style domain: everything EXCEPT other instances
                # of the SAME class (this instance + background + other
                # classes remain visible, so false positives elsewhere still
                # get penalized).
                other_same_class = torch.from_numpy(binary_gt_np).to(device=device) & (~comp)
                domain = valid_b & (~other_same_class)
                per_instance_losses.append(
                    _binary_ce_tversky(p_c, y_c, domain.float(),
                                       tversky_alpha, tversky_beta,
                                       alpha_weight, eps))
        if per_instance_losses:
            per_class_losses.append(torch.stack(per_instance_losses).mean())

    if not per_class_losses:
        return torch.zeros((), device=device)
    return torch.stack(per_class_losses).mean()
