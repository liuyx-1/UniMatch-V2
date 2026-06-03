"""Temporal Thin-Structure Consistency (TTSC) — for surgical video
segmentation of small targets (cystic_artery, calot_triangle, needle tip)
and thin elongated structures (cystic_duct, instrument shaft, suture).

Inspired by:
  Few-Shot Video Object Segmentation in X-Ray Angiography Using Local
  Matching and Spatio-Temporal Consistency Loss (Wang et al.).

Core design:
  1) LocalDirectionalMatcher        — 8-direction × R-radius cosine
                                       matching (no global dense match,
                                       no optical flow).
  2) DynamicSmallStructureRouter    — runtime per-pixel weighting that
                                       elevates SMALL or THIN regions
                                       WITHOUT a hardcoded class list.
                                       Combines connected-component
                                       isolation + boundary-density
                                       thinness + prediction entropy.
  3) TemporalThinStructureConsistencyLoss
                                     — bundles feature / mask / edge
                                       consistency losses, all gated by
                                       the dynamic router.

Usage at training time:

    from util.temporal_thin import TemporalThinStructureConsistencyLoss

    ttsc = TemporalThinStructureConsistencyLoss(
                num_classes=cfg['nclass'],
                radius=3, n_directions=8,
                use_feat_loss=False,   # turn on if you have decoder feats
                use_mask_loss=True,
                use_edge_loss=True,
                use_contrastive=False,
            )

    # Inside training step (t+1 = teacher weak view, t = student strong view)
    losses = ttsc(
        feat_t=None, feat_tp1=None,
        prob_t=p_u_s1,                       # [B, C, H, W]   student strong
        prob_tp1=prob_u_w_cm1,               # [B, C, H, W]   teacher weak, CutMix-aligned
        boundary_t=None, boundary_tp1=None,  # optional [B, 1, H, W]
        valid_mask=valid1,                   # [B, H, W]      ignore-mask filter
    )
    loss = ... + lambda_ttsc * losses['loss_total']
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Local directional matcher
# ============================================================
class LocalDirectionalMatcher(nn.Module):
    """For each pixel i in frame t, find the most similar pixel in frame
    t+1 within a local directional neighbourhood.

    Avoids optical flow and global matching. Uses K directions × R radii
    candidates, implemented with vectorised torch.roll operations.
    """

    def __init__(self, n_directions: int = 8, radius: int = 3,
                 include_zero_shift: bool = True):
        super().__init__()
        assert n_directions in (4, 8), 'use 4 or 8 directions'
        self.n_dirs = n_directions
        self.radius = int(radius)
        self.include_zero_shift = include_zero_shift

        if n_directions == 4:
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        else:
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)]
        offsets: List[Tuple[int, int]] = []
        if include_zero_shift:
            offsets.append((0, 0))
        for r in range(1, self.radius + 1):
            for dy, dx in dirs:
                offsets.append((r * dy, r * dx))
        # store as (M, 2) buffer
        self.register_buffer(
            'offsets',
            torch.tensor(offsets, dtype=torch.int64),  # [M, 2]
            persistent=False,
        )
        self.n_candidates = len(offsets)

    @staticmethod
    def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return x / x.norm(dim=1, keepdim=True).clamp_min(eps)

    def forward(self, feat_t: torch.Tensor, feat_tp1: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_t   : [B, D, H, W]   reference frame feature
            feat_tp1 : [B, D, H, W]   other frame feature (sampling source)

        Returns:
            matched_feat_tp1: [B, D, H, W] — feat_tp1 picked at best-match position
            match_score:      [B, 1, H, W] — cosine similarity at chosen offset
            match_offset:     [B, 2, H, W] — (dy, dx) selected per pixel
        """
        if feat_t.shape != feat_tp1.shape or feat_t.ndim != 4:
            raise ValueError(f'expected matching [B,D,H,W], '
                             f'got {tuple(feat_t.shape)} vs {tuple(feat_tp1.shape)}')

        B, D, H, W = feat_t.shape
        fa = self._normalize(feat_t)               # [B, D, H, W]
        fb = self._normalize(feat_tp1)             # [B, D, H, W]

        sims: List[torch.Tensor] = []
        sampled: List[torch.Tensor] = []
        boundary_mask: List[torch.Tensor] = []

        for (dy, dx) in self.offsets.tolist():
            # shift fb by (dy, dx): out-of-bound positions are zero-padded
            shifted = torch.roll(fb, shifts=(dy, dx), dims=(2, 3))
            shifted_raw = torch.roll(feat_tp1, shifts=(dy, dx), dims=(2, 3))

            # mark invalid positions where the roll wrapped around
            valid = torch.ones(B, 1, H, W, device=fa.device, dtype=fa.dtype)
            if dy > 0:  valid[:, :,  :dy, :] = 0
            if dy < 0:  valid[:, :, dy:, :] = 0
            if dx > 0:  valid[:, :, :,  :dx] = 0
            if dx < 0:  valid[:, :, :, dx:] = 0

            s = (fa * shifted).sum(dim=1, keepdim=True)        # [B, 1, H, W]
            sims.append(s)
            sampled.append(shifted_raw)
            boundary_mask.append(valid)

        sims_stack    = torch.cat(sims,    dim=1)              # [B, M, H, W]
        sampled_stack = torch.stack(sampled, dim=1)            # [B, M, D, H, W]
        valid_stack   = torch.cat(boundary_mask, dim=1)        # [B, M, H, W]

        # invalidate roll-wrapped pixels by setting their sim to -inf
        sims_stack = sims_stack.masked_fill(valid_stack < 0.5, float('-inf'))

        # pick best candidate per pixel
        match_score, match_idx = sims_stack.max(dim=1, keepdim=True)   # [B, 1, H, W]

        # gather feature at chosen index
        idx_exp = match_idx.unsqueeze(2).expand(-1, -1, D, -1, -1)     # [B, 1, D, H, W]
        matched_feat = sampled_stack.gather(1, idx_exp).squeeze(1)     # [B, D, H, W]

        # decode offset (dy, dx) from chosen index
        offsets_buf = self.offsets.to(match_idx.device)                # [M, 2]
        match_offset = offsets_buf[match_idx.squeeze(1)].permute(0, 3, 1, 2).contiguous()
        match_offset = match_offset.to(dtype=feat_t.dtype)             # [B, 2, H, W]

        # zero-out match_score where it's still -inf (all candidates invalid)
        match_score = match_score.masked_fill(torch.isinf(match_score), 0.0)

        return matched_feat, match_score, match_offset


# ============================================================
# 2. Dynamic small-structure router (the novel piece)
# ============================================================
class DynamicSmallStructureRouter(nn.Module):
    """Decide per-pixel whether a region behaves like a SMALL TARGET or
    THIN STRUCTURE, **without** any hardcoded class list.

    Three orthogonal geometric cues:

      (a) isolation     — fraction of NEIGHBOURING pixels NOT sharing the
                          same predicted class. Connected-component size
                          proxy: more isolated → smaller object.
      (b) thinness      — local boundary density (mask gradient magnitude
                          smoothed inside a window). Thin structures show
                          high boundary-to-area ratio.
      (c) uncertainty   — per-pixel prediction entropy, normalised to [0, 1].
                          Tail-class regions tend to be uncertain.

    The three are normalised to [0, 1] and combined into a soft weight:

        w_dyn(i) = (a_w · isolation_i + b_w · thinness_i + c_w · uncertainty_i)
                  / (a_w + b_w + c_w)

    Returns w_dyn ∈ [0, 1] per pixel, used to gate the consistency losses.

    No class list, no offline analysis — fully runtime, fully differentiable
    (router weights do not produce gradients, but they are computed from
    detached predictions so the router is purely a gating mechanism).
    """

    def __init__(self,
                 isolation_ks: int = 7,
                 thinness_ks: int = 7,
                 a_w: float = 1.0,
                 b_w: float = 1.0,
                 c_w: float = 0.5,
                 base_weight: float = 0.2,
                 ):
        """
        Args:
            isolation_ks:  window for the "same-class neighbour" pool
            thinness_ks:   window for the boundary-density pool
            a_w/b_w/c_w:   mixing weights for isolation/thinness/uncertainty
            base_weight:   minimum weight every pixel gets (so non-small
                           regions still receive a small consistency signal)
        """
        super().__init__()
        if isolation_ks % 2 == 0: isolation_ks += 1
        if thinness_ks  % 2 == 0: thinness_ks  += 1
        self.iso_ks = isolation_ks
        self.thin_ks = thinness_ks
        self.a_w, self.b_w, self.c_w = float(a_w), float(b_w), float(c_w)
        self.base_weight = float(base_weight)

    @torch.no_grad()
    def _isolation(self, hard_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        hard_pred : [B, H, W]  argmax predicted class id
        Returns   : [B, H, W]  ∈ [0, 1], higher = more isolated
        """
        B, H, W = hard_pred.shape
        # one-hot: [B, C, H, W]
        oh = F.one_hot(hard_pred.clamp_max(num_classes - 1),
                       num_classes=num_classes).permute(0, 3, 1, 2).float()
        # local same-class fraction via depthwise box filter
        ker = torch.ones(1, 1, self.iso_ks, self.iso_ks,
                         device=hard_pred.device, dtype=oh.dtype)
        same_frac = F.conv2d(oh, ker.expand(num_classes, 1, -1, -1),
                             padding=self.iso_ks // 2, groups=num_classes)
        same_frac = same_frac / (self.iso_ks * self.iso_ks)            # [B, C, H, W]
        # at each pixel, take the channel of its own class
        own_frac = same_frac.gather(1, hard_pred.unsqueeze(1).long()).squeeze(1)
        return (1.0 - own_frac).clamp(0.0, 1.0)                        # [B, H, W]

    @torch.no_grad()
    def _thinness(self, hard_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
        """
        Local boundary density: ratio of boundary pixels in a window.
        Higher value → thin structure or boundary-dense region.
        """
        # boundary indicator from class index gradient (Manhattan style)
        diff_h = (hard_pred[:, 1:, :] != hard_pred[:, :-1, :]).float()
        diff_w = (hard_pred[:, :, 1:] != hard_pred[:, :, :-1]).float()
        boundary = torch.zeros_like(hard_pred, dtype=torch.float32)
        boundary[:, 1:, :] = torch.maximum(boundary[:, 1:, :], diff_h)
        boundary[:, :, 1:] = torch.maximum(boundary[:, :, 1:], diff_w)

        ker = torch.ones(1, 1, self.thin_ks, self.thin_ks,
                         device=hard_pred.device, dtype=boundary.dtype)
        dens = F.conv2d(boundary.unsqueeze(1), ker,
                        padding=self.thin_ks // 2)
        dens = dens / (self.thin_ks * self.thin_ks)
        return dens.squeeze(1).clamp(0.0, 1.0)                         # [B, H, W]

    @torch.no_grad()
    def _uncertainty(self, prob: torch.Tensor) -> torch.Tensor:
        """Normalised per-pixel entropy, [0, 1]."""
        C = prob.size(1)
        log_c = math.log2(float(max(C, 2)))
        H = -(prob * torch.log2(prob.clamp_min(1e-12))).sum(dim=1) / log_c
        return H.clamp(0.0, 1.0)                                       # [B, H, W]

    @torch.no_grad()
    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        """
        prob : [B, C, H, W] — softmax probabilities of the REFERENCE frame.
        returns : [B, H, W] ∈ [base_weight, 1.0]
        """
        if prob.ndim != 4:
            raise ValueError(f'expected prob [B,C,H,W], got {tuple(prob.shape)}')
        C = prob.size(1)
        hard = prob.argmax(dim=1)                                      # [B, H, W]
        iso  = self._isolation(hard, C)
        thn  = self._thinness(hard, C)
        unc  = self._uncertainty(prob)
        total_w = self.a_w + self.b_w + self.c_w
        w = (self.a_w * iso + self.b_w * thn + self.c_w * unc) / max(total_w, 1e-8)
        # mix with base so non-small regions still get a tiny contribution
        return self.base_weight + (1.0 - self.base_weight) * w


# ============================================================
# 3. Top-level TTSC loss
# ============================================================
class TemporalThinStructureConsistencyLoss(nn.Module):
    """Bundles feature / mask / edge / contrastive consistency losses,
    all gated by the dynamic small-structure router."""

    def __init__(self,
                 num_classes: int,
                 radius: int = 3,
                 n_directions: int = 8,
                 use_feat_loss: bool = False,
                 use_mask_loss: bool = True,
                 use_edge_loss: bool = True,
                 use_contrastive: bool = False,
                 lambda_feat: float = 0.1,
                 lambda_mask: float = 0.2,
                 lambda_edge: float = 0.1,
                 lambda_con:  float = 0.05,
                 match_threshold: float = 0.3,
                 contrastive_tau: float = 0.1,
                 router_kwargs: Optional[dict] = None,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = LocalDirectionalMatcher(
            n_directions=n_directions, radius=radius)
        self.router = DynamicSmallStructureRouter(**(router_kwargs or {}))

        self.use_feat   = use_feat_loss
        self.use_mask   = use_mask_loss
        self.use_edge   = use_edge_loss
        self.use_con    = use_contrastive
        self.l_feat = float(lambda_feat)
        self.l_mask = float(lambda_mask)
        self.l_edge = float(lambda_edge)
        self.l_con  = float(lambda_con)
        self.tau_m  = float(match_threshold)
        self.tau_c  = float(contrastive_tau)

    # ----------------------------------------- helpers
    @staticmethod
    def _gather_at_offset(src: torch.Tensor, offset: torch.Tensor
                          ) -> torch.Tensor:
        """Sample `src` at each pixel offset using grid_sample. Both
        tensors share the same spatial dims [B, C, H, W] and offset
        [B, 2, H, W] is (dy, dx) in pixel units."""
        B, C, H, W = src.shape
        device = src.device
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=src.dtype),
            torch.arange(W, device=device, dtype=src.dtype),
            indexing='ij')
        gx = (xx.unsqueeze(0).expand(B, -1, -1) + offset[:, 1]) / max(W - 1, 1)
        gy = (yy.unsqueeze(0).expand(B, -1, -1) + offset[:, 0]) / max(H - 1, 1)
        grid = torch.stack((2 * gx - 1, 2 * gy - 1), dim=-1)            # [B, H, W, 2]
        return F.grid_sample(src, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)

    # ----------------------------------------- main
    def forward(self,
                feat_t: Optional[torch.Tensor],
                feat_tp1: Optional[torch.Tensor],
                prob_t:  torch.Tensor,
                prob_tp1: torch.Tensor,
                boundary_t:  Optional[torch.Tensor] = None,
                boundary_tp1: Optional[torch.Tensor] = None,
                valid_mask: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        Notation: t = reference (e.g. student strong), t+1 = source (e.g.
        teacher weak / EMA, already CutMix-aligned to t).

        prob_t / prob_tp1 : [B, C, H, W]  (softmax)
        feat_t / feat_tp1 : [B, D, H, W]  (only if use_feat_loss; optional)
        boundary_*        : [B, 1, H, W]  (only if use_edge_loss; optional)
        valid_mask        : [B, H, W]     (ignore-mask filter)
        """
        device = prob_t.device
        prob_tp1 = prob_tp1.detach()

        # ---- 1) local directional match ---------------------------------
        if feat_t is not None and feat_tp1 is not None:
            with torch.no_grad():
                _, score, offset = self.matcher(feat_t.detach(), feat_tp1.detach())
        else:
            # fall back: use prob as feature for matching
            with torch.no_grad():
                _, score, offset = self.matcher(prob_t.detach(), prob_tp1)
        score = score.squeeze(1)                                       # [B, H, W]

        # ---- 2) dynamic small-structure routing weight -----------------
        w_dyn = self.router(prob_t.detach())                           # [B, H, W]

        # ---- 3) combined per-pixel weight -------------------------------
        gate = (score >= self.tau_m).float()
        w = w_dyn * gate
        if valid_mask is not None:
            w = w * valid_mask.to(w.dtype)
        denom = w.sum().clamp_min(1.0)

        out: Dict[str, torch.Tensor] = {}
        loss_total = torch.zeros((), device=device)

        # ---- 4) feature consistency -------------------------------------
        if self.use_feat and feat_t is not None and feat_tp1 is not None:
            matched_feat = self._gather_at_offset(feat_tp1.detach(), offset)
            diff = (feat_t - matched_feat).pow(2).sum(dim=1)            # [B, H, W]
            loss_feat = (diff * w).sum() / denom
            out['loss_feat_temp'] = loss_feat
            loss_total = loss_total + self.l_feat * loss_feat

        # ---- 5) mask (probability) consistency --------------------------
        if self.use_mask:
            matched_prob = self._gather_at_offset(prob_tp1, offset)     # [B, C, H, W]
            diff = (prob_t - matched_prob).abs().sum(dim=1)             # [B, H, W]
            loss_mask = (diff * w).sum() / denom
            out['loss_mask_temp'] = loss_mask
            loss_total = loss_total + self.l_mask * loss_mask

        # ---- 6) edge / boundary consistency -----------------------------
        if self.use_edge and boundary_t is not None and boundary_tp1 is not None:
            matched_bnd = self._gather_at_offset(boundary_tp1.detach(), offset)
            # gate by "either side has high boundary response"
            edge_gate = (boundary_t.squeeze(1) > 0.3) | (matched_bnd.squeeze(1) > 0.3)
            edge_w = w * edge_gate.to(w.dtype)
            edge_denom = edge_w.sum().clamp_min(1.0)
            diff = (boundary_t.squeeze(1) - matched_bnd.squeeze(1)).abs()
            loss_edge = (diff * edge_w).sum() / edge_denom
            out['loss_edge_temp'] = loss_edge
            loss_total = loss_total + self.l_edge * loss_edge

        # ---- 7) contrastive (optional) ----------------------------------
        if self.use_con and feat_t is not None and feat_tp1 is not None:
            with torch.no_grad():
                fa = LocalDirectionalMatcher._normalize(feat_t)
                fb = LocalDirectionalMatcher._normalize(feat_tp1)
            sims_pos = (fa * self._gather_at_offset(fb, offset)).sum(1)  # [B, H, W]
            # negative: shifted fb at orthogonal offset (offset + π/2 rotated)
            neg_offset = torch.stack([offset[:, 1], -offset[:, 0]], dim=1)
            sims_neg = (fa * self._gather_at_offset(fb, neg_offset)).sum(1)
            num = torch.exp(sims_pos / self.tau_c)
            den = num + torch.exp(sims_neg / self.tau_c)
            loss_con = (-torch.log(num / den.clamp_min(1e-12)) * w).sum() / denom
            out['loss_st_con'] = loss_con
            loss_total = loss_total + self.l_con * loss_con

        out['loss_total']  = loss_total
        out['gate_ratio']  = (w > 0).float().mean()
        out['dyn_weight_mean'] = w_dyn.mean()
        return out
