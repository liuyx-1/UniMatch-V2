"""TS-MDR: Text-Semantic Morphology Dynamic Router for surgical-video
small-target and thin-structure consistency.

Five components:

  1. TextMorphologyPrior          — uses a frozen text encoder + attribute
                                    prompts to derive class→morphology
                                    soft assignment s_c ∈ R^4 (cached).
  2. MaskMorphologyExtractor      — per-class geometric statistics from
                                    the current frame prediction
                                    (area, aspect, compactness, conn,
                                    confidence, temporal variation).
  3. TextSemanticMorphologyRouter — small MLP that fuses (s_c, q_t^c) into
                                    a per-batch-per-class soft routing
                                    π_t^c ∈ R^4; emits per-class dynamic
                                    loss weights λ_temp^c / λ_edge^c /
                                    λ_aff^c.
  4. LocalDirectionalMatcher      — 8-direction × R-radius cosine match
                                    with per-class dynamic radius
                                    inherited from the router.
  5. TSMDRConsistencyLoss         — bundles feat / mask / edge / contrastive
                                    losses gated by router weights.

The module is training-only; no inference-time cost.

Symbols (with shapes):
  s_c           : (C, 4)            text morphology prior
  q_t^c         : (B, C, 6)         per-batch per-class morphology features
  π_t^c         : (B, C, 4)         soft routing over {normal, small, thin, st}
  λ_temp^c      : (B, C)
  λ_edge^c      : (B, C)
  λ_aff^c       : (B, C)

Drop-in inside the trainer where prob_u_w (teacher weak, CutMix-aligned to
the strong view) plays the role of P_{t+1}, and pred_u_s1/s2 plays P_t.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. TextMorphologyPrior
# ============================================================
ATTRIBUTE_PROMPTS = {
    'normal':     'a normal-sized surgical region with regular shape',
    'small':      'a small surgical object or tiny anatomical structure',
    'thin':       'a thin elongated tubular surgical structure',
    'small_thin': 'a small and thin tubular or line-like surgical structure',
}
ATTR_ORDER = ('normal', 'small', 'thin', 'small_thin')


class TextMorphologyPrior(nn.Module):
    """Compute s_c ∈ R^{C×4} ONCE from a frozen text encoder + attribute
    prompts, cache it on the buffer, and return on call.

    The caller provides a Python callable ``encode_text(list[str]) → Tensor[N, D]``
    that returns L2-normalised embeddings. We do NOT take ownership of the
    encoder so it can be reused across modules (e.g. LC-PAM's SigLIP2).

    If no encoder is available, the module falls back to a deterministic
    keyword-based prior (regex over class name) so the rest of TS-MDR can
    still be tested end-to-end.
    """
    def __init__(self,
                 class_names: List[str],
                 encode_text: Optional[Callable[[List[str]], torch.Tensor]] = None,
                 tau_txt: float = 0.07,
                 background_keyword: str = 'background'):
        super().__init__()
        self.class_names = list(class_names)
        self.C = len(class_names)
        self.tau_txt = float(tau_txt)
        self.bg_key = background_keyword.lower()

        if encode_text is None:
            s = self._keyword_fallback(class_names)
        else:
            s = self._encode(encode_text, class_names)
        # zero-out background row entirely (router skips bg downstream)
        bg_idx = [i for i, n in enumerate(class_names)
                  if n.lower().startswith(self.bg_key)]
        for i in bg_idx:
            s[i] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.register_buffer('s_c', s, persistent=False)              # (C, 4)
        self.register_buffer('bg_mask',
                             torch.tensor([1.0 if i in bg_idx else 0.0
                                           for i in range(self.C)]),
                             persistent=False)                         # (C,)

    # ---- main encode path ---------------------------------------------
    @torch.no_grad()
    def _encode(self, encode_text, class_names) -> torch.Tensor:
        class_prompts = [
            f'a laparoscopic surgical structure: {n.replace("_", " ")}'
            for n in class_names
        ]
        attr_prompts = [ATTRIBUTE_PROMPTS[a] for a in ATTR_ORDER]
        e_c = encode_text(class_prompts)                              # (C, D)
        e_a = encode_text(attr_prompts)                               # (4, D)
        e_c = F.normalize(e_c.float(), dim=-1)
        e_a = F.normalize(e_a.float(), dim=-1)
        logits = (e_c @ e_a.t()) / self.tau_txt                       # (C, 4)
        return F.softmax(logits, dim=-1).detach().cpu()

    # ---- deterministic fallback ---------------------------------------
    @staticmethod
    def _keyword_fallback(class_names) -> torch.Tensor:
        small_kws = {'tip', 'needle', 'suture', 'thread', 'clip', 'cystic_artery',
                     'cystic_plate'}
        thin_kws  = {'shaft', 'wrist', 'clasper', 'duct', 'vessel'}
        st_kws    = {'cystic_artery', 'cystic_duct', 'needle', 'suture',
                     'l_hook_electrocautery'}
        s = torch.zeros(len(class_names), 4)
        for i, n in enumerate(class_names):
            ln = n.lower()
            if 'background' in ln:
                s[i] = torch.tensor([1.0, 0.0, 0.0, 0.0]); continue
            small = any(k in ln for k in small_kws)
            thin  = any(k in ln for k in thin_kws)
            st    = any(k in ln for k in st_kws)
            base = torch.tensor([0.55, 0.20, 0.15, 0.10])
            if small: base = torch.tensor([0.20, 0.50, 0.10, 0.20])
            if thin:  base = torch.tensor([0.20, 0.10, 0.50, 0.20])
            if st:    base = torch.tensor([0.10, 0.20, 0.20, 0.50])
            s[i] = base
        return s / s.sum(dim=-1, keepdim=True)

    def forward(self) -> torch.Tensor:
        return self.s_c                                                # (C, 4)


# ============================================================
# 2. MaskMorphologyExtractor
# ============================================================
class MaskMorphologyExtractor(nn.Module):
    """Per-batch per-class geometric features (no learnable params).

    Outputs q_t : (B, C, 6) = [area, aspect, compactness, comp_score,
    mean_conf, temporal_var]. All in [0, 1] after normalisation.
    """
    def __init__(self, soft_threshold: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.thr = float(soft_threshold)
        self.eps = float(eps)
        # cached previous-frame soft mask per class, set externally if desired
        self._prev_mask: Optional[torch.Tensor] = None

    def reset_prev(self):
        self._prev_mask = None

    @torch.no_grad()
    def forward(self, prob: torch.Tensor,
                prev_prob: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        prob       : (B, C, H, W) softmax
        prev_prob  : (B, C, H, W) optional reference for temporal_var.
                     If None and self._prev_mask is None, returns 0.
        """
        if prob.ndim != 4:
            raise ValueError(f'expected (B,C,H,W), got {tuple(prob.shape)}')
        B, C, H, W = prob.shape
        device = prob.device
        dtype  = prob.dtype
        m = (prob > self.thr).to(dtype)                                # (B,C,H,W)
        area_pix = m.sum(dim=(2, 3)).clamp_min(self.eps)               # (B,C)
        area = area_pix / (H * W)                                      # (B,C)

        # bbox aspect via spatial moments along axes
        ys = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)
        xs = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)
        my = (m * ys).sum(dim=(2, 3)) / area_pix
        mx = (m * xs).sum(dim=(2, 3)) / area_pix
        sy = ((m * (ys - my.view(B, C, 1, 1)) ** 2).sum(dim=(2, 3))
              / area_pix).sqrt()
        sx = ((m * (xs - mx.view(B, C, 1, 1)) ** 2).sum(dim=(2, 3))
              / area_pix).sqrt()
        large = torch.maximum(sy, sx); small = torch.minimum(sy, sx)
        aspect = (large / (small + self.eps)).clamp(1.0, 50.0)
        aspect_n = (aspect - 1.0) / 49.0                               # normalise to [0, 1]

        # compactness via perimeter (mask gradient sum) / area
        gh = (m[:, :, 1:, :] != m[:, :, :-1, :]).to(dtype)
        gw = (m[:, :, :, 1:] != m[:, :, :, :-1]).to(dtype)
        perim = gh.sum(dim=(2, 3)) + gw.sum(dim=(2, 3))                # (B,C)
        compact = (perim ** 2) / (4.0 * math.pi * area_pix + self.eps)
        compact_n = (compact / (compact + 8.0)).clamp(0.0, 1.0)        # squash

        # connectivity / fragmentation proxy: ratio of boundary length to
        # sqrt(area). Higher = more fragmented or thinner.
        comp_score = (perim / (area_pix.sqrt() + self.eps)).clamp(0.0, 40.0) / 40.0

        # mean confidence inside mask
        mean_conf = (prob * m).sum(dim=(2, 3)) / area_pix              # (B,C)

        # temporal variation: 1 - IoU(M_t^c, M_{t-1}^c)
        if prev_prob is None and self._prev_mask is None:
            temp_var = torch.zeros(B, C, device=device, dtype=dtype)
        else:
            prev = prev_prob if prev_prob is not None else self._prev_mask
            m_prev = (prev > self.thr).to(dtype)
            inter = (m * m_prev).sum(dim=(2, 3))
            union = (m + m_prev - m * m_prev).sum(dim=(2, 3)).clamp_min(self.eps)
            iou = inter / union
            temp_var = 1.0 - iou
        # update cache for next call
        self._prev_mask = m.detach()

        q = torch.stack([
            area.clamp(0, 1),                # 0 area
            aspect_n,                        # 1 elongation
            compact_n,                       # 2 perimeter complexity
            comp_score,                      # 3 fragmentation proxy
            mean_conf.clamp(0, 1),           # 4 confidence
            temp_var.clamp(0, 1),            # 5 temporal variation
        ], dim=-1)                                                     # (B, C, 6)
        return q


# ============================================================
# 3. TextSemanticMorphologyRouter
# ============================================================
class TextSemanticMorphologyRouter(nn.Module):
    """Fuses text prior s_c (C×4) with per-batch morphology q_t (B×C×6)
    via a small MLP. Outputs π_t (B×C×4) and three sets of per-class
    dynamic loss weights.

    Uses a residual mixing schedule β: π = β·s + (1−β)·softmax(MLP(...)),
    where β can be annealed from 0.8 → 0.3 externally.
    """
    ATTR_NAMES = ATTR_ORDER

    def __init__(self,
                 q_dim: int = 6,
                 hidden: int = 32,
                 dropout: float = 0.1,
                 lambda_temp_base: float = 0.1,
                 lambda_edge_base: float = 0.1,
                 lambda_aff_base:  float = 0.5,
                 beta: float = 0.5,
                 use_entropy_reg: bool = True,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__()
        # weights for translating π → λ
        defaults = dict(beta_s=1.0, beta_t=0.7, beta_st=1.5,
                        eta_t=1.0, eta_st=1.5,
                        mu_s=0.7,  mu_st=1.0)
        if weights: defaults.update(weights)
        for k, v in defaults.items(): setattr(self, k, float(v))

        self.beta = float(beta)
        self.lambda_temp_base = float(lambda_temp_base)
        self.lambda_edge_base = float(lambda_edge_base)
        self.lambda_aff_base  = float(lambda_aff_base)
        self.use_entropy_reg  = bool(use_entropy_reg)

        self.mlp = nn.Sequential(
            nn.Linear(4 + q_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 16), nn.GELU(),
            nn.Linear(16, 4),
        )

    def set_beta(self, beta: float):
        """Let the trainer schedule β over epochs."""
        self.beta = float(beta)

    def forward(self, s_c: torch.Tensor, q_t: torch.Tensor,
                bg_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        s_c     : (C, 4)
        q_t     : (B, C, 6)
        bg_mask : (C,) — 1 where background, 0 otherwise. Background is
                  forced to π = [1, 0, 0, 0] and λ_* = 0.
        """
        if s_c.ndim != 2 or s_c.size(-1) != 4:
            raise ValueError(f'expected s_c (C,4), got {tuple(s_c.shape)}')
        if q_t.ndim != 3 or q_t.size(1) != s_c.size(0):
            raise ValueError(f'expected q_t (B,C,6) matching s_c, '
                             f'got {tuple(q_t.shape)}')
        B, C, _ = q_t.shape
        s_b = s_c.unsqueeze(0).expand(B, -1, -1)                       # (B,C,4)
        h   = torch.cat([s_b, q_t], dim=-1)                            # (B,C,10)
        logits = self.mlp(h.view(B * C, -1)).view(B, C, 4)
        pi_mlp = F.softmax(logits, dim=-1)
        pi = self.beta * s_b + (1.0 - self.beta) * pi_mlp              # (B,C,4)

        # background row reset
        if bg_mask is not None:
            bg = bg_mask.view(1, -1, 1).to(pi.dtype)
            ref = torch.zeros_like(pi); ref[..., 0] = 1.0
            pi = (1.0 - bg) * pi + bg * ref

        pn, ps, pt, pst = pi.unbind(dim=-1)                            # each (B,C)
        # dynamic loss multipliers (per batch × per class)
        lam_temp = self.lambda_temp_base * (1.0 + self.beta_s * ps
                                            + self.beta_t * pt
                                            + self.beta_st * pst)
        lam_edge = self.lambda_edge_base * (1.0 + self.eta_t  * pt
                                            + self.eta_st * pst)
        lam_aff  = self.lambda_aff_base  * (1.0 + self.mu_s   * ps
                                            + self.mu_st * pst)
        if bg_mask is not None:
            mask01 = (1.0 - bg_mask.view(1, -1)).to(lam_temp.dtype)
            lam_temp = lam_temp * mask01
            lam_edge = lam_edge * mask01
            lam_aff  = lam_aff  * mask01

        out = dict(pi=pi, lam_temp=lam_temp, lam_edge=lam_edge,
                   lam_aff=lam_aff)
        if self.use_entropy_reg:
            out['loss_route_ent'] = -(pi * pi.clamp_min(1e-8).log()).sum(-1).mean()
        return out


# ============================================================
# 3b. ShapeMorphologyRouter — pure-geometry alternative
# ============================================================
class ShapeMorphologyRouter(nn.Module):
    """Drop-in replacement for :class:`TextSemanticMorphologyRouter` that
    derives the per-class soft routing π_t^c **purely from current-frame
    mask geometry** — no text encoder, no class-name keywords.

    Each of the four attribute scores is a soft (differentiable) indicator
    built from the normalised morphology features
    q_t^c = [area, aspect, compact, frag, conf, temp_var] returned by
    :class:`MaskMorphologyExtractor`. The scores are stacked and
    softmax-normalised to obtain π_t^c.

    Soft indicators:
        normal      = σ(k1·(area − a_n))   · σ(−k2·(aspect_n − r_n))
        small       = σ(−k1·(area − a_s))  · σ(−k2·(aspect_n − r_n))
        thin        = σ(k1·(area − a_th))  · σ(k2·(aspect_n − r_th))
        small_thin  = σ(−k1·(area − a_st)) · σ(k2·(aspect_n − r_st))

    With defaults:
        a_n=0.05, a_s=0.02, a_th=0.005, a_st=0.01
        r_n=0.10, r_th=0.40, r_st=0.30
        k1=120 (area sharpness), k2=15 (aspect sharpness)

    This module has NO learnable parameters; β / entropy regulariser
    machinery is preserved only as a no-op for API parity.
    """
    ATTR_NAMES = ATTR_ORDER

    def __init__(self,
                 lambda_temp_base: float = 0.1,
                 lambda_edge_base: float = 0.1,
                 lambda_aff_base:  float = 0.5,
                 # area thresholds (fractional)
                 a_n: float = 0.05, a_s: float = 0.02,
                 a_th: float = 0.005, a_st: float = 0.01,
                 # normalised-aspect thresholds (q[..., 1] ∈ [0, 1])
                 r_n: float = 0.10, r_th: float = 0.40, r_st: float = 0.30,
                 k_area: float = 120.0, k_aspect: float = 15.0,
                 use_compactness_bonus: bool = True,
                 weights: Optional[Dict[str, float]] = None):
        super().__init__()
        defaults = dict(beta_s=1.0, beta_t=0.7, beta_st=1.5,
                        eta_t=1.0, eta_st=1.5,
                        mu_s=0.7,  mu_st=1.0)
        if weights: defaults.update(weights)
        for k, v in defaults.items(): setattr(self, k, float(v))

        self.lambda_temp_base = float(lambda_temp_base)
        self.lambda_edge_base = float(lambda_edge_base)
        self.lambda_aff_base  = float(lambda_aff_base)
        # thresholds
        self.a_n, self.a_s, self.a_th, self.a_st = a_n, a_s, a_th, a_st
        self.r_n, self.r_th, self.r_st           = r_n, r_th, r_st
        self.k_area, self.k_aspect = k_area, k_aspect
        self.use_compactness_bonus = bool(use_compactness_bonus)

    # API parity (no-op)
    def set_beta(self, beta: float):
        pass

    @staticmethod
    def _sig(x):
        return torch.sigmoid(x)

    def forward(self, s_c: Optional[torch.Tensor], q_t: torch.Tensor,
                bg_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        s_c : ignored (kept for signature compat with TextSemanticRouter).
        q_t : (B, C, 6) — MaskMorphologyExtractor output.
        """
        if q_t.ndim != 3 or q_t.size(-1) < 2:
            raise ValueError(f'expected q_t (B,C,>=2), got {tuple(q_t.shape)}')
        B, C, _ = q_t.shape
        area    = q_t[..., 0]
        aspect  = q_t[..., 1]
        compact = q_t[..., 2] if q_t.size(-1) > 2 else torch.zeros_like(area)

        # ---- four soft indicator scores --------------------------------
        score_n  = self._sig(self.k_area   * (area   - self.a_n)) \
                 * self._sig(-self.k_aspect * (aspect - self.r_n))
        score_s  = self._sig(-self.k_area  * (area   - self.a_s)) \
                 * self._sig(-self.k_aspect * (aspect - self.r_n))
        score_t  = self._sig(self.k_area   * (area   - self.a_th)) \
                 * self._sig(self.k_aspect  * (aspect - self.r_th))
        score_st = self._sig(-self.k_area  * (area   - self.a_st)) \
                 * self._sig(self.k_aspect  * (aspect - self.r_st))

        # Optional compactness bonus: high compactness boosts thin/st
        if self.use_compactness_bonus:
            score_t  = score_t  * (1.0 + 0.5 * compact)
            score_st = score_st * (1.0 + 0.5 * compact)

        scores = torch.stack([score_n, score_s, score_t, score_st],
                             dim=-1)                                    # (B,C,4)
        # softmax with temperature 1.0 — scores are already in [0, ~1.5]
        pi = F.softmax(scores * 5.0, dim=-1)                            # sharpen

        # background row reset
        if bg_mask is not None:
            bg = bg_mask.view(1, -1, 1).to(pi.dtype)
            ref = torch.zeros_like(pi); ref[..., 0] = 1.0
            pi = (1.0 - bg) * pi + bg * ref

        pn, ps, pt, pst = pi.unbind(dim=-1)
        lam_temp = self.lambda_temp_base * (1.0 + self.beta_s * ps
                                            + self.beta_t * pt
                                            + self.beta_st * pst)
        lam_edge = self.lambda_edge_base * (1.0 + self.eta_t  * pt
                                            + self.eta_st * pst)
        lam_aff  = self.lambda_aff_base  * (1.0 + self.mu_s   * ps
                                            + self.mu_st * pst)
        if bg_mask is not None:
            mask01 = (1.0 - bg_mask.view(1, -1)).to(lam_temp.dtype)
            lam_temp = lam_temp * mask01
            lam_edge = lam_edge * mask01
            lam_aff  = lam_aff  * mask01

        return dict(pi=pi, lam_temp=lam_temp, lam_edge=lam_edge,
                    lam_aff=lam_aff,
                    loss_route_ent=torch.zeros((), device=pi.device))


# ============================================================
# 4. LocalDirectionalMatcher
# ============================================================
class LocalDirectionalMatcher(nn.Module):
    """8-direction × R-radius cosine match using torch.roll. Supports
    a per-class dynamic radius derived from the router (caller picks
    R_c = round( R_normal·π_n + R_small·π_s + R_thin·π_t + R_st·π_st )).
    For simplicity the matcher itself is class-agnostic; the caller
    chooses ONE global radius per call (use the maximum or weighted
    average from the router).
    """
    def __init__(self, n_directions: int = 8, radius: int = 3,
                 include_zero_shift: bool = True):
        super().__init__()
        assert n_directions in (4, 8)
        self.n_dirs = n_directions
        self.radius = int(radius)
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if n_directions == 8:
            dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        offs = [(0, 0)] if include_zero_shift else []
        for r in range(1, self.radius + 1):
            for dy, dx in dirs:
                offs.append((r * dy, r * dx))
        self.register_buffer('offsets',
                             torch.tensor(offs, dtype=torch.int64),
                             persistent=False)                          # (M, 2)

    @staticmethod
    def _norm(x, eps=1e-8):
        return x / x.norm(dim=1, keepdim=True).clamp_min(eps)

    def forward(self, feat_t: torch.Tensor, feat_tp1: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        feat_t, feat_tp1: (B, D, H, W)
        Returns:
          matched_feat_tp1: (B, D, H, W)
          match_score:      (B, 1, H, W)
          match_offset:     (B, 2, H, W)  (dy, dx)
        """
        if feat_t.shape != feat_tp1.shape:
            raise ValueError('feat_t and feat_tp1 must have the same shape')
        B, D, H, W = feat_t.shape
        fa = self._norm(feat_t); fb = self._norm(feat_tp1)
        sims, sampled, valids = [], [], []
        for dy, dx in self.offsets.tolist():
            shifted = torch.roll(fb, shifts=(dy, dx), dims=(2, 3))
            sh_raw  = torch.roll(feat_tp1, shifts=(dy, dx), dims=(2, 3))
            v = torch.ones(B, 1, H, W, device=fa.device, dtype=fa.dtype)
            if dy > 0: v[:, :, :dy, :] = 0
            if dy < 0: v[:, :, dy:, :] = 0
            if dx > 0: v[:, :, :, :dx] = 0
            if dx < 0: v[:, :, :, dx:] = 0
            sims.append((fa * shifted).sum(1, keepdim=True))
            sampled.append(sh_raw); valids.append(v)
        S = torch.cat(sims, dim=1)                                     # (B,M,H,W)
        V = torch.cat(valids, dim=1)
        S = S.masked_fill(V < 0.5, float('-inf'))
        score, idx = S.max(dim=1, keepdim=True)                        # (B,1,H,W)
        feats = torch.stack(sampled, dim=1)                            # (B,M,D,H,W)
        matched = feats.gather(1, idx.unsqueeze(2)
                               .expand(-1, -1, D, -1, -1)).squeeze(1)
        offsets = self.offsets.to(idx.device)[idx.squeeze(1)]          # (B,H,W,2)
        offsets = offsets.permute(0, 3, 1, 2).to(feat_t.dtype)         # (B,2,H,W)
        score = score.masked_fill(torch.isinf(score), 0.0)
        return matched, score, offsets


# ============================================================
# 5. TSMDRConsistencyLoss
# ============================================================
class TSMDRConsistencyLoss(nn.Module):
    """Bundles the four loss heads. Each is gated by the dynamic per-class
    weight from the router (broadcast onto per-pixel weights via argmax
    class look-up).
    """
    def __init__(self,
                 num_classes: int,
                 radius: int = 3,
                 n_directions: int = 8,
                 use_feat: bool = True,
                 use_mask: bool = True,
                 use_edge: bool = True,
                 use_con:  bool = False,
                 tau_match: float = 0.5,
                 tau_conf:  float = 0.7,
                 tau_bnd:   float = 0.3,
                 tau_con:   float = 0.1,
                 lambda_f: float = 0.05,
                 lambda_m: float = 0.2,
                 lambda_e: float = 0.1,
                 lambda_con: float = 0.03):
        super().__init__()
        self.C = num_classes
        self.matcher = LocalDirectionalMatcher(n_directions=n_directions,
                                               radius=radius)
        self.use_feat, self.use_mask, self.use_edge, self.use_con = (
            use_feat, use_mask, use_edge, use_con)
        self.tm, self.tp, self.tb, self.tc = (float(tau_match),
                                              float(tau_conf),
                                              float(tau_bnd),
                                              float(tau_con))
        self.lf, self.lm, self.le, self.lcon = (float(lambda_f),
                                                float(lambda_m),
                                                float(lambda_e),
                                                float(lambda_con))

    @staticmethod
    def _sample(src: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        B, C, H, W = src.shape
        device = src.device
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=src.dtype),
            torch.arange(W, device=device, dtype=src.dtype), indexing='ij')
        gx = (xx.unsqueeze(0).expand(B, -1, -1) + offset[:, 1]) / max(W - 1, 1)
        gy = (yy.unsqueeze(0).expand(B, -1, -1) + offset[:, 0]) / max(H - 1, 1)
        grid = torch.stack((2 * gx - 1, 2 * gy - 1), dim=-1)
        return F.grid_sample(src, grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)

    def forward(self,
                feat_t: Optional[torch.Tensor],
                feat_tp1: Optional[torch.Tensor],
                prob_t: torch.Tensor,
                prob_tp1: torch.Tensor,
                lam_temp: torch.Tensor,
                lam_edge: Optional[torch.Tensor] = None,
                boundary_t: Optional[torch.Tensor] = None,
                boundary_tp1: Optional[torch.Tensor] = None,
                valid_mask: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """
        prob_t  : (B, C, H, W)   reference (student strong)
        prob_tp1: (B, C, H, W)   source (teacher weak, aligned)
        lam_temp: (B, C)         per-class temporal multiplier from router
        lam_edge: (B, C)         per-class edge multiplier from router
        """
        device = prob_t.device
        prob_tp1 = prob_tp1.detach()

        # ---- 1) local directional match -------------------------------
        with torch.no_grad():
            ref_a = feat_t.detach() if feat_t is not None else prob_t.detach()
            ref_b = feat_tp1.detach() if feat_tp1 is not None else prob_tp1
            _, score, offset = self.matcher(ref_a, ref_b)
        score = score.squeeze(1)                                       # (B,H,W)

        # ---- 2) per-pixel base gate -----------------------------------
        hard = prob_t.argmax(dim=1)                                    # (B,H,W)
        conf_max = prob_t.detach().max(dim=1).values                   # (B,H,W)
        gate_base = (score >= self.tm) & (conf_max >= self.tp)
        if valid_mask is not None:
            gate_base = gate_base & (valid_mask > 0.5)
        gate_base = gate_base.float()

        # ---- 3) per-pixel temporal weight from lam_temp[hard] ---------
        # gather (B, C) → (B, H, W)
        w_temp = lam_temp.gather(1, hard.view(hard.size(0), -1)
                                 ).view_as(hard)                       # (B,H,W)
        w_pix = w_temp * gate_base
        denom = w_pix.sum().clamp_min(1.0)

        out: Dict[str, torch.Tensor] = {}
        loss_total = torch.zeros((), device=device)

        # ---- 4) feature consistency -----------------------------------
        if self.use_feat and feat_t is not None and feat_tp1 is not None:
            matched = self._sample(feat_tp1.detach(), offset)
            diff = (feat_t - matched).pow(2).sum(dim=1)
            l_feat = (diff * w_pix).sum() / denom
            out['loss_feat_temp'] = l_feat
            loss_total = loss_total + self.lf * l_feat

        # ---- 5) mask probability consistency --------------------------
        if self.use_mask:
            matched_p = self._sample(prob_tp1, offset)
            diff = (prob_t - matched_p).abs().sum(dim=1)
            l_mask = (diff * w_pix).sum() / denom
            out['loss_mask_temp'] = l_mask
            loss_total = loss_total + self.lm * l_mask

        # ---- 6) edge / boundary consistency ---------------------------
        if (self.use_edge and boundary_t is not None
                and boundary_tp1 is not None and lam_edge is not None):
            matched_b = self._sample(boundary_tp1.detach(), offset)
            edge_gate = ((boundary_t.squeeze(1) > self.tb)
                         | (matched_b.squeeze(1) > self.tb)).float()
            w_e_pix = lam_edge.gather(1, hard.view(hard.size(0), -1)
                                      ).view_as(hard) * gate_base * edge_gate
            ed_denom = w_e_pix.sum().clamp_min(1.0)
            diff = (boundary_t.squeeze(1) - matched_b.squeeze(1)).abs()
            l_edge = (diff * w_e_pix).sum() / ed_denom
            out['loss_edge_temp'] = l_edge
            loss_total = loss_total + self.le * l_edge

        # ---- 7) optional contrastive ---------------------------------
        if self.use_con and feat_t is not None and feat_tp1 is not None:
            with torch.no_grad():
                fa = LocalDirectionalMatcher._norm(feat_t)
                fb = LocalDirectionalMatcher._norm(feat_tp1)
            sim_pos = (fa * self._sample(fb, offset)).sum(1)           # (B,H,W)
            neg_off = torch.stack([offset[:, 1], -offset[:, 0]], dim=1)
            sim_neg = (fa * self._sample(fb, neg_off)).sum(1)
            num = (sim_pos / self.tc).exp()
            den = num + (sim_neg / self.tc).exp()
            l_con = (-((num / den.clamp_min(1e-12)).log()) * w_pix).sum() / denom
            out['loss_st_con'] = l_con
            loss_total = loss_total + self.lcon * l_con

        out['loss_total'] = loss_total
        out['gate_ratio'] = gate_base.mean()
        return out


# ============================================================
# Convenience: dynamic radius from router output
# ============================================================
def dynamic_radius(pi_batch_mean: torch.Tensor,
                   R_normal: int = 3, R_small: int = 2,
                   R_thin: int = 5,   R_st: int = 3) -> int:
    """π averaged over (B, C) → scalar; pick a single integer radius
    (rounded weighted average) for the matcher this iteration."""
    p = pi_batch_mean.mean(dim=(0, 1))                                 # (4,)
    r = (R_normal * p[0] + R_small * p[1]
         + R_thin   * p[2] + R_st    * p[3]).item()
    return max(1, int(round(r)))
