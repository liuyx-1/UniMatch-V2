"""Full hyperbolic feature pathway block.

The block sits between DINOv2 patch tokens and downstream consumers
(DPT decoder in Stage-II; affinity head in Stage-I). Every operation
except (a) the initial down-projection from 768 dim, (b) the final
LogMap0 ejection, and (c) the residual addition back to the 768-dim
patches is performed *inside* the Poincaré ball.

Pipeline:

    X [B,N,768]
        │  (Euclidean)
        ▼  visual_proj + LN
    Z_v [B,N,d]
        │
        ▼  ExpMap0
    P_h ─► HypMLP (Möbius linear + hyp_relu × L)
        │
        ▼  hyperbolic attention against text anchors T_h
    T_attended_h [B,N,d]  =  hyp_weighted_mean(T_h, softmax(-d/τ))
        │
        ▼  Möbius addition  P_fused = P_h ⊕_c T_attended_h
    P_fused_d [B,N,d]
        │
        ▼  Möbius linear up:  d → 768 (still on the ball)
    P_fused_768 [B,N,768]
        │
        ▼  LogMap0   ─►  Euclidean
        │
        ▼  X_aware = X + σ(α) ⊙ LogMap0(P_fused_768)

Stage-I supervision uses A = -dist(P_fused_d, T_h) / τ_h.
Stage-II passes X_aware into the DPT decoder.

The text anchors T_h are stored as a learnable Euclidean buffer; at
forward time we project them onto the ball with project_to_ball() so
they always lie strictly inside the safety region.
"""
from __future__ import annotations
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.hyperbolic import (
    project_to_ball, expmap0, logmap0,
    poincare_distance, mobius_add, hyp_weighted_mean,
    HypLinear, HypMLP,
)


def _build_proj(d_in: int, d_out: int, hidden: int = 256, dropout: float = 0.0):
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Linear(d_in, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out),
    )


class HyperbolicFusionBlock(nn.Module):
    """Full hyperbolic feature pathway for text-vision fusion.

    Parameters that are persisted into the Stage-I checkpoint:
        visual_proj         (768 → d)
        v_pre_ln            (LN on d)
        T_h_param           text anchors (Euclidean storage, projected on use)
        rho                 curvature       (c = softplus(rho) + eps)
        log_tau_h           temperature     (tau = softplus(log_tau_h) + eps)
        hyp_mlp             HypMLP refining P_h within the ball
        hyp_up              HypLinear (d → 768) — Möbius matrix-vector product
        alpha_bias          per-channel residual gate (σ(α) starts ≈ 0.05)
        cls_bias            per-class bias added to the final affinity logits

    Args:
        d_patch:        patch token dim (DINOv2-B/14 = 768)
        d_hyp:          hyperbolic latent dim (default 128)
        n_classes:      C' non-background classes
        hidden_layers:  number of HypMLP refinement layers; 0 disables refinement
        alpha_init:     initial value of alpha_bias (σ(-3) ≈ 0.05)
        curvature_init: initial c (after softplus)
        tau_init:       initial tau_h (after softplus)
        eps:            numeric floor
    """

    def __init__(self,
                 d_patch: int = 768,
                 d_hyp: int = 128,
                 n_classes: int = 6,
                 hidden_layers: int = 2,
                 alpha_init: float = -3.0,                # legacy gated residual init
                 curvature_init: float = 1.0,
                 tau_init: float = 0.1,
                 eps: float = 1e-5,
                 distance_scale: float = 1.0,
                 light: bool = False,
                 grad_checkpoint: bool = False,
                 # ── New: stability + regularisation knobs ───────────────
                 proj_dropout: float = 0.1,               # dropout inside visual_proj
                 fuse_dropout: float = 0.1,               # dropout on Eu_fused before residual
                 fuse_layernorm: bool = True,             # LayerNorm Eu_fused before adding
                 residual_mode: str = 'rezero',           # 'rezero' | 'sigmoid_gate'
                 tau_clamp_max: float = 2.0,              # cap softplus(log_tau_h)
                 tau_clamp_min: float = 0.05,             # floor softplus(log_tau_h)
                 distance_clamp_max: float = 20.0):       # cap Poincaré distance
        """Hyperbolic feature-fusion block.

        Residual fusion modes:
          'rezero'       — X_aware = patches + γ · LN(Eu_fused),  γ init = 0
                            (ReZero: Bachlechner et al. 2020 / adaLN-zero: Peebles & Xie 2023)
                           Strictly identity at init; γ is one scalar parameter.
          'sigmoid_gate' — X_aware = patches + σ(α_bias) · Eu_fused
                           Legacy: per-channel sigmoid gate; α_bias init=-3 → σ≈0.05.

        Args impacting stability/regularisation:
          proj_dropout         — dropout inside `_build_proj(...)` visual head
          fuse_dropout         — dropout on the fused branch (post-LN) before residual
          fuse_layernorm       — LayerNorm Eu_fused before residual (highly recommended)
          tau_clamp_{min,max}  — bound the learnable temperature; prevents softmax collapse
          distance_clamp_max   — cap Poincaré distance; prevents -inf logits at ball boundary

        light / grad_checkpoint as before."""
        super().__init__()
        self.d_patch = int(d_patch)
        self.d_hyp   = int(d_hyp)
        self.n_classes = int(n_classes)
        self.eps = float(eps)
        self.distance_scale = float(distance_scale)

        # ── Stability / regularisation config (persisted in ckpt) ──
        self.proj_dropout       = float(proj_dropout)
        self.fuse_dropout_p     = float(fuse_dropout)
        self.fuse_layernorm     = bool(fuse_layernorm)
        assert residual_mode in ('rezero', 'sigmoid_gate'), residual_mode
        self.residual_mode      = str(residual_mode)
        self.tau_clamp_min      = float(tau_clamp_min)
        self.tau_clamp_max      = float(tau_clamp_max)
        self.distance_clamp_max = float(distance_clamp_max)

        # ── Bridge to hyperbolic space (with dropout inside the head) ──
        self.visual_proj = _build_proj(self.d_patch, self.d_hyp, dropout=self.proj_dropout)
        self.v_pre_ln    = nn.LayerNorm(self.d_hyp)

        # ── Curvature and temperature (softplus-parameterised) ──
        rho_init = float(torch.log(torch.expm1(torch.tensor(max(curvature_init, 1e-4)))))
        tau_init_  = float(torch.log(torch.expm1(torch.tensor(max(tau_init, 1e-4)))))
        self.rho       = nn.Parameter(torch.tensor(rho_init))
        self.log_tau_h = nn.Parameter(torch.tensor(tau_init_))

        # ── Text anchors (Euclidean storage; projected at use) ──
        self.T_h_param = nn.Parameter(torch.zeros(self.n_classes, self.d_hyp))
        nn.init.normal_(self.T_h_param, mean=0.0, std=0.02)
        self._anchor_initialised = False

        # ── Pure hyperbolic refinement (HypMLP) ──
        if hidden_layers > 0:
            dims = [self.d_hyp] + [self.d_hyp] * hidden_layers
            self.hyp_mlp = HypMLP(dims, c=curvature_init, eps=self.eps)
        else:
            self.hyp_mlp = None

        # ── Up-projection d_hyp → d_patch ──
        # `light=True`  : Euclidean nn.Linear AFTER LogMap0 (saves ~20% mem)
        # `light=False` : Möbius linear inside the ball (truer hyperbolic path)
        self.light = bool(light)
        self.grad_checkpoint = bool(grad_checkpoint)
        if self.light:
            self.hyp_up    = None
            self.eu_up     = nn.Linear(self.d_hyp, self.d_patch)
        else:
            self.hyp_up = HypLinear(self.d_hyp, self.d_patch,
                                     c=curvature_init, eps=self.eps)
            self.eu_up  = None

        # ── Residual gate ──
        # Two modes coexist in code so old ckpts (legacy `sigmoid_gate`) still load.
        # Default = 'rezero': strictly identity at init, single learnable scalar γ.
        if self.residual_mode == 'sigmoid_gate':
            self.alpha_bias = nn.Parameter(torch.full((self.d_patch,), float(alpha_init)))
            self.register_parameter('rezero_gamma', None)
        else:  # 'rezero'
            self.rezero_gamma = nn.Parameter(torch.zeros(1))
            # Keep alpha_bias as a non-trainable placeholder so checkpoint shapes
            # never collide (we just don't use it).
            self.register_parameter('alpha_bias', None)

        # ── LayerNorm on fused branch (highly recommended for stability) ──
        if self.fuse_layernorm:
            self.fuse_ln = nn.LayerNorm(self.d_patch)
        else:
            self.fuse_ln = nn.Identity()

        # ── Dropout on fused branch ──
        if self.fuse_dropout_p > 0.0:
            self.fuse_drop = nn.Dropout(p=self.fuse_dropout_p)
        else:
            self.fuse_drop = nn.Identity()

        # ── Class bias on final affinity ──
        self.cls_bias = nn.Parameter(torch.zeros(self.n_classes))

    # ─────────────────────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────────────────────
    def get_c(self)   -> torch.Tensor: return F.softplus(self.rho)       + self.eps
    def get_tau(self) -> torch.Tensor:
        # Floor + ceiling on temperature: prevents softmax collapse
        # (tau→0 ⇒ one-hot attention with vanishing grad on the rest;
        #  tau→∞ ⇒ uniform attention with no discrimination).
        return F.softplus(self.log_tau_h).clamp(min=self.tau_clamp_min,
                                                  max=self.tau_clamp_max) + self.eps

    def get_T_h(self) -> torch.Tensor:
        """Return text anchors projected onto the open Poincaré ball."""
        return project_to_ball(self.T_h_param, c=self.get_c(), eps=self.eps)

    def init_T_from_text_features(self, raw_text_features: torch.Tensor):
        """Initialise T_h_param from external text features (e.g. averaged
        SigLIP-2 template embeddings already passed through a Stage-I-style
        text_proj head). Expects shape [C', d_hyp]."""
        assert raw_text_features.shape == (self.n_classes, self.d_hyp), \
            f'expected [{self.n_classes}, {self.d_hyp}], got {tuple(raw_text_features.shape)}'
        with torch.no_grad():
            # Place anchors on the ball at the same direction (so the
            # initialisation is exactly the Stage-I text vector but moved
            # to the hyperbolic latent).
            t = raw_text_features.to(self.T_h_param.device)
            t = expmap0(t, c=self.get_c(), eps=self.eps)        # → ball
            self.T_h_param.data.copy_(logmap0(t,                 # store tangent so
                                              c=self.get_c(),     # learning is unbounded
                                              eps=self.eps))
        self._anchor_initialised = True

    # ─────────────────────────────────────────────────────────────
    # forward
    # ─────────────────────────────────────────────────────────────
    def forward(self,
                patches: torch.Tensor,
                return_attn: bool = True) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            patches:   [B, N, 768]  DINOv2 patch tokens
        Returns dict with:
            X_aware     [B, N, 768]   Euclidean text-aware patches (for DPT)
            P_fused     [B, N, d_hyp] hyperbolic fused latents (for affinity loss)
            A           [B, N, C']    affinity logits (= -dist/tau + cls_bias)
            T_h         [C', d_hyp]   text anchors on the ball
            attn        [B, N, C']    softmax-attention used for aggregation
            c           scalar tensor — current curvature
            tau         scalar tensor — current temperature
        """
        # Gradient checkpointing path: recompute activations in backward
        if self.grad_checkpoint and self.training and patches.requires_grad:
            from torch.utils.checkpoint import checkpoint
            # checkpoint() doesn't support keyword args → wrap minimally.
            def _ckpt_fn(p): return self._forward_no_ckpt(p)
            return checkpoint(_ckpt_fn, patches, use_reentrant=False)
        return self._forward_no_ckpt(patches)

    def _forward_no_ckpt(self, patches):
        B, N, dpatch = patches.shape
        assert dpatch == self.d_patch, f'expected patch dim {self.d_patch}, got {dpatch}'
        c   = self.get_c()
        tau = self.get_tau()

        # ── (1) Bridge to ball ──
        Z_v = self.v_pre_ln(self.visual_proj(patches))           # [B, N, d_hyp]
        P_h = expmap0(Z_v, c=c, eps=self.eps)                     # [B, N, d_hyp]  in ball

        # ── (2) Hyperbolic refinement (HypMLP keeps points on the ball) ──
        if self.hyp_mlp is not None:
            P_h = self.hyp_mlp(P_h, c=c)

        # ── (3) Hyperbolic attention with text anchors ──
        T_h = self.get_T_h()                                      # [C', d_hyp]
        # broadcast:  [B, N, 1, d] vs [1, 1, C', d]
        d = poincare_distance(
            P_h.unsqueeze(-2),
            T_h.view(1, 1, self.n_classes, self.d_hyp),
            c=c, eps=self.eps,
        )                                                          # [B, N, C']
        # Cap distance to prevent exp(-inf) → 0 cascades that kill attention grads
        # at boundary points. 20 is already deep in the softmax tail.
        d = d.clamp(max=self.distance_clamp_max)
        attn = F.softmax(-d / tau, dim=-1)

        # ── (4) Möbius weighted Fréchet mean of T_h ──
        # broadcast T_h along (B, N) so each patch gets its own mean
        T_exp = T_h.view(1, 1, self.n_classes, self.d_hyp).expand(B, N, -1, -1)
        T_attended_h = hyp_weighted_mean(T_exp, attn, c=c, eps=self.eps)  # [B, N, d_hyp]

        # ── (5) Möbius addition: refined patch ⊕ text-attended summary ──
        P_fused = mobius_add(P_h, T_attended_h, c=c, eps=self.eps)         # [B, N, d_hyp]

        # ── (6) Up-projection to patch dim ──
        if self.light:
            P_fused_tangent = logmap0(P_fused, c=c, eps=self.eps)          # [B, N, d_hyp]
            Eu_fused = self.eu_up(P_fused_tangent)                          # [B, N, 768]
        else:
            P_fused_768 = self.hyp_up(P_fused, c=c)                         # [B, N, 768]
            Eu_fused = logmap0(P_fused_768, c=c, eps=self.eps)              # [B, N, 768]

        # ── (7) Stable residual fusion ──
        # LN constrains Eu_fused to unit variance per channel; dropout regularises;
        # the residual scale is a single learnable scalar (rezero) or a per-channel
        # sigmoid gate (legacy).
        Eu_fused = self.fuse_ln(Eu_fused)
        Eu_fused = self.fuse_drop(Eu_fused)
        if self.residual_mode == 'sigmoid_gate':
            alpha = torch.sigmoid(self.alpha_bias)                          # [768] ∈ (0,1)
            X_aware = patches + alpha.view(1, 1, -1) * Eu_fused
        else:  # 'rezero'
            X_aware = patches + self.rezero_gamma * Eu_fused                # γ scalar, init 0

        # ── (8) Stage-I affinity logits — distance against the FUSED latents.
        # NOTE: we deliberately do not reuse `d` (which is P_h vs T_h,
        # the attention distance) — affinity-loss target should reflect
        # the post-fusion representation P_fused.
        d_fused = poincare_distance(
            P_fused.unsqueeze(-2),
            T_h.view(1, 1, self.n_classes, self.d_hyp),
            c=c, eps=self.eps,
        )                                                                   # [B, N, C']
        A = -self.distance_scale * d_fused / tau.clamp_min(self.eps)
        A = A + self.cls_bias.view(1, 1, -1)

        return {
            'X_aware':  X_aware,
            'P_fused':  P_fused,
            'A':        A,
            'T_h':      T_h,
            'attn':     attn,
            'c':        c,
            'tau':      tau,
            'Z_v':      Z_v,     # tangent-space latent;  reused by Stage-II gate
        }

    # ─────────────────────────────────────────────────────────────
    # ckpt persistence
    # ─────────────────────────────────────────────────────────────
    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        sd = {
            'visual_proj':  self.visual_proj.state_dict(),
            'v_pre_ln':     self.v_pre_ln.state_dict(),
            'T_h_param':    self.T_h_param.detach().cpu(),
            'rho':          self.rho.detach().cpu(),
            'log_tau_h':    self.log_tau_h.detach().cpu(),
            'cls_bias':     self.cls_bias.detach().cpu(),
            'd_patch':      self.d_patch,
            'd_hyp':        self.d_hyp,
            'n_classes':    self.n_classes,
            'distance_scale': self.distance_scale,
            'eps':          self.eps,
            'hidden_layers': 0 if self.hyp_mlp is None else len(self.hyp_mlp.layers),
            'light':         bool(self.light),
            'grad_checkpoint': bool(self.grad_checkpoint),
            # New stability + regularisation config (also needed at load)
            'residual_mode':        self.residual_mode,
            'fuse_layernorm':       bool(self.fuse_layernorm),
            'fuse_dropout_p':       float(self.fuse_dropout_p),
            'proj_dropout':         float(self.proj_dropout),
            'tau_clamp_min':        float(self.tau_clamp_min),
            'tau_clamp_max':        float(self.tau_clamp_max),
            'distance_clamp_max':   float(self.distance_clamp_max),
        }
        # Residual-mode-specific weights
        if self.residual_mode == 'sigmoid_gate':
            sd['alpha_bias'] = self.alpha_bias.detach().cpu()
        else:
            sd['rezero_gamma'] = self.rezero_gamma.detach().cpu()
        # Up-projection (mutually exclusive)
        if self.light:
            sd['eu_up']  = self.eu_up.state_dict()
        else:
            sd['hyp_up'] = self.hyp_up.state_dict()
        # Fused-branch LayerNorm (Identity has no state_dict to save — skip)
        if self.fuse_layernorm:
            sd['fuse_ln'] = self.fuse_ln.state_dict()
        if self.hyp_mlp is not None:
            sd['hyp_mlp'] = self.hyp_mlp.state_dict()
        return sd

    def load_state_dict_from_save(self, sd: Dict[str, torch.Tensor]) -> None:
        # Cross-version safety
        was_light = bool(sd.get('light', False))
        if was_light != self.light:
            raise RuntimeError(
                f"checkpoint was saved with light={was_light} but this block "
                f"was constructed with light={self.light}; cannot mix Möbius "
                f"and Euclidean up-projection weights."
            )
        was_residual = str(sd.get('residual_mode', 'sigmoid_gate'))  # legacy default
        if was_residual != self.residual_mode:
            raise RuntimeError(
                f"checkpoint was saved with residual_mode='{was_residual}' but "
                f"this block was constructed with residual_mode='{self.residual_mode}'."
            )
        self.visual_proj.load_state_dict(sd['visual_proj'])
        self.v_pre_ln.load_state_dict(sd['v_pre_ln'])
        self.T_h_param.data.copy_(sd['T_h_param'])
        self.rho.data.copy_(sd['rho'])
        self.log_tau_h.data.copy_(sd['log_tau_h'])
        if self.light:
            self.eu_up.load_state_dict(sd['eu_up'])
        else:
            self.hyp_up.load_state_dict(sd['hyp_up'])
        if self.residual_mode == 'sigmoid_gate':
            self.alpha_bias.data.copy_(sd['alpha_bias'])
        else:
            self.rezero_gamma.data.copy_(sd['rezero_gamma'])
        if 'fuse_ln' in sd and self.fuse_layernorm:
            self.fuse_ln.load_state_dict(sd['fuse_ln'])
        self.cls_bias.data.copy_(sd['cls_bias'])
        if 'hyp_mlp' in sd and self.hyp_mlp is not None:
            self.hyp_mlp.load_state_dict(sd['hyp_mlp'])
        self._anchor_initialised = True

    # ─────────────────────────────────────────────────────────────
    # freeze / unfreeze (Stage-II plateau control)
    # ─────────────────────────────────────────────────────────────
    def freeze_all(self):
        for p in self.parameters(): p.requires_grad_(False)
    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad_(True)
    def freeze_anchors_only(self):
        """Leave alpha_bias + up-projection trainable for fast Stage-II warm-in."""
        for p in self.parameters(): p.requires_grad_(False)
        self.alpha_bias.requires_grad_(True)
        up = self.eu_up if self.light else self.hyp_up
        if up is not None:
            for p in up.parameters(): p.requires_grad_(True)
