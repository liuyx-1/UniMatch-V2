"""Weight-sharing Cross-Modal Adapter for dense surgical segmentation.

Faithful re-implementation of the core adapter from
    Jiang et al., "Cross-Modal Adapter for Vision-Language Retrieval",
    Pattern Recognition 159 (2025) 111144.

We reuse the paper's structure / hyperparameters / initialisation:
    down(D, r)  →  non-linearity  →  [ up_unique(r, D-d_s) || up_share(r, d_s) ]
    output = concat(...) along the last dim
    init: N(0, init_std=0.01), bias=0
    activation: quick_gelu (CLIP-style)

The KEY weight is `up_share`, a single Linear layer used by BOTH the
visual and text path — this is the only cross-modal coupling.  Per-modality
weights are `{v,t}_down` and `{v,t}_up_unique`.

Two classes:
  - CrossModalAdapter       — the adapter primitive  (mirrors the paper)
  - CMAFusionBlock          — Stage-II wrapper: takes DINOv2 patches, returns
                                text-aware patches X_aware  for DPT injection,
                                plus image-level affinity logits A for Stage-I
                                multi-label BCE + Stage-II dense scattering.
"""
from __future__ import annotations
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── activations ─────────────────────────────────────────────────────────
def _quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """CLIP's QuickGELU (the activation the paper uses)."""
    return x * torch.sigmoid(1.702 * x)


def _get_activation(name: str):
    n = name.lower()
    if n in ('quick_gelu', 'quickgelu'): return _quick_gelu
    if n == 'gelu':                       return F.gelu
    if n == 'gelu_new':                   return F.gelu
    if n == 'relu':                       return F.relu
    if n == 'silu' or n == 'swish':       return F.silu
    raise ValueError(f'unknown non_linearity: {name}')


# ── adapter primitive ───────────────────────────────────────────────────
class CrossModalAdapter(nn.Module):
    """Weight-sharing bottleneck adapter (mirrors LeapLabTHU/Cross-Modal-Adapter).

    Two parallel adapter paths (vision / text), each with its own
    `{v,t}_down` and `{v,t}_up_unique`, sharing a single `up_share` that
    bridges the two modalities.

    Args:
        d_in:        input feature dim
        d_out:       output feature dim
        reduction:   bottleneck dim  `r`        (paper recommends 8)
        share_dim:   shared up dim   `d_s`      (paper recommends 32)
        non_linearity: 'quick_gelu' | 'gelu' | 'relu' | 'silu'
        dropout:     dropout after non-linearity
        init_std:    std for N(0, std) init    (paper recommends 0.01)
        use_layernorm_before: pre-adapter LN (per-modality)
    """

    def __init__(self,
                 d_in: int = 768,
                 d_out: int = 128,
                 reduction: int = 8,
                 share_dim: int = 32,
                 non_linearity: str = 'quick_gelu',
                 dropout: float = 0.1,
                 init_std: float = 0.01,
                 use_layernorm_before: bool = True):
        super().__init__()
        assert reduction >= 1, 'reduction (r) must be >= 1'
        assert 0 <= share_dim <= d_out, \
            f'share_dim (d_s)={share_dim} must be in [0, d_out={d_out}]'

        self.d_in  = int(d_in)
        self.d_out = int(d_out)
        self.r     = int(reduction)
        self.d_s   = int(share_dim)
        self.unique_dim = self.d_out - self.d_s
        self.activation_name = str(non_linearity)
        self._activation = _get_activation(non_linearity)
        self.init_std = float(init_std)

        # ── Pre-LN (per modality, in input dim) ────────────────────────
        if use_layernorm_before:
            self.v_pre_ln = nn.LayerNorm(self.d_in)
            self.t_pre_ln = nn.LayerNorm(self.d_in)
        else:
            self.v_pre_ln = nn.Identity()
            self.t_pre_ln = nn.Identity()

        # ── Down projection (modality-specific) ────────────────────────
        self.v_down = nn.Linear(self.d_in, self.r)
        self.t_down = nn.Linear(self.d_in, self.r)

        # ── Up projection: split into unique + shared ──────────────────
        if self.unique_dim > 0:
            self.v_up_unique = nn.Linear(self.r, self.unique_dim)
            self.t_up_unique = nn.Linear(self.r, self.unique_dim)
        else:
            self.v_up_unique = None
            self.t_up_unique = None

        # ── SHARED up linear (the cross-modal bridge) ──────────────────
        if self.d_s > 0:
            self.up_share = nn.Linear(self.r, self.d_s)
        else:
            self.up_share = None

        # ── Dropout ────────────────────────────────────────────────────
        self.dropout = nn.Dropout(p=float(dropout)) if dropout > 0 else nn.Identity()

        # ── Init (N(0, init_std), bias=0) ──────────────────────────────
        self._init_weights(self.init_std)

    def _init_weights(self, std: float):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─── two parallel paths ─────────────────────────────────────────────
    def _project(self, x, down, up_unique, pre_ln):
        """Common adapter body — used by both forward_visual / forward_text."""
        x = pre_ln(x)
        h = self._activation(down(x))     # [..., r]
        h = self.dropout(h)
        chunks = []
        if up_unique is not None:
            chunks.append(up_unique(h))   # [..., unique_dim]
        if self.up_share is not None:
            chunks.append(self.up_share(h))   # [..., d_s]
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-1)

    def forward_visual(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in] → [..., d_out]"""
        return self._project(x, self.v_down, self.v_up_unique, self.v_pre_ln)

    def forward_text(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in] → [..., d_out]"""
        return self._project(x, self.t_down, self.t_up_unique, self.t_pre_ln)

    def extra_repr(self) -> str:
        return (f'd_in={self.d_in}, d_out={self.d_out}, r={self.r}, '
                f'd_s={self.d_s}, unique_dim={self.unique_dim}, '
                f'non_linearity={self.activation_name}, '
                f'init_std={self.init_std}')


# ── Stage-I / Stage-II fusion block ─────────────────────────────────────
class CMAFusionBlock(nn.Module):
    """Cross-Modal Adapter fusion block for dense semantic segmentation.

    Given DINOv2 patch tokens [B, N, 768] and per-class text features
    [C', 768] (stored from SigLIP), produce:

      X_aware [B, N, 768]   text-aware patches for DPT decoder injection
      A       [B, N, C']    per-patch affinity logits for the Stage-I/II loss
      attn    [B, N, C']    softmax-attention used for aggregation
      Z_v     [B, N, d_lat] adapted visual latent (cosine compares against T)
      T       [C', d_lat]   adapted text anchors  (re-computed every forward)

    Pipeline (no hyperbolic math anywhere):

        patches ─► adapter.forward_visual ──► Z_v
        text    ─► adapter.forward_text   ──► T
        sim   = Zn · Tn^T                    (cosine)
        attn  = softmax(sim / τ)
        T_per_patch = attn @ T               (text aggregation)
        Eu_fused = up_proj(T_per_patch)      → 768
        Eu_fused = LayerNorm + Dropout
        X_aware = patches + γ · Eu_fused     (ReZero residual, γ init = 0)
        A       = sim / τ + cls_bias

    Two configurable hyperparameters (paper-style):
        reduction  r   (default 8)
        share_dim  d_s (default 32)
    """

    def __init__(self,
                 d_patch: int = 768,
                 d_lat: int = 128,
                 n_classes: int = 6,
                 reduction: int = 8,
                 share_dim: int = 32,
                 non_linearity: str = 'quick_gelu',
                 adapter_dropout: float = 0.1,
                 adapter_init_std: float = 0.01,
                 fuse_dropout: float = 0.1,
                 fuse_layernorm: bool = True,
                 tau_init: float = 0.07,
                 tau_clamp_min: float = 0.01,
                 tau_clamp_max: float = 2.0):
        super().__init__()
        self.d_patch = int(d_patch)
        self.d_lat   = int(d_lat)
        self.n_classes = int(n_classes)
        self.tau_clamp_min = float(tau_clamp_min)
        self.tau_clamp_max = float(tau_clamp_max)

        # ── Cross-modal adapter (the shared bridge) ───────────────────
        self.adapter = CrossModalAdapter(
            d_in=self.d_patch, d_out=self.d_lat,
            reduction=reduction, share_dim=share_dim,
            non_linearity=non_linearity,
            dropout=adapter_dropout, init_std=adapter_init_std,
            use_layernorm_before=True,
        )

        # ── Text features storage (raw SigLIP outputs, [C', d_in]) ──────
        # Stored as a buffer so it moves with .cuda() / .to() but isn't
        # trained.  Init via init_text_features(raw_text_features).
        self.register_buffer('text_features_raw',
                             torch.zeros(self.n_classes, self.d_patch),
                             persistent=True)
        self._anchor_initialised = False

        # ── Learnable temperature  τ = clamp(softplus(log_tau), …) ─────
        log_tau_init = float(math.log(math.expm1(max(tau_init, 1e-4))))
        self.log_tau = nn.Parameter(torch.tensor(log_tau_init))

        # ── Up projection d_lat → d_patch  (for spatial injection) ─────
        self.up_proj = nn.Linear(self.d_lat, self.d_patch)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=float(adapter_init_std))
        nn.init.zeros_(self.up_proj.bias)

        # ── Stability on the fused branch ──────────────────────────────
        self.fuse_layernorm_enabled = bool(fuse_layernorm)
        if self.fuse_layernorm_enabled:
            self.fuse_ln = nn.LayerNorm(self.d_patch)
        else:
            self.fuse_ln = nn.Identity()
        self.fuse_dropout_p = float(fuse_dropout)
        self.fuse_drop = nn.Dropout(p=self.fuse_dropout_p) if fuse_dropout > 0 else nn.Identity()

        # ── ReZero scalar gate  γ  (init 0 → strict identity at start) ─
        self.rezero_gamma = nn.Parameter(torch.zeros(1))

        # ── Class bias on final affinity logits ────────────────────────
        self.cls_bias = nn.Parameter(torch.zeros(self.n_classes))

    # ─── helpers ────────────────────────────────────────────────────────
    def get_tau(self) -> torch.Tensor:
        return F.softplus(self.log_tau).clamp(min=self.tau_clamp_min,
                                                max=self.tau_clamp_max)

    def init_text_features(self, raw_text_features: torch.Tensor):
        """Store raw (Stage-I-style, d_in-dim) per-class text features.

        We re-project these every forward through `adapter.forward_text`,
        so when the adapter is fine-tuned the text anchors automatically
        track the changes.
        """
        assert raw_text_features.shape == (self.n_classes, self.d_patch), (
            f'expected [{self.n_classes}, {self.d_patch}], '
            f'got {tuple(raw_text_features.shape)}'
        )
        with torch.no_grad():
            self.text_features_raw.copy_(
                raw_text_features.to(self.text_features_raw.device).float())
        self._anchor_initialised = True

    def get_T(self) -> torch.Tensor:
        """Return [C', d_lat] text anchors (computed live every call)."""
        return self.adapter.forward_text(self.text_features_raw)

    # ─── forward ────────────────────────────────────────────────────────
    def forward(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        assert self._anchor_initialised, 'call init_text_features() first'
        tau = self.get_tau()

        Z_v = self.adapter.forward_visual(patches)             # [B, N, d_lat]
        T   = self.get_T()                                      # [C', d_lat]

        # Cosine similarity (normalise once → numerically tame)
        Zn  = F.normalize(Z_v, dim=-1)
        Tn  = F.normalize(T,   dim=-1)
        sim = Zn @ Tn.t()                                       # [B, N, C']
        attn = F.softmax(sim / tau.clamp_min(1e-6), dim=-1)

        # Per-patch affinity logits (used by Stage-I image-level BCE &
        # patch BCE, and by Stage-II dense scatter to build R).
        A = sim / tau.clamp_min(1e-6) + self.cls_bias.view(1, 1, -1)

        # Aggregate text representation per patch (in the adapted latent),
        # then up-project to patch dim.
        text_per_patch = attn @ T                               # [B, N, d_lat]
        Eu_fused = self.up_proj(text_per_patch)                 # [B, N, d_patch]
        Eu_fused = self.fuse_ln(Eu_fused)
        Eu_fused = self.fuse_drop(Eu_fused)

        # ReZero residual: γ scalar, init 0  → identity at start
        X_aware = patches + self.rezero_gamma * Eu_fused

        return {
            'X_aware': X_aware,
            'Z_v':     Z_v,
            'T':       T,
            'A':       A,
            'attn':    attn,
            'tau':     tau,
        }

    # ─── persistence ────────────────────────────────────────────────────
    def state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        sd = {
            'adapter':            self.adapter.state_dict(),
            'text_features_raw':  self.text_features_raw.detach().cpu(),
            'log_tau':            self.log_tau.detach().cpu(),
            'up_proj':            self.up_proj.state_dict(),
            'rezero_gamma':       self.rezero_gamma.detach().cpu(),
            'cls_bias':           self.cls_bias.detach().cpu(),
            'd_patch':            self.d_patch,
            'd_lat':              self.d_lat,
            'n_classes':          self.n_classes,
            'r':                  self.adapter.r,
            'd_s':                self.adapter.d_s,
            'non_linearity':      self.adapter.activation_name,
            'fuse_layernorm':     bool(self.fuse_layernorm_enabled),
            'fuse_dropout_p':     float(self.fuse_dropout_p),
            'tau_clamp_min':      float(self.tau_clamp_min),
            'tau_clamp_max':      float(self.tau_clamp_max),
        }
        if self.fuse_layernorm_enabled:
            sd['fuse_ln'] = self.fuse_ln.state_dict()
        return sd

    def load_state_dict_from_save(self, sd: Dict[str, torch.Tensor]) -> None:
        self.adapter.load_state_dict(sd['adapter'])
        self.text_features_raw.copy_(sd['text_features_raw'])
        self.log_tau.data.copy_(sd['log_tau'])
        self.up_proj.load_state_dict(sd['up_proj'])
        if self.fuse_layernorm_enabled and 'fuse_ln' in sd and sd['fuse_ln'] is not None:
            self.fuse_ln.load_state_dict(sd['fuse_ln'])
        self.rezero_gamma.data.copy_(sd['rezero_gamma'])
        self.cls_bias.data.copy_(sd['cls_bias'])
        self._anchor_initialised = True

    # ─── freeze / unfreeze (Stage-II plateau control) ───────────────────
    def freeze_all(self):
        for p in self.parameters(): p.requires_grad_(False)

    def unfreeze_all(self):
        for p in self.parameters(): p.requires_grad_(True)
