"""Rein-style PEFT for a frozen DINOv2 backbone (dense segmentation).

Inspired by Wei et al., "Stronger, Fewer, Superior: Harnessing Vision
Foundation Models for Domain Generalized Semantic Segmentation" (CVPR 2024).

Unlike the post-hoc HOM-lite adapter in ``visual_adapter.py`` (which only
refines the 4 extracted feature maps AFTER the frozen ViT), Rein injects a
tiny learnable refinement BETWEEN transformer layers, so the correction
propagates through the subsequent frozen blocks — the backbone's internal
representation actually adapts to the surgical domain while DINOv2 stays
frozen.

Core mechanism (per adapted layer l), faithful to the paper's essentials:
    f  : block-l token features            [B, T, C]
    T_l: learnable token dictionary        [m, d]   (low-rank generated)
    S  = softmax(down(f) @ T_l^T / sqrt(d))         attention features->tokens
    Δf = up( gelu( S @ T_l ) )
    f' = f + scale_l * Δf                  (scale_l zero-init -> identity start)

Parameter efficiency ("fewer"): the token dictionaries share a single basis
across layers (per-layer low-rank coefficients A_l @ shared_basis), and the
down/up bottleneck keeps d << C. For ViT-B (C=768) with m=100, r=16, d=64 the
whole module is ~1.3M params vs the 86.6M frozen backbone.

Zero-init ``scale_l`` means training starts exactly at the frozen-backbone
output, which is what makes PEFT on a frozen VFM stable.
"""
import math

import torch
import torch.nn as nn


class ReinLayer(nn.Module):
    """One per-layer token-attention refinement block."""

    def __init__(self, dim, num_tokens, token_dim, shared_basis, dropout=0.0):
        super().__init__()
        self.token_dim = int(token_dim)
        # low-rank per-layer token coefficients; basis is shared across layers
        r = shared_basis.shape[0]
        self.token_coef = nn.Parameter(torch.empty(int(num_tokens), r))
        nn.init.normal_(self.token_coef, std=0.02)
        self._shared_basis = [shared_basis]           # held by ref, not re-registered

        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, self.token_dim)
        self.up = nn.Linear(self.token_dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        # LayerScale-style gate: ONLY the scale is zero-init, so Δf == 0 at the
        # start (identity output) while the scale still receives gradient (= Δf,
        # which is non-zero because `up` is normally initialized). Zero-initing
        # BOTH up and scale would be a dead init (no Rein param ever gets grad).
        self.scale = nn.Parameter(torch.zeros(1))

    def tokens(self):
        # [m, r] @ [r, d] -> [m, d]
        return self.token_coef @ self._shared_basis[0]

    def forward(self, x):
        # x: [B, T, C] (T = 1 CLS + N patches; refinement is token-count agnostic)
        tok = self.tokens()                                   # [m, d]
        fd = self.down(self.norm(x))                          # [B, T, d]
        attn = torch.softmax(fd @ tok.t() / math.sqrt(self.token_dim), dim=-1)  # [B,T,m]
        agg = attn @ tok                                      # [B, T, d]
        delta = self.up(self.act(agg))                        # [B, T, C]
        return x + self.scale * self.drop(delta)


class ReinAdapter(nn.Module):
    """Holds the per-block Rein layers. Called as ``rein(block_idx, x)`` from
    inside the frozen DINOv2 block loop (see DinoVisionTransformer)."""

    def __init__(self, dim, num_blocks, num_tokens=100, token_rank=16,
                 token_dim=64, dropout=0.0, adapt_layers=None):
        super().__init__()
        self.num_blocks = int(num_blocks)
        # which block indices get a Rein refinement (default: all)
        if adapt_layers is None:
            adapt_layers = list(range(self.num_blocks))
        self.adapt_layers = sorted(set(int(i) for i in adapt_layers))
        self._idx_to_slot = {b: k for k, b in enumerate(self.adapt_layers)}

        # single token basis shared across all adapted layers ("fewer" params)
        self.shared_basis = nn.Parameter(torch.empty(int(token_rank), int(token_dim)))
        nn.init.normal_(self.shared_basis, std=0.02)

        self.layers = nn.ModuleList([
            ReinLayer(dim, num_tokens, token_dim, self.shared_basis, dropout=dropout)
            for _ in self.adapt_layers
        ])

    def forward(self, block_idx, x):
        slot = self._idx_to_slot.get(int(block_idx))
        if slot is None:
            return x
        return self.layers[slot](x)
