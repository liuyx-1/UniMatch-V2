"""HPTA-MoE: a Mixture-of-Experts token-weighting upgrade of the HOM-lite
visual adapter (see visual_adapter.py for the single-block baseline).

Motivation
----------
The original HPTA block mixes exactly two branches (local depthwise-conv +
global MLP) with a single input-aware 2-way gate. HPTA-MoE generalizes this to
**four experts with per-token soft routing**, and — crucially — lets the router
**coordinate with the text (LC-PAM) and edge (EDGE/MGER) modules**: when those
are active, their per-token signals condition the router, so e.g. boundary
tokens lean on the local expert and text-salient tokens lean on the semantic
expert. When text/edge are absent the router falls back to token-only routing,
so the block is a drop-in replacement.

Experts (all act on the bottleneck z of dim h):
  1. local     — depthwise 3x3 conv (+GELU+1x1): spatial / boundary detail
  2. global    — mean-pooled context broadcast through an MLP: semantic
  3. lowrank   — channel bottleneck MLP (h -> h/2 -> h): channel re-mixing
  4. identity  — pass-through: lets a token opt out of adaptation

Router: softmax over the 4 experts, computed PER TOKEN from
[z ; proj(text_cond) ; proj(edge_cond)]. A zero-initialized residual scale
(gamma) makes the whole block an identity map at initialization.
"""
import torch
import torch.nn as nn


class MoETokenAdapterBlock(nn.Module):
    def __init__(self, dim, reduction=8, dropout=0.0,
                 text_cond_dim=0, edge_cond_dim=0, router_hidden=None):
        super().__init__()
        if int(reduction) <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        h = max(8, int(dim) // int(reduction))
        self.h = h
        self.num_experts = 4
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, h)

        # ---- experts ----
        self.local = nn.Sequential(
            nn.Conv2d(h, h, kernel_size=3, padding=1, groups=h),
            nn.GELU(),
            nn.Conv2d(h, h, kernel_size=1),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(h, h), nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(h, h),
        )
        lr = max(4, h // 2)
        self.lowrank = nn.Sequential(
            nn.Linear(h, lr), nn.GELU(), nn.Linear(lr, h),
        )
        # expert 4 = identity (no params)

        # ---- per-token router, optionally conditioned on text / edge ----
        self.text_cond_dim = int(text_cond_dim)
        self.edge_cond_dim = int(edge_cond_dim)
        rh = router_hidden or max(4, h // 2)
        self.text_proj = nn.Linear(self.text_cond_dim, rh) if self.text_cond_dim > 0 else None
        self.edge_proj = nn.Linear(self.edge_cond_dim, rh) if self.edge_cond_dim > 0 else None
        router_in = h + (rh if self.text_proj is not None else 0) \
                      + (rh if self.edge_proj is not None else 0)
        self.router = nn.Sequential(
            nn.Linear(router_in, rh), nn.GELU(), nn.Linear(rh, self.num_experts),
        )

        self.up = nn.Linear(h, dim)
        self.drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, patch_h, patch_w, text_cond=None, edge_cond=None):
        b, n, _ = x.shape
        if n != patch_h * patch_w:
            raise RuntimeError(
                f"MoETokenAdapterBlock expected N={patch_h * patch_w}, got {n}")
        z = self.down(self.norm(x))                       # [B, N, h]

        # experts
        z_map = z.transpose(1, 2).reshape(b, self.h, patch_h, patch_w)
        e_local = self.local(z_map).flatten(2).transpose(1, 2)   # [B, N, h]
        pooled = z.mean(dim=1, keepdim=True)                     # [B, 1, h]
        e_global = self.global_mlp(pooled).expand(-1, n, -1)      # [B, N, h]
        e_lowrank = self.lowrank(z)                              # [B, N, h]
        e_identity = z                                          # [B, N, h]
        experts = torch.stack([e_local, e_global, e_lowrank, e_identity], dim=2)  # [B,N,4,h]

        # per-token router (coordinates with text / edge when available)
        r_parts = [z]
        if self.text_proj is not None and text_cond is not None:
            r_parts.append(self.text_proj(text_cond))
        elif self.text_proj is not None:
            r_parts.append(z.new_zeros(b, n, self.text_proj.out_features))
        if self.edge_proj is not None and edge_cond is not None:
            r_parts.append(self.edge_proj(edge_cond))
        elif self.edge_proj is not None:
            r_parts.append(z.new_zeros(b, n, self.edge_proj.out_features))
        w = torch.softmax(self.router(torch.cat(r_parts, dim=-1)), dim=-1)   # [B,N,4]

        u = (w.unsqueeze(-1) * experts).sum(dim=2)              # [B, N, h]
        return x + self.gamma * self.drop(self.up(u))


class DinoDPTAdapterMoE(nn.Module):
    """MoE visual adapter over DINOv2 intermediate features (drop-in for
    DinoDPTAdapter). Optionally conditions each block's router on shared
    per-token text / edge signals."""

    def __init__(self, dim, num_layers=4, reduction=8, dropout=0.0,
                 text_cond_dim=0, edge_cond_dim=0):
        super().__init__()
        self.text_cond_dim = int(text_cond_dim)
        self.edge_cond_dim = int(edge_cond_dim)
        self.blocks = nn.ModuleList([
            MoETokenAdapterBlock(dim, reduction=reduction, dropout=dropout,
                                 text_cond_dim=text_cond_dim, edge_cond_dim=edge_cond_dim)
            for _ in range(int(num_layers))
        ])

    def forward(self, features, patch_h, patch_w, text_cond=None, edge_cond=None):
        if len(features) != len(self.blocks):
            raise RuntimeError(
                f"DinoDPTAdapterMoE expected {len(self.blocks)} features, got {len(features)}")
        return [blk(feat, patch_h, patch_w, text_cond=text_cond, edge_cond=edge_cond)
                for blk, feat in zip(self.blocks, features)]
