"""Stage-2 side branch wrapping Stage-1 Patch-Text Affinity.

Single visual encoder design: this module does NOT hold its own DINOv2.
It takes PRE-COMPUTED patch tokens [B, N, 768] from the DPT main backbone
and produces:
   - prior_full [B, n_orig, H, W]    dense per-class affinity (cos sim with T_anchor)
   - gate       [B, 1,      H, W]    per-pixel mixing weight, sigmoid output
   - cls_logits [B, C_new]            image-level top-k pooled affinity score

Trainable params:
   - gate_conv             always trainable from step 0
   - visual_proj           default FROZEN (loaded from Stage 1), unfrozen mid-training
                              by plateau-trigger
   - T_anchor, cls_bias, log_tau   same lifecycle as visual_proj

text_proj is NOT loaded at Stage 2: T_anchor was precomputed in Stage 1 from
   text_encoder→text_proj→ensemble pool, and is stored directly in the ckpt.
   At Stage 2 we treat T_anchor as a raw learnable tensor (can be unfrozen).

Fusion: L_final = (1 - g·mask) * L_DPT + g·mask * L_prior
   mask: 1 for non-bg classes (in original DPT class order), 0 for bg.
"""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_proj(d_in: int, d_out: int, hidden: int = 256, dropout: float = 0.0):
    return nn.Sequential(
        nn.LayerNorm(d_in),
        nn.Linear(d_in, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, d_out),
    )


class AffinitySideBranch(nn.Module):
    """Operates on patch tokens produced elsewhere; no DINOv2 inside."""

    def __init__(self, ckpt_path: str, n_orig_classes: int,
                 gate_bias_init: float = -3.0):
        super().__init__()
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.ckpt_path = str(ckpt_path)
        self.dataset = ckpt.get('dataset', 'unknown')

        d_text   = 768
        d_latent = int(ckpt.get('d_latent', 128))
        self.d_text   = d_text
        self.d_latent = d_latent
        self.topk     = int(ckpt.get('topk', 5))

        # visual_proj — loaded from Stage 1 ckpt, default frozen
        self.visual_proj = _build_proj(d_text, d_latent)
        self.visual_proj.load_state_dict(ckpt['visual_proj'])

        # T_anchor: precomputed in Stage 1, stored directly. Learnable param so it can be unfrozen.
        T_anchor = ckpt['T_anchor'].float()                      # [C_new, d_latent]
        self.T_anchor = nn.Parameter(T_anchor.clone(), requires_grad=False)
        self.cls_bias = nn.Parameter(ckpt['cls_bias'].float().clone(), requires_grad=False)
        self.log_tau  = nn.Parameter(ckpt['log_tau'].float().clone(), requires_grad=False)

        # Class index mappings (orig DPT-id space → new affinity-id space, -1 for bg)
        orig_to_new = {int(k): int(v) for k, v in ckpt['orig_to_new'].items()}
        self.n_orig = int(n_orig_classes)
        idx = [orig_to_new.get(c, -1) for c in range(self.n_orig)]
        self.register_buffer('idx_orig_to_new',
                              torch.tensor(idx, dtype=torch.long))
        self.register_buffer('non_bg_mask',
                              (self.idx_orig_to_new >= 0).to(torch.float32).view(1, -1, 1, 1))

        # gate_conv: 1×1 on the d_latent feature, init near 0 (bias=-3 → σ≈0.05)
        self.gate_conv = nn.Conv2d(d_latent, 1, kernel_size=1)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.constant_(self.gate_conv.bias, float(gate_bias_init))

        # Default freeze proj-side
        self.freeze_proj()
        self._proj_unfrozen = False

    # ---- freeze / unfreeze proj group ----
    def freeze_proj(self):
        for p in self.visual_proj.parameters(): p.requires_grad_(False)
        self.T_anchor.requires_grad_(False)
        self.cls_bias.requires_grad_(False)
        self.log_tau.requires_grad_(False)
        self._proj_unfrozen = False

    def unfreeze_proj(self):
        for p in self.visual_proj.parameters(): p.requires_grad_(True)
        self.T_anchor.requires_grad_(True)
        self.cls_bias.requires_grad_(True)
        self.log_tau.requires_grad_(True)
        self._proj_unfrozen = True

    def proj_parameters(self):
        return list(self.visual_proj.parameters()) + \
               [self.T_anchor, self.cls_bias, self.log_tau]

    # ---- main forward: receives patches, returns prior + gate + cls ----
    def forward(self, patches: torch.Tensor, target_h: int, target_w: int, patch_hw=None):
        """
        patches:  [B, N, 768]   pre-computed last-layer DINOv2 patch tokens
                                   (from DPT main backbone, before any dropout)
        target_h, target_w: spatial size of the DPT logits we will fuse with.

        Returns dict:
            prior        [B, n_orig, target_h, target_w]
            gate         [B, 1,      target_h, target_w]   sigmoid output in [0,1]
            cls_logits   [B, C_new]                          image-level top-k pooled
            non_bg_mask  [1, n_orig, 1, 1]                   1 for non-bg, 0 for bg
        """
        B, N, _ = patches.shape
        if patch_hw is None:
            side = int(round(N ** 0.5))
            patch_h, patch_w = side, side
        else:
            patch_h, patch_w = [int(v) for v in patch_hw]
        if patch_h * patch_w != N:
            raise RuntimeError(
                f"AffinitySideBranch patch grid mismatch: patch_h*patch_w={patch_h}*{patch_w} "
                f"but got N={N} patch tokens."
            )
        # latent: per-token projection, L2-norm
        p_lat = F.normalize(self.visual_proj(patches), dim=-1)         # [B, N, d_lat]
        # IMPORTANT: read params via .clone() so multiple forward() calls in the same
        # training step (img_x + img_u_s1 + img_u_s2) don't share storage with the
        # original parameter tensors. This avoids version-counter conflicts when
        # the autograd engine saves the param for backward in several places.
        T = F.normalize(self.T_anchor.clone(), dim=-1)                  # [C_new, d_lat]
        tau = self.log_tau.clone().exp().clamp(1e-3, 1.0)
        cls_bias = self.cls_bias.clone()                                # [C_new]

        A = (p_lat @ T.T) / tau                                         # [B, N, C_new]

        # Image-level cls: top-k pool over patches + bias
        k = min(self.topk, N)
        topk_vals = A.topk(k, dim=1).values                             # [B, k, C_new]
        cls_logits = topk_vals.mean(dim=1) + cls_bias                   # [B, C_new]

        # Dense map (logits, not softmax): + bias for consistent units
        dense = A.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)     # [B, C_new, H_p, W_p]
        dense = dense + cls_bias.view(1, -1, 1, 1)
        dense_up = F.interpolate(dense, size=(target_h, target_w),
                                  mode='bilinear', align_corners=False)

        # Scatter into original DPT class index space
        idx = self.idx_orig_to_new.clamp(min=0)                         # bg→0 placeholder
        gathered = dense_up.index_select(dim=1, index=idx)              # [B, n_orig, H, W]
        prior_full = gathered * self.non_bg_mask                        # bg channels zeroed

        # Gate from same projected latent (reshaped to spatial)
        p_lat_2d = p_lat.permute(0, 2, 1).reshape(B, -1, patch_h, patch_w)  # [B, d_lat, H_p, W_p]
        gate_logits = self.gate_conv(p_lat_2d)                          # [B, 1, H_p, W_p]
        gate_up = F.interpolate(gate_logits, size=(target_h, target_w),
                                 mode='bilinear', align_corners=False)
        gate = torch.sigmoid(gate_up)

        return {
            'prior': prior_full,
            'gate':  gate,
            'cls_logits': cls_logits,
            'non_bg_mask': self.non_bg_mask,
        }

    def fuse(self, dpt_logits: torch.Tensor, side_out: dict) -> torch.Tensor:
        """Per-pixel gated fusion. Bg channels untouched (mask zeros their gate)."""
        eff_gate = side_out['gate'] * side_out['non_bg_mask']
        return (1.0 - eff_gate) * dpt_logits + eff_gate * side_out['prior']

    # ---- checkpoint helpers ----
    def state_dict_for_save(self):
        return {
            'visual_proj': self.visual_proj.state_dict(),
            'T_anchor':    self.T_anchor.detach().cpu(),
            'log_tau':     self.log_tau.detach().cpu(),
            'cls_bias':    self.cls_bias.detach().cpu(),
            'gate_conv':   self.gate_conv.state_dict(),
            'proj_unfrozen': bool(self._proj_unfrozen),
            'idx_orig_to_new': self.idx_orig_to_new.detach().cpu(),
        }

    def load_state_dict_from_save(self, sd):
        self.visual_proj.load_state_dict(sd['visual_proj'])
        self.T_anchor.data.copy_(sd['T_anchor'])
        self.log_tau.data.copy_(sd['log_tau'])
        self.cls_bias.data.copy_(sd['cls_bias'])
        self.gate_conv.load_state_dict(sd['gate_conv'])
        if sd.get('proj_unfrozen', False):
            self.unfreeze_proj()
        else:
            self.freeze_proj()
