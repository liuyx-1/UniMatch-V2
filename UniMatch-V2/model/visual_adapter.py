import torch
import torch.nn as nn


class SpatialHOMAdapterBlock(nn.Module):
    """Lightweight local-global adapter for DINO patch tokens.

    The block is intentionally placed outside the frozen DINOv2 backbone. It
    adapts the intermediate token features before the DPT head, adding local
    spatial bias through depthwise convolution and input-aware branch mixing.
    """

    def __init__(self, dim, reduction=8, dropout=0.0):
        super().__init__()
        if int(reduction) <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        hidden = max(8, int(dim) // int(reduction))
        gate_hidden = max(4, hidden // 4)
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, hidden)

        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
        )
        self.branch_gate = nn.Sequential(
            nn.Linear(hidden, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 2),
        )
        self.up = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, patch_h, patch_w):
        b, n, _ = x.shape
        if n != patch_h * patch_w:
            raise RuntimeError(
                f"SpatialHOMAdapterBlock expected N={patch_h * patch_w}, got {n}"
            )

        z = self.down(self.norm(x))
        z_map = z.transpose(1, 2).reshape(b, z.shape[-1], patch_h, patch_w)

        local = self.local(z_map).flatten(2).transpose(1, 2)
        pooled = z.mean(dim=1)
        global_feat = self.global_mlp(pooled).unsqueeze(1).expand_as(z)
        gate = torch.softmax(self.branch_gate(pooled), dim=-1).view(b, 1, 2)
        fused = gate[..., 0:1] * local + gate[..., 1:2] * global_feat
        fused = self.drop(self.up(fused))
        return x + self.gamma * fused


class DinoDPTAdapter(nn.Module):
    """HOM-lite visual adapter over DINOv2 intermediate features."""

    def __init__(self, dim, num_layers=4, reduction=8, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            SpatialHOMAdapterBlock(dim, reduction=reduction, dropout=dropout)
            for _ in range(int(num_layers))
        ])

    def forward(self, features, patch_h, patch_w):
        if len(features) != len(self.blocks):
            raise RuntimeError(
                f"DinoDPTAdapter expected {len(self.blocks)} features, got {len(features)}"
            )
        return [
            block(feat, patch_h, patch_w)
            for block, feat in zip(self.blocks, features)
        ]
