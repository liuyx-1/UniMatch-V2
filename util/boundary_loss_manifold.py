"""Manifold-tangent enhancement for the existing boundary auxiliary head."""
from __future__ import annotations

import torch

from util.boundary_loss import BoundaryAuxHead, boundary_loss
from util.manifold_tangent import tangent_alignment_loss


class BoundaryAuxHeadTangent(BoundaryAuxHead):
    """Same forward contract as BoundaryAuxHead, with opt-in feature access."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        layers = list(self.body)
        h = x
        for m in layers[:-1]:
            h = m(h)
        out = layers[-1](h)
        return out, h


def boundary_loss_with_tangent(b_logits: torch.Tensor,
                                 hidden_feat: torch.Tensor,
                                 seg_logits: torch.Tensor,
                                 mask: torch.Tensor,
                                 num_classes: int,
                                 ignore_index: int = 255,
                                 pos_weight: float = 20.0,
                                 valid_mask: torch.Tensor | None = None,
                                 cutmix_box: torch.Tensor | None = None,
                                 cutmix_edge_thickness: int = 2,
                                 dice_weight: float = 0.5,
                                 tan_detach_seg: bool = True,
                                 ) -> tuple[torch.Tensor, torch.Tensor]:
    loss_b = boundary_loss(
        b_logits, mask, num_classes,
        ignore_index=ignore_index,
        pos_weight=pos_weight,
        valid_mask=valid_mask,
        cutmix_box=cutmix_box,
        cutmix_edge_thickness=cutmix_edge_thickness,
        dice_weight=dice_weight,
    )
    loss_tan = tangent_alignment_loss(
        seg_logits, hidden_feat,
        valid_mask=valid_mask,
        detach_a=tan_detach_seg,
    )
    return loss_b, loss_tan
