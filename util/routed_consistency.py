"""Unified routed consistency loss for the unlabeled stream.

This module composes the existing TCR entropy gate and TS-MDR routing /
directional matching pieces.  It is intentionally trainer-facing: the caller
passes already-computed student and CutMix-aligned teacher probabilities.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from util.temporal import entropy_gated_consistency, normalized_entropy
from util.temporal_tsmdr import (
    MaskMorphologyExtractor,
    ShapeMorphologyRouter,
    TextMorphologyPrior,
    TextSemanticMorphologyRouter,
    TSMDRConsistencyLoss,
)


class UniformMorphologyRouter(nn.Module):
    """Uniform route used for the TCR-degenerate path."""

    def __init__(self, num_classes: int, lambda_temp_base: float = 0.1,
                 lambda_edge_base: float = 0.1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.lambda_temp_base = float(lambda_temp_base)
        self.lambda_edge_base = float(lambda_edge_base)

    def set_beta(self, beta: float):
        pass

    def forward(self, s_c: Optional[torch.Tensor], q_t: torch.Tensor,
                bg_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, C, _ = q_t.shape
        device = q_t.device
        dtype = q_t.dtype
        pi = q_t.new_full((B, C, 4), 0.25)
        lam_temp = q_t.new_full((B, C), self.lambda_temp_base)
        lam_edge = q_t.new_full((B, C), self.lambda_edge_base)
        if bg_mask is not None:
            keep = (1.0 - bg_mask.view(1, -1)).to(dtype=dtype, device=device)
            lam_temp = lam_temp * keep
            lam_edge = lam_edge * keep
        return {
            'pi': pi,
            'lam_temp': lam_temp,
            'lam_edge': lam_edge,
            'lam_aff': lam_temp,
            'loss_route_ent': torch.zeros((), device=device, dtype=dtype),
        }


class RoutedConsistencyLoss(nn.Module):
    """TCR/TS-MDR unified consistency.

    route=none and beta=0 is numerically identical to the current fixed TCR
    branch because it directly delegates to ``entropy_gated_consistency``.
    route=text/shape with beta>0 reuses TS-MDR's router and directional loss.
    """

    def __init__(self,
                 num_classes: int,
                 route: str = 'none',
                 class_names=None,
                 encode_text=None,
                 radius: int = 3,
                 use_edge: bool = False,
                 lambda_temp_base: float = 0.1,
                 lambda_edge_base: float = 0.1):
        super().__init__()
        route = str(route).lower()
        if route not in ('none', 'shape', 'text'):
            raise ValueError(f'unsupported route: {route}')
        self.route = route
        self.num_classes = int(num_classes)
        self.txt = None
        self.ext = MaskMorphologyExtractor(soft_threshold=0.5)

        if route == 'none':
            self.router = UniformMorphologyRouter(
                num_classes, lambda_temp_base=lambda_temp_base,
                lambda_edge_base=lambda_edge_base)
        elif route == 'shape':
            self.router = ShapeMorphologyRouter(
                lambda_temp_base=lambda_temp_base,
                lambda_edge_base=lambda_edge_base)
        else:
            if class_names is None:
                raise ValueError('class_names is required for route=text')
            self.txt = TextMorphologyPrior(class_names, encode_text=encode_text)
            self.router = TextSemanticMorphologyRouter(
                q_dim=6, hidden=32, dropout=0.1,
                lambda_temp_base=lambda_temp_base,
                lambda_edge_base=lambda_edge_base)

        self.directional = TSMDRConsistencyLoss(
            num_classes=num_classes,
            radius=int(radius), n_directions=8,
            use_feat=False, use_mask=True,
            use_edge=bool(use_edge), use_con=False,
            lambda_m=0.2, lambda_e=0.1)

    def set_route_beta(self, beta: float):
        if hasattr(self.router, 'set_beta'):
            self.router.set_beta(beta)

    def _prior(self, prob_ref: torch.Tensor):
        if self.txt is not None:
            return self.txt(), self.txt.bg_mask
        s_c = prob_ref.new_full((self.num_classes, 4), 0.25)
        bg_mask = prob_ref.new_zeros((self.num_classes,))
        return s_c, bg_mask

    @staticmethod
    def _tcr_pair(prob_s: torch.Tensor, prob_t: torch.Tensor,
                  valid_mask: torch.Tensor) -> torch.Tensor:
        return entropy_gated_consistency(prob_s, prob_t, ignore_mask=valid_mask)

    @staticmethod
    def _weighted_tcr_pair(prob_s: torch.Tensor, prob_t: torch.Tensor,
                           valid_mask: torch.Tensor,
                           class_weight: torch.Tensor) -> torch.Tensor:
        base = entropy_gated_consistency(
            prob_s, prob_t, ignore_mask=valid_mask, reduction='none')
        gate = (normalized_entropy(prob_t.detach())
                < normalized_entropy(prob_s)).float()
        gate = gate * valid_mask.to(device=gate.device, dtype=gate.dtype)
        hard = prob_t.detach().argmax(dim=1)
        w_pix = class_weight.gather(1, hard.view(hard.size(0), -1)).view_as(hard)
        weighted_gate = gate * w_pix
        denom = weighted_gate.sum().clamp_min(1.0)
        return (base * w_pix).sum() / denom

    def forward_pair(self,
                     prob_s: torch.Tensor,
                     prob_t: torch.Tensor,
                     valid_mask: torch.Tensor,
                     beta: float = 0.0,
                     route_cache: Optional[Dict[str, torch.Tensor]] = None
                     ) -> Dict[str, torch.Tensor]:
        if route_cache is not None and self.route != 'none':
            class_weight = route_cache['lam_temp'] / max(
                float(getattr(self.router, 'lambda_temp_base', 1.0)), 1e-8)
            loss_tcr = self._weighted_tcr_pair(
                prob_s, prob_t, valid_mask, class_weight)
        else:
            loss_tcr = self._tcr_pair(prob_s, prob_t, valid_mask)
        loss_dir = torch.zeros((), device=prob_s.device, dtype=prob_s.dtype)
        route_ent = torch.zeros((), device=prob_s.device, dtype=prob_s.dtype)
        route = route_cache

        if float(beta) != 0.0:
            if route is None:
                with torch.no_grad():
                    q_t = self.ext(prob_s.detach(), prev_prob=prob_t.detach())
                    s_c, bg_mask = self._prior(prob_s)
                route = self.router(s_c, q_t, bg_mask=bg_mask)
            losses = self.directional(
                feat_t=None, feat_tp1=None,
                prob_t=prob_s,
                prob_tp1=prob_t,
                lam_temp=route['lam_temp'], lam_edge=route['lam_edge'],
                boundary_t=None, boundary_tp1=None,
                valid_mask=valid_mask)
            loss_dir = losses['loss_total']
            route_ent = route.get('loss_route_ent', route_ent)

        return {
            'loss_tcr': loss_tcr,
            'loss_dir': loss_dir,
            'loss_total': loss_tcr + float(beta) * loss_dir,
            'loss_route_ent': route_ent,
            'route': route,
        }

    def forward(self,
                prob_s1: torch.Tensor,
                prob_s2: torch.Tensor,
                prob_t1: torch.Tensor,
                prob_t2: torch.Tensor,
                valid1: torch.Tensor,
                valid2: torch.Tensor,
                beta: float = 0.0,
                route_epoch_beta: Optional[float] = None
                ) -> Dict[str, torch.Tensor]:
        if route_epoch_beta is not None:
            self.set_route_beta(float(route_epoch_beta))

        route = None
        if self.route != 'none':
            with torch.no_grad():
                q_t1 = self.ext(prob_s1.detach(), prev_prob=prob_t1.detach())
                q_t2 = self.ext(prob_s2.detach(), prev_prob=prob_t2.detach())
                q_t = 0.5 * (q_t1 + q_t2)
                s_c, bg_mask = self._prior(prob_s1)
            route = self.router(s_c, q_t, bg_mask=bg_mask)

        out1 = self.forward_pair(prob_s1, prob_t1, valid1, beta=beta,
                                 route_cache=route)
        out2 = self.forward_pair(prob_s2, prob_t2, valid2, beta=beta,
                                 route_cache=route)
        loss_tcr = 0.5 * (out1['loss_tcr'] + out2['loss_tcr'])
        loss_dir = 0.5 * (out1['loss_dir'] + out2['loss_dir'])
        return {
            'loss_tcr': loss_tcr,
            'loss_dir': loss_dir,
            'loss_total': loss_tcr + float(beta) * loss_dir,
            'loss_route_ent': out1['loss_route_ent'],
            'route': route,
        }
