"""Hyperbolic (Poincaré ball) utilities — full operator set.

Implements the Hyperbolic Neural Network operations of
Ganea, Bécigneul, Hofmann (NeurIPS 2018), adapted to the
Patch-Text Affinity setting of UniMatch-V2 Surgical.

All curvatures `c` are positive scalars; the ball has radius 1/sqrt(c).
Numerically guarded:
  * `project_to_ball` keeps points strictly inside the safety ball
    of radius (1-eps)/sqrt(c)
  * arcosh / arctanh arguments are clamped to safe ranges
  * every denominator uses clamp_min(eps)

Operator list:
  project_to_ball              ─ safety projection
  expmap0,  logmap0            ─ tangent ↔ ball at origin
  poincare_distance            ─ geodesic distance
  hyperbolic_affinity          ─ neg-distance affinity (used by Stage-I head)

  mobius_add                   ─ x ⊕_c y   (Möbius addition)
  mobius_scalar_mul            ─ r ⊙_c x   (Möbius scalar multiplication)
  mobius_matvec                ─ M ⊗_c x   (Möbius matrix-vector product, via
                                            ExpMap0(M · LogMap0(x)))
  hyp_linear                   ─ Möbius affine layer (W ⊗_c x) ⊕_c b
  hyp_relu                     ─ ExpMap0(ReLU(LogMap0(x)))
  hyp_weighted_mean            ─ ExpMap0(Σ w_i · LogMap0(x_i))   (Fréchet-mean
                                            approximation, ball-stable)

  HypLinear                    ─ nn.Module wrapping hyp_linear (with weight + bias)
  HypMLP                       ─ stack of HypLinear + hyp_relu
"""
from __future__ import annotations
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────
# basic
# ─────────────────────────────────────────────────────────────────────
def _as_scalar(c, like: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(c):
        return c.to(device=like.device, dtype=like.dtype)
    return torch.tensor(float(c), device=like.device, dtype=like.dtype)


def project_to_ball(x: torch.Tensor, c, eps: float = 1e-5) -> torch.Tensor:
    c_t = _as_scalar(c, x).clamp_min(eps)
    sqrt_c = c_t.sqrt()
    max_norm = (1.0 - eps) / sqrt_c
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    cond = (norm > max_norm).to(x.dtype)
    return cond * (x / norm * max_norm) + (1.0 - cond) * x


# ─────────────────────────────────────────────────────────────────────
# Exp / Log at origin
# ─────────────────────────────────────────────────────────────────────
def expmap0(u: torch.Tensor, c, eps: float = 1e-5) -> torch.Tensor:
    """tangent → ball:  expmap0_c(u) = tanh(√c |u|) · u / (√c |u|)"""
    c_t = _as_scalar(c, u).clamp_min(eps)
    sqrt_c = c_t.sqrt()
    norm = u.norm(dim=-1, keepdim=True).clamp_min(eps)
    coeff = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)
    return project_to_ball(coeff * u, c_t, eps=eps)


def logmap0(x: torch.Tensor, c, eps: float = 1e-5) -> torch.Tensor:
    """ball → tangent:  logmap0_c(x) = arctanh(√c |x|) · x / (√c |x|)"""
    c_t = _as_scalar(c, x).clamp_min(eps)
    sqrt_c = c_t.sqrt()
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    arg = (sqrt_c * norm).clamp(-1.0 + eps, 1.0 - eps)
    return (torch.atanh(arg) / (sqrt_c * norm)) * x


# ─────────────────────────────────────────────────────────────────────
# Distance and affinity
# ─────────────────────────────────────────────────────────────────────
def poincare_distance(x: torch.Tensor, y: torch.Tensor,
                      c, eps: float = 1e-5) -> torch.Tensor:
    c_t = _as_scalar(c, x).clamp_min(eps)
    sqrt_c = c_t.sqrt()
    x2 = (x * x).sum(dim=-1)
    y2 = (y * y).sum(dim=-1)
    diff = x - y
    diff2 = (diff * diff).sum(dim=-1)
    denom_x = (1.0 - c_t * x2).clamp_min(eps)
    denom_y = (1.0 - c_t * y2).clamp_min(eps)
    arg = 1.0 + 2.0 * c_t * diff2 / (denom_x * denom_y)
    arg = arg.clamp_min(1.0 + eps)
    return torch.acosh(arg) / sqrt_c


def hyperbolic_affinity(P_h: torch.Tensor,
                        T_h: torch.Tensor,
                        tau_h: torch.Tensor,
                        class_bias: Optional[torch.Tensor] = None,
                        c=1.0,
                        eps: float = 1e-5,
                        distance_scale: float = 1.0) -> torch.Tensor:
    Pe = P_h.unsqueeze(-2)
    Te = T_h.view(1, 1, T_h.shape[0], T_h.shape[1])
    dist = poincare_distance(Pe, Te, c=c, eps=eps)
    tau_safe = tau_h
    if not torch.is_tensor(tau_safe):
        tau_safe = torch.tensor(float(tau_safe), device=P_h.device, dtype=P_h.dtype)
    tau_safe = tau_safe.to(device=P_h.device, dtype=P_h.dtype).clamp_min(eps)
    A = -float(distance_scale) * dist / tau_safe
    if class_bias is not None:
        A = A + class_bias.view(1, 1, -1)
    return A


# ─────────────────────────────────────────────────────────────────────
# Möbius operations
# ─────────────────────────────────────────────────────────────────────
def mobius_add(x: torch.Tensor, y: torch.Tensor, c, eps: float = 1e-5) -> torch.Tensor:
    """x ⊕_c y on the Poincaré ball.

        x ⊕ y = ((1 + 2c<x,y> + c|y|²) x + (1 - c|x|²) y) /
                (1 + 2c<x,y> + c²|x|²|y|²)
    """
    c_t = _as_scalar(c, x).clamp_min(eps)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * c_t * xy + c_t * y2) * x + (1.0 - c_t * x2) * y
    den = (1.0 + 2.0 * c_t * xy + c_t * c_t * x2 * y2).clamp_min(eps)
    return project_to_ball(num / den, c_t, eps=eps)


def mobius_scalar_mul(r, x: torch.Tensor, c, eps: float = 1e-5) -> torch.Tensor:
    """r ⊙_c x = tanh(r · arctanh(√c|x|)) · x / (√c|x|)."""
    c_t = _as_scalar(c, x).clamp_min(eps)
    sqrt_c = c_t.sqrt()
    norm = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    arg = (sqrt_c * norm).clamp(-1.0 + eps, 1.0 - eps)
    if not torch.is_tensor(r):
        r = torch.tensor(float(r), device=x.device, dtype=x.dtype)
    return project_to_ball(
        torch.tanh(r * torch.atanh(arg)) * x / (sqrt_c * norm),
        c_t, eps=eps,
    )


def mobius_matvec(W: torch.Tensor, x: torch.Tensor, c, eps: float = 1e-5) -> torch.Tensor:
    """Möbius matrix-vector product   M ⊗_c x  via the tangent-space trick:
            M ⊗_c x = ExpMap0(M · LogMap0(x))
    Supports W [out, in], x [..., in] → output [..., out].
    """
    tangent = logmap0(x, c=c, eps=eps)                 # [..., in]
    mapped  = F.linear(tangent, W)                      # [..., out]
    return expmap0(mapped, c=c, eps=eps)


def hyp_linear(W: torch.Tensor, b: torch.Tensor, x: torch.Tensor,
               c, eps: float = 1e-5) -> torch.Tensor:
    """Möbius affine:  (W ⊗_c x) ⊕_c expmap0(b).

    `b` lives in the tangent space at origin (so it's a plain Euclidean
    parameter and we lift it to the ball with expmap0).
    """
    Mx = mobius_matvec(W, x, c=c, eps=eps)
    b_h = expmap0(b.expand_as(Mx), c=c, eps=eps)
    return mobius_add(Mx, b_h, c=c, eps=eps)


def hyp_relu(x: torch.Tensor, c, eps: float = 1e-5) -> torch.Tensor:
    """Non-linearity at the origin: ExpMap0(ReLU(LogMap0(x)))."""
    return expmap0(F.relu(logmap0(x, c=c, eps=eps)), c=c, eps=eps)


def hyp_weighted_mean(points: torch.Tensor, weights: torch.Tensor,
                      c, eps: float = 1e-5) -> torch.Tensor:
    """Distance-aware aggregation of points on the ball.

    Args:
        points:   [..., K, d]  on Poincaré ball (last but one dim = items)
        weights:  [..., K]     non-negative, summing to 1 along K
    Returns:
        [..., d] on Poincaré ball — ExpMap0(Σ w_k · LogMap0(points_k)).

    This is the standard tangent-space Fréchet-mean approximation;
    a closed-form true Fréchet mean has no simple expression on the
    Poincaré ball, but this approximation is widely used (Mathieu et al.
    2019; Khrulkov et al. 2020) and is numerically stable.
    """
    tangent = logmap0(points, c=c, eps=eps)                       # [..., K, d]
    w = weights.unsqueeze(-1)                                      # [..., K, 1]
    mean_t = (w * tangent).sum(dim=-2)                              # [..., d]
    return expmap0(mean_t, c=c, eps=eps)


# ─────────────────────────────────────────────────────────────────────
# nn.Module wrappers
# ─────────────────────────────────────────────────────────────────────
class HypLinear(nn.Module):
    """Hyperbolic linear layer:  y = (W ⊗_c x) ⊕_c expmap0(b).

    Weight and bias are stored as Euclidean parameters; the Möbius math
    is applied at forward time.  Initialisation matches a small-norm
    Kaiming on W (so layers stay near the origin → small distortion).
    """
    def __init__(self, in_features: int, out_features: int, c=1.0, eps: float = 1e-5,
                 bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        with torch.no_grad():
            self.weight.mul_(0.5)                                  # smaller init for stability
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        if torch.is_tensor(c):
            self.register_buffer('_c_buf', c.clone().detach())
        else:
            self.register_buffer('_c_buf', torch.tensor(float(c)))

    def forward(self, x: torch.Tensor, c=None) -> torch.Tensor:
        c_use = c if c is not None else self._c_buf
        if self.bias is None:
            return mobius_matvec(self.weight, x, c=c_use, eps=self.eps)
        return hyp_linear(self.weight, self.bias, x, c=c_use, eps=self.eps)


class HypMLP(nn.Module):
    """Stack of HypLinear with hyp_relu between layers.

    `dims = [in, hidden, ..., out]`  — uses len(dims)-1 layers.
    The last layer is linear (no activation) so the output stays
    representative.  Curvature `c` is passed in at forward time so
    a parent module can share a single learnable `rho → softplus`
    curvature with multiple HypMLPs.
    """
    def __init__(self, dims: Sequence[int], c=1.0, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.layers = nn.ModuleList([
            HypLinear(dims[i], dims[i + 1], c=c, eps=eps) for i in range(len(dims) - 1)
        ])

    def forward(self, x: torch.Tensor, c=None) -> torch.Tensor:
        c_use = c if c is not None else self.layers[0]._c_buf
        for i, lin in enumerate(self.layers):
            x = lin(x, c=c_use)
            if i < len(self.layers) - 1:
                x = hyp_relu(x, c=c_use, eps=self.eps)
        return x
