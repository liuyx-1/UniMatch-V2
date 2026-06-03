"""Unit tests for the full hyperbolic feature pathway."""
from __future__ import annotations
import math, sys, unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.hyperbolic import (
    project_to_ball, expmap0, logmap0, poincare_distance,
    mobius_add, mobius_scalar_mul, mobius_matvec,
    hyp_linear, hyp_relu, hyp_weighted_mean,
    HypLinear, HypMLP,
)
from model.hyperbolic_fusion import HyperbolicFusionBlock


def _in_ball(x, c, tol: float = 1e-4):
    """Verify x lies in the OPEN Poincaré ball of radius 1/sqrt(c).

    `project_to_ball` (with default eps=1e-5) clamps norms to at most
    (1-1e-5)/sqrt(c). We just check the strict mathematical condition
    `norm < 1/sqrt(c)` with a tiny numerical tolerance.
    """
    r = 1.0 / math.sqrt(c)
    max_n = float(x.norm(dim=-1).max())
    return max_n < r + tol


class TestMoebiusOps(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.B, self.N, self.d = 2, 7, 16
        self.c = 1.0
        self.eps = 1e-5

    def test_logmap0_inverse_of_expmap0(self):
        u = torch.randn(self.B, self.N, self.d) * 0.3
        x = expmap0(u, c=self.c, eps=self.eps)
        u2 = logmap0(x, c=self.c, eps=self.eps)
        self.assertTrue(torch.allclose(u, u2, atol=1e-4),
                        f'expmap/logmap not inverse, max err={(u-u2).abs().max():.2e}')

    def test_mobius_add_zero_identity(self):
        x = expmap0(torch.randn(self.N, self.d) * 0.3, c=self.c)
        zero = torch.zeros_like(x)
        y = mobius_add(x, zero, c=self.c, eps=self.eps)
        self.assertTrue(torch.allclose(x, y, atol=1e-5))

    def test_mobius_add_stays_in_ball(self):
        x = expmap0(torch.randn(self.N, self.d) * 0.5, c=self.c)
        y = expmap0(torch.randn(self.N, self.d) * 0.5, c=self.c)
        z = mobius_add(x, y, c=self.c, eps=self.eps)
        self.assertTrue(_in_ball(z, self.c), 'mobius_add escaped the ball')

    def test_mobius_scalar_mul_one_identity(self):
        x = expmap0(torch.randn(self.N, self.d) * 0.3, c=self.c)
        y = mobius_scalar_mul(1.0, x, c=self.c, eps=self.eps)
        self.assertTrue(torch.allclose(x, y, atol=1e-5))

    def test_mobius_matvec_identity(self):
        W = torch.eye(self.d)
        x = expmap0(torch.randn(self.N, self.d) * 0.3, c=self.c)
        y = mobius_matvec(W, x, c=self.c, eps=self.eps)
        self.assertTrue(torch.allclose(x, y, atol=1e-4))

    def test_hyp_linear_stays_in_ball(self):
        d_in, d_out = 8, 12
        W = torch.randn(d_out, d_in) * 0.1
        b = torch.randn(d_out) * 0.05
        x = expmap0(torch.randn(self.N, d_in) * 0.3, c=self.c)
        y = hyp_linear(W, b, x, c=self.c, eps=self.eps)
        self.assertTrue(_in_ball(y, self.c), 'hyp_linear escaped the ball')
        self.assertEqual(y.shape, (self.N, d_out))

    def test_hyp_weighted_mean_uniform_equals_centroid(self):
        K, d = 5, 8
        pts = expmap0(torch.randn(K, d) * 0.2, c=self.c)
        w = torch.full((K,), 1.0 / K)
        mu = hyp_weighted_mean(pts.unsqueeze(0), w.unsqueeze(0), c=self.c)
        # Tangent mean of pts should match.
        ref = expmap0(logmap0(pts, c=self.c).mean(0, keepdim=True), c=self.c)
        self.assertTrue(torch.allclose(mu, ref.squeeze(0).unsqueeze(0), atol=1e-5))

    def test_no_nan_under_stress(self):
        torch.manual_seed(123)
        for _ in range(8):
            c = float(torch.rand(1).item() * 2.0 + 0.05)
            scale = float(torch.rand(1).item() * 4.0 + 0.5)
            x = expmap0(torch.randn(self.N, self.d) * scale, c=c)
            y = expmap0(torch.randn(self.N, self.d) * scale, c=c)
            z = mobius_add(x, y, c=c)
            z2 = hyp_relu(z, c=c)
            self.assertFalse(torch.isnan(z2).any())
            self.assertFalse(torch.isinf(z2).any())
            self.assertTrue(_in_ball(z2, c))


class TestHypLayers(unittest.TestCase):
    def test_HypLinear_module(self):
        torch.manual_seed(0)
        d_in, d_out = 8, 16
        lin = HypLinear(d_in, d_out, c=1.0)
        x = expmap0(torch.randn(3, d_in) * 0.3, c=1.0)
        y = lin(x)
        self.assertEqual(y.shape, (3, d_out))
        self.assertTrue(_in_ball(y, 1.0))

    def test_HypMLP_module(self):
        torch.manual_seed(0)
        mlp = HypMLP([8, 16, 16, 8], c=1.0)
        x = expmap0(torch.randn(4, 8) * 0.3, c=1.0)
        y = mlp(x)
        self.assertEqual(y.shape, (4, 8))
        self.assertTrue(_in_ball(y, 1.0))


class TestHyperbolicFusionBlock(unittest.TestCase):
    def test_forward_shapes_and_safety(self):
        torch.manual_seed(0)
        block = HyperbolicFusionBlock(
            d_patch=768, d_hyp=32, n_classes=4,
            hidden_layers=2, alpha_init=-3.0,
            curvature_init=1.0, tau_init=0.1, eps=1e-5,
        )
        # Initialise anchors with random "text features" in d_hyp
        raw = torch.randn(4, 32) * 0.3
        block.init_T_from_text_features(raw)

        X = torch.randn(2, 9, 768) * 0.5
        out = block(X)
        self.assertEqual(out['X_aware'].shape, (2, 9, 768))
        self.assertEqual(out['P_fused'].shape, (2, 9, 32))
        self.assertEqual(out['A'].shape,       (2, 9, 4))
        self.assertEqual(out['attn'].shape,    (2, 9, 4))
        c = float(out['c'])
        self.assertTrue(_in_ball(out['P_fused'], c))
        self.assertTrue(_in_ball(out['T_h'], c))
        for k in ('X_aware', 'P_fused', 'A', 'attn'):
            self.assertFalse(torch.isnan(out[k]).any(), f'NaN in {k}')
            self.assertFalse(torch.isinf(out[k]).any(), f'Inf in {k}')

    def test_alpha_near_zero_keeps_X_aware_near_X(self):
        torch.manual_seed(1)
        block = HyperbolicFusionBlock(
            d_patch=64, d_hyp=16, n_classes=3,
            hidden_layers=1, alpha_init=-6.0,  # σ ≈ 0.0025
            curvature_init=1.0, tau_init=0.1,
        )
        block.init_T_from_text_features(torch.randn(3, 16) * 0.3)
        X = torch.randn(1, 4, 64)
        out = block(X)
        diff = (out['X_aware'] - X).abs().mean()
        self.assertLess(float(diff), 0.05,
                         f'X_aware drifted too far at alpha_init=-6 (diff={diff:.3f})')

    def test_state_dict_roundtrip(self):
        torch.manual_seed(2)
        block = HyperbolicFusionBlock(
            d_patch=128, d_hyp=16, n_classes=5,
            hidden_layers=2, curvature_init=1.0, tau_init=0.1)
        block.init_T_from_text_features(torch.randn(5, 16) * 0.3)
        sd = block.state_dict_for_save()

        block2 = HyperbolicFusionBlock(
            d_patch=128, d_hyp=16, n_classes=5,
            hidden_layers=2, curvature_init=1.0, tau_init=0.1)
        block2.load_state_dict_from_save(sd)

        X = torch.randn(1, 3, 128)
        with torch.no_grad():
            o1 = block(X); o2 = block2(X)
        self.assertTrue(torch.allclose(o1['A'], o2['A'], atol=1e-5))
        self.assertTrue(torch.allclose(o1['X_aware'], o2['X_aware'], atol=1e-5))


if __name__ == '__main__':
    unittest.main()
