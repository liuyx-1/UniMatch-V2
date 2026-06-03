"""Unit tests for the hyperbolic patch-text affinity module.

Covers the six checks requested when wiring hyperbolic affinity into
the Stage-I AffinityClassifier and the Stage-II AffinitySideBranch:

  1. expmap0 output norm is strictly below 1/sqrt(c)
  2. poincare_distance(x, x) ≈ 0
  3. hyperbolic_affinity returns shape [B, N, C']
  4. no NaN for random inputs (covers near-ball-boundary stress)
  5. AffinityClassifier forward works in 'hyperbolic' mode (no network access)
  6. AffinitySideBranch consumes a hyperbolic Stage-I checkpoint and fuses cleanly

The Stage-I and Stage-II tests construct *fake* Stage-I checkpoints in a
temp dir; they do not download any model.
"""
from __future__ import annotations
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the repo importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from util.hyperbolic import (project_to_ball, expmap0,
                              poincare_distance, hyperbolic_affinity)


class TestHyperbolicUtils(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.B, self.N, self.C, self.d = 2, 9, 4, 8
        self.eps = 1e-5

    # ── 1. ExpMap0 output stays strictly inside the ball ──────────────
    def test_expmap0_in_ball(self):
        for c_val in (0.1, 0.5, 1.0, 2.5):
            u = torch.randn(self.B, self.N, self.d) * 5.0   # large tangent vectors
            ph = expmap0(u, c=c_val, eps=self.eps)
            radius = 1.0 / math.sqrt(c_val)
            norms = ph.norm(dim=-1)
            self.assertTrue(torch.all(norms < radius),
                            f'c={c_val}: max norm {norms.max():.6f} >= 1/sqrt(c)={radius:.6f}')

    # ── 2. Poincaré distance from a point to itself is ~ 0 ────────────
    def test_poincare_distance_self_is_zero(self):
        c_val = 1.0
        x = torch.randn(self.B, self.N, self.d) * 0.3
        x = expmap0(x, c=c_val, eps=self.eps)
        d = poincare_distance(x, x, c=c_val, eps=self.eps)
        self.assertEqual(d.shape, (self.B, self.N))
        self.assertTrue(torch.all(d.abs() < 1e-3),
                        f'self-distance not ~0: max={d.abs().max():.6f}')

    # ── 3. Affinity tensor shape ──────────────────────────────────────
    def test_hyperbolic_affinity_shape(self):
        c_val = 1.0
        Ph = expmap0(torch.randn(self.B, self.N, self.d) * 0.2,
                     c=c_val, eps=self.eps)
        Th = expmap0(torch.randn(self.C, self.d) * 0.2,
                     c=c_val, eps=self.eps)
        tau_h = torch.tensor(0.1)
        A = hyperbolic_affinity(Ph, Th, tau_h=tau_h, c=c_val, eps=self.eps)
        self.assertEqual(tuple(A.shape), (self.B, self.N, self.C))

    # ── 4. No NaN over random and boundary-near inputs ────────────────
    def test_no_nan(self):
        torch.manual_seed(123)
        for _ in range(20):
            c_val = float(torch.rand(1).item() * 2.0 + 0.05)
            scale = float(torch.rand(1).item() * 4.0 + 0.5)  # mix small + huge
            u = torch.randn(self.B, self.N, self.d) * scale
            ph = expmap0(u, c=c_val, eps=self.eps)
            th = expmap0(torch.randn(self.C, self.d) * scale,
                         c=c_val, eps=self.eps)
            A = hyperbolic_affinity(ph, th,
                                     tau_h=torch.tensor(0.1),
                                     c=c_val, eps=self.eps)
            self.assertFalse(torch.isnan(A).any(), f'NaN in A at c={c_val} scale={scale}')
            self.assertFalse(torch.isinf(A).any(), f'Inf in A at c={c_val} scale={scale}')

    # ── 5. AffinityClassifier forward in 'hyperbolic' mode ────────────
    @unittest.skipUnless(
        os.environ.get('RUN_NET_TESTS', '') == '1',
        'Stage-I forward needs SigLIP weights; set RUN_NET_TESTS=1 to enable.'
    )
    def test_stage1_forward_hyperbolic(self):
        from model.affinity_cls import AffinityClassifier
        cls_names = ['shaft', 'wrist', 'clasper', 'gallbladder']
        m = AffinityClassifier(
            'google/siglip2-base-patch16-256',
            cls_names,
            d_latent=64, topk=3,
            vision_backbone='siglip',
            affinity_metric='hyperbolic',
            hyperbolic_dim=64,
            hyperbolic_curvature_init=1.0,
            hyperbolic_temperature_init=0.1,
        )
        m.init_T_from_templates({c: [f'a photo of {c}'] for c in cls_names})
        x = torch.randn(2, 3, 256, 256)
        out = m(x)
        self.assertEqual(out['cls'].shape, (2, len(cls_names)))
        self.assertFalse(torch.isnan(out['cls']).any())
        self.assertFalse(torch.isnan(out['affinity']).any())

    # ── 6. AffinitySideBranch consumes a hyperbolic Stage-I checkpoint ─
    def test_stage2_side_branch_loads_hyperbolic_ckpt(self):
        """Build a tiny synthetic hyperbolic ckpt and load it via AffinitySideBranch."""
        from util.affinity_side import AffinitySideBranch
        d_text, d_latent = 768, 16
        n_new = 3
        n_orig = 4  # one bg + 3 fg
        c_val = 1.0

        # Visual projection has the exact (LN, Linear, GELU, Dropout, Linear) layout.
        v_proj = nn.Sequential(
            nn.LayerNorm(d_text), nn.Linear(d_text, 256),
            nn.GELU(), nn.Dropout(0.0), nn.Linear(256, d_latent),
        )
        v_pre_ln = nn.LayerNorm(d_latent)
        T_h = expmap0(torch.randn(n_new, d_latent) * 0.2, c=c_val).detach()

        ckpt = {
            'visual_proj': v_proj.state_dict(),
            'T_anchor': T_h.cpu(),
            'log_tau':  torch.tensor(0.0),
            'cls_bias': torch.zeros(n_new),
            'd_latent': d_latent,
            'topk': 3,
            'pooling_mode': 'topk',
            'threshold':       torch.full((n_new,), 0.3),
            'threshold_init':  0.3,
            'threshold_gamma': 0.1,
            'threshold_learnable': False,
            'min_selected':    1,
            'hard_threshold_eval': True,
            'orig_to_new': {1: 0, 2: 1, 3: 2},
            # hyperbolic config
            'affinity_metric': 'hyperbolic',
            'hyperbolic_dim': d_latent,
            'hyperbolic_eps': 1e-5,
            'hyperbolic_distance_scale': 1.0,
            'hyperbolic_rho':       torch.tensor(math.log(math.expm1(1.0))),
            'hyperbolic_log_tau_h': torch.tensor(math.log(math.expm1(0.1))),
            'hyperbolic_v_pre_ln':  v_pre_ln.state_dict(),
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'affinity_fake.pt'
            torch.save(ckpt, p)

            side = AffinitySideBranch(str(p), n_orig_classes=n_orig)
            self.assertEqual(side.affinity_metric, 'hyperbolic')

            # forward — 9 patches (3x3) on a small spatial output
            patches = torch.randn(2, 9, d_text)
            out = side(patches, target_h=14, target_w=14, patch_hw=(3, 3))
            self.assertEqual(tuple(out['prior'].shape), (2, n_orig, 14, 14))
            self.assertEqual(tuple(out['gate'].shape),  (2, 1, 14, 14))
            self.assertEqual(tuple(out['cls_logits'].shape), (2, n_new))
            for k in ('prior', 'gate', 'cls_logits'):
                self.assertFalse(torch.isnan(out[k]).any(), f'NaN in {k}')

            # Fusion check
            dpt = torch.randn(2, n_orig, 14, 14)
            fused = side.fuse(dpt, out)
            self.assertEqual(tuple(fused.shape), (2, n_orig, 14, 14))
            self.assertFalse(torch.isnan(fused).any())


if __name__ == '__main__':
    unittest.main()
