"""Tests for the DPT.encode / decode / patches_override hooks.

Verifies:
  1. forward without patches_override is bitwise-identical to encode→decode chain
  2. patches_override replaces features[-1] exactly
  3. providing patches_override with the same patches → identical output
  4. providing a different tensor changes the output
  5. shape mismatch on patches_override raises AssertionError

Skipped automatically if DINOv2 backbone weights are not present on disk.
"""
from __future__ import annotations
import os, sys, unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _has_dinov2_weights() -> bool:
    candidates = [
        '/root/autodl-tmp/code/UniMatch-V2/pretrained/dinov2_vitb14_pretrain.pth',
        './pretrained/dinov2_vitb14_pretrain.pth',
        '/data/code/UniMatch-V2/pretrained/dinov2_vitb14_pretrain.pth',
    ]
    return any(os.path.isfile(p) for p in candidates)


@unittest.skipUnless(_has_dinov2_weights(),
                      'DPT needs DINOv2 weights at ./pretrained/dinov2_vitb14_pretrain.pth')
class TestDPTPatchOverride(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from model.semseg.dpt import DPT
        cfg = {'encoder_size': 'base', 'features': 128,
               'out_channels': [96, 192, 384, 768], 'nclass': 7}
        cls.model = DPT(**cfg).cuda().eval() if torch.cuda.is_available() else DPT(**cfg).eval()
        # tiny input that's a multiple of 14
        cls.x = torch.randn(1, 3, 224, 224).to(next(cls.model.parameters()).device)

    def test_forward_equals_encode_decode(self):
        with torch.no_grad():
            out_a = self.model(self.x, return_cls=False)
            features, _, ph, pw = self.model.encode(self.x, return_cls=False)
            out_b = self.model.decode(features, ph, pw, comp_drop=False)
        self.assertEqual(out_a.shape, out_b.shape)
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-5),
                        f'mismatch: max diff = {(out_a-out_b).abs().max():.2e}')

    def test_patches_override_same_value_is_identity(self):
        with torch.no_grad():
            out_a, cls_tok, patches = self.model(self.x, return_cls=True)
            out_b, _, _ = self.model(self.x, return_cls=True, patches_override=patches)
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-5),
                        'override with same patches must be identity')

    def test_patches_override_different_value_changes_output(self):
        with torch.no_grad():
            out_a, _, patches = self.model(self.x, return_cls=True)
            # Perturb by a small constant — output should change
            patches2 = patches + 0.5
            out_b, _, _ = self.model(self.x, return_cls=True, patches_override=patches2)
        self.assertFalse(torch.allclose(out_a, out_b, atol=1e-3),
                        'override with different patches must change output')

    def test_patches_override_shape_mismatch_raises(self):
        with torch.no_grad():
            _, _, patches = self.model(self.x, return_cls=True)
            bad = patches[:, :-1, :]               # wrong N
            with self.assertRaises(AssertionError):
                self.model(self.x, return_cls=True, patches_override=bad)


if __name__ == '__main__':
    unittest.main()
