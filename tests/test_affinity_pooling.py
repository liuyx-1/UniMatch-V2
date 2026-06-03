import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.affinity_cls import AffinityPooling


class AffinityPoolingTest(unittest.TestCase):
    def setUp(self):
        self.A = torch.tensor([[[0.1, 0.8],
                                [0.9, 0.2],
                                [0.7, 0.1]]], dtype=torch.float32)

    def test_topk_matches_original_pooling(self):
        pool = AffinityPooling(num_classes=2, mode='topk', topk=2)
        out = pool(self.A)
        expected = torch.tensor([[(0.9 + 0.7) / 2.0, (0.8 + 0.2) / 2.0]])
        self.assertTrue(torch.allclose(out['image_logits'], expected))
        self.assertIsNone(out['selection_mask'])

    def test_soft_threshold_is_differentiable_and_finite(self):
        A = self.A.clone().requires_grad_(True)
        pool = AffinityPooling(num_classes=2, mode='soft_threshold',
                               threshold_init=0.3, gamma=0.1,
                               threshold_learnable=True)
        out = pool(A)
        self.assertFalse(torch.isnan(out['image_logits']).any())
        self.assertEqual(tuple(out['selection_mask'].shape), (1, 3, 2))
        out['image_logits'].sum().backward()
        self.assertIsNotNone(A.grad)
        self.assertIsNotNone(pool.theta.grad)

    def test_hard_threshold_selects_expected_patches(self):
        pool = AffinityPooling(num_classes=2, mode='hard_threshold',
                               threshold_init=0.3, threshold_learnable=False,
                               min_selected=1)
        out = pool(self.A)
        expected_mask = torch.tensor([[[0.0, 1.0],
                                       [1.0, 0.0],
                                       [1.0, 0.0]]])
        self.assertTrue(torch.equal(out['selection_mask'], expected_mask))
        self.assertTrue(torch.equal(out['selected_count'], torch.tensor([[2.0, 1.0]])))

    def test_empty_hard_selection_uses_fallback_without_nan(self):
        pool = AffinityPooling(num_classes=2, mode='hard_threshold',
                               topk=1, threshold_init=2.0,
                               threshold_learnable=False, min_selected=1)
        out = pool(self.A)
        expected = torch.tensor([[0.9, 0.8]])
        self.assertFalse(torch.isnan(out['image_logits']).any())
        self.assertTrue(torch.allclose(out['image_logits'], expected))
        self.assertTrue(torch.equal(out['selected_count'], torch.zeros(1, 2)))


if __name__ == '__main__':
    unittest.main()
