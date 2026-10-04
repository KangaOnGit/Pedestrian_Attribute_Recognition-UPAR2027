from __future__ import annotations

import unittest

import torch

from src.builders.loss import build_loss


class FocalLossTests(unittest.TestCase):
    def test_focal_loss_balances_each_attribute_from_training_targets(self) -> None:
        targets = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ]
        )

        loss = build_loss("focal", targets)

        self.assertIsInstance(loss.pos_weight, torch.Tensor)
        self.assertIsInstance(loss.class_weight, torch.Tensor)
        torch.testing.assert_close(loss.pos_weight, torch.tensor([3.0, 1.0 / 3.0]))
        torch.testing.assert_close(loss.class_weight, torch.tensor([2.0 / 3.0, 2.0]))

        logits = torch.zeros_like(targets, requires_grad=True)
        value = loss(logits, targets)
        self.assertTrue(torch.isfinite(value))
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_focal_loss_rejects_invalid_training_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "training_targets"):
            build_loss("focal", torch.tensor([[0.0, 2.0]]))


if __name__ == "__main__":
    unittest.main()
