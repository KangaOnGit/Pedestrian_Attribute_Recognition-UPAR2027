from __future__ import annotations

import unittest

import torch

from src.builders.loss import build_loss
from src.losses.focal_loss import FocalLoss


class FocalLossTests(unittest.TestCase):
    def test_fractional_gamma_has_finite_gradients_for_saturated_correct_logits(self) -> None:
        loss = build_loss("focal")
        logits = torch.tensor(
            [[20.0, -20.0], [100.0, -100.0]],
            requires_grad=True,
        )
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        value = loss(logits, targets)
        value.backward()

        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_zero_gamma_matches_weighted_binary_cross_entropy_for_saturated_logits(self) -> None:
        loss = FocalLoss(alpha=0.5, gamma=0.0)
        logits = torch.tensor([[100.0, -100.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0]])

        value = loss(logits, targets)
        value.backward()

        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.detach(),
            targets,
        )
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(value, expected * 0.5)

    def test_log_space_focal_weight_matches_probability_formula_at_moderate_logits(self) -> None:
        alpha = 0.5
        gamma = 0.2
        loss = build_loss("focal")
        logits = torch.tensor([[-2.0, 0.5, 3.0]])
        targets = torch.tensor([[0.0, 1.0, 1.0]])
        probabilities = logits.sigmoid()
        p_t = probabilities * targets + (1 - probabilities) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        expected = (
            alpha_t
            * (1 - p_t).pow(gamma)
            * torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
        ).mean()

        torch.testing.assert_close(loss(logits, targets), expected)

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
