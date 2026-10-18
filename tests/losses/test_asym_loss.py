import unittest
from unittest.mock import patch

import torch

from src.builders.loss import build_loss
from src.losses.asym_loss import AsymmetricLossOptimized


class AsymmetricLossTests(unittest.TestCase):
    def test_loss_builder_creates_asymmetric_loss(self) -> None:
        loss = build_loss("asym")

        self.assertIsInstance(loss, AsymmetricLossOptimized)
        self.assertEqual(loss.gamma_neg, 4)
        self.assertEqual(loss.gamma_pos, 1)
        self.assertEqual(loss.clip, 0.05)

    def test_asymmetric_loss_has_finite_gradients(self) -> None:
        loss = build_loss("asym")
        logits = torch.tensor(
            [[20.0, -20.0], [-100.0, 100.0]],
            requires_grad=True,
        )
        targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

        value = loss(logits, targets)
        value.backward()

        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_cca_defaults_to_asymmetric_but_allows_loss_override(self) -> None:
        from scripts.train import parse_args

        with patch("sys.argv", ["train.py", "--architecture", "cca"]):
            self.assertEqual(parse_args().loss, "asym")

        with patch(
            "sys.argv",
            ["train.py", "--architecture", "cca", "--loss", "bce"],
        ):
            self.assertEqual(parse_args().loss, "bce")

    def test_other_architectures_keep_configured_default_loss(self) -> None:
        from scripts.train import parse_args

        with patch(
            "sys.argv",
            ["train.py", "--architecture", "image", "--roi-generator", "none"],
        ):
            self.assertEqual(parse_args().loss, "FocalAsym")


if __name__ == "__main__":
    unittest.main()
