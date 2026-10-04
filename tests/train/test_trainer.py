from __future__ import annotations

import unittest

import torch
from torch import nn

from src.train.trainer import Trainer


class TrainerGradientDiagnosticsTests(unittest.TestCase):
    def test_gradient_diagnostics_report_global_and_component_norms(self) -> None:
        model = nn.Sequential(nn.Linear(2, 1), nn.Linear(1, 1))
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.device = torch.device("cpu")

        model[0].weight.grad = torch.tensor([[3.0, 4.0]])
        model[0].bias.grad = torch.tensor([12.0])
        model[1].weight.grad = torch.tensor([[0.0]])

        metrics, nonfinite_gradients = trainer._gradient_diagnostics()

        self.assertEqual(nonfinite_gradients, [])
        self.assertEqual(metrics["grad_parameter_count"], 3.0)
        self.assertEqual(metrics["grad_trainable_parameter_count"], 4.0)
        self.assertEqual(metrics["grad_missing_parameter_count"], 1.0)
        self.assertEqual(metrics["grad_nonfinite_count"], 0.0)
        self.assertAlmostEqual(metrics["grad_norm"], 13.0)
        self.assertAlmostEqual(metrics["grad_norm/0"], 13.0)
        self.assertAlmostEqual(metrics["grad_norm/1"], 0.0)
        self.assertEqual(metrics["grad_missing/1"], 1.0)

    def test_gradient_diagnostics_name_nonfinite_parameters(self) -> None:
        model = nn.Linear(2, 1)
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.device = torch.device("cpu")
        model.weight.grad = torch.tensor([[float("nan"), 1.0]])
        model.bias.grad = torch.tensor([float("inf")])

        metrics, nonfinite_gradients = trainer._gradient_diagnostics()

        self.assertEqual(nonfinite_gradients, ["weight", "bias"])
        self.assertEqual(metrics["grad_nonfinite_count"], 2.0)
        self.assertTrue(torch.isnan(torch.tensor(metrics["grad_norm"])))


if __name__ == "__main__":
    unittest.main()
