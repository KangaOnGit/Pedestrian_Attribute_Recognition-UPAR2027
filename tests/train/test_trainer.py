from __future__ import annotations

import unittest

import torch
from torch import nn
from torch.optim import SGD

from src.train.trainer import Trainer


class TrainerGradientDiagnosticsTests(unittest.TestCase):
    def build_minimal_trainer(
        self,
        model: nn.Module,
        criterion: nn.Module,
    ) -> Trainer:
        trainer = Trainer.__new__(Trainer)
        trainer.model = model
        trainer.device = torch.device("cpu")
        trainer.train_loader = [
            (
                torch.ones(2, 3),
                torch.ones(2, 3),
                torch.zeros(2, 1),
            )
        ]
        trainer.optimizer = SGD(model.parameters(), lr=0.1)
        trainer.criterion = criterion
        trainer.logging_steps = 100
        trainer.global_step = 0
        trainer.epochs = 1
        trainer.wandb_run = None
        return trainer

    def test_nonfinite_loss_stops_before_backward_or_optimizer_step(self) -> None:
        class NonfiniteLoss(nn.Module):
            def forward(
                self,
                logits: torch.Tensor,
                labels: torch.Tensor,
            ) -> torch.Tensor:
                return (
                    logits.sum() + labels.sum()
                ) * torch.tensor(float("nan"))

        model = nn.Linear(3, 1)
        trainer = self.build_minimal_trainer(model, NonfiniteLoss())
        parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

        with self.assertRaisesRegex(FloatingPointError, "Non-finite loss"):
            trainer.train_epoch(epoch=1)

        for before, after in zip(parameters_before, model.parameters()):
            torch.testing.assert_close(after, before)

    def test_nonfinite_gradients_stop_before_optimizer_step(self) -> None:
        model = nn.Linear(3, 1)
        model.weight.register_hook(
            lambda gradient: torch.full_like(gradient, float("nan"))
        )
        trainer = self.build_minimal_trainer(model, nn.BCEWithLogitsLoss())
        parameters_before = [parameter.detach().clone() for parameter in model.parameters()]

        with self.assertRaisesRegex(FloatingPointError, "Non-finite gradients"):
            trainer.train_epoch(epoch=1)

        for before, after in zip(parameters_before, model.parameters()):
            torch.testing.assert_close(after, before)
        self.assertIsNone(model.weight.grad)

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
