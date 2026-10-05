from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import torch
from torch import nn
from torch.optim import SGD

from src.metrics.base import eval_metrics
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

    def test_train_saves_best_challenge_average_checkpoint(self) -> None:
        trainer = Trainer.__new__(Trainer)
        trainer.epochs = 2
        trainer.start_epoch = 1
        trainer.global_step = 0
        trainer.best_mA = float("-inf")
        trainer.best_f1 = float("-inf")
        trainer.best_challenge_avg = float("-inf")
        trainer.wandb_run = None
        trainer.scheduler = Mock()
        trainer.optimizer = SGD(nn.Linear(1, 1).parameters(), lr=0.1)
        trainer.criterion = nn.Identity()
        trainer.eval = Mock(
            side_effect=[
                (
                    0.5,
                    eval_metrics(
                        avg=0.7,
                        mA=0.6,
                        label_f1=0.5,
                        inst_acc=0.4,
                        inst_prec=0.3,
                        inst_rec=0.2,
                        inst_f1=0.1,
                    ),
                ),
                (
                    0.4,
                    eval_metrics(
                        avg=0.65,
                        mA=0.61,
                        label_f1=0.55,
                        inst_acc=0.4,
                        inst_prec=0.3,
                        inst_rec=0.2,
                        inst_f1=0.1,
                    ),
                ),
            ],
        )
        trainer.train_epoch = Mock(return_value=1.0)

        with TemporaryDirectory() as output_dir:
            trainer.output_dir = Path(output_dir)
            saved_checkpoints: list[tuple[str, eval_metrics]] = []
            trainer.save_checkpoint = lambda epoch, metrics, filename: (
                saved_checkpoints.append((filename, metrics))
            )

            trainer.train()

        challenge_checkpoints = [
            metrics
            for filename, metrics in saved_checkpoints
            if filename == "best_challenge_avg.pt"
        ]
        self.assertEqual(len(challenge_checkpoints), 1)
        self.assertEqual(challenge_checkpoints[0].avg, 0.7)
        self.assertEqual(trainer.best_challenge_avg, 0.7)
        self.assertEqual(trainer.best_f1, 0.55)


if __name__ == "__main__":
    unittest.main()
