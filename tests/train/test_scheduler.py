import unittest

import torch
from torch.optim import SGD
from torch.nn import Parameter

from src.train.scheduler import build_scheduler


class SchedulerTests(unittest.TestCase):
    def test_cosine_decay_preserves_differential_learning_rates(self) -> None:
        head_parameter = Parameter(torch.ones(1))
        backbone_parameter = Parameter(torch.ones(1))
        optimizer = SGD(
            [
                {"params": [head_parameter], "lr": 1e-3},
                {"params": [backbone_parameter], "lr": 1e-5},
            ]
        )
        scheduler = build_scheduler(optimizer, num_epochs=4)

        self.assertAlmostEqual(
            optimizer.param_groups[0]["lr"]
            / optimizer.param_groups[1]["lr"],
            100,
        )

        for _ in range(4):
            optimizer.step()
            scheduler.step()

        self.assertAlmostEqual(
            optimizer.param_groups[0]["lr"]
            / optimizer.param_groups[1]["lr"],
            100,
        )
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-7)


if __name__ == "__main__":
    unittest.main()
