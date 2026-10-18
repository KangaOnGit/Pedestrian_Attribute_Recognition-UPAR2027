import unittest

import torch
from torch import nn

from src.builders.optimizer import build_optimizer


class OptimizerTests(unittest.TestCase):
    def test_cca_backbones_use_lower_learning_rate(self) -> None:
        class CCAWithFinetunedBackbones(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.siglip2 = nn.Module()
                self.siglip2.text_model = nn.Linear(4, 4)
                self.siglip2.vision_model = nn.Linear(4, 4)
                self.siglip2.logit_scale = nn.Parameter(torch.ones(1))
                self.context = nn.Parameter(torch.ones(4))

        model = CCAWithFinetunedBackbones()
        optimizer = build_optimizer(
            model,
            optim_name="adamw",
            lr=1e-3,
            weight_decay=1e-5,
        )

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optimizer.param_groups[1]["lr"], 1e-5)
        self.assertEqual(
            {
                id(parameter)
                for parameter in optimizer.param_groups[1]["params"]
            },
            {
                id(parameter)
                for parameter in model.siglip2.text_model.parameters()
            }
            | {
                id(parameter)
                for parameter in model.siglip2.vision_model.parameters()
            },
        )
        self.assertIn(
            id(model.siglip2.logit_scale),
            {
                id(parameter)
                for parameter in optimizer.param_groups[0]["params"]
            },
        )


if __name__ == "__main__":
    unittest.main()
