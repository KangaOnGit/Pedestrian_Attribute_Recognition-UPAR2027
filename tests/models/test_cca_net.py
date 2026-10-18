import math
import unittest

import torch
from torch import nn

from src.models.cca_net import CCA_NET


class FixedVisionEncoder(nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "image_features": images,
            "shallow_features": images,
        }


class FixedTextEncoder(nn.Module):
    def forward(self) -> torch.Tensor:
        return torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
        )


class CcaNetLogitTests(unittest.TestCase):
    def test_forward_returns_siglip_scaled_and_biased_logits(self) -> None:
        model = CCA_NET.__new__(CCA_NET)
        nn.Module.__init__(model)
        model.siglip2 = nn.Module()
        model.siglip2.logit_scale = nn.Parameter(
            torch.tensor([math.log(4.0)])
        )
        model.siglip2.logit_bias = nn.Parameter(torch.tensor([-1.0]))
        model.vision_encoder = FixedVisionEncoder()
        model.text_encoder = FixedTextEncoder()
        model.con_text = False

        images = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
        )
        logits = model(images)

        torch.testing.assert_close(
            logits,
            torch.tensor([[3.0, -1.0], [-1.0, 3.0]]),
        )
        self.assertGreater(float(logits.max()), 1.0)
        self.assertLess(float(logits.min()), 0.0)


if __name__ == "__main__":
    unittest.main()
