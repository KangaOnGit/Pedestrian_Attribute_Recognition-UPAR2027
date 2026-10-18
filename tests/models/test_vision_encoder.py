from types import SimpleNamespace
import unittest

import torch
from torch import nn

from src.models.vision_encoder import VisionEncoder


class RecordingVisionEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inputs: torch.Tensor | None = None

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        self.inputs = pixel_values
        hidden_states = pixel_values.flatten(2).transpose(1, 2)
        return SimpleNamespace(
            last_hidden_state=hidden_states,
            pooler_output=hidden_states.mean(dim=1),
        )


class VisionEncoderInputTests(unittest.TestCase):
    def test_raw_augmented_image_is_passed_to_siglip_vision(self) -> None:
        backbone = RecordingVisionEncoder()
        encoder = VisionEncoder(
            vision_encoder=backbone,
            shallow_hidden_dim=4,
            shallow_out_dim=4,
        )
        images = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(
            2, 3, 4, 4
        )

        output = encoder(images)

        self.assertIs(backbone.inputs, images)
        self.assertEqual(tuple(output["image_features"].shape), (2, 3))
        self.assertEqual(tuple(output["patch_features"].shape), (2, 16, 3))


if __name__ == "__main__":
    unittest.main()
