from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.builders.optimizer import build_optimizer
from src.models.feature_encoder import DINOv3FeatureEncoder, ImageAttributeModel


class FakeBatch(dict):
    def to(self, device: torch.device) -> FakeBatch:
        return FakeBatch(
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.items()
            }
        )


class FakeDINOv3(nn.Module):
    config = SimpleNamespace(hidden_size=3, patch_size=4, num_register_tokens=0)

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, pixel_values: torch.Tensor) -> SimpleNamespace:
        pooled = pixel_values.mean(dim=(-1, -2)) * self.weight
        patch_offsets = torch.arange(
            12,
            device=pixel_values.device,
            dtype=pixel_values.dtype,
        )[None, :, None]
        patch_tokens = pooled[:, None, :] + patch_offsets
        return SimpleNamespace(
            pooler_output=pooled,
            last_hidden_state=torch.cat((pooled[:, None, :], patch_tokens), dim=1),
        )


class FakeImageProcessor:
    def __init__(self) -> None:
        self.images: torch.Tensor | None = None
        self.do_rescale: bool | None = None

    def __call__(
        self,
        images: torch.Tensor,
        return_tensors: str,
        do_rescale: bool,
    ) -> FakeBatch:
        assert return_tensors == "pt"
        self.images = images.detach().clone()
        self.do_rescale = do_rescale
        return FakeBatch({"pixel_values": images})


class FeatureEncoderTests(unittest.TestCase):
    def test_dinov3_receives_rgb_pixels_and_train_returns_encoder(self) -> None:
        processor = FakeImageProcessor()
        with (
            patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=processor,
            ),
            patch(
                "transformers.AutoModel.from_pretrained",
                return_value=FakeDINOv3(),
            ),
        ):
            encoder = DINOv3FeatureEncoder(
                hidden_dim=8,
                model_id="test/dinov3",
                trainable=False,
            )
            normalized_images = torch.zeros(2, 3, 16, 12)
            features = encoder(normalized_images)

        self.assertEqual(tuple(features.shape), (2, 8))
        self.assertEqual(encoder.projection[0].in_features, 32)
        self.assertIs(encoder.train(), encoder)
        self.assertFalse(encoder.model.training)
        self.assertIsNotNone(processor.images)
        self.assertFalse(processor.do_rescale)
        torch.testing.assert_close(
            processor.images[0, :, 0, 0],
            torch.tensor((0.485, 0.456, 0.406)),
        )

    def test_image_attribute_model_returns_logits_and_backpropagates(self) -> None:
        model = ImageAttributeModel(
            num_classes=5,
            hidden_dim=16,
            backbone_type="conv",
        )
        logits = model(torch.randn(2, 3, 64, 32))

        self.assertEqual(tuple(logits.shape), (2, 5))
        logits.sum().backward()
        self.assertIsNotNone(model.classifier[2].weight.grad)
        self.assertIsNotNone(model.attribute_queries.grad)
        self.assertIsNotNone(model.visual_attention.in_proj_weight.grad)
        self.assertIsNotNone(model.label_interaction.layers[0].self_attn.in_proj_weight.grad)

    def test_image_attribute_model_uses_pretrained_region_tokens(self) -> None:
        with (
            patch(
                "transformers.AutoImageProcessor.from_pretrained",
                return_value=FakeImageProcessor(),
            ),
            patch(
                "transformers.AutoModel.from_pretrained",
                return_value=FakeDINOv3(),
            ),
        ):
            model = ImageAttributeModel(
                num_classes=5,
                hidden_dim=16,
                backbone_type="dinov3",
            )
            images = torch.randn(2, 3, 16, 12)
            visual_tokens = model.backbone.forward_patch_tokens(images)
            logits = model(images)

        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(tuple(visual_tokens.shape), (2, 13, 16))

    def test_optimizer_uses_lower_learning_rate_for_finetuned_backbone(self) -> None:
        class ModelWithPretrainedBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Module()
                self.backbone.model = nn.Linear(4, 4)
                self.classifier = nn.Linear(4, 2)

        optimizer = build_optimizer(
            ModelWithPretrainedBackbone(),
            optim_name="adamw",
            lr=1e-3,
            weight_decay=1e-5,
        )

        self.assertEqual(len(optimizer.param_groups), 2)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        self.assertEqual(optimizer.param_groups[1]["lr"], 1e-5)


if __name__ == "__main__":
    unittest.main()
