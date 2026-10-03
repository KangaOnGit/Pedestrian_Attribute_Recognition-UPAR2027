from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.models import SparseROIAttributeModel


class FakeBatch(dict):
    def to(self, device: torch.device) -> FakeBatch:
        return FakeBatch(
            {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in self.items()
            }
        )


class FakeSam3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, **_inputs: object) -> object:
        return tuple(_inputs)


class FakeSam3Processor:
    def __init__(self) -> None:
        self.seen_prompts: list[str] = []

    def __call__(
        self,
        images: object,
        text: str,
        return_tensors: str,
    ) -> FakeBatch:
        assert return_tensors == "pt"
        assert images is not None
        self.seen_prompts.append(text)
        return FakeBatch({"original_sizes": torch.tensor([[48, 24]])})

    def post_process_instance_segmentation(
        self,
        outputs: object,
        threshold: float,
        mask_threshold: float,
        target_sizes: list[list[int]],
    ) -> list[dict[str, torch.Tensor]]:
        assert outputs is not None
        assert threshold == 0.5
        assert mask_threshold == 0.5
        assert target_sizes == [[48, 24]]
        return [{"boxes": torch.tensor([[1.0, 2.0, 20.0, 40.0]])}]


class SparseROIAttributeModelTests(unittest.TestCase):
    def build_model(self, **kwargs: object) -> SparseROIAttributeModel:
        return SparseROIAttributeModel(
            num_classes=7,
            hidden_dim=16,
            num_experts=3,
            attention_k=2,
            roi_size=(16, 8),
            **kwargs,
        )

    def test_forward_with_boxes_has_expected_shape_and_gradients(self) -> None:
        model = self.build_model()
        images = torch.randn(2, 3, 64, 32)
        boxes = torch.tensor(
            [
                [[2, 3, 20, 50], [0, 0, 0, 0]],
                [[4, 5, 25, 60], [10, 10, 30, 40]],
            ],
            dtype=torch.float32,
        )

        logits = model(images, boxes)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(len(model.segmentation_experts), 5)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertIsNotNone(model.roi_router[0].weight.grad)
        self.assertGreater(model.roi_router[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(
            model.segmentation_router[0].weight.grad.abs().sum().item(),
            0,
        )

    def test_sam3_is_initialized_and_generates_prompt_boxes(self) -> None:
        processor = FakeSam3Processor()
        with (
            patch("transformers.Sam3Model.from_pretrained", return_value=FakeSam3()) as load_model,
            patch("transformers.Sam3Processor.from_pretrained", return_value=processor) as load_processor,
        ):
            model = self.build_model(
                prompts=("person", "backpack"),
                roi_generator="sam3",
                sam3_model_id="test/sam3",
            )
            logits = model(torch.randn(2, 3, 48, 24))

        self.assertEqual(tuple(logits.shape), (2, 7))
        load_model.assert_called_once_with("test/sam3")
        load_processor.assert_called_once_with("test/sam3")
        self.assertEqual(processor.seen_prompts, ["person", "backpack"] * 2)

    def test_dinov3_can_be_used_as_full_image_backbone(self) -> None:
        class FakeDINOv3(nn.Module):
            config = SimpleNamespace(hidden_size=12)

            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, **inputs: torch.Tensor) -> object:
                batch_size = inputs["pixel_values"].shape[0]
                return SimpleNamespace(
                    pooler_output=torch.ones(batch_size, 12) * self.weight
                )

        class FakeImageProcessor:
            def __call__(
                self,
                images: list[object],
                return_tensors: str,
            ) -> FakeBatch:
                assert return_tensors == "pt"
                return FakeBatch({"pixel_values": torch.ones(len(images), 3, 32, 32)})

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
            model = self.build_model(image_backbone_type="dinov3")
            logits = model(torch.randn(2, 3, 48, 24))

        self.assertEqual(tuple(logits.shape), (2, 7))

    def test_no_valid_rois_uses_full_image_branch(self) -> None:
        model = self.build_model()
        images = torch.randn(2, 3, 64, 32)
        padded_boxes = torch.zeros(2, 2, 4)

        logits = model(images, padded_boxes)

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
