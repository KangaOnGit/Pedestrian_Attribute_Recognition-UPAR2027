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
        return [
            {
                "boxes": torch.tensor(
                    [
                        [1.0, 2.0, 20.0, 40.0],
                        [5.0, 6.0, 22.0, 42.0],
                    ]
                ),
                "scores": torch.tensor([0.4, 0.9]),
            }
        ]


class SparseROIAttributeModelTests(unittest.TestCase):
    def build_model(self, **kwargs: object) -> SparseROIAttributeModel:
        kwargs.setdefault("roi_generator", "none")
        return SparseROIAttributeModel(
            num_classes=7,
            hidden_dim=16,
            num_experts=3,
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

        logits = model(images, images, boxes)
        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertFalse(hasattr(model, "segmentation_experts"))
        self.assertFalse(hasattr(model, "segmentation_router"))
        self.assertFalse(
            any(
                isinstance(layer, nn.MultiheadAttention)
                for layer in model.modules()
            )
        )
        self.assertEqual(model.classifier[0].in_features, 32)
        self.assertTrue(torch.isfinite(logits).all())
        logits.sum().backward()
        self.assertIsNotNone(model.roi_router[0].weight.grad)
        self.assertGreater(model.roi_router[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(
            model.classifier[0].weight.grad.abs().sum().item(),
            0,
        )

    def test_roi_features_are_not_attenuated_by_router_probability(self) -> None:
        model = self.build_model()
        rois = torch.randn(1, 1, 16)
        valid = torch.ones(1, 1, dtype=torch.bool)
        routing_logits = torch.tensor([[8.0, 0.0, 0.0]])

        with patch.object(model.roi_router, "forward", return_value=routing_logits):
            features, present = model._encode_rois(rois, valid)

        expected = model.roi_expert_projection[0](
            model.roi_experts[0](rois[:, 0])
        )
        torch.testing.assert_close(features[0, 0], expected[0])
        self.assertTrue(present[0, 0])

    def test_roi_box_pools_matching_patch_grid_cells(self) -> None:
        model = self.build_model()
        patch_grid = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        boxes = torch.tensor([[[25.0, 25.0, 75.0, 75.0]]])

        pooled, valid = model._pool_roi_features(
            patch_grid,
            boxes,
            image_height=100,
            image_width=100,
        )

        torch.testing.assert_close(pooled, torch.tensor([[[7.5]]]))
        self.assertTrue(valid[0, 0])

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
            images = torch.randn(2, 3, 48, 24)
            images_detector = torch.rand(2, 3, 48, 24)
            generator = model.roi_proposal_generator
            assert generator is not None
            generated_boxes = generator(images, model.prompts)
            logits = model(images, images, images_detector=images_detector)

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(tuple(generated_boxes.shape), (2, 2, 4))
        torch.testing.assert_close(
            generated_boxes,
            torch.tensor([[[5.0, 6.0, 22.0, 42.0]] * 2] * 2),
        )
        load_model.assert_called_once_with("test/sam3")
        load_processor.assert_called_once_with("test/sam3")
        self.assertEqual(processor.seen_prompts, ["person", "backpack"] * 4)

    def test_yoloe_batches_images_and_generates_prompt_boxes(self) -> None:
        class FakeYOLOE:
            def __init__(self, model_id: str) -> None:
                self.model_id = model_id
                self.prompt_calls: list[list[str]] = []
                self.prediction_calls: list[dict[str, object]] = []

            def set_classes(self, prompts: list[str]) -> None:
                self.prompt_calls.append(prompts)

            def predict(self, **kwargs: object) -> list[object]:
                self.prediction_calls.append(kwargs)
                source = kwargs["source"]
                assert isinstance(source, torch.Tensor)
                prompt_index = (len(self.prediction_calls) - 1) % 2
                if prompt_index == 0:
                    return [
                        SimpleNamespace(
                            boxes=SimpleNamespace(
                                xyxy=torch.tensor(
                                    [
                                        [1.0, 2.0, 20.0, 40.0],
                                        [3.0, 4.0, 30.0, 44.0],
                                    ]
                                ),
                                conf=torch.tensor([0.4, 0.9]),
                            )
                        ),
                        SimpleNamespace(boxes=None),
                    ]
                return [
                    SimpleNamespace(
                        boxes=SimpleNamespace(
                            xyxy=torch.tensor([[5.0, 6.0, 35.0, 46.0]]),
                            conf=torch.tensor([0.75]),
                        )
                    ),
                    SimpleNamespace(
                        boxes=SimpleNamespace(
                            xyxy=torch.tensor(
                                [
                                    [7.0, 8.0, 37.0, 47.0],
                                    [8.0, 9.0, 38.0, 49.0],
                                ]
                            ),
                            conf=torch.tensor([0.3, 0.8]),
                        )
                    ),
                ]

        fake_module = SimpleNamespace(YOLOE=FakeYOLOE)
        with patch.dict("sys.modules", {"ultralytics": fake_module}):
            model = self.build_model(
                prompts=("person", "backpack"),
                roi_generator="yoloe",
                yoloe_model_id="test/yoloe",
                yoloe_score_threshold=0.5,
            )
            images_detector = torch.empty(2, 3, 48, 24)
            images_detector[:, 0] = 0.0
            images_detector[:, 1] = 0.5
            images_detector[:, 2] = 1.0
            imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            images = (images_detector - imagenet_mean) / imagenet_std
            generator = model.roi_proposal_generator
            assert generator is not None
            generated_boxes = generator(images_detector, model.prompts)
            logits = model(images, images, images_detector=images_detector)

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(generator.model.model_id, "test/yoloe")
        self.assertEqual(
            generator.model.prompt_calls,
            [["person"], ["backpack"], ["person"], ["backpack"]],
        )
        self.assertEqual(len(generator.model.prediction_calls), 4)
        self.assertEqual(tuple(generated_boxes.shape), (2, 2, 4))
        torch.testing.assert_close(
            generated_boxes,
            torch.tensor(
                [
                    [[3.0, 4.0, 30.0, 44.0], [5.0, 6.0, 35.0, 46.0]],
                    [[0.0, 0.0, 0.0, 0.0], [8.0, 9.0, 38.0, 49.0]],
                ]
            ),
        )
        first_call = generator.model.prediction_calls[0]
        self.assertEqual(first_call["conf"], 0.5)
        self.assertEqual(first_call["device"], "cpu")
        self.assertFalse(first_call["verbose"])
        source = first_call["source"]
        self.assertIsInstance(source, torch.Tensor)
        self.assertEqual(tuple(source.shape), (2, 3, 48, 24))
        torch.testing.assert_close(
            source[0, :, 0, 0],
            torch.tensor([0.0, 0.5, 1.0]),
        )

    def test_yoloe_one_pass_keeps_all_boxes_and_pads_batch(self) -> None:
        class FakeYOLOE:
            def __init__(self, model_id: str) -> None:
                self.model_id = model_id
                self.prompt_calls: list[list[str]] = []
                self.prediction_calls: list[dict[str, object]] = []

            def set_classes(self, prompts: list[str]) -> None:
                self.prompt_calls.append(prompts)

            def predict(self, **kwargs: object) -> list[object]:
                self.prediction_calls.append(kwargs)
                return [
                    SimpleNamespace(
                        boxes=SimpleNamespace(
                            xyxy=torch.tensor(
                                [
                                    [1.0, 2.0, 10.0, 20.0],
                                    [3.0, 4.0, 30.0, 40.0],
                                    [5.0, 6.0, 50.0, 60.0],
                                ]
                            ),
                            conf=torch.tensor([0.6, 0.9, 0.8]),
                            cls=torch.tensor([0.0, 0.0, 1.0]),
                        )
                    ),
                    SimpleNamespace(
                        boxes=SimpleNamespace(
                            xyxy=torch.tensor(
                                [
                                    [7.0, 8.0, 70.0, 80.0],
                                    [9.0, 10.0, 90.0, 100.0],
                                ]
                            ),
                            conf=torch.tensor([0.75, 0.85]),
                            cls=torch.tensor([1.0, 1.0]),
                        )
                    ),
                ]

        with patch.dict(
            "sys.modules",
            {"ultralytics": SimpleNamespace(YOLOE=FakeYOLOE)},
        ):
            model = self.build_model(
                prompts=("person", "backpack"),
                roi_generator="yoloe",
                yoloe_model_id="test/yoloe",
                yoloe_score_threshold=0.3,
                yoloe_prompt_mode="one-pass",
            )
            generator = model.roi_proposal_generator
            assert generator is not None
            boxes = generator(torch.rand(2, 3, 48, 24), model.prompts)

        self.assertEqual(generator.model.prompt_calls, [["person", "backpack"]])
        self.assertEqual(len(generator.model.prediction_calls), 1)
        torch.testing.assert_close(
            boxes,
            torch.tensor(
                [
                    [
                        [1.0, 2.0, 10.0, 20.0],
                        [3.0, 4.0, 30.0, 40.0],
                        [5.0, 6.0, 50.0, 60.0],
                    ],
                    [
                        [7.0, 8.0, 70.0, 80.0],
                        [9.0, 10.0, 90.0, 100.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ],
                ]
            ),
        )

    def test_dinov3_can_be_used_as_full_image_backbone(self) -> None:
        class FakeDINOv3(nn.Module):
            config = SimpleNamespace(
                hidden_size=12,
                patch_size=8,
                num_register_tokens=0,
            )

            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(()))

            def forward(self, **inputs: torch.Tensor) -> object:
                batch_size = inputs["pixel_values"].shape[0]
                return SimpleNamespace(
                    pooler_output=torch.ones(batch_size, 12) * self.weight,
                    last_hidden_state=torch.ones(batch_size, 17, 12) * self.weight,
                )

        class FakeImageProcessor:
            def __call__(
                self,
                images: list[object],
                return_tensors: str,
                do_rescale: bool,
            ) -> FakeBatch:
                assert return_tensors == "pt"
                assert not do_rescale
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
            images = torch.randn(2, 3, 48, 24)
            full_image_boxes = torch.tensor(
                [[[0.0, 0.0, 24.0, 48.0]], [[0.0, 0.0, 24.0, 48.0]]]
            )
            logits = model(images, images, full_image_boxes)

        self.assertEqual(tuple(logits.shape), (2, 7))

    def test_no_valid_rois_uses_full_image_branch(self) -> None:
        model = self.build_model()
        images = torch.randn(2, 3, 64, 32)
        padded_boxes = torch.zeros(2, 2, 4)

        logits = model(images, images, padded_boxes)

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
