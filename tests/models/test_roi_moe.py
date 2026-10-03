import unittest

import torch

from src.models import SparseROIAttributeModel


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

    def test_detector_callback_and_sparse_roi_dispatch(self) -> None:
        received_prompts: list[tuple[str, ...]] = []

        def propose(images: torch.Tensor, prompts: tuple[str, ...]) -> torch.Tensor:
            received_prompts.append(prompts)
            return images.new_tensor([[[1, 1, 20, 40]]]).expand(images.shape[0], -1, -1)

        model = self.build_model(prompts=("person", "backpack"), proposal_generator=propose)
        router_output = model.roi_router[-1]
        with torch.no_grad():
            router_output.weight.zero_()
            router_output.bias.copy_(torch.tensor([0.0, 10.0, 0.0]))
        calls = [0, 0, 0]
        handles = [
            expert.register_forward_hook(
                lambda *_, expert_index=index: calls.__setitem__(
                    expert_index,
                    calls[expert_index] + 1,
                )
            )
            for index, expert in enumerate(model.roi_experts)
        ]
        try:
            logits = model(torch.randn(2, 3, 48, 24))
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertEqual(received_prompts, [("person", "backpack")])
        self.assertEqual(calls, [0, 1, 0])

    def test_no_valid_rois_uses_full_image_branch(self) -> None:
        model = self.build_model()
        images = torch.randn(2, 3, 64, 32)
        padded_boxes = torch.zeros(2, 2, 4)

        logits = model(images, padded_boxes)

        self.assertEqual(tuple(logits.shape), (2, 7))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
