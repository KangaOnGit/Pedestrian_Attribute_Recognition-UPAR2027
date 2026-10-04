from collections.abc import Sequence
from typing import Protocol

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import Sam3Model, Sam3Processor


class _YOLOBoxes(Protocol):
    xyxy: Tensor
    conf: Tensor


class _YOLOResult(Protocol):
    boxes: _YOLOBoxes | None


class _YOLOEModel(Protocol):
    def set_classes(self, classes: list[str]) -> object: ...

    def predict(
        self,
        *,
        source: Tensor,
        conf: float,
        device: str,
        verbose: bool,
    ) -> Sequence[_YOLOResult]: ...


class Sam3PromptBoxGenerator:
    """Keep the highest-confidence SAM 3 box for each prompt and image."""

    def __init__(
        self,
        model_id: str,
        score_threshold: float,
    ) -> None:

        self.model = Sam3Model.from_pretrained(model_id)
        self.processor = Sam3Processor.from_pretrained(model_id)

        self.score_threshold = score_threshold
        
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.inference_mode()
    def __call__(
        self,
        images_no_aug: Float[Tensor, "B 3 H W"],
        prompts: Sequence[str],
    ) -> Float[Tensor, "B P 4"]:
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("SAM 3 requires a non-empty list of non-blank prompts")

        self.model.to(images_no_aug.device)
        self.model.eval()
        prompt_boxes = images_no_aug.new_zeros(
            (images_no_aug.shape[0], len(prompts), 4)
        )

        for image_index, image in enumerate(images_no_aug):
            for prompt_index, prompt in enumerate(prompts):
                inputs = self.processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt",
                ).to(images_no_aug.device)
                outputs = self.model(**inputs)

                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.score_threshold,
                    mask_threshold=0.5,
                    target_sizes=[inputs["original_sizes"].tolist()[0]],
                )[0]

                boxes = results["boxes"].to(
                    device=images_no_aug.device,
                    dtype=images_no_aug.dtype,
                )
                scores = results["scores"].to(device=images_no_aug.device)
                if boxes.ndim != 2 or boxes.shape[-1] != 4:
                    raise ValueError("SAM 3 post-processing must return boxes shaped [R, 4]")
                if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
                    raise ValueError("SAM 3 must return one confidence score per box")
                if boxes.shape[0] > 0:
                    highest_confidence = scores.argmax()
                    prompt_boxes[image_index, prompt_index] = boxes[highest_confidence]

        return prompt_boxes

class YOLOEPromptBoxGenerator:
    """Detect each prompt separately and keep its highest-confidence box per image."""

    def __init__(
        self,
        model_id: str,
        score_threshold: float,
    ) -> None:
        from ultralytics import YOLOE

        self.model: _YOLOEModel = YOLOE(model_id)
        self.score_threshold = score_threshold

    @torch.inference_mode()
    def __call__(
        self,
        images_no_aug: Float[Tensor, "B 3 H W"],
        prompts: Sequence[str],
    ) -> Float[Tensor, "B P 4"]:
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("YOLO-E requires a non-empty list of non-blank prompts")
        if images_no_aug.ndim != 4 or images_no_aug.shape[1] != 3:
            raise ValueError("YOLO-E requires RGB images shaped [B, 3, H, W]")
        if not torch.is_floating_point(images_no_aug):
            raise ValueError("YOLO-E expects floating-point, ImageNet-normalized images")

        # P is prompt_length
        # Each image in batch will have P bounding boxes with its 4 coordinates
        prompt_boxes: Float[torch.Tensor, "B P 4"] = images_no_aug.new_zeros(
            (images_no_aug.shape[0], len(prompts), 4)
        )
        for prompt_index, prompt in enumerate(prompts):
            self.model.set_classes([prompt])
            results = self.model.predict(
                source=images_no_aug,
                conf=self.score_threshold,
                device=str(images_no_aug.device),
                verbose=False,
            )
            if len(results) != images_no_aug.shape[0]:
                raise ValueError("YOLO-E must return one result per input image")

            for image_index, result in enumerate(results):
                if result.boxes is None:
                    continue
                
                # R is number of bbox generated
                boxes: Float[torch.Tensor, "R 4"] = torch.as_tensor(
                    result.boxes.xyxy,
                    device=images_no_aug.device,
                    dtype=images_no_aug.dtype,
                )
                confidences: Float[torch.Tensor, "R"] = torch.as_tensor(
                    result.boxes.conf,
                    device=images_no_aug.device,
                )
                if boxes.ndim != 2 or boxes.shape[-1] != 4:
                    raise ValueError("YOLO-E must return boxes shaped [R, 4]")
                if confidences.ndim != 1 or confidences.shape[0] != boxes.shape[0]:
                    raise ValueError("YOLO-E must return one confidence score per box")
                
                # if atleast 1 boxexist 
                if boxes.shape[0] > 0:
                    highest_confidence: Int[torch.Tensor, "1"]  = confidences.argmax()
                    prompt_boxes[image_index, prompt_index] = boxes[highest_confidence]

        return prompt_boxes