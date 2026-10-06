from collections.abc import Sequence
from typing import Literal, Protocol

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
        
        prompt_names = tuple(prompts)
        if prompt_names != self._active_prompts:
            self.model.set_classes(list(prompt_names))
            self._active_prompts = prompt_names

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
    """Detect prompts on RGB pixels in [0, 1] and keep each prompt's top box."""

    def __init__(
        self,
        model_id: str,
        score_threshold: float,
        prompt_mode: Literal["loop", "one-pass"] = "one-pass",
    ) -> None:
        if prompt_mode not in ("loop", "one-pass"):
            raise ValueError("prompt_mode must be 'loop' or 'one-pass'")

        from ultralytics import YOLOE

        self.model: _YOLOEModel = YOLOE(model_id)
        self.score_threshold = score_threshold
        self.prompt_mode = prompt_mode

    @torch.inference_mode()
    def __call__(
        self,
        images_rgb: Float[Tensor, "B 3 H W"],
        prompts: Sequence[str],
    ) -> Float[Tensor, "B R 4"]:

        if self.prompt_mode == "one-pass":
            
            self.model.set_classes(list(prompts))
            results = self.model.predict(
                source=images_rgb,
                conf=self.score_threshold,
                device=str(images_rgb.device),
                verbose=False,
            )
            batch_boxes: list[Float[torch.Tensor, "R 4"]] = []
            for image_index, result in enumerate(results):
                if result.boxes is None:
                    boxes = images_rgb.new_empty((0, 4))
                else:
                    boxes = torch.as_tensor(
                        result.boxes.xyxy,
                        device=images_rgb.device,
                        dtype=images_rgb.dtype,
                    )
                batch_boxes.append(boxes)

            max_rois = max((boxes.shape[0] for boxes in batch_boxes), default=0)
            padded: Float[torch.Tensor, "B Rmax 4"] = images_rgb.new_zeros(
                (images_rgb.shape[0], max_rois, 4)
            )
            for batch_index, boxes in enumerate(batch_boxes):
                padded[batch_index, : boxes.shape[0]] = boxes
            return padded

        prompt_boxes: Float[torch.Tensor, "B R 4"] = images_rgb.new_zeros(
            (images_rgb.shape[0], len(prompts), 4)
        )
        for prompt_index, prompt in enumerate(prompts):
            self.model.set_classes([prompt])
            results = self.model.predict(
                source=images_rgb,
                conf=self.score_threshold,
                device=str(images_rgb.device),
                verbose=False,
            )
            for image_index, result in enumerate(results):
                if result.boxes is None:
                    continue

                boxes: Float[torch.Tensor, "R 4"] = torch.as_tensor(
                    result.boxes.xyxy,
                    device=images_rgb.device,
                    dtype=images_rgb.dtype,
                )
                confidences: Float[torch.Tensor, "R"] = torch.as_tensor(
                    result.boxes.conf,
                    device=images_rgb.device,
                )
        return prompt_boxes