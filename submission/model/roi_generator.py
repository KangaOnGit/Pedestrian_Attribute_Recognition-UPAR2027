from collections.abc import Sequence
from typing import Literal, Protocol

import torch
from jaxtyping import Float, Int
from torch import Tensor

from pathlib import Path

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

class YOLOEPromptBoxGenerator:
    """Detect prompts on RGB pixels in [0, 1] and keep each prompt's top box."""

    def __init__(
        self,
        score_threshold: float,
        model_id: str | Path = "yoloe-26s-seg.pt",
        prompt_mode: Literal["loop", "one-pass"] = "one-pass",
    ) -> None:
        if prompt_mode not in ("loop", "one-pass"):
            raise ValueError("prompt_mode must be 'loop' or 'one-pass'")

        from ultralytics import YOLOE
        
        model_path = Path(model_id).expanduser()

        if not model_path.is_absolute():
            model_path = model_path.resolve()

        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLOE weights not found: {model_path}"
            )
            
        self.model: _YOLOEModel = YOLOE(str(model_path))
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