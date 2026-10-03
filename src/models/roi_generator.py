import torch

from torch import Tensor
from transformers import Sam3Model, Sam3Processor
from collections.abc import Sequence
from src.utils.config import load_config

class Sam3PromptBoxGenerator:
    """Load SAM 3 and turn a prompt list into pixel-coordinate boxes."""

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
    def __call__(self, images_no_aug: Tensor, prompts: Sequence[str]) -> Tensor:
        if not prompts or any(not prompt.strip() for prompt in prompts):
            raise ValueError("SAM 3 requires a non-empty list of non-blank prompts")

        self.model.to(images_no_aug.device)
        self.model.eval()
        batch_boxes: list[Tensor] = []
        
        for image in images_no_aug:
            image_boxes: list[Tensor] = []
            
            for prompt in prompts:
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
                
                boxes = results["boxes"].to(device=images_no_aug.device, dtype=images_no_aug.dtype)
                if boxes.ndim != 2 or boxes.shape[-1] != 4:
                    raise ValueError("SAM 3 post-processing must return boxes shaped [R, 4]")
                image_boxes.append(boxes)

            valid_boxes = [boxes for boxes in image_boxes if boxes.shape[0] > 0]
            if valid_boxes:
                batch_boxes.append(torch.cat(valid_boxes, dim=0))
            else:
                batch_boxes.append(images_no_aug.new_empty((0, 4)))

        max_rois = max((boxes.shape[0] for boxes in batch_boxes), default=0)
        padded = images_no_aug.new_zeros((images_no_aug.shape[0], max_rois, 4))
        for batch_index, boxes in enumerate(batch_boxes):
            padded[batch_index, : boxes.shape[0]] = boxes
        return padded