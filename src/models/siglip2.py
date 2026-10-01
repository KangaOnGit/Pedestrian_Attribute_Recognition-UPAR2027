import math

import torch
from torch import Tensor, nn
from transformers import Siglip2VisionModel


class SigLIP2Baseline(nn.Module):
    def __init__(
        self,
        num_attributes: int = 40,
        model_name: str = "google/siglip2-so400m-patch16-256",
        freeze_backbone: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.backbone = Siglip2VisionModel.from_pretrained(
            model_name,
            ignore_mismatched_sizes=True,
        )

        self.backbone.requires_grad_(not freeze_backbone)

        hidden_dim = self.backbone.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_attributes),
        )

    def _to_siglip_inputs(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if images.dim() == 3:
            images = images.unsqueeze(0)

        if images.size(1) not in (1, 3):
            raise ValueError(
                f"Expected 1 or 3 channels, got {images.shape[1]}"
            )

        patch_size = self.backbone.config.patch_size
        max_num_patches = getattr(
            self.backbone.config,
            "max_num_patches",
            256,
        )

        def get_scaled_image_size(scale: float, size: int) -> int:
            scaled = size * scale
            scaled = math.ceil(scaled / patch_size) * patch_size
            return max(patch_size, int(scaled))

        eps = 1e-5
        scale_min, scale_max = eps / 10, 100.0
        image_height, image_width = images.shape[-2], images.shape[-1]

        while (scale_max - scale_min) >= eps:
            scale = (scale_min + scale_max) / 2
            target_height = get_scaled_image_size(scale, image_height)
            target_width = get_scaled_image_size(scale, image_width)
            num_patches = (target_height // patch_size) * (target_width // patch_size)
            if num_patches <= max_num_patches:
                scale_min = scale
            else:
                scale_max = scale

        target_height = get_scaled_image_size(scale_min, image_height)
        target_width = get_scaled_image_size(scale_min, image_width)

        if target_height != image_height or target_width != image_width:
            images = torch.nn.functional.interpolate(
                images,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )

        h = images.shape[-2] // patch_size
        w = images.shape[-1] // patch_size
        num_patches = h * w

        patches = (
            images.unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
            .permute(0, 2, 3, 4, 5, 1)
            .reshape(images.shape[0], num_patches, -1)
        )

        if num_patches < max_num_patches:
            pad_len = max_num_patches - num_patches
            patches = torch.nn.functional.pad(
                patches,
                (0, 0, 0, pad_len),
                value=0.0,
            )
            attention_mask = torch.ones(
                (images.shape[0], max_num_patches),
                device=images.device,
                dtype=torch.int32,
            )
            attention_mask[:, -pad_len:] = 0
        else:
            patches = patches[:, :max_num_patches]
            attention_mask = torch.ones(
                (images.shape[0], max_num_patches),
                device=images.device,
                dtype=torch.int32,
            )

        spatial_shapes = torch.tensor(
            [[h, w] for _ in range(images.shape[0])],
            device=images.device,
            dtype=torch.long,
        )

        return patches, attention_mask, spatial_shapes

    def forward(self, images: Tensor) -> Tensor:
        pixel_values, pixel_attention_mask, spatial_shapes = self._to_siglip_inputs(images)

        outputs = self.backbone(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )

        features = outputs.pooler_output
        logits = self.classifier(features)

        return logits