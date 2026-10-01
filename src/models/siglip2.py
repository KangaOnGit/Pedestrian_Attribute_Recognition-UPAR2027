import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


DEFAULT_SIGLIP2_CHECKPOINT = "google/siglip2-so400m-patch16-256"


class Backbone(nn.Module):
    """SigLIP2 vision encoder adapted to this project's image tensors."""

    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP2_CHECKPOINT,
        pretrained: bool = True,
        freeze: bool = False,
        max_num_patches: int = 256,
    ) -> None:
        super().__init__()

        if max_num_patches < 1:
            raise ValueError("max_num_patches must be at least 1.")

        try:
            from transformers import Siglip2VisionModel
        except ImportError as error:
            raise ImportError(
                "SigLIP2 requires transformers>=4.50.0; install the updated requirements."
            ) from error

        if pretrained:
            self.encoder = Siglip2VisionModel.from_pretrained(model_name)
        else:
            config = Siglip2VisionModel.config_class.from_pretrained(model_name)
            self.encoder = Siglip2VisionModel(config)

        self.patch_size: int = int(self.encoder.config.patch_size)
        self.output_dim: int = int(self.encoder.config.hidden_size)
        self.max_num_patches = max_num_patches
        self.encoder.requires_grad_(not freeze)

        self.register_buffer(
            "imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def _prepare_patches(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [batch, 3, height, width].")

        batch_size, _, height, width = images.shape
        patch_size = self.patch_size
        scale = min(
            1.0,
            math.sqrt(
                self.max_num_patches
                / (math.ceil(height / patch_size) * math.ceil(width / patch_size))
            ),
        )
        target_height = max(patch_size, int(height * scale) // patch_size * patch_size)
        target_width = max(patch_size, int(width * scale) // patch_size * patch_size)

        while (target_height // patch_size) * (target_width // patch_size) > self.max_num_patches:
            if target_height >= target_width:
                target_height -= patch_size
            else:
                target_width -= patch_size

        pixels = (images * self.imagenet_std + self.imagenet_mean).clamp(0, 1)
        pixels = pixels.mul(2).sub(1)
        if (target_height, target_width) != (height, width):
            pixels = F.interpolate(
                pixels,
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            )

        grid_height = target_height // patch_size
        grid_width = target_width // patch_size
        patches = (
            pixels.unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
            .permute(0, 2, 3, 4, 5, 1)
            .reshape(batch_size, grid_height * grid_width, -1)
        )
        patch_count = patches.shape[1]
        patch_mask = torch.ones(
            (batch_size, patch_count),
            dtype=torch.long,
            device=images.device,
        )
        spatial_shapes = torch.tensor(
            [grid_height, grid_width],
            dtype=torch.long,
            device=images.device,
        ).expand(batch_size, -1)
        return patches, patch_mask, spatial_shapes

    def forward(self, images: Tensor) -> Tensor:
        patches, patch_mask, spatial_shapes = self._prepare_patches(images)
        outputs = self.encoder(
            pixel_values=patches,
            pixel_attention_mask=patch_mask,
            spatial_shapes=spatial_shapes,
        )
        if outputs.pooler_output is None:
            raise RuntimeError("SigLIP2 did not return pooled vision features.")
        return outputs.pooler_output


class AttributeHead(nn.Module):
    """Linear head producing one logit for each independent attribute."""

    def __init__(
        self,
        input_dim: int,
        num_attributes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_attributes < 1:
            raise ValueError("num_attributes must be at least 1.")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_attributes)

    def forward(self, features: Tensor) -> Tensor:
        return self.classifier(self.dropout(features))


class SigLIP2AttributeModel(nn.Module):
    """SigLIP2 backbone and multi-label pedestrian attribute head."""

    def __init__(
        self,
        num_attributes: int = 40,
        model_name: str = DEFAULT_SIGLIP2_CHECKPOINT,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.0,
        max_num_patches: int = 256,
    ) -> None:
        super().__init__()
        self.backbone = Backbone(
            model_name=model_name,
            pretrained=pretrained,
            freeze=freeze_backbone,
            max_num_patches=max_num_patches,
        )
        self.head = AttributeHead(
            input_dim=self.backbone.output_dim,
            num_attributes=num_attributes,
            dropout=dropout,
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.head(self.backbone(images))