from __future__ import annotations

import torch
from torch.nn import functional as F

from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoModel

class DINOv3FeatureEncoder(nn.Module):
    """Use a pretrained DINOv3 model to encode full images."""

    def __init__(
        self,
        hidden_dim: int,
        model_id: str,
        trainable: bool,
    ) -> None:
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)

        self.trainable = trainable
        
        for parameter in self.model.parameters():
            parameter.requires_grad_(trainable)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.token_projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        if not trainable:
            self.model.eval()

    def train(self, mode: bool = True) -> DINOv3FeatureEncoder:
        super().train(mode)
        if not self.trainable:
            self.model.eval()
        return self

    def forward(self, images: Tensor) -> Tensor:
        regions = self.forward_regions(images)
        return self.projection(regions.flatten(start_dim=1))

    def forward_regions(self, images: Tensor) -> Tensor:
        """Return global and top/middle/bottom visual features."""
        pooled_features, patch_grid = self._encode_image(images)
        if patch_grid is None:
            region_features = pooled_features.new_zeros(
                (pooled_features.shape[0], 3, pooled_features.shape[1])
            )
        else:
            region_features = F.adaptive_avg_pool2d(
                patch_grid,
                output_size=(3, 1),
            ).squeeze(-1).transpose(1, 2)
        return self.token_projection(
            torch.cat((pooled_features[:, None], region_features), dim=1)
        )

    def forward_patch_tokens(self, images: Tensor) -> Tensor:
        """Return the global token and each spatially encoded patch token."""
        pooled_features, patch_grid = self._encode_image(images)
        if patch_grid is None:
            region_features = pooled_features.new_zeros(
                (pooled_features.shape[0], 3, pooled_features.shape[1])
            )
            return self.token_projection(
                torch.cat((pooled_features[:, None], region_features), dim=1)
            )
        patch_tokens = patch_grid.flatten(start_dim=2).transpose(1, 2)
        return self.token_projection(
            torch.cat((pooled_features[:, None], patch_tokens), dim=1)
        )

    def _encode_image(self, images: Tensor) -> tuple[Tensor, Tensor | None]:
        input_mean = images.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        input_std = images.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        rgb_images = (images * input_std + input_mean).clamp(0, 1)
        processor_inputs = self.processor(
            images=rgb_images,
            return_tensors="pt",
            do_rescale=False,
        )
        model_device = next(self.model.parameters()).device
        processor_inputs = processor_inputs.to(model_device)
        if self.trainable:
            outputs = self.model(**processor_inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**processor_inputs)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        pooled_features = getattr(outputs, "pooler_output", None)
        if pooled_features is None:
            if not isinstance(last_hidden_state, Tensor) or last_hidden_state.ndim != 3:
                raise RuntimeError(
                    "DINOv3 must return pooler_output or rank-3 last_hidden_state"
                )
            pooled_features = last_hidden_state[:, 0]

        patch_size = getattr(self.model.config, "patch_size", None)
        register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        pixel_values = processor_inputs["pixel_values"]
        patch_grid = None
        if (
            isinstance(last_hidden_state, Tensor)
            and isinstance(patch_size, int)
            and patch_size > 0
            and pixel_values.shape[-2] % patch_size == 0
            and pixel_values.shape[-1] % patch_size == 0
        ):
            grid_height = pixel_values.shape[-2] // patch_size
            grid_width = pixel_values.shape[-1] // patch_size
            patch_tokens = last_hidden_state[:, 1 + register_tokens :]
            if patch_tokens.shape[1] == grid_height * grid_width:
                patch_grid = patch_tokens.reshape(
                    patch_tokens.shape[0],
                    grid_height,
                    grid_width,
                    patch_tokens.shape[-1],
                ).permute(0, 3, 1, 2)
        return pooled_features, patch_grid


class ConvFeatureEncoder(nn.Module):
    """Small convolutional encoder that returns one vector per input image."""

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        feature_map = self._feature_map(x)
        return F.adaptive_avg_pool2d(feature_map, 1).flatten(start_dim=1)

    def forward_regions(self, x: Tensor) -> Tensor:
        feature_map = self._feature_map(x)
        global_feature = F.adaptive_avg_pool2d(feature_map, 1).flatten(start_dim=1)
        region_features = F.adaptive_avg_pool2d(feature_map, (3, 1)).squeeze(-1).transpose(1, 2)
        return torch.cat((global_feature[:, None], region_features), dim=1)

    def _feature_map(self, x: Tensor) -> Tensor:
        return self.features[:-2](x)


class ImageAttributeModel(nn.Module):
    """Predict attributes from a pretrained full-image feature representation."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        *,
        backbone_type: str = "dinov3",
        dinov3_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        backbone_trainable: bool = False,
    ) -> None:
        super().__init__()
        if num_classes < 1 or hidden_dim < 1:
            raise ValueError("num_classes and hidden_dim must be positive")
        if backbone_type not in ("conv", "dinov3"):
            raise ValueError("backbone_type must be 'conv' or 'dinov3'")

        self.backbone = (
            DINOv3FeatureEncoder(
                hidden_dim=hidden_dim,
                model_id=dinov3_model_id,
                trainable=backbone_trainable,
            )
            if backbone_type == "dinov3"
            else ConvFeatureEncoder(in_channels=3, hidden_dim=hidden_dim)
        )
        if hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by 4 for label attention")
        self.attribute_queries = nn.Parameter(torch.empty(1, num_classes, hidden_dim))
        nn.init.normal_(self.attribute_queries, std=hidden_dim**-0.5)
        self.visual_attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        self.label_interaction = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        features = (
            self.backbone.forward_patch_tokens(images)
            if isinstance(self.backbone, DINOv3FeatureEncoder)
            else self.backbone.forward_regions(images)
        )
        if (
            features.ndim != 3
            or features.shape[0] != images.shape[0]
            or features.shape[2] != self.hidden_dim
        ):
            raise ValueError(
                "image backbone must return [B, visual_tokens, hidden_dim] features"
            )
        queries = self.attribute_queries.expand(images.shape[0], -1, -1)
        attribute_features, _ = self.visual_attention(
            queries,
            features,
            features,
            need_weights=False,
        )
        attribute_features = self.label_interaction(attribute_features)
        return self.classifier(attribute_features).squeeze(-1)