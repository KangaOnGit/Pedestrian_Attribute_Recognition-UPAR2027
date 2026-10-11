from __future__ import annotations

import torch
from torch.nn import functional as F

from torch import Tensor, nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel, BatchFeature

class DINOv3FeatureEncoder(nn.Module):
    """Use a pretrained DINOv3 model to encode full images."""

    def __init__(
        self,
        hidden_dim: int,
        model_id: str,
        trainable: bool,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(
            model_id,
            local_files_only=True,
        )
        
        if load_pretrained:
            self.model = AutoModel.from_pretrained(
                model_id,
                local_files_only=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                model_id,
                local_files_only=True,
            )
            self.model = AutoModel.from_config(config)

        if not trainable:
            self.model.eval()
        self.trainable: bool = trainable
        for parameter in self.model.parameters():
            parameter.requires_grad_(trainable)

        # [B, Hd*4] -> [B, Hd]
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # [B, ..., DinoHd] -> [B, Hd]
        self.token_projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def train(self, mode: bool = True) -> DINOv3FeatureEncoder:
        super().train(mode)
        if not self.trainable:
            self.model.eval()
        return self

    def forward(self,
                images_aug: torch.Tensor
                ) -> torch.Tensor:
        regions: torch.Tensor = self.forward_regions(images_aug)
        
        flat_region: torch.Tensor = regions.flatten(start_dim=1)
        
        return self.projection(flat_region)

    def forward_with_patch_grid(
        self,
        images_no_aug: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return the full-image feature and projected spatial patch features."""
        pooled_features, patch_grid = self._encode_image(images_no_aug)
        if patch_grid is None:
            raise RuntimeError(
                "DINOv3 must expose patch tokens matching its configured patch grid"
            )

        region_features: torch.Tensor = F.adaptive_avg_pool2d(
            patch_grid,
            output_size=(3, 1),
        ).squeeze(-1).transpose(1, 2)
        regions: torch.Tensor = self.token_projection(
            torch.cat((pooled_features[:, None], region_features), dim=1)
        )
        image_features: torch.Tensor = self.projection(
            regions.flatten(start_dim=1)
        )
        patch_features: torch.Tensor = self.token_projection(
            patch_grid.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        return image_features, patch_features

    def forward_regions(self,
                        images_aug: torch.Tensor
                        ) -> torch.Tensor:
        """Return global and top/middle/bottom visual features."""
        
        pooled_features, patch_grid = self._encode_image(images_aug)
        
        if patch_grid is None:
            region_features: torch.Tensor = pooled_features.new_zeros(
                (pooled_features.shape[0], 3, pooled_features.shape[1])
            )
        else:
            region_features: torch.Tensor = F.adaptive_avg_pool2d(
                patch_grid,
                output_size=(3, 1), # torch.Tensor
            ).squeeze(-1).transpose(1, 2)
            
        return self.token_projection(
            # (Top, Mid, Bot) = 3 + Global = 4
            # torch.Tensor
            torch.cat((pooled_features[:, None], region_features), dim=1)
        )

    def forward_patch_tokens(self,
                             images_aug: torch.Tensor
                             ) -> torch.Tensor:
        """
        Return the global token and each spatially encoded patch token.
            VT: 1 + number_of_patch_tokens
        """
        
        pooled_features, patch_grid = self._encode_image(images_aug)
        if patch_grid is None:
            region_features: torch.Tensor = pooled_features.new_zeros(
                (pooled_features.shape[0], 3, pooled_features.shape[1])
            )
            return self.token_projection(
                # torch.Tensor
                torch.cat((pooled_features[:, None], region_features), dim=1)
            )
            
        patch_tokens: torch.Tensor = patch_grid.flatten(start_dim=2).transpose(1, 2)
        return self.token_projection(
            # Gh*Gw + 1 = VT
            # [B VT DinoHd]
            torch.cat((pooled_features[:, None], patch_tokens), dim=1)
        )

    def _encode_image(self,
                      images_no_aug: torch.Tensor
                      ) -> tuple[
                          torch.Tensor,
                          torch.Tensor | None,
                      ]:
        """
        Args:
            images_no_aug (Tensor): Image that has only went through
                Normalization, Resize and ToTensor
                    mean: [0.485, 0.456, 0.406]
                    std: [0.229, 0.224, 0.225]
        """
        processor_inputs = self.processor(
            images=images_no_aug,
            return_tensors="pt",
            do_rescale=False,
        )
        model_device = next(self.model.parameters()).device
        
        # Processor will resize the Image to its own Height and Width
        processor_inputs: BatchFeature = processor_inputs.to(model_device)
        if self.trainable:
            outputs = self.model(**processor_inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**processor_inputs)
                
        last_hidden_state: Tensor | None = getattr(
            outputs,
            "last_hidden_state",
            None,
        )
        pooled_features: Tensor | None = getattr(
            outputs,
            "pooler_output",
            None,
        )
        if pooled_features is None:
            if not isinstance(last_hidden_state, Tensor) or last_hidden_state.ndim != 3:
                raise RuntimeError(
                    "DINOv3 must return pooler_output or rank-3 last_hidden_state"
                )
            pooled_features = last_hidden_state[:, 0]

        patch_size = getattr(self.model.config, "patch_size", None)
        register_tokens = getattr(self.model.config, "num_register_tokens", 0)
        pixel_values: torch.Tensor = processor_inputs["pixel_values"]
        patch_grid: torch.Tensor | None = None
        if (
            isinstance(last_hidden_state, Tensor)
            and isinstance(patch_size, int)
            and patch_size > 0
            and pixel_values.shape[-2] % patch_size == 0
            and pixel_values.shape[-1] % patch_size == 0
        ):
            grid_height: int = pixel_values.shape[-2] // patch_size
            grid_width: int = pixel_values.shape[-1] // patch_size
            patch_tokens: torch.Tensor = last_hidden_state[
                :, 1 + register_tokens :
            ]
            if patch_tokens.shape[1] == grid_height * grid_width:
                patch_grid: torch.Tensor = patch_tokens.reshape(
                    patch_tokens.shape[0],
                    grid_height,
                    grid_width,
                    patch_tokens.shape[-1],
                ).permute(0, 3, 1, 2)
        return pooled_features, patch_grid
from torch import Tensor, nn
from torch.nn import functional as F

class ConvFeatureEncoder(nn.Module):
    """Small convolutional encoder."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 32),
            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),

            nn.Conv2d(
                64,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

    def forward_spatial_features(
        self,
        images: Tensor,
    ) -> Tensor:
        return self.features(images)

    def forward(
        self,
        images: Tensor,
    ) -> Tensor:
        features = self.forward_spatial_features(images)
        return F.adaptive_avg_pool2d(
            features,
            1,
        ).flatten(1)

    def forward_regions(
        self,
        images: Tensor,
    ) -> Tensor:

        feature_map = self.forward_spatial_features(images)

        global_feature = F.adaptive_avg_pool2d(
            feature_map,
            1,
        ).flatten(1)

        region_features = F.adaptive_avg_pool2d(
            feature_map,
            (3, 1),
        ).squeeze(-1).transpose(1, 2)

        return torch.cat(
            (
                global_feature[:, None],
                region_features,
            ),
            dim=1,
        )
class ImageAttributeModel(nn.Module):
    """Predict attributes from a pooled full-image feature representation."""

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        *,
        backbone_type: str = "dinov3",
        dinov3_model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
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
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(self,
                images: torch.Tensor
                ) -> torch.Tensor:
        """
        K is the number of classes
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        features: torch.Tensor = self.backbone(images)
        if features.ndim != 2 or features.shape != (images.shape[0], self.hidden_dim):
            raise ValueError(
                "image backbone must return [B, hidden_dim] features"
            )
        return self.classifier(features)