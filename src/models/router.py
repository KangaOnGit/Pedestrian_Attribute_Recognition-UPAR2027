from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from jaxtyping import Bool, Float

from src.models.roi_generator import (
    Sam3PromptBoxGenerator,
    YOLOEPromptBoxGenerator,
)
from src.models.feature_encoder import (
    DINOv3FeatureEncoder,
    ConvFeatureEncoder,
)
from src.models.experts import ROIExpert
from src.losses.aux_loss import aux_moe_loss


class ROIRouter(nn.Module):
    """
    Produces routing logits for each ROI

    The router uses both:
      1. spatial ROI information
      2. global pooled ROI information
      3. ROI geometry

    Input:
        ROI feature maps [N, D, H, W]

    Output:
        routing logits [N, num_experts]
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        geometry_dim: int = 6,
    ) -> None:
        super().__init__()

        router_dim = max(hidden_dim // 2, 64)

        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                router_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                min(8, router_dim),
                router_dim,
            ),
            nn.GELU(),
            nn.Conv2d(
                router_dim,
                router_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                min(8, router_dim),
                router_dim,
            ),
            nn.GELU(),
        )

        self.pool_projection = nn.Sequential(
            nn.Linear(
                router_dim * 2,
                router_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(router_dim),
        )

        self.geometry_projection = nn.Sequential(
            nn.Linear(
                geometry_dim,
                router_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(router_dim),
        )
        self.router = nn.Sequential(
            nn.Linear(
                router_dim * 2,
                router_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(router_dim),
            nn.Linear(
                router_dim,
                num_experts,
            ),
        )

    def forward(
        self,
        roi_features: Tensor,
        roi_geometry: Tensor,
    ) -> Tensor:
        """
        Args:
            roi_features:
                [N, D, H, W]

            roi_geometry:
                [N, 6]

        Returns:
            [N, num_experts]
        """

        x = self.spatial_encoder(roi_features)

        avg_pool = F.adaptive_avg_pool2d(
            x,
            1,
        ).flatten(1)

        max_pool = F.adaptive_max_pool2d(
            x,
            1,
        ).flatten(1)

        appearance = self.pool_projection(
            torch.cat(
                (
                    avg_pool,
                    max_pool,
                ),
                dim=1,
            )
        )

        geometry = self.geometry_projection(
            roi_geometry
        )

        router_features = torch.cat(
            (
                appearance,
                geometry,
            ),
            dim=1,
        )

        return self.router(router_features)