"""Sparse mixture-of-experts model for full-image and prompted-ROI features."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from src.models.roi_generator import Sam3PromptBoxGenerator, YOLOEPromptBoxGenerator
from src.models.feature_encoder import DINOv3FeatureEncoder, ConvFeatureEncoder

class ROIExpert(nn.Module):
    """Processes the ROIs routed to one expert."""

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = ConvFeatureEncoder(in_channels, hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, rois: Tensor) -> Tensor:
        return self.projection(self.encoder(rois))
class SparseROIAttributeModel(nn.Module):
    """Combine sparse prompted-ROI experts and a full-image path.

    ``roi_generator="yoloe"`` loads YOLO-E during construction and uses
    ``prompts`` to generate pixel-coordinate XYXY boxes. Boxes may also be
    passed directly to ``forward``.
    Custom encoders must accept image batches and return ``[N, hidden_dim]``.
    """

    def __init__(
        self,
        
        num_classes: int,
        hidden_dim: int,
        num_experts: int,
        attention_k: int,
        
        *,
        in_channels: int = 3,
        roi_top_k: int = 1,
        
        segmentation_top_k: int = 3,
        num_attention_heads: int = 4,
        
        roi_size: tuple[int, int] = (96, 48),
        prompts: Sequence[str] = (),
        roi_generator: Literal["none", "sam3", "yoloe"] = "yoloe",
        sam3_model_id: str = "facebook/sam3",
        sam3_score_threshold: float = 0.5,
        yoloe_model_id: str = "yoloe-11s-seg.pt",
        yoloe_score_threshold: float = 0.5,
        
        image_backbone_type: Literal["conv", "dinov3"] = "conv",
        dinov3_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        dinov3_trainable: bool = False,
        image_backbone: nn.Module | None = None,
        roi_backbones: Sequence[nn.Module] | None = None,
        
    ) -> None:
        super().__init__()
        if min(num_classes, hidden_dim, num_experts, attention_k) < 1:
            raise ValueError("num_classes, hidden_dim, num_experts, and attention_k must be positive")
        num_segmentation_experts = 5
        if not 1 <= roi_top_k <= num_experts:
            raise ValueError("roi_top_k must be between 1 and num_experts")
        if not 1 <= segmentation_top_k <= num_segmentation_experts:
            raise ValueError(
                "segmentation_top_k must be between 1 and num_segmentation_experts"
            )
        if num_attention_heads < 1 or hidden_dim % num_attention_heads:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        if in_channels < 1 or len(roi_size) != 2 or min(roi_size) < 1:
            raise ValueError("in_channels and both roi_size dimensions must be positive")
        if roi_generator not in ("none", "sam3", "yoloe"):
            raise ValueError("roi_generator must be 'none', 'sam3', or 'yoloe'")
        if roi_generator != "none" and not prompts:
            raise ValueError(f"prompts must be provided when roi_generator={roi_generator!r}")
        if image_backbone_type not in ("conv", "dinov3"):
            raise ValueError("image_backbone_type must be 'conv' or 'dinov3'")
        if in_channels != 3 and (
            roi_generator != "none" or image_backbone_type == "dinov3"
        ):
            raise ValueError("ROI detectors and DINOv3 require three-channel RGB inputs")
        if not 0 <= sam3_score_threshold <= 1:
            raise ValueError("sam3_score_threshold must be between 0 and 1")
        if not 0 <= yoloe_score_threshold <= 1:
            raise ValueError("yoloe_score_threshold must be between 0 and 1")
        if roi_backbones is not None and len(roi_backbones) != num_experts:
            raise ValueError("roi_backbones must contain exactly num_experts encoders")

        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.roi_top_k = roi_top_k
        self.segmentation_top_k = segmentation_top_k
        self.roi_size = roi_size
        self.prompts = tuple(prompts)
        self.roi_generator_name = roi_generator
        self.roi_proposal_generator = (
            Sam3PromptBoxGenerator(
                model_id=sam3_model_id,
                score_threshold=sam3_score_threshold,
            )
            if roi_generator == "sam3"
            else (
                YOLOEPromptBoxGenerator(
                    model_id=yoloe_model_id,
                    score_threshold=yoloe_score_threshold,
                )
                if roi_generator == "yoloe"
                else None
            )
        )

        self.num_classes = num_classes
        self.image_backbone = (
            image_backbone
            if image_backbone is not None
            else (
                DINOv3FeatureEncoder(
                    hidden_dim=hidden_dim,
                    model_id=dinov3_model_id,
                    trainable=dinov3_trainable,
                )
                if image_backbone_type == "dinov3"
                else ConvFeatureEncoder(in_channels, hidden_dim)
            )
        )
        self.roi_experts = nn.ModuleList(
            [
                ROIExpert(
                    in_channels,
                    hidden_dim,
                )
                if roi_backbones is None
                else nn.Sequential(
                    roi_backbones[index],
                    nn.LayerNorm(hidden_dim),
                )
                for index in range(num_experts)
            ]
        )
        self.roi_router = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )
        self.roi_expert_projection = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.attention_queries = nn.Parameter(torch.empty(1, attention_k, hidden_dim))
        nn.init.normal_(self.attention_queries, std=hidden_dim**-0.5)
        self.roi_attention = nn.MultiheadAttention(
            hidden_dim,
            num_attention_heads,
            batch_first=True,
        )
        self.attention_k = attention_k

        fused_dim = hidden_dim * (attention_k + 1)
        self.segmentation_router = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_segmentation_experts),
        )
        self.segmentation_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, num_classes),
                )
                for _ in range(num_segmentation_experts)
            ]
        )

    def forward(self, images: Tensor, roi_boxes: Tensor | None = None) -> Tensor:
        """Return multi-label class logits shaped ``[B, num_classes]``.

        ``roi_boxes`` uses pixel-coordinate ``(x1, y1, x2, y2)`` values and
        may include zero-area boxes as padding; these are ignored.
        """
        if images.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        
        batch_size, channels, image_height, image_width = images.shape
        
        if batch_size < 1 or channels < 1 or image_height < 1 or image_width < 1:
            raise ValueError("images must have non-empty batch, channel, and spatial dimensions")
        if channels != self.in_channels:
            raise ValueError(f"images must have {self.in_channels} channels")

        if roi_boxes is None and self.roi_proposal_generator is not None:
            roi_boxes = self.roi_proposal_generator(images, self.prompts)
        if roi_boxes is None:
            roi_boxes = images.new_empty((batch_size, 0, 4))
        roi_boxes = torch.as_tensor(roi_boxes, device=images.device, dtype=images.dtype)
        
        if roi_boxes.ndim != 3 or roi_boxes.shape[0] != batch_size or roi_boxes.shape[-1] != 4:
            raise ValueError("roi_boxes must have shape [B, R, 4]")
        if not torch.isfinite(roi_boxes).all():
            raise ValueError("roi_boxes must contain only finite coordinates")

        roi_crops, roi_valid = self._crop_rois(images, roi_boxes)
        roi_features, expert_present = self._encode_rois(roi_crops, roi_valid, batch_size)

        safe_padding_mask = ~expert_present
        if safe_padding_mask.numel():
            safe_padding_mask = safe_padding_mask.clone()
            safe_padding_mask[~expert_present.any(dim=1), 0] = False
        queries = self.attention_queries.expand(batch_size, -1, -1)
        attended, _ = self.roi_attention(
            queries,
            roi_features,
            roi_features,
            key_padding_mask=safe_padding_mask,
            need_weights=False,
        )
        attended = attended * expert_present.any(dim=1)[:, None, None]
        roi_branch = attended.reshape(batch_size, self.attention_k * self.hidden_dim)

        image_features = self.image_backbone(images)
        self._validate_encoder_output(
            image_features,
            batch_size,
            self.hidden_dim,
            "image_backbone",
        )
        fused_features = torch.cat((image_features, roi_branch), dim=1)
        return self._route_segmentation_experts(fused_features)

    def _crop_rois(self, images: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
        _, _, image_height, image_width = images.shape
        num_rois = boxes.shape[1]
        valid = (
            (boxes[..., 2] > boxes[..., 0])
            & (boxes[..., 3] > boxes[..., 1])
        )
        if num_rois == 0:
            return images.new_empty((0, images.shape[1], *self.roi_size)), valid

        box_batch, box_index = torch.where(valid)
        valid_boxes = boxes[box_batch, box_index]
        x1 = valid_boxes[:, 0].clamp(0, image_width)
        y1 = valid_boxes[:, 1].clamp(0, image_height)
        x2 = valid_boxes[:, 2].clamp(0, image_width)
        y2 = valid_boxes[:, 3].clamp(0, image_height)
        in_bounds = (x2 > x1) & (y2 > y1)
        valid[box_batch[~in_bounds], box_index[~in_bounds]] = False
        box_batch, box_index = box_batch[in_bounds], box_index[in_bounds]
        x1, y1, x2, y2 = x1[in_bounds], y1[in_bounds], x2[in_bounds], y2[in_bounds]
        if not box_batch.numel():
            return images.new_empty((0, images.shape[1], *self.roi_size)), valid

        roi_height, roi_width = self.roi_size
        x_steps = (torch.arange(roi_width, device=images.device, dtype=images.dtype) + 0.5)
        y_steps = (torch.arange(roi_height, device=images.device, dtype=images.dtype) + 0.5)
        sample_x = x1[:, None] + (x2 - x1)[:, None] * (x_steps / roi_width)
        sample_y = y1[:, None] + (y2 - y1)[:, None] * (y_steps / roi_height)
        grid_x = (2 * sample_x / image_width - 1)[:, None, :].expand(-1, roi_height, -1)
        grid_y = (2 * sample_y / image_height - 1)[:, :, None].expand(-1, -1, roi_width)
        grid = torch.stack((grid_x, grid_y), dim=-1)
        crops = F.grid_sample(
            images[box_batch],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        return crops, valid

    def _encode_rois(
        self,
        rois: Tensor,
        valid: Tensor,
        batch_size: int,
    ) -> tuple[Tensor, Tensor]:
        roi_features = rois.new_zeros((batch_size, self.num_experts, self.hidden_dim))
        expert_present = torch.zeros(
            (batch_size, self.num_experts),
            dtype=torch.bool,
            device=rois.device,
        )
        if not valid.any():
            return roi_features, expert_present

        valid_batches, _ = torch.where(valid)
        roi_statistics = torch.cat(
            (
                rois.mean(dim=(-1, -2)),
                rois.std(dim=(-1, -2), unbiased=False),
            ),
            dim=-1,
        )
        routing_logits = self.roi_router(roi_statistics)
        probabilities = routing_logits.softmax(dim=-1)
        selected = torch.zeros_like(probabilities, dtype=torch.bool)
        selected.scatter_(-1, probabilities.topk(self.roi_top_k, dim=-1).indices, True)
        selected_weights = probabilities * selected

        per_expert_features = []
        for expert_index, (expert, projection) in enumerate(
            zip(self.roi_experts, self.roi_expert_projection)
        ):
            routed = selected[:, expert_index]
            if not routed.any():
                per_expert_features.append(
                    roi_features.new_zeros((batch_size, self.hidden_dim))
                )
                continue
            batch_indices = valid_batches[routed]
            chosen_rois = rois[routed]
            encoded = expert(chosen_rois)
            self._validate_encoder_output(
                encoded,
                len(batch_indices),
                self.hidden_dim,
                f"roi_experts[{expert_index}]",
            )
            encoded = projection(encoded)
            weighted = encoded * selected_weights[routed, expert_index, None]
            expert_counts = selected_weights.new_zeros((batch_size,)).index_add(
                0,
                batch_indices,
                torch.ones_like(selected_weights[routed, expert_index]),
            )
            expert_weights = selected_weights.new_zeros((batch_size,)).index_add(
                0,
                batch_indices,
                selected_weights[routed, expert_index],
            )
            expert_features = roi_features.new_zeros((batch_size, self.hidden_dim)).index_add(
                0,
                batch_indices,
                weighted,
            )
            per_expert_features.append(
                expert_features
                / expert_weights.clamp_min(torch.finfo(expert_weights.dtype).tiny)[:, None]
            )
            expert_present[:, expert_index] = expert_counts > 0
        return torch.stack(per_expert_features, dim=1), expert_present

    def _route_segmentation_experts(self, fused_features: Tensor) -> Tensor:
        batch_size = fused_features.shape[0]
        probabilities = self.segmentation_router(fused_features).softmax(dim=-1)
        selected_probabilities, selected_indices = probabilities.topk(
            self.segmentation_top_k,
            dim=-1,
        )
        selected_weights = selected_probabilities / selected_probabilities.sum(
            dim=-1,
            keepdim=True,
        )
        sparse_weights = torch.zeros_like(probabilities).scatter(
            -1,
            selected_indices,
            selected_weights,
        )
        selected_weights = sparse_weights + probabilities - probabilities.detach()
        logits = fused_features.new_zeros(
            (batch_size, self.num_classes)
        )
        for expert_index, expert in enumerate(self.segmentation_experts):
            image_indices, topk_slots = torch.where(selected_indices == expert_index)
            if not image_indices.numel():
                continue
            expert_logits = expert(fused_features[image_indices])
            weighted_logits = expert_logits * selected_weights[image_indices, topk_slots, None]
            logits = logits.index_add(0, image_indices, weighted_logits)
        return logits

    @staticmethod
    def _validate_encoder_output(
        features: Tensor,
        batch_size: int,
        hidden_dim: int,
        name: str,
    ) -> None:
        if features.ndim != 2 or features.shape != (batch_size, hidden_dim):
            raise ValueError(f"{name} must return a rank-2 tensor shaped [N, hidden_dim]")


__all__ = [
    "ConvFeatureEncoder",
    "DINOv3FeatureEncoder",
    "Sam3PromptBoxGenerator",
    "YOLOEPromptBoxGenerator",
    "SparseROIAttributeModel",
]
