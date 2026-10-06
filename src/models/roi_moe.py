from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from src.models.roi_generator import Sam3PromptBoxGenerator, YOLOEPromptBoxGenerator
from src.models.feature_encoder import DINOv3FeatureEncoder, ConvFeatureEncoder
from src.losses.aux_loss import aux_moe_loss
from jaxtyping import Bool, Float, Int
class SparseROIAttributeModel(nn.Module):
    """Route DINO-pooled ROI features through experts and fuse with full-image features.

    ``roi_generator="yoloe"`` loads YOLO-E during construction and uses
    ``prompts`` to generate pixel-coordinate XYXY boxes. Boxes may also be
    passed directly to ``forward``.
    """

    def __init__(
        self,
        
        num_classes: int,
        hidden_dim: int,
        num_experts: int,
        
        *,
        in_channels: int = 3,
        roi_top_k: int = 2,
        prompts: Sequence[str] = (),
        roi_generator: Literal["none", "sam3", "yoloe"] = "yoloe",
        sam3_model_id: str = "facebook/sam3",
        sam3_score_threshold: float = 0.5,
        yoloe_model_id: str = "yoloe-26s-seg.pt",
        yoloe_score_threshold: float = 0.3,
        yoloe_prompt_mode: Literal["loop", "one-pass"] = "one-pass",
        
        image_backbone_type: Literal["conv", "dinov3"] = "dinov3",
        dinov3_model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        dinov3_trainable: bool = False,
        image_backbone: nn.Module | None = None,
        
    ) -> None:
        super().__init__()
        if min(num_classes, hidden_dim, num_experts) < 1:
            raise ValueError("num_classes, hidden_dim, and num_experts must be positive")
        if not 1 <= roi_top_k <= num_experts:
            raise ValueError("roi_top_k must be between 1 and num_experts")
        if in_channels < 1:
            raise ValueError("in_channels must be positive")
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
        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.roi_top_k = roi_top_k
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
                    prompt_mode=yoloe_prompt_mode,
                )
                if roi_generator == "yoloe"
                else None
            )
        )

        self.num_classes: int = num_classes
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
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                )
                for index in range(num_experts)
            ]
        )
        
        # [B, R, Hd] -> [B, R, E]
        self.roi_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )
        
        # [B, ..., Hd] -> [B, ..., Hd]
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
        fused_dim: int = hidden_dim * (1 + num_experts)

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self,
                images_aug: Float[torch.Tensor, "B 3 H W"],
                images_no_aug: Float[torch.Tensor, "B 3 H W"],
                roi_boxes: Tensor | None = None,
                *,
                images_detector: Float[torch.Tensor, "B 3 H W"] | None = None,
                ) -> tuple[Float[Tensor, "B K"], Float[Tensor, ""]]:
        """Return multi-label class logits shaped ``[B, num_classes]``.

        ``roi_boxes`` uses pixel-coordinate ``(x1, y1, x2, y2)`` values and
        may include zero-area boxes as padding; these are ignored.
        """
        if images_aug.ndim != 4:
            raise ValueError("images must have shape [B, C, H, W]")
        if images_no_aug.ndim != 4 or images_no_aug.shape[:2] != images_aug.shape[:2]:
            raise ValueError("images_no_aug must match images_aug batch and channel dimensions")
        
        batch_size: int = images_aug.shape[0]
        channels: int = images_aug.shape[1]
        image_height: int = images_aug.shape[2]
        image_width: int = images_aug.shape[3]
        
        if batch_size < 1 or channels < 1 or image_height < 1 or image_width < 1:
            raise ValueError("images must have non-empty batch, channel, and spatial dimensions")
        if channels != self.in_channels:
            raise ValueError(f"images must have {self.in_channels} channels")

        # P is len(prompts)
        if roi_boxes is None and self.roi_proposal_generator is not None:
            if images_detector is None:
                raise ValueError(
                    "images_detector is required for automatic ROI generation; "
                    "it must contain resized RGB pixels in the [0, 1] range"
                )
            roi_boxes: Float[torch.Tensor, "B R 4"] = self.roi_proposal_generator(
                images_detector,
                self.prompts,
            )
            
        if roi_boxes is None:
            roi_boxes: Float[torch.Tensor, "B 0 4"] = images_aug.new_empty((batch_size, 0, 4))
        
        # make sure its on same device
        roi_boxes: Float[Tensor, "B P 4"] = torch.as_tensor(
            roi_boxes,
            device=images_no_aug.device,
            dtype=images_no_aug.dtype,
        )
        
        if roi_boxes.ndim != 3 or roi_boxes.shape[0] != batch_size or roi_boxes.shape[-1] != 4:
            raise ValueError("roi_boxes must have shape [B, P, 4]")
        if not torch.isfinite(roi_boxes).all():
            raise ValueError("roi_boxes must contain only finite coordinates")

        if isinstance(self.image_backbone, DINOv3FeatureEncoder):
            image_features, patch_grid = self.image_backbone.forward_with_patch_grid(
                images_no_aug
            )
        elif isinstance(self.image_backbone, ConvFeatureEncoder):
            patch_grid = self.image_backbone.forward_spatial_features(images_no_aug)
            image_features = F.adaptive_avg_pool2d(patch_grid, 1).flatten(start_dim=1)
        else:
            raise TypeError(
                "image_backbone must expose DINOv3 or convolutional spatial features"
            )

        coordinate_height, coordinate_width = (
            images_detector.shape[-2:]
            if images_detector is not None
            else images_no_aug.shape[-2:]
        )
        pooled_rois, roi_valid = self._pool_roi_features(
            patch_grid,
            roi_boxes,
            coordinate_height,
            coordinate_width,
        )
        roi_features, expert_present, aux_loss = self._encode_rois(pooled_rois, roi_valid)
        
        roi_features = roi_features * expert_present[..., None].float()

        roi_flat = roi_features.flatten(1)

        self._validate_encoder_output(
            image_features,
            batch_size,
            self.hidden_dim,
            "image_backbone",
        )
        fused_features: Float[torch.Tensor, "B Fused"] = torch.cat(
            (image_features, roi_flat),
            dim=1,
        )
        return self.classifier(fused_features), aux_loss

    def _pool_roi_features(
        self,
        feature_grid: Float[torch.Tensor, "B D Gh Gw"],
        boxes: Float[torch.Tensor, "B R 4"],
        image_height: int,
        image_width: int,
    ) -> tuple[Float[torch.Tensor, "B R D"], Bool[torch.Tensor, "B R"]]:
        batch_size, _, grid_height, grid_width = feature_grid.shape
        if boxes.ndim != 3 or boxes.shape[0] != batch_size or boxes.shape[-1] != 4:
            raise ValueError("roi_boxes must have shape [B, R, 4]")
        if image_height < 1 or image_width < 1:
            raise ValueError("image dimensions for ROI pooling must be positive")

        boxes = boxes.to(device=feature_grid.device, dtype=feature_grid.dtype)
        x1 = boxes[..., 0].clamp(0, image_width) * (grid_width / image_width)
        y1 = boxes[..., 1].clamp(0, image_height) * (grid_height / image_height)
        x2 = boxes[..., 2].clamp(0, image_width) * (grid_width / image_width)
        y2 = boxes[..., 3].clamp(0, image_height) * (grid_height / image_height)
        valid = (x2 > x1) & (y2 > y1)
        if boxes.shape[1] == 0:
            return feature_grid.new_empty((batch_size, 0, feature_grid.shape[1])), valid

        patch_left = torch.arange(
            grid_width,
            device=feature_grid.device,
            dtype=feature_grid.dtype,
        )
        patch_top = torch.arange(
            grid_height,
            device=feature_grid.device,
            dtype=feature_grid.dtype,
        )
        selected = (
            ((patch_left + 1)[None, None, None, :] > x1[..., None, None])
            & (patch_left[None, None, None, :] < x2[..., None, None])
            & ((patch_top + 1)[None, None, :, None] > y1[..., None, None])
            & (patch_top[None, None, :, None] < y2[..., None, None])
            & valid[..., None, None]
        )
        patch_mask = selected.to(feature_grid.dtype)
        patch_counts = patch_mask.sum(dim=(-1, -2))
        pooled = torch.einsum(
            "brhw,bdhw->brd",
            patch_mask,
            feature_grid,
        ) / patch_counts.clamp_min(1)[..., None]
        valid = valid & (patch_counts > 0)
        return pooled, valid

    def _encode_rois(
        self,
        roi_features: Float[torch.Tensor, "B R Hd"],
        valid: Bool[torch.Tensor, "B R"],
    ) -> tuple[Float[Tensor, "B E Hd"], Bool[Tensor, "B E"],  Float[Tensor, ""]]:
        batch_size = roi_features.shape[0]
        expert_features_by_image = roi_features.new_zeros(
            (batch_size, self.num_experts, self.hidden_dim)
        )
        expert_present: Bool[torch.Tensor, "B E"] = torch.zeros(
            (batch_size, self.num_experts),
            dtype=torch.bool,
            device=roi_features.device,
        )
        if not valid.any():
            return expert_features_by_image, expert_present, roi_features.new_zeros(())

        valid_indices: tuple[torch.Tensor, torch.Tensor] = torch.where(valid)
        valid_batches: Int[torch.Tensor, "N"] = valid_indices[0]
        valid_features: Float[torch.Tensor, "N Hd"] = roi_features[valid]
        routing_logits: Float[torch.Tensor, "N E"] = self.roi_router(valid_features)
        probabilities: Float[torch.Tensor, "N E"] = routing_logits.softmax(dim=-1)
        
        topk = probabilities.topk(
            self.roi_top_k,
            dim=-1,
        )

        selected = torch.zeros_like(
            probabilities,
            dtype=torch.bool,
        )

        selected.scatter_(
            -1,
            topk.indices,
            True,
        )

        topk_weights = topk.values

        topk_weights = topk_weights / topk_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-8)

        selected_weights = torch.zeros_like(probabilities)

        selected_weights.scatter_(
            -1,
            topk.indices,
            topk_weights,
        )

        per_expert_features: list[Float[torch.Tensor, "B Hd"]] = []
        for expert_index, (expert, projection) in enumerate(
            zip(self.roi_experts, self.roi_expert_projection)
        ):
            routed: Bool[torch.Tensor, "N"] = selected[:, expert_index]
            if not routed.any():
                per_expert_features.append(
                    expert_features_by_image.new_zeros((batch_size, self.hidden_dim))
                )
                continue
            batch_indices: Int[torch.Tensor, "N"] = valid_batches[routed]
            chosen_features: Float[torch.Tensor, "N Hd"] = valid_features[routed]
            encoded: Float[Tensor, "N Hd"] = chosen_features + expert(chosen_features)
            self._validate_encoder_output(
                encoded,
                len(batch_indices),
                self.hidden_dim,
                f"roi_experts[{expert_index}]",
            )
            encoded = encoded + projection(encoded)
            weighted: Float[Tensor, "N Hd"] = encoded * selected_weights[
                routed,
                expert_index,
                None,
            ]
            expert_counts: Float[Tensor, "B"] = selected_weights.new_zeros(
                (batch_size,)
            ).index_add(
                0,
                batch_indices,
                torch.ones_like(selected_weights[routed, expert_index]),
            )
            expert_features: Float[Tensor, "B Hd"] = expert_features_by_image.new_zeros(
                (batch_size, self.hidden_dim)
            ).index_add(
                0,
                batch_indices,
                weighted,
            )
            per_expert_features.append(
                expert_features
                / expert_counts.clamp_min(1)[:, None]
            )
            expert_present[:, expert_index] = expert_counts > 0
        stacked_features: Float[Tensor, "B E Hd"] = torch.stack(
            per_expert_features,
            dim=1,
        )
        
        aux_loss = aux_moe_loss(
            router_probs = probabilities,
            expert_indices = topk.indices,
            num_experts = self.num_experts,
        )
        
        return stacked_features, expert_present, aux_loss

    @staticmethod
    def _validate_encoder_output(
        features: torch.Tensor,
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
