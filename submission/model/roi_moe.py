from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from model.roi_generator import YOLOEPromptBoxGenerator
from model.feature_encoder import DINOv3FeatureEncoder, ConvFeatureEncoder
from losses.aux_loss import aux_moe_loss
from model.experts import ROIExpert
from model.router import ROIRouter

from torchvision.ops import roi_align
class SparseROIAttributeModel(nn.Module):

    def __init__(
        self,
        num_classes: int,
        hidden_dim: int,
        num_experts: int,

        *,
        in_channels: int = 3,

        roi_top_k: int = 2,
        roi_size: tuple[int, int] = (14, 14),

        prompts: Sequence[str] = (),
        roi_generator: Literal[
            "none",
            "sam3",
            "yoloe",
        ] = "yoloe",

        sam3_model_id: str = "facebook/sam3",
        sam3_score_threshold: float = 0.5,

        yoloe_model_id: str = "yoloe-26s-seg.pt",
        yoloe_score_threshold: float = 0.3,
        yoloe_prompt_mode: Literal[
            "loop",
            "one-pass",
        ] = "one-pass",

        image_backbone_type: Literal[
            "conv",
            "dinov3",
        ] = "dinov3",

        dinov3_model_id: str = (
            "facebook/dinov3-vitb16-pretrain-lvd1689m"
        ),

        dinov3_trainable: bool = False,
        dinov3_load_pretrained: bool = True,

        image_backbone: nn.Module | None = None,
    ) -> None:

        super().__init__()

        self.hidden_dim = hidden_dim
        self.in_channels = in_channels
        self.num_experts = num_experts
        self.roi_top_k = roi_top_k
        self.roi_size = roi_size

        self.prompts = tuple(prompts)
        self.roi_generator_name = roi_generator
        self.roi_proposal_generator = (
            YOLOEPromptBoxGenerator(
                model_id=yoloe_model_id,
                score_threshold=yoloe_score_threshold,
                prompt_mode=yoloe_prompt_mode,
            )
        )

        self.image_backbone = (
            image_backbone
            if image_backbone is not None
            else (
                DINOv3FeatureEncoder(
                    hidden_dim=hidden_dim,
                    model_id=dinov3_model_id,
                    trainable=dinov3_trainable,
                    load_pretrained = dinov3_load_pretrained,
                )
                if image_backbone_type == "dinov3"
                else ConvFeatureEncoder(
                    in_channels,
                    hidden_dim,
                )
            )
        )

        self.roi_router = ROIRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
        )

        self.roi_experts = nn.ModuleList(
            [
                ROIExpert(
                    in_channels=hidden_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(num_experts)
            ]
        )

        fused_dim = hidden_dim * (
            1 + num_experts
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(
                fused_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(
                hidden_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        images_aug: torch.Tensor,
        images_no_aug: torch.Tensor,
        roi_boxes: Tensor | None = None,
        *,
        images_detector: torch.Tensor | None = None,
    ) -> tuple[
        Tensor,
        Tensor,
    ]:

        batch_size = images_aug.shape[0]
        if (
            roi_boxes is None
            and self.roi_proposal_generator is not None
        ):
            roi_boxes = self.roi_proposal_generator(
                images_detector,
                self.prompts,
            )

        if roi_boxes is None:
            roi_boxes = images_no_aug.new_empty(
                (
                    batch_size,
                    0,
                    4,
                )
            )

        roi_boxes: torch.Tensor = torch.as_tensor(
            roi_boxes,
            device=images_no_aug.device,
            dtype=images_no_aug.dtype,
        )

        if isinstance(
            self.image_backbone,
            DINOv3FeatureEncoder,
        ):
            (
                image_features,
                patch_grid,
            ) = self.image_backbone.forward_with_patch_grid(
                images_no_aug
            )
        elif isinstance(
            self.image_backbone,
            ConvFeatureEncoder,
        ):

            patch_grid = (
                self.image_backbone
                .forward_spatial_features(
                    images_no_aug
                )
            )

            image_features = (
                F.adaptive_avg_pool2d(
                    patch_grid,
                    1,
                )
                .flatten(1)
            )
        else:
            raise TypeError(
                "image_backbone must be "
                "DINOv3FeatureEncoder or "
                "ConvFeatureEncoder"
            )

        self._validate_encoder_output(
            image_features,
            batch_size,
            self.hidden_dim,
            "image_backbone",
        )

        expert_features, aux_loss = (
            self._encode_rois(
                feature_grid=patch_grid,
                roi_boxes=roi_boxes,
                image_height=(
                    images_detector.shape[-2]
                    if images_detector is not None
                    else images_no_aug.shape[-2]
                ),
                image_width=(
                    images_detector.shape[-1]
                    if images_detector is not None
                    else images_no_aug.shape[-1]
                ),
            )
        )

        expert_features: torch.Tensor = expert_features.flatten(1)

        # [B,D] + [B,E*D]
        fused_features: torch.Tensor = torch.cat(
            (
                image_features,
                expert_features,
            ),
            dim=1,
        )

        logits: torch.Tensor = self.classifier(
            fused_features
        )

        return logits, aux_loss

    def _encode_rois(
        self,
        feature_grid: torch.Tensor,
        roi_boxes: torch.Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        batch_size: int = feature_grid.shape[0]
        num_rois: int = roi_boxes.shape[1]

        device: torch.device = feature_grid.device
        dtype: torch.dtype = feature_grid.dtype

        # =========================================================
        # No ROIs
        # =========================================================

        if num_rois == 0:

            empty_expert_features: torch.Tensor = feature_grid.new_zeros(
                (
                    batch_size,
                    self.num_experts,
                    self.hidden_dim,
                )
            )

            empty_aux_loss: torch.Tensor = feature_grid.new_zeros(())

            return (
                empty_expert_features,
                empty_aux_loss,
            )

        # =========================================================
        # Convert boxes to feature-grid coordinates
        #
        # ROIAlign expects:
        #
        # [batch_idx, x1, y1, x2, y2]
        # =========================================================

        boxes: torch.Tensor = roi_boxes.to(
            device=device,
            dtype=dtype,
        )

        grid_height: int = feature_grid.shape[-2]
        grid_width: int = feature_grid.shape[-1]

        scale_x: float = (
            grid_width / image_width
        )

        scale_y: float = (
            grid_height / image_height
        )

        x1: torch.Tensor = (
            boxes[..., 0]
            * scale_x
        )

        y1: torch.Tensor = (
            boxes[..., 1]
            * scale_y
        )

        x2: torch.Tensor = (
            boxes[..., 2]
            * scale_x
        )

        y2: torch.Tensor = (
            boxes[..., 3]
            * scale_y
        )

        x1: torch.Tensor = x1.clamp(
            0,
            grid_width,
        )

        x2: torch.Tensor = x2.clamp(
            0,
            grid_width,
        )

        y1: torch.Tensor = y1.clamp(
            0,
            grid_height,
        )

        y2: torch.Tensor = y2.clamp(
            0,
            grid_height,
        )

        valid: torch.Tensor = (
            (x2 > x1)
            & (y2 > y1)
        )

        # =========================================================
        # Flatten boxes
        # =========================================================

        flat_boxes: torch.Tensor = torch.stack(
            (
                x1,
                y1,
                x2,
                y2,
            ),
            dim=-1,
        ).reshape(
            -1,
            4,
        )

        batch_indices: torch.Tensor = (
            torch.arange(
                batch_size,
                device=device,
                dtype=torch.long,
            )
            [:, None]
            .expand(
                batch_size,
                num_rois,
            )
            .reshape(-1)
        )

        batch_indices_float: torch.Tensor = batch_indices.to(dtype)

        roi_boxes_with_batch: torch.Tensor = torch.cat(
            (
                batch_indices_float[:, None],
                flat_boxes,
            ),
            dim=1,
        )

        # =========================================================
        # ROIAlign
        # =========================================================

        roi_features_flat: torch.Tensor = roi_align(
            input=feature_grid,
            boxes=roi_boxes_with_batch,
            output_size=self.roi_size,
            spatial_scale=1.0,
            sampling_ratio=2,
            aligned=True,
        )

        roi_height: int = self.roi_size[0]
        roi_width: int = self.roi_size[1]

        roi_features: torch.Tensor = roi_features_flat.reshape(
            batch_size,
            num_rois,
            self.hidden_dim,
            roi_height,
            roi_width,
        )

        # =========================================================
        # Geometry
        # =========================================================

        x1_normalized: torch.Tensor = (
            boxes[..., 0]
            / image_width
        )

        y1_normalized: torch.Tensor = (
            boxes[..., 1]
            / image_height
        )

        x2_normalized: torch.Tensor = (
            boxes[..., 2]
            / image_width
        )

        y2_normalized: torch.Tensor = (
            boxes[..., 3]
            / image_height
        )

        width_normalized: torch.Tensor = (
            boxes[..., 2]
            - boxes[..., 0]
        ) / image_width

        height_normalized: torch.Tensor = (
            boxes[..., 3]
            - boxes[..., 1]
        ) / image_height

        normalized_geometry: torch.Tensor = torch.stack(
            (
                x1_normalized,
                y1_normalized,
                x2_normalized,
                y2_normalized,
                width_normalized,
                height_normalized,
            ),
            dim=-1,
        )

        # =========================================================
        # Flatten ROI dimension
        # =========================================================

        num_flat_rois: int = (
            batch_size * num_rois
        )

        roi_features: torch.Tensor = roi_features.reshape(
            num_flat_rois,
            self.hidden_dim,
            roi_height,
            roi_width,
        )

        geometry: torch.Tensor = normalized_geometry.reshape(
            num_flat_rois,
            6,
        )

        valid_flat: torch.Tensor = valid.reshape(-1)

        # =========================================================
        # Router
        # =========================================================

        routing_logits: torch.Tensor = self.roi_router(
            roi_features,
            geometry,
        )

        routing_probs: torch.Tensor = F.softmax(
            routing_logits,
            dim=-1,
        )

        valid_float: torch.Tensor = valid_flat.to(dtype)

        valid_float_expanded: torch.Tensor = valid_float[:, None]

        # Invalid/padded ROIs should not contribute.
        routing_probs: torch.Tensor = (
            routing_probs
            * valid_float_expanded
        )

        # =========================================================
        # Top-k routing
        # =========================================================

        routing_k: int = min(
            self.roi_top_k,
            self.num_experts,
        )

        topk_result: torch.return_types.topk = (
            routing_probs.topk(
                k=routing_k,
                dim=-1,
            )
        )

        topk_weights: torch.Tensor = topk_result.values

        topk_indices: torch.Tensor = topk_result.indices

        selected_weight_sum: torch.Tensor = topk_weights.sum(
            dim=-1,
            keepdim=True,
        )

        selected_weight_sum_safe: torch.Tensor = selected_weight_sum.clamp_min(
            1e-8
        )

        # Normalize only across selected experts.
        topk_weights: torch.Tensor = (
            topk_weights
            / selected_weight_sum_safe
        )

        # =========================================================
        # Run experts
        # =========================================================

        encoded_rois: torch.Tensor = roi_features.new_zeros(
            (
                num_flat_rois,
                self.hidden_dim,
            )
        )

        for expert_idx, expert in enumerate(
            self.roi_experts
        ):

            selected: torch.Tensor = (
                topk_indices
                == expert_idx
            )

            has_selected: torch.Tensor = selected.any()

            if not has_selected:
                continue

            roi_mask: torch.Tensor = selected.any(
                dim=-1
            )

            roi_indices: torch.Tensor = (
                roi_mask
                .nonzero(
                    as_tuple=False,
                )
                .squeeze(-1)
            )

            expert_input: torch.Tensor = roi_features[
                roi_indices
            ]

            expert_output: torch.Tensor = expert(
                expert_input
            )

            selected_for_rows: torch.Tensor = selected[
                roi_indices
            ]

            expert_weights: torch.Tensor = torch.zeros(
                roi_indices.shape[0],
                device=device,
                dtype=dtype,
            )

            for k_idx in range(
                routing_k
            ):

                selected_at_k: torch.Tensor = selected_for_rows[
                    ...,
                    k_idx,
                ]

                if not selected_at_k.any():
                    continue

                selected_weights: torch.Tensor = topk_weights[
                    roi_indices,
                    k_idx,
                ]

                expert_weights[
                    selected_at_k
                ] = selected_weights[
                    selected_at_k
                ]

            weighted_expert_output: torch.Tensor = (
                expert_output
                * expert_weights[:, None]
            )

            encoded_rois[
                roi_indices
            ] = weighted_expert_output

        # =========================================================
        # Reshape back to [B, R, D]
        # =========================================================

        encoded_rois: torch.Tensor = encoded_rois.reshape(
            batch_size,
            num_rois,
            self.hidden_dim,
        )

        # =========================================================
        # Aggregate ROI embeddings PER EXPERT
        # =========================================================

        expert_features: torch.Tensor = feature_grid.new_zeros(
            (
                batch_size,
                self.num_experts,
                self.hidden_dim,
            )
        )

        expert_counts: torch.Tensor = feature_grid.new_zeros(
            (
                batch_size,
                self.num_experts,
            )
        )

        routing_weights: torch.Tensor = topk_weights.reshape(
            batch_size,
            num_rois,
            routing_k,
        )

        routing_indices: torch.Tensor = topk_indices.reshape(
            batch_size,
            num_rois,
            routing_k,
        )

        # =========================================================
        # Aggregate each top-k route
        # =========================================================

        for k_idx in range(
            routing_k
        ):

            indices: torch.Tensor = routing_indices[
                ...,
                k_idx,
            ]

            weights: torch.Tensor = routing_weights[
                ...,
                k_idx,
            ]

            weights: torch.Tensor = (
                weights
                * valid.to(dtype)
            )

            for expert_idx in range(
                self.num_experts
            ):

                mask: torch.Tensor = (
                    indices
                    == expert_idx
                )

                if not mask.any():
                    continue

                mask_float: torch.Tensor = mask.to(dtype)

                weighted_assignment: torch.Tensor = (
                    weights
                    * mask_float
                )

                weighted_features: torch.Tensor = (
                    encoded_rois
                    * weighted_assignment[
                        ...,
                        None,
                    ]
                )

                aggregated_features: torch.Tensor = weighted_features.sum(
                    dim=1
                )

                expert_features[
                    :,
                    expert_idx,
                ] += aggregated_features

                assignment_mass: torch.Tensor = weighted_assignment.sum(
                    dim=1
                )

                expert_counts[
                    :,
                    expert_idx,
                ] += assignment_mass

        # =========================================================
        # Weighted mean instead of sum
        # =========================================================

        expert_counts_expanded: torch.Tensor = expert_counts[
            ...,
            None,
        ]

        expert_counts_safe: torch.Tensor = expert_counts_expanded.clamp_min(
            1e-6
        )

        expert_features: torch.Tensor = (
            expert_features
            / expert_counts_safe
        )

        # =========================================================
        # Explicitly zero experts with no assignment
        # =========================================================

        no_assignment: torch.Tensor = (
            expert_counts
            <= 1e-6
        )

        no_assignment_expanded: torch.Tensor = no_assignment[
            ...,
            None,
        ]

        zero_expert_features: torch.Tensor = torch.zeros_like(
            expert_features
        )

        expert_features: torch.Tensor = torch.where(
            no_assignment_expanded,
            zero_expert_features,
            expert_features,
        )

        # =========================================================
        # Auxiliary MoE loss
        # =========================================================

        valid_router_probs: torch.Tensor = routing_probs[
            valid_flat
        ]

        valid_topk_indices: torch.Tensor = topk_indices[
            valid_flat
        ]

        num_valid_rois: int = (
            valid_router_probs.shape[0]
        )

        if num_valid_rois > 0:

            aux_loss: torch.Tensor = aux_moe_loss(
                router_probs=valid_router_probs,
                expert_indices=valid_topk_indices,
                num_experts=self.num_experts,
            )

        else:

            aux_loss: torch.Tensor = feature_grid.new_zeros(())

        return (
            expert_features,
            aux_loss,
        )
        
    @staticmethod
    def _validate_encoder_output(
        features: Tensor,
        batch_size: int,
        hidden_dim: int,
        name: str,
    ) -> None:

        if (
            features.ndim != 2
            or features.shape
            != (
                batch_size,
                hidden_dim,
            )
        ):
            raise ValueError(
                f"{name} must return "
                f"[B, hidden_dim]"
            )
        
    def _crop_roi_features(
        self,
        feature_grid: Tensor,
        boxes: Tensor,
        image_height: int,
        image_width: int,
    ) -> tuple[
        list[Tensor],
        list[tuple[int, int]],
        Tensor,
    ]:
        """
        Map detector-space ROI boxes to the DINO feature grid and
        crop each ROI independently.

        Returns:
            crops:
                List of [D, h_i, w_i] tensors.

            locations:
                (batch_idx, roi_idx) for each crop.

            valid:
                [B, R] validity mask.
        """

        batch_size: int = feature_grid.shape[0]
        channels: int = feature_grid.shape[1]
        grid_height: int = feature_grid.shape[2]
        grid_width: int = feature_grid.shape[3]

        num_rois: int = boxes.shape[1]

        device: torch.device = feature_grid.device
        dtype: torch.dtype = feature_grid.dtype

        boxes: Tensor = boxes.to(
            device=device,
            dtype=dtype,
        )

        # ROI coordinates -> DINO feature-map coordinates

        scale_x: float = grid_width / image_width
        scale_y: float = grid_height / image_height

        x1: Tensor = boxes[..., 0] * scale_x
        y1: Tensor = boxes[..., 1] * scale_y
        x2: Tensor = boxes[..., 2] * scale_x
        y2: Tensor = boxes[..., 3] * scale_y

        x1: Tensor = x1.clamp(
            min=0,
            max=grid_width,
        )

        x2: Tensor = x2.clamp(
            min=0,
            max=grid_width,
        )

        y1: Tensor = y1.clamp(
            min=0,
            max=grid_height,
        )

        y2: Tensor = y2.clamp(
            min=0,
            max=grid_height,
        )

        valid: Tensor = (
            (x2 > x1)
            & (y2 > y1)
        )

        crops: list[Tensor] = []
        locations: list[tuple[int, int]] = []

        for b in range(batch_size):
            batch_idx: int = b

            for r in range(num_rois):
                roi_idx: int = r

                roi_valid: Tensor = valid[
                    batch_idx,
                    roi_idx,
                ]

                if not roi_valid:
                    continue

                left_float: Tensor = x1[
                    batch_idx,
                    roi_idx,
                ]

                top_float: Tensor = y1[
                    batch_idx,
                    roi_idx,
                ]

                right_float: Tensor = x2[
                    batch_idx,
                    roi_idx,
                ]

                bottom_float: Tensor = y2[
                    batch_idx,
                    roi_idx,
                ]

                left_floor: Tensor = torch.floor(
                    left_float,
                )

                top_floor: Tensor = torch.floor(
                    top_float,
                )

                right_ceil: Tensor = torch.ceil(
                    right_float,
                )

                bottom_ceil: Tensor = torch.ceil(
                    bottom_float,
                )

                left: int = int(left_floor.item())
                top: int = int(top_floor.item())

                right: int = int(right_ceil.item())
                bottom: int = int(bottom_ceil.item())

                left: int = max(
                    0,
                    min(left, grid_width - 1),
                )

                top: int = max(
                    0,
                    min(top, grid_height - 1),
                )

                right: int = max(
                    left + 1,
                    min(right, grid_width),
                )

                bottom: int = max(
                    top + 1,
                    min(bottom, grid_height),
                )

                crop: Tensor = feature_grid[
                    batch_idx,
                    :,
                    top:bottom,
                    left:right,
                ]

                crops.append(crop)

                location: tuple[int, int] = (
                    batch_idx,
                    roi_idx,
                )

                locations.append(location)

        return crops, locations, valid

__all__ = [
    "ConvFeatureEncoder",
    "DINOv3FeatureEncoder",
    "Sam3PromptBoxGenerator",
    "YOLOEPromptBoxGenerator",
    "ROIExpert",
    "ROIRouter",
    "SparseROIAttributeModel",
]
