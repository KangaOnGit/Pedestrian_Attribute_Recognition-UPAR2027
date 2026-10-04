import torch.nn as nn
import torch

from src.losses.focal_loss import FocalLoss
from src.utils.config import load_config

CONFIG = load_config("configs/train.yaml")["losses"]

def build_loss(
    loss_name: str,
    training_targets: torch.Tensor | None = None,
) -> nn.Module:
    """
    Build loss

    Args:
        loss_name (str): Name of loss to use
            Supported:
                - bce
                - weighted_bce
                - focal (focal loss)
        training_targets (torch.Tensor | None): Training labels shaped [N, K].
            Used to balance positive and negative classes for focal loss.

    Returns:
        nn.Module: loss of choice
    """

    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_name == "weighted_bce":
        pos_weight = CONFIG[loss_name]["weight"]
        pos_weight_tensor = torch.tensor(
            pos_weight,
            dtype=torch.float32,
        )

        return nn.BCEWithLogitsLoss(
            pos_weight=pos_weight_tensor,
        )

    if loss_name == "focal":
        pos_weight = None
        class_weight = None
        if training_targets is not None:
            if (
                training_targets.ndim != 2
                or training_targets.shape[0] == 0
                or training_targets.shape[1] == 0
                or not torch.isfinite(training_targets).all()
                or ((training_targets < 0) | (training_targets > 1)).any()
            ):
                raise ValueError("training_targets must be a non-empty [N, K] tensor in [0, 1]")
            positive = training_targets.sum(dim=0)
            negative = training_targets.shape[0] - positive
            has_both_classes = (positive > 0) & (negative > 0)

            pos_weight = torch.ones_like(positive)
            pos_weight[has_both_classes] = (
                negative[has_both_classes] / positive[has_both_classes]
            )
            class_weight = torch.ones_like(positive)
            class_weight[has_both_classes] = (
                training_targets.shape[0] / (2 * negative[has_both_classes])
            )

        return FocalLoss(
            alpha=CONFIG[loss_name]["alpha"],
            gamma=CONFIG[loss_name]["gamma"],
            pos_weight=pos_weight,
            class_weight=class_weight,
        )

    raise ValueError(
        f"Unknown loss '{loss_name}'. "
        "Expected: bce, weighted_bce, or focal."
    )