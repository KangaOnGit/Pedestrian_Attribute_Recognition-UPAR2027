import torch.nn as nn
import torch

from src.losses.focal_loss import FocalLoss
from src.losses.asym_loss import AsymmetricLossOptimized
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
                - FocalAsym
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
        return FocalLoss(
            alpha=CONFIG[loss_name]["alpha"],
            gamma=CONFIG[loss_name]["gamma"],
        )
        
    if loss_name == "FocalAsym":
        focal = FocalLoss(
            alpha=CONFIG["focal"]["alpha"],
            gamma=CONFIG["focal"]["gamma"],
        )
        
        asym = AsymmetricLossOptimized()
        return focal, asym
        

    raise ValueError(
        f"Unknown loss '{loss_name}'. "
        "Expected: bce, weighted_bce, or focal or FocalAsym"
    )