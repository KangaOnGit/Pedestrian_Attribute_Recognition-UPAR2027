import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float


class FocalLoss(nn.Module):
    """
    Binary focal loss for multilabel classification.

    Args:
        alpha:
            Weight applied to positive examples.

        gamma:
            Focusing parameter. Larger values reduce the contribution
            of easy examples.

        pos_weight:
            Optional per-class multiplier for positive examples.

        class_weight:
            Optional per-class multiplier applied to each class's loss.
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
        pos_weight: torch.Tensor | None = None,
        class_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight)
        self.register_buffer("class_weight", class_weight)

    def forward(
        self,
        logits: Float[torch.Tensor, "B K"],
        targets: Float[torch.Tensor, "B K"],
    ) -> torch.Tensor:

        # Binary cross entropy for each attribute.
        ce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        prob = torch.sigmoid(logits)

        p_t = (
            prob * targets
            + (1.0 - prob) * (1.0 - targets)
        )
        
        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        focal_weight = alpha_t * (
            1.0 - p_t
        ).pow(self.gamma)

        loss = focal_weight * ce_loss
        if self.class_weight is not None:
            loss = loss * self.class_weight
        return loss.mean()