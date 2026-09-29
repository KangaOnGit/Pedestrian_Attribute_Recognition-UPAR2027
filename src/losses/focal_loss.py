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
    """

    def __init__(
        self,
        alpha: float,
        gamma: float,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        logits: Float[torch.Tensor, "B K"],
        targets: Float[torch.Tensor, "B K"],
    ) -> torch.Tensor:

        # Binary cross entropy for each attribute.
        ce_loss = F.binary_cross_entropy_with_logits(
            logits,
            targets,
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

        return (focal_weight * ce_loss).mean()