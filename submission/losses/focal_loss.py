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

        alpha_t = (
            self.alpha * targets
            + (1.0 - self.alpha) * (1.0 - targets)
        )

        if self.gamma == 0.0:
            focal_modulation = torch.ones_like(logits)
        else:
            log_one_minus_pt = torch.where(
                targets > 0.5,
                F.logsigmoid(-logits),
                F.logsigmoid(logits),
            )

            focal_modulation = torch.exp(
                self.gamma * log_one_minus_pt
            )

        loss = alpha_t * focal_modulation * ce_loss

        return loss.mean()