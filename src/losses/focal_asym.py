import torch
import torch.nn as nn

from src.losses.asym_loss import AsymmetricLossOptimized
from src.losses.focal_loss import FocalLoss
from jaxtyping import Float

class FocalAsym(nn.Module):
    
    
    def __init__(
        self,
        alpha: float,
        gamma: float,
        
        focal_weight: float,
        asym_weight: float,
        
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        disable_torch_grad_focal_loss=False,
    ) -> None:
        super().__init__()
        
        self.focal = FocalLoss(
            alpha,
            gamma
        )
        
        self.asym = AsymmetricLossOptimized(
            gamma_neg,
            gamma_pos,
            clip,
            eps,
            disable_torch_grad_focal_loss
        )
        
        self.asym_weight = asym_weight
        self.focal_weight = focal_weight
        
    def forward(
        self,
        logits: Float[torch.Tensor, "B K"],
        targets: Float[torch.Tensor, "B K"],
    ) -> torch.Tensor:
        
        return self.focal(logits, targets) * self.focal_weight + self.asym(logits, targets) * self.asym_weight
    