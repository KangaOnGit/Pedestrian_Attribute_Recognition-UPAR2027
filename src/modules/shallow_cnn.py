
import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float

class ShallowCNN(nn.Module):
    """Trainable low-level visual feature extractor."""

    def __init__(
        self,
        output_dim: int = 1024,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.projection = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self,
                images: Float[torch.Tensor, "B 3 H W"],
                ) -> Float[torch.Tensor, "B E"]:
        return self.norm(self.projection(self.features(images)))