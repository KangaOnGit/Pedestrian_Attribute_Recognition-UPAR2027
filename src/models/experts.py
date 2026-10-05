from torch import Tensor, nn
from src.models.feature_encoder import ConvFeatureEncoder
from jaxtyping import Float

class ROIExpert(nn.Module):
    """Processes the ROIs routed to one expert."""

    def __init__(self,
                 in_channels: int,
                 hidden_dim: int) -> None:
        
        super().__init__()
        self.encoder = ConvFeatureEncoder(in_channels, hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self,
                rois: Tensor) -> Tensor:
        return rois + self.projection(self.encoder(rois))