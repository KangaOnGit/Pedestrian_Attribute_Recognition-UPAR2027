from torch import Tensor, nn
from src.models.feature_encoder import ConvFeatureEncoder


class ROIExpert(nn.Module):
    """Processes ROIs routed to one expert."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.encoder = ConvFeatureEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, rois: Tensor) -> Tensor:
        """
        Args:
            rois:
                [N, C, H, W]

        Returns:
            [N, hidden_dim]
        """

        x = self.encoder(rois)

        # Residual MLP in embedding space.
        x = x + self.projection(x)

        return self.norm(x)