
import logging
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

class VisualResidualGate(nn.Module):
    def __init__(
        self,
        shallow_dim: int,
        ctx_len: int,
        embed_dim: int,
        debug: bool = False,
    ):
        super().__init__()

        self.ctx_len = ctx_len
        self.embed_dim = embed_dim
        self.debug = debug

        self.visual_proj = nn.Sequential(
            nn.Linear(shallow_dim, ctx_len * embed_dim),
            nn.GELU(),
        )

        self.gate_proj = nn.Linear(
            shallow_dim, ctx_len * embed_dim
        )

        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        shallow_features: torch.Tensor,
    ) -> torch.Tensor:

        B = shallow_features.shape[0]

        visual: torch.Tensor = self.visual_proj(shallow_features)
        gate: torch.Tensor = torch.sigmoid(self.gate_proj(shallow_features))

        visual = visual.reshape(B, self.ctx_len, self.embed_dim)
        gate = gate.reshape(B, self.ctx_len, self.embed_dim)
        if self.debug:
            log.debug("visual shape: %s", visual.shape)
            log.debug("gate shape: %s", gate.shape)

        residual: torch.Tensor = self.alpha * gate * visual
        if self.debug:
            log.debug("residual shape: %s", residual.shape)

        return residual