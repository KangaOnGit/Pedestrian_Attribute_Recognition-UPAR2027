
import torch
import torch.nn as nn
from jaxtyping import Float

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
        shallow_features: Float[torch.Tensor, "B E"],
    ) -> Float[torch.Tensor, "B ctx_len E"]:

        B = shallow_features.shape[0]

        visual: Float[torch.Tensor, "B E*ctx_len"] = self.visual_proj(shallow_features)
        gate: Float[torch.Tensor, "B E*ctx_len"] = torch.sigmoid(self.gate_proj(shallow_features))

        visual: Float[torch.Tensor, "B ctx_len E"] = visual.reshape(B, self.ctx_len, self.embed_dim)
        gate : Float[torch.Tensor, "B ctx_len E"]= gate.reshape(B, self.ctx_len, self.embed_dim)
        if self.debug:
            print(f"visual shape: {visual.shape}")
            print(f"gate shape: {gate.shape}")

        residual: Float[torch.Tensor, "B ctx_len E"] = self.alpha * gate * visual
        if self.debug:
            print(f"residual shape: {residual.shape}")

        return residual