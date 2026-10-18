import torch
import torch.nn as nn

class VisualResidualGate(nn.Module):
    def __init__(self, shallow_dim: int, embed_dim: int):
        super().__init__()

        self.visual_proj = nn.Sequential(
            nn.Linear(shallow_dim, embed_dim),
            nn.GELU(),
        )

        self.gate_proj = nn.Linear(shallow_dim, embed_dim)

        # Start near a no-op, so the text embeddings are initially preserved.
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, text_emb, shallow_features):
        """
        text_emb:         [40, D] or [B, 40, D]
        shallow_features: [B, shallow_dim]

        returns:          [B, 40, D]
        """
        visual = self.visual_proj(shallow_features)  # [B, D]
        gate = torch.sigmoid(self.gate_proj(shallow_features))

        residual = self.alpha * gate * visual       # [B, D]

        if text_emb.ndim == 2:
            text_emb = text_emb.unsqueeze(0)         # [1, 40, D]

        return text_emb + residual.unsqueeze(1)      # [B, 40, D]