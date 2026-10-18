import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassSelfAttention(nn.Module):
    """
    Self-attention across attribute classes

    Input:
        x: [num_classes, embed_dim]
           or [batch_size, num_classes, embed_dim]

    Output:
        Same shape as input

    Each attribute class can attend to every other class.
    The attention parameters are independent of SigLIP 2's
    pretrained token-level attention parameters
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Pre-normalization, following the Transformer block structure
        self.layer_norm1 = nn.LayerNorm(
            embed_dim, eps=layer_norm_eps
        )
        self.layer_norm2 = nn.LayerNorm(
            embed_dim, eps=layer_norm_eps
        )

        # Independent Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Feed-forward network, analogous to SigLIP's MLP
        hidden_dim = embed_dim * mlp_ratio
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.mlp_dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # [B, N, D] -> [B, H, N, D/H]
        batch_size, num_classes, _ = x.shape

        return (
            x.reshape(
                batch_size,
                num_classes,
                self.num_heads,
                self.head_dim,
            )
            .transpose(1, 2)
        )

    def forward(self, x, return_attn=False):
        # Support both [N, D] and [B, N, D]
        squeeze_batch = x.ndim == 2

        if squeeze_batch:
            x = x.unsqueeze(0)

        if x.ndim != 3 or x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected [N, {self.embed_dim}] or "
                f"[B, N, {self.embed_dim}], got {tuple(x.shape)}"
            )

        # Self-attention across classes
        residual = x
        h = self.layer_norm1(x)

        q = self._split_heads(self.q_proj(h))
        k = self._split_heads(self.k_proj(h))
        v = self._split_heads(self.v_proj(h))

        scores = (q @ k.transpose(-2, -1)) * (
            self.head_dim ** -0.5
        )
        attn = scores.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        h = attn @ v

        # [B, H, N, Dh] -> [B, N, D]
        h = h.transpose(1, 2).contiguous()
        h = h.reshape(x.shape[0], x.shape[1], self.embed_dim)
        h = self.proj_dropout(self.out_proj(h))

        x = residual + h

        # MLP sublayer with residual connection
        residual = x
        h = self.layer_norm2(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.mlp_dropout(h)
        h = self.fc2(h)
        h = self.mlp_dropout(h)

        x = residual + h

        if squeeze_batch:
            x = x.squeeze(0)

        if return_attn:
            # [B, H, N, N], or [H, N, N] for unbatched input
            if squeeze_batch:
                attn = attn.squeeze(0)
            return x, attn

        return x