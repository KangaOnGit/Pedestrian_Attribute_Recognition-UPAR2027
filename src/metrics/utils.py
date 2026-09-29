import torch
from jaxtyping import Float

def _safe_divide(
    numerator: Float[torch.Tensor, "..."],
    denominator: Float[torch.Tensor, "..."],
) -> Float[torch.Tensor, "..."]:
    return torch.where(
        denominator != 0,
        numerator / denominator,
        torch.zeros_like(numerator),
    )