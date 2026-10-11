import torch
from jaxtyping import Float, Int
import torch.nn.functional as F

def aux_moe_loss(
    router_probs: Float[torch.Tensor, "N E"],
    expert_indices: Int[torch.Tensor, "N K"],
    num_experts: int,
    ):
    """Auxilliary Loss for MoE

    Args:
        router_probs: Router probabilities before topk
        expert_indices: Router that was picked after topk
        num_experts (int)

    N: total routed tokens/rois
    E: num_experts
    K: topK experts
    Returns:
        _type_: _description    _
    """
    
    expert_probability = router_probs.mean(dim=0)
    assignments = F.one_hot(
        expert_indices,
        num_classes=num_experts,
    ).float()

    expert_fraction = assignments.mean(dim=(0, 1))

    loss = num_experts * torch.sum(
        expert_fraction * expert_probability
    )

    return loss