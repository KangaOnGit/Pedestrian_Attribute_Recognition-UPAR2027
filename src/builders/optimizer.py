import torch.nn as nn

from torch.optim import AdamW, SGD, Optimizer
from src.utils.config import load_config

CONFIG = load_config("configs/train.yaml")

def build_optimizer(
    model: nn.Module,
    optim_name: str,
    lr: float,
    weight_decay: float
) -> Optimizer:
    """
    Build optimizer

    Args:
        model (nn.Module)
        optim_name (str): Optimizer name
        lr (float): Learning Rate
        weight_decay (float)

    Returns:
        Optimizer of choice from optim_name
    """
    backbone_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.startswith(("backbone.model.", "image_backbone.model."))
    ]
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_parameter_ids
    ]
    parameter_groups = [{"params": head_parameters, "lr": lr}]
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": lr * 0.01}
        )

    if optim_name == "adamw":
        return AdamW(
            parameter_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(CONFIG["optim"][optim_name]["beta"]),
        )

    if optim_name == "sgd":
        return SGD(
            parameter_groups,
            lr=lr,
            weight_decay=weight_decay,
            momentum=CONFIG["optim"][optim_name]["momentum"]
        )

    raise ValueError(
        f"Unknown optimizer '{optim_name}'. "
        "Expected: adamw or sgd."
    )