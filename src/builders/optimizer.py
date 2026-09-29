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
    if optim_name == "adamw":
        return AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(CONFIG["optim"][optim_name]["beta"]),
        )

    if optim_name == "sgd":
        return SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=CONFIG["optim"][optim_name]["momentum"]
        )

    raise ValueError(
        f"Unknown optimizer '{optim_name}'. "
        "Expected: adamw or sgd."
    )