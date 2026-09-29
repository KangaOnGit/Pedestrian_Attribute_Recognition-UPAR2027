
from torch.optim import Optimizer

import torch.optim as optim

def build_scheduler(
    optimizer: Optimizer,
    num_epochs: int,
    lr: float,
) -> optim.lr_scheduler.SequentialLR:

    if num_epochs < 1:
        raise ValueError("num_epochs must be at least 1.")

    warmup_epochs = min(
        num_epochs,
        max(1, int(0.05 * num_epochs)),
    )
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.05,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )

    decay = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs - warmup_epochs),
        eta_min=lr * 0.01,
    )

    schedule = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_epochs],
    )

    return schedule