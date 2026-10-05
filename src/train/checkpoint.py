import logging
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Optimizer

from src.metrics.base import eval_metrics

log = logging.getLogger(__name__)


def load_model_weights(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    map_location: torch.device,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            f"Model weights file must contain a state dictionary: {checkpoint_path}"
        )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint.get("state_dict", checkpoint),
    )
    if not isinstance(state_dict, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise ValueError(
            f"Model weights file does not contain a valid state dictionary: "
            f"{checkpoint_path}"
        )

    model.load_state_dict(state_dict)
    log.info("Loaded model weights: %s", checkpoint_path)


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    best_mA: float,
    best_f1: float,
    best_challenge_avg: float,
    metrics: eval_metrics,
    config: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "last_epoch": epoch,
        "global_step": global_step,
        "best_mA": best_mA,
        "best_f1": best_f1,
        "best_label_f1": best_f1,
        "best_challenge_avg": best_challenge_avg,
        "metrics": asdict(metrics),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }

    torch.save(checkpoint, path)

    log.info("Saved checkpoint: %s", path)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Any,
    map_location: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    log.info("Resumed checkpoint: %s", checkpoint_path)

    return checkpoint