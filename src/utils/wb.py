import os
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

def init_wandb(
    project: str,
    config: dict[str, Any],
    entity: str | None = None,
    run_name: str | None = None,
) -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "W&B is enabled but wandb is not installed. "
            "Run: pip install wandb"
        ) from exc

    api_key = os.getenv("WANDB_API_KEY")

    if not api_key:
        raise RuntimeError(
            "WANDB_API_KEY is not set. "
            "Add it to .env or your environment."
        )

    wandb.login(key=api_key)

    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
    )

    # Use epoch as the x-axis for epoch-level metrics.
    run.define_metric("epoch")

    for metric in (
        "train_loss",
        "eval_loss",
        "challenge_avg",
        "mA",
        "label_f1",
        "inst_acc",
        "inst_prec",
        "inst_rec",
        "inst_f1",
        "learning_rate",
    ):
        run.define_metric(metric, step_metric="epoch")

    return run

def log_wandb(
    run: Any | None,
    metrics: dict[str, Any],
    step: int,
) -> None:
    if run is not None:
        run.log(metrics, step=step)