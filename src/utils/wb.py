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

    return wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config,
    )


def log_wandb(
    run: Any | None,
    metrics: dict[str, Any],
    step: int,
) -> None:
    if run is not None:
        run.log(metrics, step=step)