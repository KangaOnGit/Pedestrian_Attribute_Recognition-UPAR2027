import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import Trainer

from src.utils.config import HF_TOKEN

log = logging.getLogger(__name__)


def push_hub(
    name: str,
    trainer: "Trainer",
) -> None:
    trainer.push_to_hub(name, token=HF_TOKEN)


def push_folder_to_hub(
    folder_path: str | Path,
    repo_id: str,
    private: bool = False,
    commit_message: str = "Upload trained model",
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Hugging Face Hub is not installed. "
            "Run: pip install huggingface_hub"
        ) from exc

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    log.info("Uploaded model to Hugging Face: %s", repo_id)