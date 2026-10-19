from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor_linux"
sys.path.insert(0, str(ROOT))
if sys.platform.startswith("linux"):
    sys.path.insert(0, str(VENDOR))

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageFile
from torch import Tensor, nn

try:
    from models.cca_net import CCA_NET
except Exception:
    import traceback
    traceback.print_exc()
    raise

# ---------------------------------------------------------------------
# Paths and inference configuration
# ---------------------------------------------------------------------
                
PROMPTS_PATH = ROOT / "prompts.yaml"
WEIGHTS_PATH = ROOT / "best_challenge_avg.pt"

IMAGE_SIZE = (256, 256)  # (height, width)
NUM_CLASSES = 40
BATCH_SIZE = 32
CTX_LEN = 16
SHALLOW_HIDDEN_DIM = 128
CONDITION_TEXT = False

MEAN = torch.tensor(
    [0.5, 0.5, 0.5],
    dtype=torch.float32,
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.5, 0.5, 0.5],
    dtype=torch.float32,
).view(1, 3, 1, 1)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL: nn.Module | None = None
PROMPTS: list[str] | None = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a YAML mapping in {path}"
        )

    return config

# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------

def load_model_weights(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    map_location: torch.device,
) -> None:
    """Load model weights from a state-dict checkpoint."""

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )

    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "Checkpoint must contain a state dictionary."
        )

    # Support common checkpoint formats.
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, Mapping):
        raise ValueError(
            "Invalid checkpoint state dictionary."
        )

    if not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in state_dict.items()
    ):
        raise ValueError("Checkpoint contains an invalid state dictionary.")

    if state_dict and all(
        key.startswith("module.") for key in state_dict
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict, strict=True)

# ---------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------

def load_model() -> None:
    """Initialize the model and load trained weights."""

    global MODEL, PROMPTS

    if MODEL is not None:
        return

    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Trained CCA checkpoint not found: {WEIGHTS_PATH}"
        )

    config = load_config(PROMPTS_PATH)
    prompts = config.get("prompts")
    category_ids = config.get("category_ids")
    if not isinstance(prompts, list) or not all(
        isinstance(prompt, str) for prompt in prompts
    ):
        raise ValueError(f"Expected a list of prompt strings in {PROMPTS_PATH}")
    if len(prompts) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} prompts in {PROMPTS_PATH}, got {len(prompts)}"
        )
    if not isinstance(category_ids, list) or not all(
        isinstance(category_id, int) for category_id in category_ids
    ):
        raise ValueError(
            f"Expected a list of category IDs in {PROMPTS_PATH}"
        )
    if len(category_ids) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} category IDs in {PROMPTS_PATH}, "
            f"got {len(category_ids)}"
        )

    PROMPTS = prompts

    model = CCA_NET(
        prompts=prompts,
        ctx_len=CTX_LEN,
        csc=False,
        hybrid=True,
        category_ids=category_ids,
        condition_text=CONDITION_TEXT,
        freeze_backbone=True,
        shallow_hidden_dim=SHALLOW_HIDDEN_DIM,
    )

    load_model_weights(
        WEIGHTS_PATH,
        model=model,
        map_location=torch.device("cpu"),
    )

    model.to(DEVICE)
    model.eval()

    MODEL = model

    LOGGER.info("Model loaded successfully on %s", DEVICE)

# ---------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------

def load_image(
    image_path: str | Path,
) -> Tensor:
    """Load and normalize an RGB image for CCA-Net."""
    height, width = IMAGE_SIZE
    image_path = Path(image_path)

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(
            (width, height),
            resample=Image.Resampling.BILINEAR,
        )

        array = np.asarray(image, dtype=np.uint8).copy()

    # [H, W, C] -> [C, H, W], float32 in [0, 1].
    image_tensor = (
        torch.from_numpy(array)
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
    )

    return (
        (image_tensor - MEAN.squeeze(0))
        / STD.squeeze(0)
    ).contiguous()

# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------

@torch.inference_mode()
def forward_batch(
    images: Tensor,
) -> Tensor:
    if MODEL is None:
        raise RuntimeError(
            "Model is not initialized. Call load_model() first."
        )

    logits = MODEL(images.to(DEVICE))
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    if not isinstance(logits, Tensor):
        raise TypeError("Model output must contain a logits tensor.")

    if logits.ndim != 2 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected logits shape [B, {NUM_CLASSES}], "
            f"got {tuple(logits.shape)}."
        )

    return logits.float()

@torch.inference_mode()

def predict_batch(
    samples: list[dict[str, Any]],
) -> list[list[float]]:
    if not samples:
        return []

    if MODEL is None:
        load_model()

    assert MODEL is not None

    results: list[list[float]] = []

    for start in range(0, len(samples), BATCH_SIZE):
        batch_samples = samples[start : start + BATCH_SIZE]

        images = torch.stack(
            [
                load_image(sample["image_path"])
                for sample in batch_samples
            ],
            dim=0,
        )

        logits = forward_batch(images)

        probabilities = torch.sigmoid(logits).cpu()

        probabilities = torch.nan_to_num(
            probabilities,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        for sample, row in zip(
            batch_samples,
            probabilities.tolist(),
        ):
            if len(row) != len(sample["attribute_names"]):
                raise ValueError(
                    "Prediction length does not match attribute_names "
                    f"for sample index {sample.get('index', 'unknown')}."
                )

            results.append(row)

    return results

def predict_image(sample: dict[str, Any]) -> list[float]:
    try:
        result = predict_batch([sample])[0]

        if len(result) != NUM_CLASSES:
            raise ValueError(
                f"Expected {NUM_CLASSES} probabilities, got {len(result)}"
            )

        if not all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in result):
            raise ValueError("Prediction contains invalid probabilities")

        return result

    except Exception:
        LOGGER.exception(
            "Prediction failed for index=%s, path=%s",
            sample.get("index"),
            sample.get("image_path"),
        )
        raise

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test CCA-Net inference."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to a real RGB image for end-to-end inference testing.",
    )
    args = parser.parse_args()

    try:
        load_model()

        assert MODEL is not None

        LOGGER.info("Checkpoint loading: OK")

        if args.image is None:
            LOGGER.warning(
                "Model loaded, but full inference was not tested. "
                "Pass --image /path/to/image to run the complete pipeline."
            )
        else:
            if not args.image.is_file():
                raise FileNotFoundError(
                    f"Test image not found: {args.image}"
                )

            sample = {
                "image_path": str(args.image),
                "attribute_names": [f"attribute_{i}" for i in range(NUM_CLASSES)],
                "index": 0,
            }

            predictions = predict_image(sample)

            assert len(predictions) == NUM_CLASSES
            assert all(np.isfinite(p) for p in predictions)
            assert all(0.0 <= p <= 1.0 for p in predictions)

            LOGGER.info(
                "End-to-end inference: OK | "
                "classes=%d | min_probability=%.6f | "
                "max_probability=%.6f",
                len(predictions),
                min(predictions),
                max(predictions),
            )

            print("First 10 probabilities:", predictions[:10])

    except Exception:
        LOGGER.exception("Inference smoke test failed.")
        raise
    
# cd submission
# python run.py --image .\00001.png