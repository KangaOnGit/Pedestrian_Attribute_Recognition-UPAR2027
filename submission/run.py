from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
sys.path.insert(0, str(VENDOR))

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageFile
from torch import Tensor, nn

from model.roi_moe import SparseROIAttributeModel

# ---------------------------------------------------------------------
# Paths and inference configuration
# ---------------------------------------------------------------------
                
PROMPTS_PATH = ROOT / "prompts.yaml"
YOLOE_WEIGHTS_PATH = ROOT / "yoloe-26s-seg.pt"
DINOV3_MODEL_PATH = ROOT / "dinov3_big_configs"
WEIGHTS_PATH = ROOT / "best_challenge_avg.pt"

IMAGE_SIZE = (128, 128)  # (height, width)
NUM_CLASSES = 40
BATCH_SIZE = 64

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32,
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32,
).view(1, 3, 1, 1)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL: nn.Module | None = None
PROMPTS: Any = None

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

    # Remove the DistributedDataParallel prefix if present.
    if state_dict and all(
        isinstance(key, str) and key.startswith("module.")
        for key in state_dict
    ):
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }

    # Strict loading catches architecture/checkpoint mismatches.
    #model.load_state_dict(state_dict, strict=True)
    

    # Match transformer-layer names from the training checkpoint
    # to the current Hugging Face DINOv3 module structure.
    model_keys = set(model.state_dict().keys())
    compatible_state_dict = {}

    for key, value in state_dict.items():
        new_key = key

        old_prefix = "image_backbone.model.layer."
        new_prefix = "image_backbone.model.model.layer."

        if key.startswith(old_prefix):
            candidate = new_prefix + key[len(old_prefix):]
            if candidate in model_keys:
                new_key = candidate

        if new_key in compatible_state_dict:
            raise ValueError(f"Duplicate checkpoint key after mapping: {new_key}")

        compatible_state_dict[new_key] = value

    # Fail if any trained parameter is missing or unexpected.
    model.load_state_dict(compatible_state_dict, strict=True)

# ---------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------

def load_model() -> None:
    """Initialize the model and load trained weights."""

    global MODEL, PROMPTS

    if MODEL is not None:
        return

    config = load_config(PROMPTS_PATH)

    if "prompts" not in config:
        raise KeyError(
            f"'prompts' is missing from {PROMPTS_PATH}"
        )

    PROMPTS = config["prompts"]

    model = SparseROIAttributeModel(
        num_classes=NUM_CLASSES,
        hidden_dim=128,
        num_experts=5,
        roi_top_k=2,
        prompts=PROMPTS,
        roi_generator="yoloe",
        yoloe_prompt_mode="one-pass",
        image_backbone_type="dinov3",
        dinov3_trainable=False,
        yoloe_model_id=str(YOLOE_WEIGHTS_PATH),
        dinov3_model_id=str(DINOV3_MODEL_PATH),
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
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Returns:
        images_aug:     [3, H, W], normalized RGB.
        images_no_aug:  [3, H, W], normalized RGB.
        images_detector:[3, H, W], RGB in [0, 1].
    """
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

    # Detector input: unnormalized RGB in [0, 1].
    images_detector = image_tensor.contiguous()

    # Backbone inputs: ImageNet-normalized RGB.
    images_no_aug = (
        (image_tensor - MEAN.squeeze(0))
        / STD.squeeze(0)
    ).contiguous()

    # No stochastic augmentation during inference.
    # This is the deterministic counterpart of the normalized
    # training augmentation branch.
    images_aug = images_no_aug.clone()

    return images_aug, images_no_aug, images_detector

# ---------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------

@torch.inference_mode()
def forward_batch(
    images_aug: Tensor,
    images_no_aug: Tensor,
    images_detector: Tensor,
) -> Tensor:
    if MODEL is None:
        raise RuntimeError(
            "Model is not initialized. Call load_model() first."
        )

    images_aug = images_aug.to(DEVICE)
    images_no_aug = images_no_aug.to(DEVICE)
    images_detector = images_detector.to(DEVICE)

    output = MODEL(
        images_aug,
        images_no_aug,
        images_detector=images_detector,
    )

    logits = output[0] if isinstance(output, (tuple, list)) else output

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

        image_pairs = [
            load_image(sample["image_path"])
            for sample in batch_samples
        ]

        images_aug = torch.stack(
            [pair[0] for pair in image_pairs],
            dim=0,
        )

        images_no_aug = torch.stack(
            [pair[1] for pair in image_pairs],
            dim=0,
        )

        images_detector = torch.stack(
            [pair[2] for pair in image_pairs],
            dim=0,
        )

        logits = forward_batch(
            images_aug,
            images_no_aug,
            images_detector,
        )

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
    return predict_batch([sample])[0]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test SparseROIAttributeModel inference."
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