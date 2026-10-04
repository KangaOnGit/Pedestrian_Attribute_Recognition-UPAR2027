
import albumentations as A
import logging

from albumentations.pytorch import ToTensorV2
from src.utils.config import load_config

log = logging.getLogger(__name__)

CONFIG = load_config("configs/augmentation.yaml")

def build_transforms(
    dataset: str | None = None,
    height: int = 256,
    width: int = 256,
    horizontal_flip: bool = True,
    random_resized_crop: bool = False,
    affine: bool = True,
    safe_rotate: bool = False,
    coarse_dropout: bool = True,
    gaussian_blur: bool = True,
    color_jitter: bool = True,
    rgb_shift: bool = True,
    image_compression: bool = True,
    split: str = "train",
    normalize: bool = True,
) -> A.Compose:

    """
    Build the Albumentations training pipeline.

    Args:
        dataset:
            Dataset name: "market", "pa", "peta", or None.

            The dataset-specific resize statistics are stored in the
            YAML for reference/analysis. The actual model input size
            is always CONFIG["resize"]["width"] x CONFIG["resize"]["height"].

        horizontal_flip: Enable/disable horizontal flipping.

        random_resized_crop: Enable/disable RandomResizedCrop.

        affine: Enable/disable affine transformations.

        safe_rotate: Enable/disable SafeRotate.

        coarse_dropout: Enable/disable CoarseDropout.

        gaussian_blur: Enable/disable GaussianBlur.

        color_jitter: Enable/disable ColorJitter.

        rgb_shift: Enable/disable RGBShift.

        image_compression: Enable/disable ImageCompression.

        normalize: Apply the configured ImageNet normalization.

        Returns:
            A.Compose: Albumentations transformation pipeline.
    """
    
    if split == "train":
        log.info(f"Split is {split}, using train configs for augments")
        config = CONFIG["train"]
    else:
        log.info(f"Split is {split}, using eval configs for augments")
        config = CONFIG["eval"]
        
    transforms: list[A.BasicTransform] = []

    log.info(f"Currently augmenting for Dataset {dataset}. "
             f"Image will be resized to (H, W) = ({height}, {width})")
    
    transforms.append(
        A.Resize(
            height=height,
            width=width,
        )
    )
    
    if random_resized_crop:
        c = config["random_resized_crop"]

        transforms.append(
            A.RandomResizedCrop(
                size=(height, width),
                scale=tuple(c["scale"]),
                ratio=tuple(c["ratio"]),
                p=c["p"],
            )
        )

    # Horizontal Flip
    if horizontal_flip:
        c = config["horizontal_flip"]

        transforms.append(
            A.HorizontalFlip(
                p=c["p"],
            )
        )

    # Affine
    if affine:
        c = config["affine"]

        transforms.append(
            A.Affine(
                scale=tuple(c["scale"]),
                translate_percent=tuple(c["translate_percent"]),
                rotate=tuple(c["rotate"]),
                shear=tuple(c["shear"]),
                p=c["p"],
            )
        )

    # Safe Rotate
    if safe_rotate:
        c = config["safe_rotate"]

        transforms.append(
            A.SafeRotate(
                limit=c["limit"],
                p=c["p"],
            )
        )

    # Coarse Dropout
    if coarse_dropout:
        c = config["coarse_dropout"]

        transforms.append(
            A.CoarseDropout(
                num_holes_range=tuple(c["num_holes_range"]),
                hole_height_range=tuple(c["hole_height_range"]),
                hole_width_range=tuple(c["hole_width_range"]),
                p=c["p"],
            )
        )

    # Gaussian Blur
    if gaussian_blur:
        c = config["gaussian_blur"]

        transforms.append(
            A.GaussianBlur(
                blur_limit=tuple(c["blur_range"]),
                p=c["p"],
            )
        )

    # Color Jitter
    if color_jitter:
        c = config["color_jitter"]

        transforms.append(
            A.ColorJitter(
                brightness=tuple(c["brightness_range"]),
                contrast=tuple(c["contrast_range"]),
                saturation=tuple(c["saturation_range"]),
                hue=tuple(c["hue_range"]),
                p=c["p"],
            )
        )

    # RGB Shift
    if rgb_shift:
        c = config["rgb_shift"]

        transforms.append(
            A.RGBShift(
                r_shift_limit=c["r_shift_limit"],
                g_shift_limit=c["g_shift_limit"],
                b_shift_limit=c["b_shift_limit"],
                p=c["p"],
            )
        )

    # Image Compression
    if image_compression:
        c = config["image_compression"]

        transforms.append(
            A.ImageCompression(
                quality_range=tuple(c["quality_range"]),
                p=c["p"],
            )
        )

    if normalize:
        c = config["norm"]
        transforms.append(
            A.Normalize(
                mean=c["mean"],
                std=c["std"],
            )
        )

    transforms.append(
        ToTensorV2()
    )
    log.info(f"Augmentations: {A.Compose(transforms)}")
    return A.Compose(transforms)