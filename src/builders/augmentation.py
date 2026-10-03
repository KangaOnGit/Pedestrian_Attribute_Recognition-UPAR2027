def build_augmentation_flags(augmentation_config,):
    """Convert the YAML augmentation config into dataset constructor flags."""
    return {
        "horizontal_flip": "horizontal_flip" in augmentation_config,
        "random_resized_crop": "random_resized_crop" in augmentation_config,
        "affine": "affine" in augmentation_config,
        "safe_rotate": "safe_rotate" in augmentation_config,
        "coarse_dropout": "coarse_dropout" in augmentation_config,
        "gaussian_blur": "gaussian_blur" in augmentation_config,
        "color_jitter": "color_jitter" in augmentation_config,
        "rgb_shift": "rgb_shift" in augmentation_config,
        "image_compression": "image_compression" in augmentation_config,
    }