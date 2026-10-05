from src.builders.augmentation import build_augmentation_flags
from src.train.load_data import UPAR_dataset
from src.utils.config import load_config, resolve_path
from src.models.upar.vision import CLIP_MEAN, CLIP_STD
from torch.utils.data import DataLoader

CONFIG = load_config("configs/train.yaml")
AUGMENTATION_CONFIG = load_config("configs/augmentation.yaml")

TRAIN_AUGMENTATION = AUGMENTATION_CONFIG["train"]
EVAL_AUGMENTATION = AUGMENTATION_CONFIG["eval"]

def build_dataloaders(args,
                      device,
                      height: int,
                      width: int):
    """Build training and validation datasets/loaders."""

    train_aug_flags = build_augmentation_flags(TRAIN_AUGMENTATION)
    eval_aug_flags = build_augmentation_flags(EVAL_AUGMENTATION)
    normalization = (
        {"normalization_mean": CLIP_MEAN, "normalization_std": CLIP_STD}
        if args.architecture == "upar"
        else {}
    )

    train_dataset = UPAR_dataset(
        data_path=str(resolve_path(args.train_csv)),
        split="train",
        data_name=args.dataset,
        num_samples=(
            None
            if args.all_training_data
            else args.train_num_samples
        ),
        aug=args.augment,
        include_detector_images=(
            args.architecture == "roi_moe" and args.roi_generator != "none"
        ),
        **train_aug_flags,
        **normalization,
        height = height,
        width = width,
    )

    eval_dataset = UPAR_dataset(
        data_path=str(resolve_path(args.val_csv)),
        split="eval",
        data_name=args.dataset,
        num_samples=args.eval_num_samples,
        aug=False,
        **eval_aug_flags,
        **normalization,
        height = height,
        width = width,
        include_detector_images=(
            args.architecture == "roi_moe" and args.roi_generator != "none"
        ),
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    return (
        train_dataset,
        eval_dataset,
        train_loader,
        eval_loader,
    )