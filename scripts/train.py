import argparse
import logging
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
import torch

from src.builders.data_loaders import build_dataloaders
from src.train.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.wb import init_wandb
from src.utils.config import resolve_path

from src.models.feature_encoder import ImageAttributeModel
from src.models.roi_moe import SparseROIAttributeModel


TRAIN_CONFIG = load_config("configs/train.yaml")
HYPER_PARAM = TRAIN_CONFIG["hyper_param"]

MISC_CONFIG = load_config("configs/miscs.yaml")

DEFAULT_DATASET = TRAIN_CONFIG["data"].get("data_name")

if DEFAULT_DATASET == "None":
    DEFAULT_DATASET = None
    
PROMPTS = load_config("configs/prompts.yaml")["prompts"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a pedestrian attribute recognition model."
    )

    # Data
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path(TRAIN_CONFIG["data"]["data_path"]),
        help="Training annotation CSV.",
    )

    parser.add_argument(
        "--val-csv",
        type=Path,
        default=Path("data/annotations/val.csv"),
        help="Validation annotation CSV.",
    )

    parser.add_argument(
        "--dataset",
        choices=("market", "pa", "peta"),
        default=DEFAULT_DATASET,
        help="Optionally train/evaluate on one dataset subset.",
    )

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of training samples.",
    )
    
    parser.add_argument(
        "--eval-num-samples",
        type=int,
        default=None,
        help="Number of samples to perform validation on.",
    )

    parser.add_argument(
        "--all-training-data",
        action="store_true",
        help="Use the complete training CSV.",
    )
    
    parser.add_argument(
        "--architecture",
        choices=("image", "roi_moe"),
        default="image",
        help="Use the pretrained image baseline or the prompted ROI mixture-of-experts.",
    )
    parser.add_argument(
        "--backbone",
        choices=("dinov3", "conv"),
        default="dinov3",
        help="Image feature encoder. DINOv3 is pretrained; conv is trained from scratch.",
    )
    parser.add_argument(
        "--finetune-backbone",
        action="store_true",
        help="Fine-tune DINOv3 instead of keeping its pretrained weights frozen.",
    )
    parser.add_argument(
        "--roi-generator",
        choices=("none", "sam3", "yoloe"),
        default="none",
        help="Proposal generator when --architecture=roi_moe.",
    )
    parser.add_argument(
        "--yoloe-prompt-mode",
        choices=("loop", "one-pass"),
        default="one-pass",
        help="Detect YOLO-E prompts separately or together in one inference pass.",
    )
    parser.add_argument(
        "--hidden-dim",
        "--hidden",
        dest="hidden_dim",
        type=int,
        default=None,
        help="Model hidden dimension (defaults to 256 for image and 128 for roi_moe).",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=5,
        help="Number of ROI experts for --architecture=roi_moe.",
    )
    parser.add_argument(
        "--roi-top-k",
        type=int,
        default=1,
        help="Number of ROI experts selected per ROI.",
    )
    # Training
    parser.add_argument(
        "--batch-size",
        type=int,
        default=HYPER_PARAM["batch_size"],
    )
    
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=HYPER_PARAM["logging_steps"],
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=HYPER_PARAM["epochs"],
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=float(HYPER_PARAM["lr"]),
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=float(HYPER_PARAM["weight_decay"]),
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=HYPER_PARAM["width"],
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=HYPER_PARAM["height"],
    )

    parser.add_argument(
        "--optimizer",
        choices=("adamw", "sgd"),
        default=HYPER_PARAM["optim"],
    )

    parser.add_argument(
        "--loss",
        choices=("bce", "weighted_bce", "focal"),
        default=HYPER_PARAM["loss"],
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=HYPER_PARAM["workers"],
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(HYPER_PARAM["output_dir"]),
    )

    # Runtime
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cuda, cuda:0, or cpu.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=TRAIN_CONFIG["default"]["seed"],
    )

    # Augmentation
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=HYPER_PARAM["aug"],
        help="Enable configured training augmentations.",
    )

    # Checkpointing
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model, optimizer, scheduler, and epoch from a training checkpoint.",
    )
    checkpoint_group.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Initialize model weights from a model state dictionary or training checkpoint.",
    )

    # Hugging Face
    parser.add_argument(
        "--hub-repo-id",
        default=None,
        help="Hugging Face repository ID.",
    )

    parser.add_argument(
        "--hub-private",
        action="store_true",
        help="Create the Hugging Face repository as private.",
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )

    parser.add_argument(
        "--wandb-project",
        default= MISC_CONFIG["WANDB_PROJECT"],
        help="Weights & Biases project name.",
    )

    parser.add_argument(
        "--wandb-entity",
        default=MISC_CONFIG["WANDB_ENTITY"],
        help="Weights & Biases entity/team.",
    )

    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Weights & Biases run name.",
    )

    return parser.parse_args()

def validate_args(args):
    """Validate command-line arguments."""

    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    if args.workers < 0:
        raise SystemExit("--workers cannot be negative")

    if args.train_num_samples is not None and args.train_num_samples < 1:
        raise SystemExit("--num-samples must be at least 1")

    if args.lr <= 0:
        raise SystemExit("--lr must be greater than 0")

    if args.weight_decay < 0:
        raise SystemExit("--weight-decay cannot be negative")

    if args.architecture == "image" and args.roi_generator != "none":
        raise SystemExit("--roi-generator can only be used with --architecture=roi_moe")

    if args.finetune_backbone and args.backbone != "dinov3":
        raise SystemExit("--finetune-backbone requires --backbone=dinov3")

    if args.hidden_dim is not None and args.hidden_dim < 1:
        raise SystemExit("--hidden-dim must be at least 1")

    if args.architecture == "roi_moe":
        if args.num_experts < 1:
            raise SystemExit("--num-experts must be at least 1")
        if not 1 <= args.roi_top_k <= args.num_experts:
            raise SystemExit("--roi-top-k must be between 1 and --num-experts")


def main():
    args = parse_args()

    os.chdir(PROJECT_ROOT)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    log = logging.getLogger(__name__)

    validate_args(args)

    set_seed(args.seed)

    # Device
    device = torch.device(
        args.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but is not available."
        )

    log.info("Using device: %s", device)

    # Data
    log.info("Loading dataset: %s", args.dataset)
    (
        train_dataset,
        eval_dataset,
        train_loader,
        eval_loader,
    ) = build_dataloaders(args,
                          device,
                          height = args.height,
                          width = args.width)

    num_attributes = len(train_dataset.label_columns)

    log.info(
        "Training samples: %d",
        len(train_dataset),
    )

    log.info(
        "Validation samples: %d",
        len(eval_dataset),
    )

    log.info(
        "Number of attributes: %d",
        num_attributes,
    )

    # Model
    log.info("Loading model...")
    if args.architecture == "image":
        model = ImageAttributeModel(
            num_classes=num_attributes,
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else 256,
            backbone_type=args.backbone,
            backbone_trainable=args.finetune_backbone,
        )
    else:
        model = SparseROIAttributeModel(
            num_classes=num_attributes,
            hidden_dim=args.hidden_dim if args.hidden_dim is not None else 128,
            num_experts=args.num_experts,
            roi_top_k=args.roi_top_k,
            prompts=PROMPTS if args.roi_generator != "none" else (),
            roi_generator=args.roi_generator,
            yoloe_prompt_mode=args.yoloe_prompt_mode,
            image_backbone_type=args.backbone,
            dinov3_trainable=args.finetune_backbone,
        )

    # W&B
    wandb_run = None
    if args.wandb:
        wandb_config = {
            key: str(value)
            if isinstance(value, Path)
            else value
            for key, value in vars(args).items()
        }

        wandb_config.update(
            {
                "num_attributes": num_attributes,
                "device": str(device),
                "train_samples": len(train_dataset),
                "eval_samples": len(eval_dataset),
            }
        )

        wandb_run = init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            config=wandb_config,
        )

    # Trainer
    output_dir = resolve_path(args.output_dir)

    log.info("Building trainer...")
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        eval_loader=eval_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        output_dir=str(output_dir),
        optim_name=args.optimizer,
        loss_name=args.loss,
        logging_steps=args.logging_steps,
        wandb_run=wandb_run,
    )

    # Resume
    if args.resume is not None:
        resume_path = resolve_path(args.resume)

        log.info(
            "Resuming training from %s",
            resume_path,
        )

        trainer.load_checkpoint(resume_path)
    elif args.weights is not None:
        weights_path = resolve_path(args.weights)

        log.info(
            "Loading model weights from %s",
            weights_path,
        )

        trainer.load_weights(weights_path)

    # Training
    log.info(
        "Starting %s training on %s",
        args.backbone,
        device,
    )

    try:
        trainer.train()

        if args.hub_repo_id:
            log.info(
                "Uploading model to Hugging Face: %s",
                args.hub_repo_id,
            )

            trainer.push_to_hub(
                args.hub_repo_id,
                private=args.hub_private,
            )

    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()