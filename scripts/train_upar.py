"""Train the CLIP-based UPAR model independently from the main training CLI."""

import argparse
import csv
import logging
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed
from src.utils.wb import init_wandb, log_wandb

TRAIN_CONFIG = load_config("configs/train.yaml")
HYPER_PARAM = TRAIN_CONFIG["hyper_param"]
MISC_CONFIG = load_config("configs/miscs.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the CLIP-based UPAR pedestrian attribute model."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="UPAR data root containing annotations/task1/{train,val}/gt.csv.",
    )
    parser.add_argument(
        "--image-roots",
        type=Path,
        nargs="*",
        default=(),
        help="Additional roots used to resolve image paths in gt.csv.",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("src/models/upar/weights"),
        help="Directory containing clip_visual_fp16.pt.",
    )
    parser.add_argument(
        "--clip-model",
        default="ViT-B/16",
        help="OpenAI CLIP model used to initialize the UPAR model.",
    )
    parser.add_argument("--n-ctx", type=int, default=4)
    parser.add_argument(
        "--ctx-mode",
        choices=("shared", "polarity", "csc"),
        default="shared",
    )
    parser.add_argument("--ctx-init", default="")
    parser.add_argument(
        "--part-cuts",
        type=float,
        nargs=2,
        default=(0.32, 0.68),
        metavar=("HEAD", "LEGS"),
    )
    parser.add_argument("--adapter-reduction", type=int, default=4)
    parser.add_argument("--init-scale", type=float, default=100.0)
    parser.add_argument(
        "--grad-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--use-grl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the dataset-domain adversarial head.",
    )
    parser.add_argument(
        "--domain-loss-weight",
        type=float,
        default=1.0,
        help="Weight for the domain-classification loss.",
    )

    parser.add_argument("--train-num-samples", type=int, default=None)
    parser.add_argument("--eval-num-samples", type=int, default=None)
    parser.add_argument(
        "--all-training-data",
        action="store_true",
        help="Use the complete training annotation file.",
    )
    parser.add_argument("--batch-size", type=int, default=HYPER_PARAM["batch_size"])
    parser.add_argument(
        "--logging-steps", type=int, default=HYPER_PARAM["logging_steps"]
    )
    parser.add_argument("--epochs", type=int, default=HYPER_PARAM["epochs"])
    parser.add_argument("--lr", type=float, default=float(HYPER_PARAM["lr"]))
    parser.add_argument(
        "--weight-decay", type=float, default=float(HYPER_PARAM["weight_decay"])
    )
    parser.add_argument("--width", type=int, default=HYPER_PARAM["width"])
    parser.add_argument("--height", type=int, default=HYPER_PARAM["height"])
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "sgd"),
        default=HYPER_PARAM["optim"],
    )
    parser.add_argument("--workers", type=int, default=HYPER_PARAM["workers"])
    parser.add_argument(
        "--output-dir", type=Path, default=Path(HYPER_PARAM["output_dir"])
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["default"]["seed"])
    parser.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=HYPER_PARAM["aug"],
        help="Enable UPAR training augmentations.",
    )

    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume a UPAR checkpoint, including optimizer and scheduler.",
    )
    checkpoint_group.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Initialize trainable UPAR parameters from a checkpoint.",
    )

    parser.add_argument("--hub-repo-id", default=None)
    parser.add_argument("--hub-private", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", default=MISC_CONFIG["WANDB_PROJECT"]
    )
    parser.add_argument("--wandb-entity", default=MISC_CONFIG["WANDB_ENTITY"])
    parser.add_argument("--wandb-run-name", default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise SystemExit("--epochs must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.workers < 0:
        raise SystemExit("--workers cannot be negative")
    if args.logging_steps < 1:
        raise SystemExit("--logging-steps must be at least 1")
    if args.train_num_samples is not None and args.train_num_samples < 1:
        raise SystemExit("--train-num-samples must be at least 1")
    if args.eval_num_samples is not None and args.eval_num_samples < 1:
        raise SystemExit("--eval-num-samples must be at least 1")
    if args.lr <= 0:
        raise SystemExit("--lr must be greater than 0")
    if args.weight_decay < 0:
        raise SystemExit("--weight-decay cannot be negative")
    if args.height % 16 or args.width % 16 or args.height < 48:
        raise SystemExit("UPAR image dimensions must be multiples of 16 and height >= 48")
    if args.n_ctx < 1:
        raise SystemExit("--n-ctx must be at least 1")
    if args.adapter_reduction < 1:
        raise SystemExit("--adapter-reduction must be at least 1")
    if args.init_scale <= 0:
        raise SystemExit("--init-scale must be greater than 0")
    if args.domain_loss_weight < 0:
        raise SystemExit("--domain-loss-weight cannot be negative")
    if not 0 < args.part_cuts[0] < args.part_cuts[1] < 1:
        raise SystemExit("--part-cuts must be increasing values between 0 and 1")


def build_dataloaders(args: argparse.Namespace, device: torch.device):
    from src.models.upar.upar_train.data import (
        DOMAIN_NAMES,
        PARDataset,
        PathResolver,
        find_data_root,
        infer_domain,
        make_transforms,
        read_gt_csv,
    )

    data_root = find_data_root(
        str(resolve_path(args.data_root)) if args.data_root is not None else ""
    )
    image_roots = [
        data_root,
        PROJECT_ROOT,
        *(resolve_path(path) for path in args.image_roots),
    ]
    resolver = PathResolver(image_roots)
    train_transform, eval_transform = make_transforms(args.height, args.width)

    def make_dataset(split, transform):
        annotation_path = data_root / "annotations" / "task1" / split / "gt.csv"
        names, labels = read_gt_csv(annotation_path)
        paths, missing = [], []
        for name in names:
            path = resolver(name)
            if path is None:
                missing.append(name)
            else:
                paths.append(path)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} image(s) referenced by {annotation_path} could not "
                f"be resolved; examples: {missing[:5]}. Add their locations with "
                "--image-roots."
            )
        domains = [
            infer_domain(name, path) for name, path in zip(names, paths)
        ]
        return PARDataset(paths, labels, domains, transform)

    train_dataset = make_dataset(
        "train", train_transform if args.augment else eval_transform
    )
    eval_dataset = make_dataset("val", eval_transform)

    def select_samples(dataset, count, seed):
        if count is None:
            return dataset
        if count > len(dataset):
            raise ValueError(
                f"Requested {count} samples from a dataset with {len(dataset)} rows."
            )
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
        return Subset(dataset, indices)

    train_dataset = select_samples(
        train_dataset,
        None if args.all_training_data else args.train_num_samples,
        args.seed,
    )
    eval_dataset = select_samples(eval_dataset, args.eval_num_samples, args.seed + 1)
    if not len(train_dataset) or not len(eval_dataset):
        raise ValueError("UPAR training and validation datasets must not be empty.")
    if args.use_grl:
        selected_domains = (
            [
                train_dataset.dataset.domains[index]
                for index in train_dataset.indices
            ]
            if isinstance(train_dataset, Subset)
            else train_dataset.domains
        )
        if not any(domain >= 0 for domain in selected_domains):
            raise ValueError(
                "Domain-adversarial training was enabled, but no training images "
                "could be assigned to a known domain. Check the image paths."
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
    return train_dataset, eval_dataset, train_loader, eval_loader, len(DOMAIN_NAMES)


def build_model(args: argparse.Namespace, device: torch.device, n_domains: int):
    try:
        import clip
    except ImportError as exc:
        raise ImportError(
            "Training the updated UPAR model requires OpenAI CLIP. "
            "Install the `clip` package before running train_upar.py."
        ) from exc

    from src.models.upar.attributes import PART_OF_ATTR, PROMPT_PAIRS
    from src.models.upar.model import PartCLIPPAR

    clip_model, _ = clip.load(args.clip_model, device=device, jit=False)
    clip_model.float()
    weights_dir = resolve_path(args.weights_dir)
    visual_path = weights_dir / "clip_visual_fp16.pt"
    if not visual_path.is_file():
        raise FileNotFoundError(f"UPAR CLIP vision weights were not found: {visual_path}")
    visual_state = torch.load(visual_path, map_location="cpu", weights_only=True)
    clip_model.visual.load_state_dict(visual_state, strict=True)

    config = SimpleNamespace(
        img_h=args.height,
        img_w=args.width,
        part_cuts=args.part_cuts,
        n_ctx=args.n_ctx,
        ctx_mode=args.ctx_mode,
        ctx_init=args.ctx_init,
        grad_ckpt=args.grad_checkpointing,
        adapter_reduction=args.adapter_reduction,
        init_scale=args.init_scale,
        use_grl=args.use_grl,
    )
    return PartCLIPPAR(
        clip_model,
        config,
        PROMPT_PAIRS,
        PART_OF_ATTR,
        n_domains=n_domains,
    )


def train(
    model,
    train_loader,
    eval_loader,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    wandb_run,
    log: logging.Logger,
) -> Path:
    from src.models.upar.upar_train.losses import AsymmetricLoss
    from src.models.upar.upar_train.metrics import par_metrics
    from src.builders.optimizer import build_optimizer
    from src.train.scheduler import build_scheduler

    model.to(device)
    criterion = AsymmetricLoss().to(device)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.lr)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "upar_train_results.csv"
    columns = (
        "epoch",
        "train_loss",
        "train_asl_loss",
        "train_domain_loss",
        "eval_loss",
        "eval_asl_loss",
        "eval_domain_loss",
        "mA",
        "F1",
        "HM",
        "learning_rate",
    )
    if not result_path.exists() or not result_path.stat().st_size:
        with result_path.open("w", newline="", encoding="utf-8") as result_file:
            csv.DictWriter(result_file, fieldnames=columns).writeheader()

    start_epoch, global_step, best_hm = 1, 0, float("-inf")

    def load_model_state(path):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid UPAR checkpoint: {path}")
        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict", checkpoint.get("model", checkpoint)),
        )
        if not isinstance(state, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise ValueError(f"Checkpoint has no valid UPAR model state: {path}")
        model.load_trainable(state)
        return checkpoint

    if args.resume is not None:
        resume_path = resolve_path(args.resume)
        checkpoint = load_model_state(resume_path)
        if (
            "optimizer_state_dict" not in checkpoint
            or "scheduler_state_dict" not in checkpoint
        ):
            raise ValueError(
                f"UPAR resume checkpoint is missing optimizer/scheduler state: {resume_path}"
            )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_hm = float(checkpoint.get("best_hm", float("-inf")))
        log.info("Resumed UPAR training from %s at epoch %d", resume_path, start_epoch)
    elif args.weights is not None:
        weights_path = resolve_path(args.weights)
        load_model_state(weights_path)
        log.info("Loaded UPAR trainable weights from %s", weights_path)

    def evaluate():
        model.eval()
        losses, attr_losses, domain_losses = [], [], []
        logits_all, labels_all = [], []
        with torch.inference_mode():
            for images, labels, domains in eval_loader:
                images = images.to(device)
                labels = labels.to(device)
                domains = domains.to(device)
                parts, glob = model.image_parts(images)
                logits = model.logits(parts, glob, model.text_feats())
                mask = labels >= 0
                attr_loss = criterion(logits, labels.clamp(0, 1), mask)
                domain_loss = logits.new_zeros(())
                if model.domain_head is not None:
                    valid_domains = domains >= 0
                    if valid_domains.any():
                        domain_logits = model.domain_logits(parts, 1.0)
                        domain_loss = F.cross_entropy(
                            domain_logits[valid_domains], domains[valid_domains]
                        )
                loss = attr_loss + args.domain_loss_weight * domain_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite UPAR validation loss.")
                losses.append(loss.cpu())
                attr_losses.append(attr_loss.cpu())
                domain_losses.append(domain_loss.cpu())
                logits_all.append(logits.float().cpu())
                labels_all.append(labels.cpu())
        if not logits_all:
            raise ValueError("UPAR validation data loader is empty.")
        metrics = par_metrics(
            torch.cat(logits_all).sigmoid().numpy(),
            torch.cat(labels_all).numpy(),
        )
        return (
            torch.stack(losses).mean().item(),
            torch.stack(attr_losses).mean().item(),
            torch.stack(domain_losses).mean().item(),
            metrics,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = total_asl_loss = total_domain_loss = 0.0
        for batch_idx, (images, labels, domains) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            domains = domains.to(device)
            progress = ((epoch - 1) + (batch_idx - 1) / len(train_loader)) / args.epochs
            grl_lambda = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0

            optimizer.zero_grad(set_to_none=True)
            parts, glob = model.image_parts(images)
            logits = model.logits(parts, glob, model.text_feats())
            mask = labels >= 0
            attr_loss = criterion(logits, labels.clamp(0, 1), mask)
            domain_loss = logits.new_zeros(())
            if model.domain_head is not None:
                valid_domains = domains >= 0
                if valid_domains.any():
                    domain_logits = model.domain_logits(parts, grl_lambda)
                    domain_loss = F.cross_entropy(
                        domain_logits[valid_domains], domains[valid_domains]
                    )
            loss = attr_loss + args.domain_loss_weight * domain_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite UPAR loss at epoch {epoch}, batch {batch_idx}."
                )
            loss.backward()
            bad_gradients = [
                name
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
                and parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
            ]
            if bad_gradients:
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    "Non-finite UPAR gradients: " + ", ".join(bad_gradients[:10])
                )
            optimizer.step()

            total_loss += loss.item()
            total_asl_loss += attr_loss.item()
            total_domain_loss += domain_loss.item()
            global_step += 1
            if batch_idx % args.logging_steps == 0:
                log.info(
                    "UPAR epoch %d/%d batch %d/%d | loss %.5f (ASL %.5f, domain %.5f) | lambda %.3f",
                    epoch,
                    args.epochs,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                    attr_loss.item(),
                    domain_loss.item(),
                    grl_lambda,
                )

        eval_loss, eval_asl_loss, eval_domain_loss, metrics = evaluate()
        learning_rate = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "train_asl_loss": total_asl_loss / len(train_loader),
            "train_domain_loss": total_domain_loss / len(train_loader),
            "eval_loss": eval_loss,
            "eval_asl_loss": eval_asl_loss,
            "eval_domain_loss": eval_domain_loss,
            "mA": metrics["mA"],
            "F1": metrics["F1"],
            "HM": metrics["HM"],
            "learning_rate": learning_rate,
        }
        log.info(
            "UPAR epoch %d/%d | train loss %.5f | eval loss %.5f | mA %.4f | F1 %.4f | HM %.4f",
            epoch,
            args.epochs,
            row["train_loss"],
            eval_loss,
            metrics["mA"],
            metrics["F1"],
            metrics["HM"],
        )
        with result_path.open("a", newline="", encoding="utf-8") as result_file:
            csv.DictWriter(result_file, fieldnames=columns).writerow(row)
        log_wandb(wandb_run, row)
        scheduler.step()

        trainable_state = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "best_hm": max(best_hm, metrics["HM"]),
            "model_state_dict": trainable_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": {
                key: metrics[key] for key in ("mA", "F1", "HM", "Acc", "Prec", "Rec")
            },
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        torch.save(checkpoint, output_dir / "last_epoch.pt")
        if metrics["HM"] > best_hm:
            best_hm = metrics["HM"]
            checkpoint["best_hm"] = best_hm
            torch.save(checkpoint, output_dir / "best_hm.pt")
    return result_path


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)
    validate_args(args)
    set_seed(args.seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but is not available.")
    log.info("Using device: %s", device)

    train_dataset, eval_dataset, train_loader, eval_loader, num_domains = (
        build_dataloaders(args, device)
    )
    log.info(
        "UPAR training samples: %d | validation samples: %d | attributes: 40 | domains: %d",
        len(train_dataset),
        len(eval_dataset),
        num_domains,
    )
    model = build_model(args, device, num_domains)

    wandb_run = None
    if args.wandb:
        wandb_config = {
            key: (
                str(value)
                if isinstance(value, Path)
                else [str(item) for item in value]
                if isinstance(value, (tuple, list))
                and all(isinstance(item, Path) for item in value)
                else value
            )
            for key, value in vars(args).items()
        }
        wandb_config.update(
            {
                "num_attributes": 40,
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

    output_dir = resolve_path(args.output_dir)
    try:
        train(
            model,
            train_loader,
            eval_loader,
            args,
            device,
            output_dir,
            wandb_run,
            log,
        )
        if args.hub_repo_id:
            from src.utils.hf_hub import push_folder_to_hub

            log.info("Uploading UPAR output to Hugging Face: %s", args.hub_repo_id)
            push_folder_to_hub(
                output_dir,
                args.hub_repo_id,
                private=args.hub_private,
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
