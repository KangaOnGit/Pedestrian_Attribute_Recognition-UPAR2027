import logging
import csv
from pathlib import Path
from jaxtyping import Float

import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.optim as optim
from torch.utils.data import DataLoader

from src.builders.optimizer import build_optimizer
from src.train.scheduler import build_scheduler
from src.builders.loss import build_loss
from src.train.checkpoint import load_checkpoint, load_model_weights, save_checkpoint
from src.utils.config import load_config
from src.utils.hf_hub import push_folder_to_hub
from src.utils.wb import log_wandb
from src.metrics.run import run_metrics
from src.metrics.base import eval_metrics
from src.models.roi_moe import SparseROIAttributeModel

CONFIG = load_config("configs/train.yaml")

log = logging.getLogger(__name__)

METRIC_COLUMNS = (
    "epoch",
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
)

class Trainer:
    
    def __init__(
        self,
        
        model: nn.Module,
        device: torch.device,
        
        train_loader: DataLoader,
        eval_loader: DataLoader,

        epochs: int,
        lr: float,
        weight_decay: float,
        
        output_dir: str,
        
        optim_name: str,
        loss_name: str,
        
        logging_steps: int,
        wandb_run: object | None = None,
    ) -> None:
        """
        init trainer class
        Args:
            model (nn.Module)
            device (torch.device)
            train_loader (DataLoader)
            eval_loader (DataLoader)
            epochs (int)
            lr (float)
            weight_decay (float)
            output_dir (str): output directory name
            optim_name (str)
            loss_name (str)
            logging_steps (int, optional)
        """
        self.model: nn.Module = model
        self.device: torch.device = device
        
        self.train_loader: DataLoader = train_loader
        self.eval_loader: DataLoader = eval_loader

        self.logging_steps = logging_steps
        self.wandb_run = wandb_run
        self.epochs: int = epochs
        self.start_epoch = 1
        self.global_step = 0
        self.best_mA = float("-inf")
        self.best_f1 = float("-inf")
        self.best_challenge_avg = float("-inf")
        self.config = CONFIG

        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.to(self.device)
        target_tensor = None
        if loss_name == "focal":
            training_targets = getattr(train_loader.dataset, "labels", None)
            if training_targets is None:
                raise ValueError(
                    "Focal loss requires the training dataset to expose its labels "
                    "for per-attribute class balancing."
                )
            target_tensor = torch.as_tensor(training_targets, dtype=torch.float32)
        self.criterion: nn.Module = build_loss(loss_name, target_tensor)

        # -------- Loss, Optim, Scheduler -------------
        self.criterion.to(self.device)
        
        self.optimizer: Optimizer = build_optimizer(
            self.model,
            optim_name,
            lr,
            weight_decay
        )

        self.scheduler: optim.lr_scheduler.SequentialLR = build_scheduler(
            self.optimizer,
            self.epochs,
            lr,
        )
        # ------------------------------------------

    def train(self) -> None:
        """
        It's in the name
        """

        log.info(f"Training for {self.epochs} epochs")
        log.info(f"Optimizer: {self.optimizer} | Scheduler: {self.scheduler} | Loss: {self.criterion}")

        csv_path = self.output_dir / "train_results.csv"
        if not csv_path.exists() or csv_path.stat().st_size == 0:
            with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                csv.DictWriter(csv_file, fieldnames=METRIC_COLUMNS).writeheader()

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            eval_loss, eval_result = self.eval()
            learning_rate = self.optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "challenge_avg": eval_result.avg,
                "mA": eval_result.mA,
                "label_f1": eval_result.label_f1,
                "inst_acc": eval_result.inst_acc,
                "inst_prec": eval_result.inst_prec,
                "inst_rec": eval_result.inst_rec,
                "inst_f1": eval_result.inst_f1,
                "learning_rate": learning_rate,
            }
            self._log_epoch(row, csv_path)
            log_wandb(self.wandb_run, row)
            self.scheduler.step()

            is_best_mA = eval_result.mA > self.best_mA
            if is_best_mA:
                self.best_mA = eval_result.mA

            is_best_f1 = eval_result.label_f1 > self.best_f1
            if is_best_f1:
                self.best_f1 = eval_result.label_f1

            is_best_challenge_avg = eval_result.avg > self.best_challenge_avg
            if is_best_challenge_avg:
                self.best_challenge_avg = eval_result.avg

            if is_best_mA:
                self.save_checkpoint(epoch, eval_result, "best_mA.pt")

            if is_best_f1:
                self.save_checkpoint(epoch, eval_result, "best_f1.pt")

            if is_best_challenge_avg:
                self.save_checkpoint(epoch, eval_result, "best_challenge_avg.pt")

            self.save_checkpoint(epoch, eval_result, "last_epoch.pt")

    def _log_epoch(self, row: dict[str, float | int], csv_path: Path) -> None:
        log.info(
            f"epoch %d/{self.epochs} | train_loss: %.5f | eval_loss: %.5f | challenge_avg: %.5f "
            "mA: %.5f | label_f1: %.5f | inst_acc: %.5f | inst_prec: %.5f "
            "inst_rec: %.5f | inst_f1: %.5f | learning_rate: %.8f",
            *(row[column] for column in METRIC_COLUMNS),
        )

        with csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            csv.DictWriter(csv_file, fieldnames=METRIC_COLUMNS).writerow(row)

    def train_epoch(
        self,
        epoch: int,
    ) -> float:
        self.model.train()

        total_loss: float = 0.0

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            if len(batch) == 4:
                images_aug, images_no_aug, images_detector, labels = batch
                images_detector = images_detector.to(self.device)
            else:
                images_aug, images_no_aug, labels = batch
                images_detector = None

            images_aug: Float[torch.Tensor, "B C H W"] = images_aug.to(self.device)
            images_no_aug: Float[torch.Tensor, "B C H W"] = images_no_aug.to(self.device)
            labels: Float[torch.Tensor, "B K"] = labels.to(self.device)
            
            self.optimizer.zero_grad()

            if isinstance(self.model, SparseROIAttributeModel):
                logits: Float[torch.Tensor, "B K"] = self.model(
                    images_aug,
                    images_no_aug,
                    images_detector=images_detector,
                )
            else:
                logits = self.model(images_aug)

            if not torch.isfinite(logits).all():
                log.error(
                    "Non-finite logits at epoch %d batch %d (global step %d). "
                    "logits=%s images_aug=%s images_no_aug=%s images_detector=%s labels=%s",
                    epoch,
                    batch_idx,
                    self.global_step + 1,
                    self._tensor_health(logits),
                    self._tensor_health(images_aug),
                    self._tensor_health(images_no_aug),
                    (
                        self._tensor_health(images_detector)
                        if images_detector is not None
                        else None
                    ),
                    self._tensor_health(labels),
                )
                raise FloatingPointError(
                    f"Non-finite model logits at epoch {epoch}, batch {batch_idx}"
                )

            loss = self.criterion(logits, labels)
            if not torch.isfinite(loss):
                log.error(
                    "Non-finite loss at epoch %d batch %d (global step %d). "
                    "loss=%s logits=%s labels=%s",
                    epoch,
                    batch_idx,
                    self.global_step + 1,
                    self._tensor_health(loss),
                    self._tensor_health(logits),
                    self._tensor_health(labels),
                )
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {batch_idx}"
                )

            loss.backward()
            should_log_gradients = batch_idx % self.logging_steps == 0
            nonfinite_gradient_names = self._nonfinite_gradient_names()
            if nonfinite_gradient_names:
                gradient_metrics, _ = self._gradient_diagnostics()
                log.error(
                    "Non-finite gradients at epoch %d batch %d (global step %d); "
                    "optimizer step skipped. Non-finite parameters (%d): %s; "
                    "global_norm=%s",
                    epoch,
                    batch_idx,
                    self.global_step + 1,
                    len(nonfinite_gradient_names),
                    ", ".join(nonfinite_gradient_names[:20]),
                    gradient_metrics["grad_norm"],
                )
                log_wandb(
                    self.wandb_run,
                    {
                        **gradient_metrics,
                        "train_batch_loss": float(loss.detach().item()),
                        "global_step": self.global_step + 1,
                        "epoch": epoch,
                        "batch": batch_idx,
                    },
                )
                self.optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite gradients at epoch {epoch}, batch {batch_idx}; "
                    "optimizer step was skipped"
                )

            if should_log_gradients:
                gradient_metrics, _ = self._gradient_diagnostics()
                log.info(
                    "Gradient diagnostics | epoch %d | batch %d/%d | "
                    "global_step %d | global_norm %.6g | gradients %d/%d | "
                    "missing %d | nonfinite %d | per_module_norm %s | "
                    "missing_by_module %s",
                    epoch,
                    batch_idx,
                    len(self.train_loader),
                    self.global_step + 1,
                    gradient_metrics["grad_norm"],
                    int(gradient_metrics["grad_parameter_count"]),
                    int(gradient_metrics["grad_trainable_parameter_count"]),
                    int(gradient_metrics["grad_missing_parameter_count"]),
                    int(gradient_metrics["grad_nonfinite_count"]),
                    {
                        name.removeprefix("grad_norm/"): value
                        for name, value in gradient_metrics.items()
                        if name.startswith("grad_norm/")
                    },
                    {
                        name.removeprefix("grad_missing/"): value
                        for name, value in gradient_metrics.items()
                        if name.startswith("grad_missing/")
                    },
                )
                log_wandb(
                    self.wandb_run,
                    {
                        **gradient_metrics,
                        "train_batch_loss": float(loss.detach().item()),
                        "global_step": self.global_step + 1,
                        "epoch": epoch,
                        "batch": batch_idx,
                    },
                )

            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx) % self.logging_steps == 0:
                log.info(
                    "Epoch [%d/%d] | "
                    "Batch [%d/%d] | "
                    "Loss %.5f | "
                    "LR %.8f",
                    epoch,
                    self.epochs,
                    batch_idx + 1,
                    len(self.train_loader),
                    loss.item(),
                    self.optimizer.param_groups[0]["lr"],
                )

            self.global_step += 1

        if not self.train_loader:
            raise ValueError("Training data loader is empty.")
        return total_loss / len(self.train_loader)

    def _nonfinite_gradient_names(self) -> list[str]:
        gradient_tensors = [
            (name, parameter.grad)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradient_tensors:
            return []

        finite_flags = torch.stack(
            [
                torch.isfinite(gradient).all()
                for _, gradient in gradient_tensors
            ]
        )
        if finite_flags.all().item():
            return []
        return [
            name
            for (name, _), is_finite in zip(gradient_tensors, finite_flags)
            if not is_finite.item()
        ]

    @staticmethod
    def _tensor_health(tensor: torch.Tensor) -> dict[str, str | float | int]:
        detached = tensor.detach()
        finite = torch.isfinite(detached)
        finite_values = detached[finite]
        return {
            "shape": str(tuple(detached.shape)),
            "finite": int(finite.sum().item()),
            "nan": int(torch.isnan(detached).sum().item()),
            "inf": int(torch.isinf(detached).sum().item()),
            "finite_min": (
                float(finite_values.min().item()) if finite_values.numel() else float("nan")
            ),
            "finite_max": (
                float(finite_values.max().item()) if finite_values.numel() else float("nan")
            ),
        }

    def _gradient_diagnostics(self) -> tuple[dict[str, float], list[str]]:
        """Summarize gradient norms by model component and identify non-finite gradients."""
        squared_norms: dict[str, torch.Tensor] = {}
        missing_gradients: dict[str, int] = {}
        nonfinite_gradients: list[str] = []
        parameter_count = 0
        trainable_parameter_count = 0

        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue

            trainable_parameter_count += 1
            component = name.split(".", maxsplit=1)[0]
            gradient = parameter.grad
            if gradient is None:
                missing_gradients[component] = missing_gradients.get(component, 0) + 1
                continue

            parameter_count += 1
            detached_gradient = gradient.detach()
            squared_norm = detached_gradient.float().square().sum()
            if component in squared_norms:
                squared_norms[component] = squared_norms[component] + squared_norm
            else:
                squared_norms[component] = squared_norm

            if not torch.isfinite(detached_gradient).all():
                nonfinite_gradients.append(name)

        global_squared_norm = (
            torch.stack(tuple(squared_norms.values())).sum()
            if squared_norms
            else torch.zeros((), device=self.device)
        )
        metrics = {
            "grad_norm": float(global_squared_norm.sqrt().item()),
            "grad_parameter_count": float(parameter_count),
            "grad_trainable_parameter_count": float(trainable_parameter_count),
            "grad_missing_parameter_count": float(
                sum(missing_gradients.values())
            ),
            "grad_nonfinite_count": float(len(nonfinite_gradients)),
        }
        metrics.update(
            {
                f"grad_norm/{component}": float(squared_norm.sqrt().item())
                for component, squared_norm in squared_norms.items()
            }
        )
        metrics.update(
            {
                f"grad_missing/{component}": float(count)
                for component, count in missing_gradients.items()
            }
        )
        return metrics, nonfinite_gradients

    @torch.no_grad()
    def eval(
        self,
    ) -> tuple[float, eval_metrics]:

        self.model.eval()
        val_loss: float = 0.0
        all_predictions: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for batch in self.eval_loader:
            if len(batch) == 4:
                images_aug, images_no_aug, images_detector, labels = batch
                images_detector = images_detector.to(self.device)
            else:
                images_aug, images_no_aug, labels = batch
                images_detector = None
            
            images_aug: Float[torch.Tensor, "B C H W"] = images_aug.to(self.device)
            images_no_aug: Float[torch.Tensor, "B C H W"] = images_no_aug.to(self.device)
            labels: Float[torch.Tensor, "B K"] = labels.to(self.device)

            if isinstance(self.model, SparseROIAttributeModel):
                logits: Float[torch.Tensor, "B K"] = self.model(
                    images_aug,
                    images_no_aug,
                    images_detector=images_detector,
                )
            else:
                logits = self.model(images_aug)

            loss = self.criterion(logits, labels)
            val_loss += loss.item()
            all_predictions.append(logits.sigmoid().cpu())
            all_labels.append(labels.cpu())

        if not all_predictions:
            raise ValueError("Evaluation data loader is empty.")

        eval_result = run_metrics(
            pred=torch.cat(all_predictions),
            gt=torch.cat(all_labels),
        )
        return val_loss / len(self.eval_loader), eval_result
        
    def save_checkpoint(
        self,
        epoch: int,
        metrics: eval_metrics,
        filename: str,
    ) -> None:
        save_checkpoint(
            self.output_dir / filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            global_step=self.global_step,
            best_mA=self.best_mA,
            best_f1=self.best_f1,
            best_challenge_avg=self.best_challenge_avg,
            metrics=metrics,
            config=self.config,
        )

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
    ) -> None:
        """
        Resume training from a checkpoint.
        """

        checkpoint = load_checkpoint(
            checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            map_location=self.device,
        )

        self.start_epoch = checkpoint.get("last_epoch", checkpoint["epoch"]) + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.best_mA = checkpoint.get("best_mA", float("-inf"))
        self.best_f1 = checkpoint.get(
            "best_label_f1",
            checkpoint.get("best_f1", float("-inf")),
        )
        self.best_challenge_avg = checkpoint.get("best_challenge_avg", float("-inf"))

        log.info(
            "Resumed checkpoint: %s",
            checkpoint_path,
        )

    def load_weights(
        self,
        weights_path: str | Path,
    ) -> None:
        load_model_weights(
            weights_path,
            model=self.model,
            map_location=self.device,
        )

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        commit_message: str = "Upload trained model",
    ) -> None:
        """
        Upload the training output directory to Hugging Face Hub.
        """

        push_folder_to_hub(
            folder_path=self.output_dir,
            repo_id=repo_id,
            private=private,
            commit_message=commit_message,
        )
