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
from src.train.checkpoint import load_checkpoint, save_checkpoint
from src.utils.config import load_config
from src.utils.hf_hub import push_folder_to_hub
from src.utils.wb import log_wandb
from src.metrics.run import run_metrics
from src.metrics.base import eval_metrics

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
        
        logging_steps: int = CONFIG["hyper_param"]["logging_steps"],
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
        self.config = CONFIG

        self.output_dir: Path = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.to(self.device)
        self.criterion: nn.Module = build_loss(loss_name)

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
            log_wandb(self.wandb_run, row, epoch)
            self.scheduler.step()

            if eval_result.mA > self.best_mA:
                self.best_mA = eval_result.mA
                self.save_checkpoint(epoch, eval_result, "best_mA.pt")

            if eval_result.label_f1 > self.best_f1:
                self.best_f1 = eval_result.label_f1
                self.save_checkpoint(epoch, eval_result, "best_f1.pt")

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

        for batch_idx, (images, labels) in enumerate(self.train_loader, start=1):

            images: Float[torch.Tensor, "B C H W"] = images.to(self.device)
            labels: Float[torch.Tensor, "B K"] = labels.to(self.device)
            
            self.optimizer.zero_grad()

            logits: Float[torch.Tensor, "B K"] = self.model(images)
            loss = self.criterion(logits, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if (batch_idx) % self.logging_steps == 0:
                log.info(
                    "Epoch [%d/%d] | "
                    "Batch [%d/%d] | "
                    "Loss %.5f | "
                    "LR %.8f",
                    epoch + 1,
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

    @torch.no_grad()
    def eval(
        self,
    ) -> tuple[float, eval_metrics]:

        self.model.eval()
        val_loss: float = 0.0
        all_predictions: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for images, labels in self.eval_loader:
            
            images: Float[torch.Tensor, "B C H W"] = images.to(self.device)
            labels: Float[torch.Tensor, "B K"] = labels.to(self.device)

            logits: Float[torch.Tensor, "B K"] = self.model(images)

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
        self.best_f1 = checkpoint.get("best_f1", float("-inf"))

        log.info(
            "Resumed checkpoint: %s",
            checkpoint_path,
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
