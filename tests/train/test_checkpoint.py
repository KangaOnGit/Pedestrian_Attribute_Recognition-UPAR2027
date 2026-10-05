from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn
from torch.optim import SGD

from src.metrics.base import eval_metrics
from src.train.checkpoint import load_model_weights, save_checkpoint


class CheckpointTests(unittest.TestCase):
    def test_load_model_weights_accepts_raw_and_training_checkpoint_files(self) -> None:
        source_model = nn.Linear(2, 1)
        with torch.no_grad():
            source_model.weight.fill_(0.5)
            source_model.bias.fill_(0.25)
        state_dict = source_model.state_dict()

        with TemporaryDirectory() as output_dir:
            raw_weights_path = Path(output_dir) / "weights.pt"
            checkpoint_path = Path(output_dir) / "checkpoint.pt"
            torch.save(state_dict, raw_weights_path)
            torch.save({"model_state_dict": state_dict, "epoch": 3}, checkpoint_path)

            for weights_path in (raw_weights_path, checkpoint_path):
                with self.subTest(weights_path=weights_path):
                    model = nn.Linear(2, 1)
                    load_model_weights(
                        weights_path,
                        model=model,
                        map_location=torch.device("cpu"),
                    )

                    torch.testing.assert_close(model.weight, source_model.weight)
                    torch.testing.assert_close(model.bias, source_model.bias)

    def test_checkpoint_records_best_label_f1_and_challenge_average(self) -> None:
        model = nn.Linear(2, 1)
        optimizer = SGD(model.parameters(), lr=0.1)
        scheduler = Mock()
        scheduler.state_dict.return_value = {}
        metrics = eval_metrics(
            avg=0.75,
            mA=0.65,
            label_f1=0.55,
            inst_acc=0.45,
            inst_prec=0.35,
            inst_rec=0.25,
            inst_f1=0.15,
        )

        with TemporaryDirectory() as output_dir:
            with patch("src.train.checkpoint.torch.save") as torch_save:
                save_checkpoint(
                    Path(output_dir) / "best_challenge_avg.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=1,
                    global_step=1,
                    best_mA=0.65,
                    best_f1=0.55,
                    best_challenge_avg=0.75,
                    metrics=metrics,
                    config=None,
                )

        checkpoint = torch_save.call_args.args[0]
        self.assertEqual(checkpoint["best_f1"], 0.55)
        self.assertEqual(checkpoint["best_label_f1"], 0.55)
        self.assertEqual(checkpoint["best_challenge_avg"], 0.75)
        self.assertEqual(checkpoint["metrics"]["avg"], 0.75)
        self.assertEqual(checkpoint["metrics"]["label_f1"], 0.55)


if __name__ == "__main__":
    unittest.main()
