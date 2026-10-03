import pandas as pd
import numpy as np
import logging
import torch

from PIL import Image
from torch.utils.data import Dataset
from src.train.augmentation import build_transforms
from jaxtyping import Float


log = logging.getLogger(__name__)


class UPAR_dataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        data_name: str | None = None,
        num_samples: int | None = None,
        aug: bool = True,
        
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
    ):
        """
        Args:
            data_path:
                Path to the CSV

            split:
                "train" or "eval"

            data_name:
                Dataset to sample from:
                "market", "pa", "peta", or None

                None -> randomly sample from the entire CSV

            num_samples:
                Number of samples to use

            aug:
                Whether to use data augmentation
        """

        # Dataset boundaries depend on which CSV we are using.
        dataset_ranges = {
            "train": {
                "market": (0, 10001),
                "pa": (10001, 89002),
                "peta": (89002, None),
            },

            "eval": {
                "market": (0, 16459),
                "pa": (16459, 26445),
                "peta": (26446, None),
            },
        }

        # Validate split
        if split not in dataset_ranges:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Expected 'train' or 'eval'."
            )

        csv = pd.read_csv(data_path)

        # Validate dataset
        if data_name is not None:
            if data_name not in dataset_ranges[split]:
                raise ValueError(
                    f"Dataset '{data_name}' is not available "
                    f"in the {split} split."
                )

            start, end = dataset_ranges[split][data_name]

            csv = csv.iloc[start:end]

        # Randomly sample
        if num_samples is not None:
            if num_samples > len(csv):
                raise ValueError(
                    f"Requested {num_samples} samples, "
                    f"but only {len(csv)} are available."
                )

            csv = csv.sample(
                n=num_samples,
                replace=False,
            )

        # Reset index after selecting samples
        self.csv: pd.DataFrame = csv.reset_index(drop=True)

        self.image_paths: list[str] = [
            "data/" + path
            for path in self.csv["# image"]
        ]

        self.label_columns: list[str] = [
            col
            for col in self.csv.columns
            if col != "# image"
        ]

        self.labels: np.ndarray = self.csv[self.label_columns].values

        self.split: str = split
        self.data_name: str | None = data_name
        self.aug: bool = aug
        self.horizontal_flip: bool = horizontal_flip
        self.random_resized_crop: bool = random_resized_crop
        self.affine: bool = affine
        self.safe_rotate: bool = safe_rotate
        self.coarse_dropout: bool = coarse_dropout
        self.gaussian_blur: bool = gaussian_blur
        self.color_jitter: bool = color_jitter
        self.rgb_shift: bool = rgb_shift
        self.image_compression: bool = image_compression
        
        self.width: int = width
        self.height: int = height

        self.transform = build_transforms(
            dataset=self.data_name,
            height = self.height,
            width = self.width,
            horizontal_flip=self.aug and self.horizontal_flip,
            random_resized_crop=self.aug and self.random_resized_crop,
            affine=self.aug and self.affine,
            safe_rotate=self.aug and self.safe_rotate,
            coarse_dropout=self.aug and self.coarse_dropout,
            gaussian_blur=self.aug and self.gaussian_blur,
            color_jitter=self.aug and self.color_jitter,
            rgb_shift=self.aug and self.rgb_shift,
            image_compression=self.aug and self.image_compression,
            split=self.split,
        )

        log.info(
            f"Using data {self.data_name} with aug {self.aug} "
            f"and {len(self.csv)} samples"
        )

    def __len__(self) -> int:
        return len(self.csv)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        Float[torch.Tensor, "C H W"],
        Float[torch.Tensor, "K"],
    ]:
        label: Float[torch.Tensor, "K"] = torch.tensor(
            self.labels[idx],
            dtype=torch.float32,
        )

        image: np.ndarray = np.array(
            Image.open(self.image_paths[idx]).convert("RGB")
        )

        image: Float[torch.Tensor, "C H W"] = self.transform(
            image=image
        )["image"]

        return image, label