from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .attributes import ATTR_NAMES
from .head import PartHead
from .vision import PatchEncoder


class UPARAttributeModel(nn.Module):
    """Trainable UPAR adapter/head with a frozen pretrained CLIP vision tower."""

    def __init__(self, weights_dir: str | Path | None = None) -> None:
        super().__init__()
        if weights_dir is None:
            weights_dir = Path(__file__).parent / "weights"
        weights_dir = Path(weights_dir)

        head_checkpoint = torch.load(
            weights_dir / "head.pt",
            map_location="cpu",
            weights_only=True,
        )
        vision_state = torch.load(
            weights_dir / "clip_visual_fp16.pt",
            map_location="cpu",
            weights_only=True,
        )
        meta = head_checkpoint["meta"]
        if tuple(meta.get("attr_names", ATTR_NAMES)) != tuple(ATTR_NAMES):
            raise ValueError("UPAR checkpoint attributes do not match the canonical order.")

        self.img_h = int(meta["img_h"])
        self.img_w = int(meta["img_w"])
        if self.img_h % 16 or self.img_w % 16:
            raise ValueError("UPAR checkpoint image dimensions must be divisible by 16.")
        grid = (self.img_h // 16, self.img_w // 16)

        self.encoder = PatchEncoder.from_clip_visual_state(vision_state, grid)
        dimension = int(head_checkpoint["text_feats"].shape[-1])
        self.head = PartHead(
            dimension,
            grid,
            tuple(meta["rows"]),
            meta["part_of_attr"],
            int(meta["adapter_reduction"]),
        )
        self.head.load_state_dict(head_checkpoint["head"], strict=True)
        self.register_buffer("text_features", head_checkpoint["text_feats"].float())

        calibration = head_checkpoint["calib"]
        self.register_buffer("calibration_a", calibration["a"].float())
        self.register_buffer("calibration_b", calibration["b"].float())
        self.register_buffer("calibration_off", calibration["off"].float())
        self.tta = bool(meta.get("tta", True))

    def train(self, mode: bool = True) -> UPARAttributeModel:
        super().train(mode)
        self.encoder.eval()
        return self

    def _forward_logits(self, images: Tensor) -> Tensor:
        parts, global_features = self.head.parts_from_tokens(self.encoder(images))
        return self.head.logits(parts, global_features, self.text_features)

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B, 3, H, W]")
        if images.shape[-2:] != (self.img_h, self.img_w):
            raise ValueError(
                f"UPAR checkpoint expects images of size {self.img_h}x{self.img_w}, "
                f"got {images.shape[-2]}x{images.shape[-1]}"
            )

        logits = self._forward_logits(images)
        if self.training:
            return logits

        if self.tta:
            flipped_logits = self._forward_logits(torch.flip(images, dims=[3]))
            logits = 0.5 * (logits + flipped_logits)

        return (
            self.calibration_a.exp() * logits
            + self.calibration_b
            + self.calibration_off
        )


__all__ = ["UPARAttributeModel"]
