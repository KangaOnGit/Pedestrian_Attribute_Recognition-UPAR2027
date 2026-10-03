import torch

from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoModel

class DINOv3FeatureEncoder(nn.Module):
    """Use a pretrained DINOv3 model to encode full images."""

    def __init__(
        self,
        hidden_dim: int,
        model_id: str,
        trainable: bool,
    ) -> None:
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)

        self.trainable = trainable
        
        for parameter in self.model.parameters():
            parameter.requires_grad_(trainable)
        self.projection = nn.Sequential(
            nn.Linear(self.model.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        if not trainable:
            self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if not self.trainable:
            self.model.eval()
        return self.model

    def forward(self, images_no_aug: Tensor) -> Tensor:
        processor_inputs = self.processor(
            images=images_no_aug,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        processor_inputs = processor_inputs.to(model_device)
        if self.trainable:
            outputs = self.model(**processor_inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**processor_inputs)
        return self.projection(outputs.pooler_output)

class ConvFeatureEncoder(nn.Module):
    """Small convolutional encoder that returns one vector per input image."""

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)