
import logging
import torch
import torch.nn as nn

from transformers import AutoModel
from modules.shallow_cnn import ShallowCNN
from models.pretrained import PRETRAINED_MODEL_PATH

log = logging.getLogger(__name__)

class VisionEncoder(nn.Module):
    """
    SigLIP2 vision encoder with a separate shallow CNN.

    Outputs:
        image_features:   [B, E]
        patch_features:   [B, H, E]
        shallow_features: [B, E]

    E = vision hidden dimension/embedding dim, 1024 for SigLIP2-Large.
    """

    def __init__(
        self,
        vision_encoder: nn.Module,
        freeze_backbone: bool = True,
        shallow_hidden_dim: int = 128,
        shallow_out_dim: int = 1024,
        debug: bool = False,
    ):
        super().__init__()

        self.vision_encoder = vision_encoder

        self.shallow_cnn = ShallowCNN(
            output_dim=shallow_out_dim,
            hidden_dim=shallow_hidden_dim,
        )

        if freeze_backbone:
            for parameter in self.vision_encoder.parameters():
                parameter.requires_grad = False
        self.debug = debug
        
    def forward(
        self,
        images: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        # shallow path
        shallow_features: torch.Tensor = self.shallow_cnn(images)
        if self.debug:
            log.debug("shallow_features shape: %s", shallow_features.shape)

        outputs = self.vision_encoder(
            pixel_values=images,
            return_dict=True,
        )

        patch_features: torch.Tensor = outputs.last_hidden_state
        if self.debug:
            log.debug("patch_features shape: %s", patch_features.shape)

        image_features: torch.Tensor = outputs.pooler_output
        if self.debug:
            log.debug("image_features shape: %s", image_features.shape)

        return {
            "image_features": image_features,
            "patch_features": patch_features,
            "shallow_features": shallow_features,
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ve = VisionEncoder(
        vision_encoder = AutoModel.from_pretrained(
            str(PRETRAINED_MODEL_PATH),
            local_files_only=True,
        ).vision_model,
        freeze_backbone = False,
        shallow_hidden_dim = 128,
        debug = True,
    )
    
    ipt = torch.randn(
        6, 3, 256, 256
    )
    
    out = ve.forward(ipt)