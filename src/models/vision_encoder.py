
import torch
import torch.nn as nn
import torch.nn.functional as F

from jaxtyping import Float
from transformers import Siglip2VisionModel, AutoModel
from src.modules.shallow_cnn import ShallowCNN

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
        vision_encoder: Siglip2VisionModel,
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
        images: Float[torch.Tensor, "B 3 H W"],
        pixel_attention_mask: torch.Tensor | None = None,
        spatial_shapes: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:

        # shallow path
        shallow_features: Float[torch.tensor, "B E"] = self.shallow_cnn(images)
        if self.debug:
            print(f"shallow_features shape: {shallow_features.shape}")

        # deep path
        outputs = self.vision_encoder(
            pixel_values=images,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            return_dict=True,
        )

        patch_features: Float[torch.tensor, "B H E"] = outputs.last_hidden_state
        if self.debug:
            print(f"patch_features shape: {patch_features.shape}")

        image_features: Float[torch.tensor, "B E"] = outputs.pooler_output
        if self.debug:
            print(f"image_features shape: {image_features.shape}")

        return {
            "image_features": image_features,
            "patch_features": patch_features,
            "shallow_features": shallow_features,
        }

if __name__ == "__main__":
    ve = VisionEncoder(
        vision_encoder = AutoModel.from_pretrained("google/siglip2-large-patch16-256").vision_model,
        freeze_backbone = False,
        shallow_hidden_dim = 128,
        debug = True,
    )
    
    ipt = torch.randn(
        6, 3, 256, 256
    )
    
    out = ve.forward(ipt)