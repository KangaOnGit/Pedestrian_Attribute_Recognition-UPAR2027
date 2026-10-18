import logging
import torch
import torch.nn as nn

from src.models.text_encoder import TextEncoder
from src.models.vision_encoder import VisionEncoder
from src.modules.visual_residual import VisualResidualGate
from transformers import AutoModel
from jaxtyping import Float

log = logging.getLogger(__name__)

class CCA_NET(nn.Module):
    def __init__(self,
                 
                 model_name: str,
                 
                 prompts: list[str],
                 ctx_len: int,
                 csc: bool = True,
                 hybrid: bool = False,
                 category_ids: list[int] | None = None,
                 condition_text: bool = False,
                 
                 debug: bool = False,
                 
                 freeze_backbone: bool = True,
                 shallow_hidden_dim: int = 128,
                 ):
        super().__init__()
        
        self.siglip2 = AutoModel.from_pretrained(model_name)
        
        self.text_encoder = TextEncoder(
            model_name = model_name,
            text_encoder = self.siglip2.text_model,
            prompts = prompts,
            ctx_len = ctx_len,
            csc = csc,
            hybrid = hybrid,
            category_ids = category_ids,
            freeze_backbone = freeze_backbone,
            condition_text = condition_text,
            debug = debug,
        )
        
        self.vision_encoder = VisionEncoder(
            vision_encoder = self.siglip2.vision_model,
            freeze_backbone = freeze_backbone,
            shallow_hidden_dim = shallow_hidden_dim,
            shallow_out_dim = self.siglip2.text_model.get_input_embeddings().embedding_dim,
            debug = debug,
        )
        self.con_text = condition_text
        self.cos = nn.CosineSimilarity(dim = 1, eps = 1e-6)
        
    def forward(self,
                img: Float[torch.Tensor, "B 3 H W"]):
        
        v_out = self.vision_encoder(img)
        
        image_features: Float[torch.Tensor, "B E"] =  v_out["image_features"]
        shallow_features: Float[torch.Tensor, "B E"] =  v_out["shallow_features"]
        
        if self.con_text:
            t_out: Float[torch.Tensor, "B 40 E"] = self.text_encoder(shallow_features)
        else:
            t_out: Float[torch.Tensor, "40 E"] = self.text_encoder()
            
        if self.con_text:
            matching: Float[torch.Tensor, "B 40"] = nn.functional.cosine_similarity(
                image_features.unsqueeze(1),
                t_out,
                dim=-1,
                eps=1e-6,
            )
        else:
            matching: Float[torch.Tensor, "B 40"] = nn.functional.cosine_similarity(
                image_features.unsqueeze(1),
                t_out.unsqueeze(0),
                dim=-1,
                eps=1e-6,
            )
        return matching
        
if __name__ == "__main__":
    import torch

    logging.basicConfig(level=logging.DEBUG)

    # Configuration
    MODEL_NAME = "google/siglip2-large-patch16-256"
    BATCH_SIZE = 3
    IMAGE_SIZE = 256
    CTX_LEN = 16

    prompts = [
        f"attribute {i}" for i in range(40)
    ]

    # Example category assignment; replace with your actual category IDs.
    category_ids = [i // 5 for i in range(40)]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    log.info("Device: %s", device)

    for condition_text in [False, True]:
        log.info("Testing condition_text=%s", condition_text)

        model = CCA_NET(
            model_name=MODEL_NAME,
            prompts=prompts,
            ctx_len=CTX_LEN,
            csc=False,
            hybrid=True,
            category_ids=category_ids,
            condition_text=condition_text,
            debug=True,
            freeze_backbone=True,
            shallow_hidden_dim=128,
        ).to(device)

        model.eval()

        images: Float[torch.Tensor, "B 3 H W"] = torch.randn(
            BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE,
            device=device,
        )

        try:
            with torch.no_grad():
                output = model(images)

            log.info("Output shape: %s", output.shape)
            log.info("Output dtype: %s", output.dtype)
            log.info("Output device: %s", output.device)
            log.info("All finite: %s", torch.isfinite(output).all().item())

            assert output.shape == (BATCH_SIZE, 40), (
                f"Expected {(BATCH_SIZE, 40)}, got {output.shape}"
            )

            log.info("Forward pass passed.")

        except Exception as e:
            log.exception("Forward pass failed: %s: %s", type(e).__name__, e)
            raise

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
            
        