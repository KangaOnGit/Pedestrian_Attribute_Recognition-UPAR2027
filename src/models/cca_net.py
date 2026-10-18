import torch
import torch.nn as nn

from src.models.text_encoder import TextEncoder
from src.models.vision_encoder import VisionEncoder
from transformers import AutoModel

class CCA_NET(nn.Module):
    def __init__(self,
                 
                 model_name: str,
                 
                 prompts: list[str],
                 ctx_len: int,
                 csc: bool = True,
                 hybrid: bool = False,
                 category_ids: list[int] | None = None,
                 
                 debug: bool = False,
                 
                 freeze_backbone: bool = True,
                 shallow_hidden_dim: int = 128,
                 ):
        super().__init__()
        
        siglip2 = AutoModel.from_pretrained(model_name)
        
        text_encoder = TextEncoder(
            model_name = model_name,
            text_encoder = siglip2.text_model,
            prompts = prompts,
            ctx_len = ctx_len,
            csc = csc,
            hybrid = hybrid,
            category_ids = category_ids,
            freeze_backbone = freeze_backbone
            debug = debug
        )
        
        vision_encoder = VisionEncoder(
            vision_encoder = siglip2.vision_model,
            freeze_backbone = freeze_backbone,
            shallow_hidden_dim = shallow_hidden_dim,
            shallow_out_dim = text_encoder.get_input_embeddings().embedding_dim,
        )
        
        
        def forward(self, img):
            pass
        