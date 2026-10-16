"""Mô hình huấn luyện: ViT đóng băng (upar.vision) -> Adapter + 3 dải + đối chiếu (upar.head) với 80 text vector (CoOp)."""
import torch
import torch.nn as nn

from .head import PartHead, compute_rows
from .vision import PatchEncoder
from .upar_train.losses import GradReverse
from .upar_train.text import PromptLearner, PromptTextEncoder


class PartCLIPPAR(nn.Module):
    def __init__(self, clip_model, c, pairs, part_of_attr, n_domains):
        super().__init__()
        self.grid = (c.img_h // 16, c.img_w // 16)
        self.rows = compute_rows(self.grid[0], c.part_cuts)          # Đầu [0,r0) | Thân [r0,r1) | Chân [r1,gh)
        dev = clip_model.positional_embedding.device
        self.visual = PatchEncoder.from_clip_visual_state(clip_model.visual.state_dict(), self.grid).to(dev)
        self.prompt = PromptLearner(clip_model, pairs, c.n_ctx, c.ctx_mode, c.ctx_init)
        self.text = PromptTextEncoder(clip_model, self.prompt.L, c.grad_ckpt)
        D = clip_model.text_projection.shape[1]
        self.head = PartHead(D, self.grid, self.rows, part_of_attr, c.adapter_reduction, c.init_scale)
        self.domain_head = (nn.Sequential(nn.Linear(3 * D, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, n_domains))
                            if (c.use_grl and n_domains > 1) else None)

    # --- nhánh ảnh: ViT(freeze) -> Adapter -> 3 part-vectors (+ global) ---
    def image_parts(self, images):
        return self.head.parts_from_tokens(self.visual(images))

    # --- nhánh văn bản: soft prompt -> Text Encoder(freeze) -> 80 text vectors [A, 2, 512] ---
    def text_feats(self):
        f = self.text(self.prompt(), self.prompt.eot_idx).float()
        return torch.nn.functional.normalize(f, dim=-1).view(self.prompt.A, 2, -1)

    def logits(self, parts, glob, tfeat):
        return self.head.logits(parts, glob, tfeat)

    def domain_logits(self, parts, lambd):
        return self.domain_head(GradReverse.apply(parts.flatten(1), lambd))

    # --- nạp trọng số trainable (tương thích checkpoint của bản cũ: adapter.* / log_tau -> head.adapter.* / head.log_tau) ---
    LEGACY_PREFIX = (("adapter.", "head.adapter."), ("log_tau", "head.log_tau"))

    def load_trainable(self, sd):
        fixed = {}
        for k, v in sd.items():
            for old, new in self.LEGACY_PREFIX:
                if k.startswith(old):
                    k = new + k[len(old):]
                    break
            fixed[k] = v
        _, unexpected = self.load_state_dict(fixed, strict=False)
        need = {n for n, p in self.named_parameters() if p.requires_grad}
        missing = sorted(need - set(fixed))
        if unexpected or missing:
            raise RuntimeError("Checkpoint không khớp kiến trúc hiện tại. Key lạ: %s | thiếu: %s. "
                               "Đặt cfg.resume=False (best.pt cũ sẽ được sao lưu) hoặc dùng out_dir khác." % (unexpected[:5], missing[:5]))
        return fixed
