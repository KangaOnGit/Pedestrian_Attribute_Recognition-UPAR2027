"""Nhánh văn bản (chỉ khi huấn luyện, cần gói OpenAI `clip`): CoOp soft prompt + Text Encoder đóng băng.
Khi xuất bài nộp, 80 text vector được tính sẵn nên run.py không cần phần này."""
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PromptLearner(nn.Module):
    """CoOp: context vector học được chèn giữa <SOS> và cụm từ mô tả.
    Thứ tự 80 prompt: [neg_0, pos_0, neg_1, pos_1, ...] (chỉ số 0 = phủ định, 1 = khẳng định)."""
    def __init__(self, clip_model, pairs, n_ctx, mode, ctx_init):
        super().__init__()
        assert mode in ("shared", "polarity", "csc")
        dev = clip_model.token_embedding.weight.device
        self.A, self.n_ctx, self.mode = len(pairs), n_ctx, mode
        texts = [t for (neg, pos) in pairs for t in (neg, pos)]
        holder = " ".join(["X"] * n_ctx)
        tok = clip.tokenize([f"{holder} {t}." for t in texts], truncate=True).to(dev)        # [M, 77]
        eot = tok.argmax(dim=-1)                                                              # vị trí <EOT>
        self.L = int(eot.max()) + 1                     # cắt phần padding: causal mask => vector tại <EOT> không đổi, nhanh hơn
        tok = tok[:, :self.L]
        with torch.no_grad():
            emb = clip_model.token_embedding(tok).float()                                     # [M, L, D]
        self.D = emb.shape[-1]
        self.register_buffer("eot_idx", eot, persistent=False)
        self.register_buffer("prefix", emb[:, :1].clone(), persistent=False)                  # <SOS>
        self.register_buffer("suffix", emb[:, 1 + n_ctx:].clone(), persistent=False)          # cụm từ + '.' + <EOT> + pad
        if ctx_init and len(ctx_init.split()) == n_ctx:
            with torch.no_grad():
                e = clip_model.token_embedding(clip.tokenize(ctx_init).to(dev)).float()
            init = e[0, 1:1 + n_ctx].clone()
        else:
            init = torch.randn(n_ctx, self.D, device=dev) * 0.02
        if mode == "shared":
            self.ctx = nn.Parameter(init)                                                     # [n_ctx, D]
        elif mode == "polarity":
            self.ctx = nn.Parameter(init[None].repeat(2, 1, 1))                               # [2, n_ctx, D]
        else:
            self.ctx = nn.Parameter(init[None, None].repeat(self.A, 2, 1, 1))                 # [A, 2, n_ctx, D]

    def forward(self):
        M = 2 * self.A
        if self.mode == "shared":
            ctx = self.ctx[None].expand(M, -1, -1)
        elif self.mode == "polarity":
            ctx = self.ctx[None].expand(self.A, -1, -1, -1).reshape(M, self.n_ctx, self.D)
        else:
            ctx = self.ctx.reshape(M, self.n_ctx, self.D)
        return torch.cat([self.prefix, ctx, self.suffix], dim=1)                              # [M, L, D]


class PromptTextEncoder(nn.Module):
    """Text Encoder đóng băng của CLIP nhận thẳng embedding (đã có soft prompt).
    - Attention tự viết bằng SDPA(is_causal=True): không đụng vào mask của CLIP gốc, không lo lệch dtype mask/q/k/v khi AMP.
    - Gradient checkpointing: chỉ lưu đầu vào mỗi block, tính lại khi backward."""
    def __init__(self, clip_model, ctx_len, grad_ckpt=True):
        super().__init__()
        self.blocks = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final, self.text_projection = clip_model.ln_final, clip_model.text_projection
        self.ctx_len, self.grad_ckpt = ctx_len, grad_ckpt
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _block(blk, x):                                                          # x: [M, L, D]
        a = blk.attn
        M, L, D = x.shape
        q, k, v = F.linear(blk.ln_1(x), a.in_proj_weight, a.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(M, L, a.num_heads, D // a.num_heads).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + a.out_proj(o.transpose(1, 2).reshape(M, L, D))
        return x + blk.mlp(blk.ln_2(x))

    def forward(self, prompt_embeds, eot_idx):
        x = prompt_embeds + self.positional_embedding[:self.ctx_len].to(prompt_embeds.dtype)
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(self._block, blk, x, use_reentrant=False)
            else:
                x = self._block(blk, x)
        x = self.ln_final(x)
        return x[torch.arange(x.shape[0], device=x.device), eot_idx] @ self.text_projection    # [M, 512]
