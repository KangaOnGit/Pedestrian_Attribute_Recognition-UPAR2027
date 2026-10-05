"""CLIP ViT-B/16 (vision tower) viết lại gọn, cùng TÊN THAM SỐ với OpenAI CLIP nên nạp thẳng state_dict `visual.*`.
Chỉ cần torch => chạy được trong môi trường chấm bài (không có gói `clip`, không có mạng).
Dùng chung cho huấn luyện (notebook) và suy luận (run.py) để hai bên luôn cho cùng kết quả."""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln_1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads)      # chỉ để giữ đúng tên/shape tham số; forward dùng SDPA bên dưới
        self.ln_2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(d, 4 * d)), ("gelu", QuickGELU()),
                                              ("c_proj", nn.Linear(4 * d, d))]))

    def forward(self, x):                                # x: [B, L, D]
        B, L, D = x.shape
        a = self.attn
        q, k, v = F.linear(self.ln_1(x), a.in_proj_weight, a.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, L, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)
        x = x + a.out_proj(o.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, d, layers, heads):
        super().__init__()
        self.resblocks = nn.ModuleList([ResBlock(d, heads) for _ in range(layers)])

    def forward(self, x):
        for blk in self.resblocks:
            x = blk(x)
        return x


class PatchEncoder(nn.Module):
    """ViT đóng băng -> TOÀN BỘ patch tokens (bỏ [CLS]) đã chiếu vào không gian chung (512-d), shape [B, gh*gw, 512].
    Nếu lưới ≠ lưới gốc (14x14 với ảnh 224) thì nội suy bicubic positional embedding."""

    def __init__(self, grid_hw=(16, 16), width=768, layers=12, out_dim=512, patch=16, base_grid=14):
        super().__init__()
        self.grid = (int(grid_hw[0]), int(grid_hw[1]))
        self.base_grid = int(base_grid)
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
        s = width ** -0.5
        self.class_embedding = nn.Parameter(s * torch.randn(width))
        self.positional_embedding = nn.Parameter(s * torch.randn(self.base_grid ** 2 + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, width // 64)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(s * torch.randn(width, out_dim))
        self.register_buffer("pos_embed", torch.zeros(self.grid[0] * self.grid[1] + 1, width), persistent=False)
        self.refresh_pos_embed()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def refresh_pos_embed(self):
        """Tính positional embedding cho lưới hiện tại (luôn tính trên CPU fp32 để kết quả giống nhau mọi nơi)."""
        pos = self.positional_embedding.detach().float().cpu()
        cls_pos, patch_pos = pos[:1], pos[1:]
        n = self.base_grid
        patch_pos = patch_pos.reshape(1, n, n, -1).permute(0, 3, 1, 2)
        if (n, n) != self.grid:
            patch_pos = F.interpolate(patch_pos, size=self.grid, mode="bicubic", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(self.grid[0] * self.grid[1], -1)
        self.pos_embed.copy_(torch.cat([cls_pos, patch_pos], 0))

    @classmethod
    def from_clip_visual_state(cls, sd, grid_hw):
        """sd: state_dict của `clip_model.visual` (hoặc các key 'visual.*' đã bỏ tiền tố), fp16 hay fp32 đều được."""
        width, patch = sd["conv1.weight"].shape[0], sd["conv1.weight"].shape[-1]
        out_dim = sd["proj"].shape[1]
        layers = len({k.split(".")[2] for k in sd if k.startswith("transformer.resblocks.")})
        base = int(round((sd["positional_embedding"].shape[0] - 1) ** 0.5))
        m = cls(grid_hw, width, layers, out_dim, patch, base)
        m.load_state_dict({k: v.detach().float().cpu() for k, v in sd.items()}, strict=True)
        m.refresh_pos_embed()
        return m

    @torch.no_grad()          # không lưu activation => tiết kiệm VRAM (nhánh ảnh không cần gradient checkpointing)
    def forward(self, x):
        x = self.conv1(x)                                                    # [B, 768, gh, gw]
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)                                     # [B, N, 768]
        cls = self.class_embedding.to(x.dtype).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed.to(x.dtype)
        x = self.transformer(self.ln_pre(x))
        x = self.ln_post(x) @ self.proj                                      # [B, 1+N, 512]
        return x[:, 1:]
