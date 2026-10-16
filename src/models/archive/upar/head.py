"""Phần 'đầu' của mô hình (Adapter -> 3 dải Đầu/Thân/Chân -> đối chiếu cosine có ràng buộc vùng) và hiệu chuẩn xác suất.
Dùng chung cho huấn luyện và suy luận."""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rows(grid_h, cuts):
    """Ranh giới hàng patch: Đầu [0,r0) | Thân [r0,r1) | Chân [r1,gh) theo tỉ lệ chiều cao `cuts`."""
    r0 = int(min(max(round(cuts[0] * grid_h), 1), grid_h - 2))
    r1 = int(min(max(round(cuts[1] * grid_h), r0 + 1), grid_h - 1))
    return r0, r1


class ResidualAdapter(nn.Module):
    """Adapter mỏng dim -> dim/reduction -> dim (residual, khởi tạo = identity nhờ up-proj bằng 0)."""

    def __init__(self, dim=512, reduction=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, dim // reduction)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // reduction, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.act(self.down(self.norm(x))))


def pool_three_parts(tokens, grid, rows):
    """tokens [B, N, D] trên lưới (gh, gw) -> 3 dải ngang [B, 3, D] (Đầu, Thân, Chân) bằng average pooling."""
    B, N, D = tokens.shape
    t = tokens.reshape(B, grid[0], grid[1], D)
    r0, r1 = rows
    return torch.stack([t[:, :r0].mean((1, 2)), t[:, r0:r1].mean((1, 2)), t[:, r1:].mean((1, 2))], dim=1)


class PartHead(nn.Module):
    def __init__(self, dim, grid, rows, part_of_attr, reduction=4, init_scale=100.0):
        super().__init__()
        self.grid, self.rows = (int(grid[0]), int(grid[1])), (int(rows[0]), int(rows[1]))
        self.adapter = ResidualAdapter(dim, reduction)
        self.log_tau = nn.Parameter(torch.full((len(part_of_attr),), math.log(init_scale)))   # 1/temperature theo từng cặp thuộc tính
        self.register_buffer("part_of_attr", torch.as_tensor(list(part_of_attr), dtype=torch.long), persistent=False)

    def parts_from_tokens(self, tok):
        """patch tokens [B, N, D] -> (3 part-vectors [B,3,D], global-vector [B,D]); đã L2-normalize."""
        tok = F.normalize(tok.float(), dim=-1)                 # chuẩn hoá từng token: triệt token có norm lớn bất thường
        tok = self.adapter(tok)                                # phần trainable duy nhất của nhánh ảnh
        parts = F.normalize(pool_three_parts(tok.float(), self.grid, self.rows), dim=-1)
        glob = F.normalize(parts.mean(1), dim=-1)              # vector toàn thân cho Age/Gender
        return parts, glob

    def logits(self, parts, glob, tfeat):
        """Đối chiếu có ràng buộc vùng: thuộc tính i chỉ so với part-vector part_of_attr[i].
        logit_i = τ_i·(cos(part, text₊) − cos(part, text₋))  (= logit của softmax 2 lớp có temperature)."""
        with torch.autocast(device_type=parts.device.type, enabled=False):      # cosine tính fp32 cho chính xác
            bank = torch.cat([parts.float(), glob.float()[:, None]], dim=1)     # [B, 4, D]: Đầu, Thân, Chân, Global
            v = bank[:, self.part_of_attr]                                      # [B, A, D]
            sim = torch.einsum("bad,akd->bak", v, tfeat.float())                # [B, A, 2] (0 = phủ định, 1 = khẳng định)
            return self.log_tau.float().exp() * (sim[..., 1] - sim[..., 0])


# ---------------- hiệu chuẩn (numpy) ----------------
def sigmoid_np(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def apply_calibration(z, a, b, off=None):
    """z' = exp(a_i)·z + b_i (+ off_i). T_i = exp(-a_i) là temperature theo từng thuộc tính."""
    zc = np.exp(np.asarray(a))[None] * np.asarray(z) + np.asarray(b)[None]
    return zc if off is None else zc + np.asarray(off)[None]
