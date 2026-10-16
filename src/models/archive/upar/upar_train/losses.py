"""Hàm mất mát ASL và Gradient Reversal Layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer: xuôi = identity, ngược = -λ·grad."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lambd * grad_out, None


class AsymmetricLoss(nn.Module):
    """ASL (Ridnik et al.): phạt lệch (γ₋ > γ₊) + dịch xác suất (clip) cho mẫu âm. Có mask cho nhãn chưa gán (-1)."""

    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gn, self.gp, self.clip, self.eps = gamma_neg, gamma_pos, clip, eps

    def forward(self, logits, target, mask):
        x = logits.float()
        p = torch.sigmoid(x)
        p_neg = (1.0 - p + self.clip).clamp(max=1.0) if self.clip > 0 else 1.0 - p
        loss = target * F.logsigmoid(x) + (1.0 - target) * torch.log(p_neg.clamp(min=self.eps))
        with torch.no_grad():                                               # trọng số focusing không truyền gradient
            pt = p * target + p_neg * (1.0 - target)                        # như bản gốc: p_neg đã dịch (clip)
            w = torch.pow(1.0 - pt, self.gp * target + self.gn * (1.0 - target))
        return -(loss * w * mask).sum() / mask.sum().clamp(min=1.0)
