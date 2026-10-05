"""Metric PAR (mA, instance F1, HM) và hiệu chuẩn hậu kỳ (temperature + bias, offset)."""
import numpy as np
import torch
import torch.nn.functional as F

from ..head import apply_calibration, sigmoid_np   # noqa: F401  (dùng chung với suy luận)


def par_metrics(prob, y, thr=0.5):
    """mA (trung bình TPR/TNR theo thuộc tính), các chỉ số instance-based (Acc/Prec/Rec/F1) và HM = harmonic mean(mA, F1_inst).
    Nhãn y ∈ {0,1,-1}; nhãn -1 bị loại khỏi mọi phép tính."""
    eps = 1e-20
    prob, y = np.asarray(prob), np.asarray(y)
    m = (y == 0) | (y == 1)
    g = (y == 1) & m
    pr = (prob >= thr) & m
    tp, fp, fn = (pr & g).sum(0).astype(float), (pr & ~g).sum(0).astype(float), (~pr & g).sum(0).astype(float)
    tn = (~pr & ~g & m).sum(0).astype(float)
    P, N = g.sum(0), (m & ~g).sum(0)
    valid = (P > 0) & (N > 0)
    tpr, tnr = tp / np.maximum(P, 1), tn / np.maximum(N, 1)
    mA = float(((tpr + tnr) / 2)[valid].mean()) if valid.any() else 0.0
    f1_attr = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    inter, pc, gc, uni = (pr & g).sum(1), pr.sum(1), g.sum(1), (pr | g).sum(1)
    acc = float((inter / (uni + eps)).mean())
    prec, rec = float((inter / (pc + eps)).mean()), float((inter / (gc + eps)).mean())
    f1 = 2 * prec * rec / (prec + rec + eps)
    hm = 2 * mA * f1 / (mA + f1 + eps)
    return dict(mA=mA, F1=float(f1), Acc=acc, Prec=prec, Rec=rec, HM=float(hm),
                mF1_attr=float(f1_attr[valid].mean()) if valid.any() else 0.0, n_valid=int(valid.sum()),
                per_attr_mA=(tpr + tnr) / 2, per_attr_F1=f1_attr)


def fmt_metrics(m):
    return f"HM={m['HM']:.4f} | mA={m['mA']:.4f} | F1_inst={m['F1']:.4f} | Acc_inst={m['Acc']:.4f} | P={m['Prec']:.4f} R={m['Rec']:.4f}"


# ---------------- Hiệu chuẩn hậu kỳ: temperature + bias theo từng thuộc tính ----------------
def fit_calibrator(z, y, iters=100, l2_a=1e-2, l2_b=1e-3, balanced=True, min_count=5):
    """Tìm a_i, b_i để z' = exp(a_i)·z + b_i (T_i = exp(-a_i)); tối thiểu hoá BCE (mặc định cân bằng 2 lớp,
    để ngưỡng 0.5 sau hiệu chuẩn gần điểm tối ưu của mA). Thuộc tính quá ít mẫu -> giữ nguyên (a=b=0)."""
    z = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y))
    m = ((y == 0) | (y == 1)).float()
    t = (y == 1).float()
    P, N = (t * m).sum(0), ((1 - t) * m).sum(0)
    if balanced:
        W = (t / P.clamp(min=1) + (1 - t) / N.clamp(min=1)) * 0.5 * m            # mỗi lớp đóng góp 50% trọng số
    else:
        W = m / (P + N).clamp(min=1)
    a = torch.zeros(z.shape[1], requires_grad=True)
    b = torch.zeros(z.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([a, b], lr=0.5, max_iter=iters, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        zc = a.exp() * z + b
        loss = (F.binary_cross_entropy_with_logits(zc, t, reduction="none") * W).sum() + l2_a * (a ** 2).sum() + l2_b * (b ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    usable = (P >= min_count) & (N >= min_count)
    a = torch.where(usable, a.detach().clamp(-1.5, 2.0), torch.zeros_like(a))
    b = torch.where(usable, b.detach(), torch.zeros_like(b))
    return a.numpy(), b.numpy()


def refine_offsets(z, y, grid=np.linspace(-2.5, 2.5, 21), rounds=2):
    """Coordinate-ascent: chọn offset logit cho từng thuộc tính để tối đa HM(mA, F1_inst). Cập nhật F1_inst tăng dần nên rất nhanh."""
    z, y = np.asarray(z), np.asarray(y)
    n, A = z.shape
    m = (y == 0) | (y == 1)
    g = (y == 1) & m
    Pn, Nn = g.sum(0), (m & ~g).sum(0)
    valid = (Pn > 0) & (Nn > 0)
    gt_cnt = g.sum(1).astype(float)
    off = np.zeros(A)
    pred = (z > 0) & m
    pc, it = pred.sum(1).astype(float), (pred & g).sum(1).astype(float)

    def per_attr_mA(pcol, i):
        tpr = (pcol & g[:, i]).sum() / max(Pn[i], 1)
        tnr = (~pcol & m[:, i] & ~g[:, i]).sum() / max(Nn[i], 1)
        return (tpr + tnr) / 2

    mA_i = np.array([per_attr_mA(pred[:, i], i) if valid[i] else 0.0 for i in range(A)])
    for _ in range(rounds):
        for i in range(A):
            if not valid[i]:
                continue
            base_pc, base_it = pc - pred[:, i], it - (pred[:, i] & g[:, i])
            sum_other = mA_i.sum() - mA_i[i]
            best_s, best_o, best_pred, best_mA = -1.0, off[i], pred[:, i], mA_i[i]
            for o in sorted(grid, key=abs):                              # hoà điểm -> ưu tiên offset nhỏ
                pcol = ((z[:, i] + o) > 0) & m[:, i]
                cpc, cit = base_pc + pcol, base_it + (pcol & g[:, i])
                Pm, Rm = (cit / (cpc + 1e-20)).mean(), (cit / (gt_cnt + 1e-20)).mean()
                f1 = 2 * Pm * Rm / (Pm + Rm + 1e-20)
                mA_new = per_attr_mA(pcol, i)
                mA = (sum_other + mA_new) / valid.sum()
                s = 2 * mA * f1 / (mA + f1 + 1e-20)
                if s > best_s + 1e-9:
                    best_s, best_o, best_pred, best_mA = s, o, pcol, mA_new
            off[i], pred[:, i], mA_i[i] = best_o, best_pred, best_mA
            pc, it = base_pc + best_pred, base_it + (best_pred & g[:, i])
    return off
