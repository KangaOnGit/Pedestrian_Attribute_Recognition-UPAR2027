import torch
from jaxtyping import Bool, Float, Int

from src.metrics.base import (
    label_based_c_matrix,
    instance_based_c_matrix,
)


def conf_matrix(
    pred: Float[torch.Tensor, "B K"],
    gt: Int[torch.Tensor, "B K"],
    threshold: float = 0.5,
) -> tuple[label_based_c_matrix, instance_based_c_matrix]:
    """
    Calculate label-based and instance-based TP, FP, TN, FN.

    Args:
        pred: Model probability predictions with shape [B, K]
        gt: Ground-truth binary labels with shape [B, K]
        threshold: Probability threshold used to convert predictions
                   to binary predictions.

    K is the number of attributes
    B is the number of instances/images/batch
    """

    pred_bool: Bool[torch.Tensor, "B K"] = pred > threshold
    gt_bool: Bool[torch.Tensor, "B K"] = gt.bool()

    # Label-based
    lb_tp = (gt_bool & pred_bool).sum(
        dim=0,
        dtype=torch.float64,
    )

    lb_fp = (~gt_bool & pred_bool).sum(
        dim=0,
        dtype=torch.float64,
    )

    lb_fn = (gt_bool & ~pred_bool).sum(
        dim=0,
        dtype=torch.float64,
    )

    lb_tn = (~gt_bool & ~pred_bool).sum(
        dim=0,
        dtype=torch.float64,
    )

    # Instance-based confusion matrix
    i_tp = (gt_bool & pred_bool).sum(
        dim=1,
        dtype=torch.float64,
    )

    i_fp = (~gt_bool & pred_bool).sum(
        dim=1,
        dtype=torch.float64,
    )

    i_fn = (gt_bool & ~pred_bool).sum(
        dim=1,
        dtype=torch.float64,
    )

    i_tn = (~gt_bool & ~pred_bool).sum(
        dim=1,
        dtype=torch.float64,
    )

    lb_c_matrix = label_based_c_matrix(
        tp=lb_tp,
        fp=lb_fp,
        fn=lb_fn,
        tn=lb_tn,
    )

    i_c_matrix = instance_based_c_matrix(
        tp=i_tp,
        fp=i_fp,
        fn=i_fn,
        tn=i_tn,
    )

    return lb_c_matrix, i_c_matrix