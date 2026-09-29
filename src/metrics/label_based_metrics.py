from src.metrics.base import label_based_c_matrix
from src.metrics.utils import _safe_divide

def label_mean_acc(
    lb_cm: label_based_c_matrix,
) -> float:
    """
    Compute label-based mean accuracy
    
    Args:
        lb_cm (label_based_c_matrix): label-based confusion matrix

    Returns:
        float: label-based mean accuracy
    """

    tpr = _safe_divide(
        lb_cm.tp,
        lb_cm.tp + lb_cm.fn,
    )

    tnr = _safe_divide(
        lb_cm.tn,
        lb_cm.tn + lb_cm.fp,
    )

    attribute_acc = (tpr + tnr) / 2.0

    return attribute_acc.mean().item()
    
def label_mean_f1(
    lb_cm: label_based_c_matrix,
) -> float:
    """
    Compute label-based mean F1

    Args:
        lb_cm (label_based_c_matrix): label-based confusion matrix

    Returns:
        float: label-based mean F1
    """

    precision = _safe_divide(
        lb_cm.tp,
        lb_cm.tp + lb_cm.fp,
    )

    recall = _safe_divide(
        lb_cm.tp,
        lb_cm.tp + lb_cm.fn,
    )

    attribute_f1 = _safe_divide(
        2.0 * precision * recall,
        precision + recall,
    )

    return attribute_f1.mean().item()