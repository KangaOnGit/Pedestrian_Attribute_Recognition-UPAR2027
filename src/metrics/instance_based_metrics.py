from src.metrics.base import instance_based_c_matrix, i_metrics
from src.metrics.utils import _safe_divide

def instance_metrics(
    i_cm: instance_based_c_matrix,
) -> i_metrics:
    """
    Compute UPAR instance-based metrics

    Metrics are computed independently for each image over its
    predicted and ground-truth attribute sets, then averaged
    across images

    Instance_Acc  = |P ∩ G| / |P ∪ G|
    Instance_Prec = |P ∩ G| / |P|
    Instance_Rec  = |P ∩ G| / |G|
    Instance_F1   = 2 * Prec * Rec / (Prec + Rec)
    
    Args:
        i_cm (instance_based_c_matrix): instance based confusion matrix

    Returns:
        i_metrics
    """

    intersection = i_cm.tp

    predicted_count = i_cm.tp + i_cm.fp

    ground_truth_count = i_cm.tp + i_cm.fn

    union = i_cm.tp + i_cm.fp + i_cm.fn

    instance_acc = _safe_divide(
        intersection,
        union,
    )

    instance_prec = _safe_divide(
        intersection,
        predicted_count,
    )

    instance_recall = _safe_divide(
        intersection,
        ground_truth_count,
    )

    instance_f1 = _safe_divide(
        2.0 * instance_prec * instance_recall,
        instance_prec + instance_recall,
    )

    return i_metrics(
        prec = instance_prec.mean().item(),
        recall = instance_recall.mean().item(),
        f1 = instance_f1.mean().item(),
        acc = instance_acc.mean().item()
    )