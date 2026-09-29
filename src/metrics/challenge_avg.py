from src.metrics.instance_based_metrics import instance_metrics
from src.metrics.label_based_metrics import label_mean_acc
from src.metrics.base import label_based_c_matrix, instance_based_c_matrix

def chal_avg(
    lb_cm: label_based_c_matrix,
    i_cm: instance_based_c_matrix,
) -> float:
    """
    Compute challenge average
    Args:
        lb_cm (label_based_c_matrix): label-based confusion matrix
        i_cm (instance_based_c_matrix): instance-based confusion matrix

    Returns:
        float: challenge average
    """
    
    mA: float = label_mean_acc(lb_cm)
    i_F1: float = instance_metrics(i_cm).acc
    
    if (mA + i_F1 == 0):
        print(f"Division by 0")
    
    return (2.0 * mA * i_F1) / (mA + i_F1)