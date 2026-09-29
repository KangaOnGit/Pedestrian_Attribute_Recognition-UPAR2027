import torch
from jaxtyping import Float, Int
from src.metrics.confusion_matrix import conf_matrix
from src.metrics.instance_based_metrics import instance_metrics
from src.metrics.label_based_metrics import label_mean_acc, label_mean_f1
from src.metrics.challenge_avg import chal_avg
from src.metrics.base import eval_metrics
import logging

log = logging.getLogger(__name__)

def run_metrics(
    pred: Float[torch.Tensor, "B K"],
    gt: Int[torch.Tensor, "B K"],
    threshold: float = 0.5,
) -> eval_metrics:
    
    lbcm, icm = conf_matrix(pred, gt, threshold=threshold)

    label_ma = label_mean_acc(lbcm)
    label_f1 = label_mean_f1(lbcm)

    instance = instance_metrics(icm)
    
    challenge_avg = chal_avg(lbcm, icm)
    
    log.info("Challenge average: %.4f", challenge_avg)
    log.info("Label mA: %.4f", label_ma)
    log.info("Label F1: %.4f", label_f1)
    log.info("Instance Accuracy: %.4f", instance.acc)
    log.info("Instance Precision: %.4f", instance.prec)
    log.info("Instance Recall: %.4f", instance.recall)
    log.info("Instance F1: %.4f", instance.f1)
    
    return eval_metrics(
        avg = challenge_avg,
        mA = label_ma,
        label_f1 = label_f1,
        inst_acc = instance.acc,
        inst_prec = instance.prec,
        inst_rec = instance.recall,
        inst_f1 = instance.f1,
    )