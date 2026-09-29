from dataclasses import dataclass

import torch
from jaxtyping import Float


@dataclass
class label_based_c_matrix:
    tp: Float[torch.Tensor, "K"]
    tn: Float[torch.Tensor, "K"]
    fp: Float[torch.Tensor, "K"]
    fn: Float[torch.Tensor, "K"]


@dataclass
class instance_based_c_matrix:
    tp: Float[torch.Tensor, "B"]
    tn: Float[torch.Tensor, "B"]
    fp: Float[torch.Tensor, "B"]
    fn: Float[torch.Tensor, "B"]
    
@dataclass
class i_metrics:
    prec: float
    recall: float
    f1: float
    acc: float
    
@dataclass
class eval_metrics:
    avg: float
    mA: float
    label_f1: float
    inst_acc: float
    inst_prec: float
    inst_rec: float
    inst_f1: float
