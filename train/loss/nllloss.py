from typing import Optional
from torch import nn, Tensor
from train.loss.base import Loss


class NLLLoss(Loss):
    def __init__(self, weight: Optional[Tensor] = None, reduction: str = "mean"):
        self.reduction: str = reduction
        super().__init__("NLLLoss", nn.NLLLoss(weight=weight, reduction=reduction))

    def get_loss(self) -> float:
        if isinstance(self.acc_loss, float):
            return 0.

        loss = self.acc_loss.data.item()
        if self.reduction == "mean":
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs: Tensor, target: Tensor):
        self.acc_loss += self.criterion(outputs, target)
        self.norm_term += 1.
