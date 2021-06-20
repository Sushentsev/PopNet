from typing import Optional
from torch import nn, Tensor
from train.loss.base import Loss


class NLLLoss(Loss):
    def __init__(self,
                 mask: Optional[Tensor] = None,
                 weight: Optional[Tensor] = None,
                 reduction: str = "mean"):
        self.__reduction: str = reduction

        if mask is not None:
            if weight is not None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        super().__init__("NLLLoss", nn.NLLLoss(weight=weight, reduction=reduction))

    def get_loss(self) -> float:
        if isinstance(self.__acc_loss, float):
            return 0.

        loss = self.__acc_loss.data.item()
        if self.__reduction == "mean":
            loss /= self.__norm_term
        return loss

    def eval_batch(self, outputs: Tensor, target: Tensor):
        self.__acc_loss += self.__criterion(outputs, target)
        self.__norm_term += 1.
