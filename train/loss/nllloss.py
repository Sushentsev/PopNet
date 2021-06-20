from typing import Optional
from torch import nn, Tensor
from train.loss.base import Loss
from train.preprocess.seq2seq.lang import PAD_TOKEN_IDX


class NLLLoss(Loss):
    def __init__(self,
                 mask: Optional[Tensor] = None,
                 weight: Optional[Tensor] = None,
                 ignore_index: int = PAD_TOKEN_IDX,
                 reduction: str = "mean"):
        if mask is not None:
            if weight is not None:
                raise ValueError("Must provide weight with a mask.")
            weight[mask] = 0

        super().__init__("NLLLoss", nn.NLLLoss(weight=weight,
                                               reduction=reduction,
                                               ignore_index=ignore_index))
        self.__reduction = reduction

    def get_loss(self) -> float:
        if isinstance(self._acc_loss, float):
            return 0.

        loss = self._acc_loss.data.item()
        if self.__reduction == "mean":
            loss /= self.__norm_term
        return loss

    def eval_batch(self, outputs: Tensor, target: Tensor):
        self._acc_loss += self.__criterion(outputs, target)
        self._norm_term += 1.
