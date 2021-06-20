from typing import Optional
from torch import Tensor, nn

from train.loss.base import Loss
from train.preprocess.seq2seq.lang import PAD_TOKEN_IDX


class CrossEntropyLoss(Loss):
    def __init__(self,
                 weight: Optional[Tensor] = None,
                 ignore_index: int = PAD_TOKEN_IDX,
                 reduction: str = "mean"):
        super().__init__("CrossEntropyLoss", nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index))
        self.__reduction = reduction

    def get_loss(self) -> float:
        if isinstance(self._acc_loss, float):
            return 0.

        loss = self._acc_loss.data.item()
        if self.__reduction == "mean":
            loss /= self._norm_term
        return loss

    def eval_batch(self, outputs: Tensor, target: Tensor):
        self._acc_loss += self._criterion(outputs, target)
        self._norm_term += 1.
