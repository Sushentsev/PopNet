from abc import ABC, abstractmethod
from typing import Union
from torch import Tensor
from torch.nn import Module


class Loss(ABC):
    def __init__(self, name: str, criterion: Module):
        self.name: str = name
        self.criterion: Module = criterion

        self.acc_loss: Union[Tensor, float] = 0.
        self.norm_term: float = 0.

    def reset(self):
        self.acc_loss = 0.
        self.norm_term = 0.

    @abstractmethod
    def get_loss(self):
        raise NotImplementedError

    @abstractmethod
    def eval_batch(self, outputs: Tensor, target: Tensor):
        raise NotImplementedError

    def backward(self):
        if isinstance(self.acc_loss, float):
            raise ValueError("No loss to back propagate.")
        self.acc_loss.backward()
