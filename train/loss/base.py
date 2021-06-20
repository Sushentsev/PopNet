from abc import ABC, abstractmethod
from typing import Union
from torch import Tensor
from torch import nn


class Loss(ABC):
    def __init__(self, name: str, criterion: nn.Module):
        self.__name: str = name
        self.__criterion: nn.Module = criterion

        self.__acc_loss: Union[Tensor, float] = 0.
        self.__norm_term: float = 0.

    def reset(self):
        self.__acc_loss = 0.
        self.__norm_term = 0.

    @abstractmethod
    def get_loss(self):
        raise NotImplementedError

    @abstractmethod
    def eval_batch(self, outputs: Tensor, target: Tensor):
        raise NotImplementedError

    def backward(self):
        if isinstance(self.__acc_loss, float):
            raise ValueError("No loss to back propagate.")
        self.__acc_loss.backward()
