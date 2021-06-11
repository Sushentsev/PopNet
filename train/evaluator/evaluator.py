import torch
from torch import nn


class Evaluator:
    def __init__(self, model: nn.Module, loss: nn.Module, batch_size: int, device):
        self.model = model
        self.loss = loss
        self.batch_size = batch_size
        self.device = device

    def evaluate(self, test_loader):
        raise NotImplementedError

