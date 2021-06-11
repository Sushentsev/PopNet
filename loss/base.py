from torch import nn, Tensor


class Loss:
    def __init__(self, name: str, criterion):
        self.name = name
        self.criterion = criterion
        self.acc_loss = 0.
        self.norm_term = 0.

    def reset(self):
        self.acc_loss = 0.
        self.norm_term = 0.

    def eval_batch(self, outputs: Tensor, target: Tensor):
        raise NotImplementedError
