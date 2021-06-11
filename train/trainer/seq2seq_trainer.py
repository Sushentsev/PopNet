import random
from typing import Optional, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from train.evaluator.evaluator import Evaluator
from train.loss.base import Loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Seq2SeqTrainer:
    def __init__(self, loss: Loss, batch_size: int, random_seed: int = 42, device):
        self.loss: Loss = loss
        self.evaluator: Evaluator = Evaluator(self.loss, batch_size, device)
        self.optimizer: Optional[Optimizer] = None

        set_seed(random_seed)

    def _train_batch(self, model: Module, input_seqs: Tensor, input_lens: Tensor,
                     target_seqs: Tensor, teacher_forcing_ratio: float = 0.):
        raise NotImplementedError

    def _train_epoch(self, model: Module, train_data: Any,
                     test_data: Optional[Any] = None, teacher_forcing_ratio: float = 0.):
        raise NotImplementedError

    def train(self, model: Module, train_data: Any,
              n_epochs: int, test_data: Optional[Any] = None,
              optimizer: Optional[Optimizer] = None, teacher_forcing_ratio: float = 0.):
        raise NotImplementedError