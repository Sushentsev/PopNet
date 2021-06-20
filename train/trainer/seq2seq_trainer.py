import random
from logging import getLogger
from typing import Optional, Any, List, Tuple

import numpy as np
import torch
from torch import Tensor, optim
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer
from torchtext.legacy.data import BucketIterator
from torch.utils.data import DataLoader

from train.dataset.seq2seq_dataset import Seq2SeqDataset
from train.evaluator.evaluator import Evaluator
from train.loss.base import Loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _collate_fn(batch: List[Tuple[List[int], List[int]]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    src_tensors = [torch.LongTensor(src) for src, _ in batch]
    trg_tensors = [torch.LongTensor(trg) for trg, _ in batch]

    src_lens = torch.LongTensor([len(src) for src in src_tensors])
    trg_lens = torch.LongTensor([len(trg) for trg in trg_tensors])

    src_padded = pad_sequence(src_tensors, batch_first=True)
    trg_padded = pad_sequence(trg_tensors, batch_first=True)

    return src_padded, src_lens, trg_padded, trg_lens


def get_dataloader(dataset: Seq2SeqDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size, shuffle, collate_fn=_collate_fn)


class Seq2SeqTrainer:
    def __init__(self, loss: Loss, batch_size: int, device, random_seed: int = 42):
        self.__loss = loss
        self.__evaluator = Evaluator(loss, batch_size, device)
        self.__batch_size = batch_size
        self.__optimizer = None
        self.__device = device
        self.__random_seed = random_seed

        self.__logger = getLogger(__name__)

    def __train_batch(self,
                      src: Tensor, src_lens: Tensor,
                      trg: Tensor, trg_lens: Tensor,
                      model: nn.Module, teacher_forcing_ratio: float):
        out = None # self.model()

        self.__loss.reset()

        model.zero_grad()
        self.__loss.backward()
        self.__optimizer.step()
        return self.__loss.get_loss()

    def __train_epochs(self,
                       model: nn.Module,
                       train_data: Seq2SeqDataset,
                       n_epochs: int,
                       dev_data: Optional[Seq2SeqDataset] = None,
                       teacher_forcing_ratio: float = 0.):

        dataloader = get_dataloader(train_data, self.__batch_size, shuffle=True)

        steps_per_epoch = len(dataloader)
        total_steps = steps_per_epoch * n_epochs
        current_step = 0

        for epoch in range(1, n_epochs + 1):
            epoch_total_loss = 0.
            self.__logger.info(f"Epoch: {epoch}, step: {current_step}")
            model.train()

            for i, batch in enumerate(dataloader):
                current_step += 1
                src, src_lens, trg, trg_lens = [tensor.to(self.__device) for tensor in batch]

                loss = self.__train_batch(src, src_lens, trg, trg_lens, model, teacher_forcing_ratio)
                epoch_total_loss += loss

            epoch_average_loss = epoch_total_loss / steps_per_epoch
            self.__logger.info(f"Finished epoch {epoch} with train average loss {round(epoch_average_loss, 4)}")

            if dev_data is not None:
                raise NotImplementedError

    def train(self,
              model: nn.Module,
              train_data: Any,
              n_epochs: int,
              dev_data: Optional[Any] = None,
              optimizer: Optional[Optimizer] = None,
              teacher_forcing_ratio: float = 0.):

        set_seed(self.__random_seed)
        if optimizer is None:
            optimizer = optim.Adam(model.parameters())
        self.__optimizer = optimizer
        self.__logger.info("Starting training...")

        self.__train_epochs(model, train_data, n_epochs, dev_data, teacher_forcing_ratio)
