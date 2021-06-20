from logging import getLogger

import torch
from torch import nn

from train.dataset.seq2seq_dataset import Seq2SeqDataset
from train.loss.base import Loss
from train.trainer.seq2seq_trainer import get_dataloader


class Evaluator:
    def __init__(self, loss: Loss, batch_size: int, device):
        self.__loss = loss
        self.__batch_size = batch_size
        self.__device = device

    def evaluate(self, model: nn.Module, dev_data: Seq2SeqDataset) -> float:
        model.eval()

        dev_loader = get_dataloader(dev_data, self.__batch_size, shuffle=False)
        steps = len(dev_loader)
        total_loss = 0.

        self.__loss.reset()

        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                src, src_lens, trg, trg_lens = [tensor.to(self.__device) for tensor in batch]

                out = model(src, src_lens, trg, trg_lens)




