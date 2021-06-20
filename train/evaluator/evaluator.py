from logging import getLogger

import torch
from torch import nn

from train.dataset.seq2seq_dataset import Seq2SeqDataset
from train.loss.base import Loss
from train.models.seq2seq.seq2seq_model import Seq2Seq
from train.trainer.dataloaders import get_dataloader


class Evaluator:
    def __init__(self, loss: Loss, batch_size: int, device):
        self.__loss = loss
        self.__batch_size = batch_size
        self.__device = device

    def evaluate(self, model: Seq2Seq, dev_data: Seq2SeqDataset) -> float:
        model.eval()

        dev_loader = get_dataloader(dev_data, self.__batch_size, shuffle=False)
        self.__loss.reset()

        with torch.no_grad():
            for batch in dev_loader:
                src, src_lens, trg, trg_lens = [tensor.to(self.__device) for tensor in batch]
                logits = model(src, src_lens, teacher_forcing_ratio=0.)  # -> (max_len, batch_size, vocab_size)

                expected = trg.permute(1, 0)[1:].contiguous().view(-1)
                actual = logits[1:trg.shape[1]].view(-1, logits.shape[2])

                self.__loss.eval_batch(actual, expected)

        return self.__loss.get_loss()
