import random
import numpy as np
import torch
import wandb
import logging

from typing import Optional
from torch import Tensor, optim
from torch.optim import Optimizer
from train.dataset.seq2seq_dataset import Seq2SeqDataset
from train.evaluator.evaluator import Evaluator
from train.loss.base import Loss
from train.models.seq2seq.seq2seq_model import Seq2Seq
from train.trainer.dataloaders import get_dataloader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Seq2SeqTrainer:
    def __init__(self, loss: Loss, batch_size: int, device, random_seed: int = 42):
        self.__loss = loss
        self.__evaluator = Evaluator(loss, batch_size, device)
        self.__batch_size = batch_size
        self.__optimizer = None
        self.__device = device
        self.__random_seed = random_seed

        self.__logger = logging.getLogger(__file__)

    def __train_batch(self,
                      src: Tensor, src_lens: Tensor,
                      trg: Tensor, trg_lens: Tensor,
                      model: Seq2Seq, teacher_forcing_ratio: float):
        loss = self.__loss
        logits = model.forward(src, src_lens, trg, trg_lens, teacher_forcing_ratio)
        # logits.shape = (seq_len, batch_size, vocab_size)

        loss.reset()

        expected = trg.permute(1, 0)[1:].contiguous().view(-1)
        actual = logits[1:].view(-1, logits.shape[2])
        loss.eval_batch(actual, expected)

        model.zero_grad()
        loss.backward()
        self.__optimizer.step()
        return loss.get_loss()

    def __train_epochs(self,
                       model: Seq2Seq,
                       train_data: Seq2SeqDataset,
                       n_epochs: int,
                       dev_data: Optional[Seq2SeqDataset] = None,
                       teacher_forcing_ratio: float = 0.):

        dataloader = get_dataloader(train_data, self.__batch_size, shuffle=True)

        for epoch in range(1, n_epochs + 1):
            model.train()

            epoch_total_loss = 0.
            epoch_steps = 0
            for i, batch in enumerate(dataloader):
                self.__logger.info(f"Epoch {epoch} of {n_epochs}, step {i} of {len(dataloader)}")
                src, src_lens, trg, trg_lens = [tensor.to(self.__device) for tensor in batch]

                loss = self.__train_batch(src, src_lens, trg, trg_lens, model, teacher_forcing_ratio)
                wandb.log({"Train loss": loss, "epoch": epoch})

                epoch_steps += 1
                epoch_total_loss += loss

            wandb.log({"Train average epoch loss": epoch_total_loss / epoch_steps, "epoch": epoch})
            self.__logger.info(f"Train loss: {round(epoch_total_loss / epoch_steps, 4)}")

            if dev_data is not None:
                dev_loss = self.__evaluator.evaluate(model, dev_data)

                wandb.log({"Dev loss": dev_loss, "epoch": epoch})
                self.__logger.info(f"Dev loss: {round(dev_loss, 4)}")

                model.train()

    def train(self,
              model: Seq2Seq,
              train_data: Seq2SeqDataset,
              n_epochs: int,
              dev_data: Optional[Seq2SeqDataset] = None,
              optimizer: Optional[Optimizer] = None,
              teacher_forcing_ratio: float = 0.):

        set_seed(self.__random_seed)
        if optimizer is None:
            optimizer = optim.Adam(model.parameters())
        self.__optimizer = optimizer
        self.__train_epochs(model, train_data, n_epochs, dev_data, teacher_forcing_ratio)
