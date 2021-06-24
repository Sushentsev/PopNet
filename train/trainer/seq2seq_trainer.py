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
    def __init__(self, loss: Loss, batch_size: int, device,
                 acc_steps: int = 1, random_seed: int = 42):
        self.__loss = loss
        self.__evaluator = Evaluator(loss, batch_size, device)
        self.__batch_size = batch_size
        self.__optimizer = None
        self.__device = device

        self.__acc_steps = acc_steps
        self.__random_seed = random_seed

        self.__logger = logging.getLogger(__file__)

    def __train_batch(self, step: int,
                      src: Tensor, src_lens: Tensor,
                      trg: Tensor, trg_lens: Tensor,
                      model: Seq2Seq, teacher_forcing_ratio: float):
        loss = self.__loss
        logits = model.forward(src, src_lens, trg, trg_lens, teacher_forcing_ratio)
        # logits.shape = (seq_len, batch_size, vocab_size)

        loss.reset()

        expected = trg.permute(1, 0)[1:].contiguous().view(-1)
        actual = logits[1:].view(-1, logits.shape[2])

        # Gradients accumulation
        loss.eval_batch(actual, expected)
        loss.divide(self.__acc_steps)
        loss.backward()

        if step % self.__acc_steps == 0:
            self.__optimizer.step()
            model.zero_grad()

        return loss.get_loss()

    def __train_epochs(self,
                       model: Seq2Seq,
                       train_data: Seq2SeqDataset,
                       n_epochs: int,
                       dev_data: Optional[Seq2SeqDataset] = None,
                       teacher_forcing_ratio: float = 0.):
        save = True
        dataloader = get_dataloader(train_data, self.__batch_size, shuffle=True)
        step = 0
        accum_loss = 0.

        for epoch in range(1, n_epochs + 1):
            model.train()

            for i, batch in enumerate(dataloader):
                step += 1
                src, src_lens, trg, trg_lens = [tensor.to(self.__device) for tensor in batch]
                self.__logger.info(f"Epoch {epoch} of {n_epochs}, step {step}")

                loss = self.__train_batch(step, src, src_lens, trg, trg_lens, model, teacher_forcing_ratio)
                accum_loss += loss

                if step % self.__acc_steps == 0:
                    wandb.log({"Train loss": accum_loss, "epoch": epoch})
                    accum_loss = 0.

            if dev_data is not None:
                dev_loss = self.__evaluator.evaluate(model, dev_data)

                wandb.log({"Dev loss": dev_loss, "epoch": epoch})
                self.__logger.info(f"Dev loss: {round(dev_loss, 4)}")

                model.train()

            if save:
                torch.save(model.state_dict(), f"weight/s2s_epoch{epoch}")

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
