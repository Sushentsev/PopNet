import random
from typing import Tuple, Optional

import torch
from torch import nn, Tensor

from train.models.seq2seq.embedding_layer import EmbeddingLayer
from utils import device


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size: int, embedding_dim: int, hidden_size: int,
                 sos_id: int, eos_id: int,
                 device, max_len: int, input_dropout: float = 0.,
                 padding_idx: Optional[int] = None,
                 embedding_weight: Optional[Tensor] = None,
                 update_embedding: bool = True):
        super().__init__()
        self.__vocab_size = vocab_size
        self.__sos_id = sos_id
        self.__eos_id = eos_id
        self.__max_len = max_len
        self.__device = device
        self.__embedding = EmbeddingLayer(vocab_size, embedding_dim, padding_idx,
                                          embedding_weight, update_embedding)
        self.__decoder = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.__linear = nn.Linear(hidden_size, vocab_size)
        self.__input_dropout = nn.Dropout(input_dropout)

    def forward(self,
                encoder_h: Tensor, encoder_c: Tensor,
                trg: Optional[Tensor] = None, trg_lens: Optional[Tensor] = None,
                teacher_forcing_ratio: float = 0.) -> Tensor:
        """
        :param encoder_h: shape (batch_size, hidden_size)
        :param encoder_c: shape (batch_size, hidden_size)
        :param trg: shape (batch_size, seq_len)
        :param trg_lens: shape (batch_size)
        :param teacher_forcing_ratio: float
        :return: output logits, shape (max_len, batch_size, vocab_size)
        """
        if (teacher_forcing_ratio > 0.) and (trg is None or trg_lens is None):
            raise ValueError("Target should be not empty if teacher forcing ration grater than zero.")

        batch_size = encoder_h.shape[0]
        hidden, cell = encoder_h.unsqueeze(0), encoder_c.unsqueeze(0)
        # hidden.shape = cell.shape = (1, batch_size, hidden_size)

        if trg is not None:
            input = trg[:, 0].view(batch_size, 1)  # <sos> tokens
            max_len = trg.shape[1]
        else:
            input = torch.LongTensor([self.__sos_id] * batch_size).view(batch_size, 1).to(self.__device)
            max_len = self.__max_len

        input = self.__input_dropout(self.__embedding(input))  # -> (batch_size, 1, embedding_dim)
        outputs = torch.zeros(max_len, batch_size, self.__vocab_size).to(self.__device)

        for t in range(1, max_len):
            output, (hidden, cell) = self.__decoder(input, (hidden, cell))
            # output.shape =              (batch_size, 1, hidden_size)
            # hidden.shape = cell.shape = (1, batch_size, hidden_size)
            logits = self.__linear(output.squeeze(1))  # -> (batch_size, vocab_size)
            outputs[t] = logits

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(axis=1)  # -> (batch_size)

            input = trg[:, t] if teacher_force else top1  # -> (batch_size)
            input = self.__input_dropout(self.__embedding(input.view(batch_size, 1)))

        return outputs
