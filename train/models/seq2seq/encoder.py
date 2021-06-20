from typing import Tuple, Optional, List

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from train.models.seq2seq.embedding_layer import EmbeddingLayer


class Encoder(nn.Module):
    """
    Bidirectional RNN encoder.
    Final hidden states cell states result from concatenations of
    hidden states and cell states in each direction.
    """
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_size: int,
                 device,
                 input_dropout: float = 0.,
                 padding_idx: Optional[int] = None,
                 embedding_weight: Optional[Tensor] = None,
                 update_embedding: bool = True):
        super().__init__()
        self.__embedding = EmbeddingLayer(vocab_size, embedding_dim, padding_idx,
                                          embedding_weight, update_embedding)
        self.__encoder = nn.LSTM(embedding_dim, hidden_size, bidirectional=True, batch_first=True)
        self.__input_dropout = nn.Dropout(input_dropout)
        self.__device = device

    def forward(self, seqs: Tensor, lens: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encoder gets batch of embedded sequences
        and returns batch of last hidden states and last cell states.

        :param seqs: embedded sequences, shape (batch_size, seq_len)
        :param lens: sequence lengths, shape (batch_size)
        :return: last hidden states, shape (batch_size, 2 * hidden_size);
        last cell states, shape (batch_size, 2 * hidden_size)
        """

        embedded = self.__embedding(seqs)  # -> (batch_size, seq_len, embedding_dim)
        embedded = self.__input_dropout(embedded)  # -> (batch_size, seq_len, embedding_dim)

        packed = pack_padded_sequence(embedded, lens, enforce_sorted=False, batch_first=True)
        _, (h_n, c_n) = self.__encoder(packed)
        # h_n.shape = (2, batch_size, hidden_size)
        # c_n.shape = (2, batch_size, hidden_size)

        h_n_forward, h_n_backward = h_n[0], h_n[1]
        c_n_forward, c_n_backward = c_n[0], c_n[1]

        h_n_cat = torch.cat((h_n_forward, h_n_backward), dim=1)  # -> (batch_size, 2 * hidden_size)
        c_n_cat = torch.cat((c_n_forward, c_n_backward), dim=1)  # -> (batch_size, 2 * hidden_size)

        return h_n_cat, c_n_cat
