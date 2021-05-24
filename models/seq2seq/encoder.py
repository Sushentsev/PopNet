from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs: Tensor, lens: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encoder in seq2seq architecture.
        Encoder gets batch of embedded sequences
        and returns batch of last hidden states and batch of last cell states.

        :param seqs: embedded sequences, shape (seq_len, batch_size, input_size)
        :param lens: sequence lengths, shape (batch_size)
        :return: last hidden states, shape (batch_size, 2 * hidden_size);
        last cell states, shape (batch_size, 2 * hidden_size)
        """

        # seqs = self.dropout(seqs)  # -> (batch_size, max_seq_len, input_size)
        packed = pack_padded_sequence(seqs, lens, enforce_sorted=False)
        _, (h_n, c_n) = self.encoder(packed)
        # h_n.shape = (2, batch_size, hidden_size)
        # c_n.shape = (2, batch_size, hidden_size)

        h_n_forward, h_n_backward = h_n[0], h_n[1]
        c_n_forward, c_n_backward = c_n[0], c_n[1]

        h_n_cat = torch.cat((h_n_forward, h_n_backward), dim=1)  # -> (batch_size, 2 * hidden_size)
        c_n_cat = torch.cat((c_n_forward, c_n_backward), dim=1)  # -> (batch_size, 2 * hidden_size)
        return h_n_cat, c_n_cat
