from typing import Tuple

from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence


class Decoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, dropout: float = 0.5):
        super().__init__()

        self.vocab_size = vocab_size
        self.decoder = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               batch_first=False,
                               dropout=dropout)

        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, seqs: Tensor, h_n: Tensor, c_n: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Decoder in seq2seq architecture.
        Decoder gets batch of embedded sequences and returns batch of output sequences.

        Available 2 options of training: predict recursively or teacher forcing.

        :param seqs: embedded sequences, shape (batch_size, input_size)
        :param h_n: encoder last hidden states, shape (batch_size, hidden_size)
        :param c_n: encoder last cell states, shape (batch_size, hidden_size)
        :return: logits, shape (batch_size, vocab_size);
        hidden states, shape (batch_size, hidden_size);
        cell states, shape (batch_size, hidden_size)
        """

        seqs = seqs.unsqueeze(0)  # -> (1, batch_size, input_size)
        h_n = h_n.unsqueeze(0)  # -> (1, batch_size, hidden_size)
        c_n = c_n.unsqueeze(0)  # -> (1, batch_size, hidden_size)

        output, (h_n, c_n) = self.decoder(seqs, (h_n, c_n))
        # output.shape = (1, batch_size, hidden_size)
        # h_n.shape = c_n.shape = (1, batch_size, hidden_size)

        logits = self.linear(output.squeeze(0))  # -> (batch_size, vocab_size)
        h_n = h_n.squeeze(0)  # -> (batch_size, hidden_size)
        c_n = c_n.squeeze(0)  # -> (batch_size, hidden_size)

        return logits, h_n, c_n
