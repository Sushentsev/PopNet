from typing import Optional, Tuple
from torch import nn, Tensor

from train.models.seq2seq.decoder import Decoder
from train.models.seq2seq.encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device):
        super().__init__()
        self.__encoder = encoder
        self.__decoder = decoder
        self.__device = device

    def forward(self,
                src: Tensor, src_lens: Tensor,
                trg: Optional[Tensor] = None, trg_lens: Optional[Tensor] = None,
                teacher_forcing_ratio: float = 0.) -> Tensor:
        """
        Gets source texts and returns logits.
        :param src: shape (batch_size, seq_len)
        :param src_lens: shape (batch_size)
        :param trg: shape (batch_size, seq_len)
        :param trg_lens: shape (batch_size)
        :param teacher_forcing_ratio: float
        :return: logits, shape (trg_len, batch_size, vocab_size)
        """
        h_n, c_n = self.__encoder(src, src_lens)
        outputs = self.__decoder(h_n, c_n, trg, trg_lens, teacher_forcing_ratio)
        return outputs
