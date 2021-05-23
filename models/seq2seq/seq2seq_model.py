import torch
import random
from torch import nn, Tensor


# https://github.com/bentrevett/pytorch-seq2seq
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
class Seq2Seq(nn.Module):
    def __init__(self, embedding_layer: nn.Module, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source: Tensor, source_lens: Tensor, target: Tensor, teacher_forcing_ratio: float = 0.5) -> Tensor:
        """

        :param source: input sequences, shape (batch_size, max_seq_len)
        :param source_lens: input lens, shape (batch_size)
        :param target: target sequences, shape (batch_size, max_seq_len)
        :param teacher_forcing_ratio: teacher forcing ratio (ratio of ground-truth inputs)
        :return: outputs, shape (target_max_len, batch_size, vocab_size)
        """

        batch_size, target_max_len = target.shape
        vocab_size = self.decoder.vocab_size

        source_embedded = self.embedding_layer(source, source_lens)  # -> (batch_size, max_seq_len, embedding_dim)
        hidden, cell = self.encoder(source_embedded)  # hidden.shape = cell.shape = (batch_size, hidden_size)

        outputs = torch.zeros(target_max_len, batch_size, vocab_size)
        seqs = target[:, 0]

        for t in range(1, target.shape[1]):
            output, hidden, cell = self.decoder(seqs, hidden, cell)
            # output.shape = (batch_size, vocab_size)
            # hidden.shape = cell.shape = (batch_size, hidden_size)

            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            predicted_tokens = output.argmax(axis=1)  # -> (batch_size)
            seqs = target[:, t] if teacher_force else predicted_tokens

        return outputs
