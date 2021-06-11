import torch
import random
from torch import nn, Tensor


# Adapted from: https://github.com/bentrevett/pytorch-seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, embedding_layer: nn.Module, encoder: nn.Module, decoder: nn.Module, device):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self,
                source: Tensor,
                source_lens: Tensor,
                target: Tensor,
                teacher_forcing_ratio: float = 0.5) -> Tensor:
        """

        :param source: input sequences, shape (seq_len, batch_size)
        :param source_lens: input lens, shape (batch_size)
        :param target: target sequences, shape (target_len, batch_size)
        :param teacher_forcing_ratio: teacher forcing ratio (ratio of ground-truth inputs)
        :return: outputs, shape (target_len, batch_size, vocab_size)
        """

        target_len, batch_size = target.shape
        vocab_size = self.decoder.vocab_size

        source_embedded = self.embedding_layer(source)  # -> (seq_len, batch_size, embedding_dim)
        hidden, cell = self.encoder(source_embedded, source_lens)
        # hidden.shape = cell.shape = (batch_size, hidden_size)

        outputs = torch.zeros(target_len, batch_size, vocab_size).to(self.device)
        input_batch = target[0]

        for t in range(1, target.shape[1]):
            output, hidden, cell = self.decoder(self.embedding_layer(input_batch), hidden, cell)
            # output.shape = (batch_size, vocab_size)
            # hidden.shape = cell.shape = (batch_size, hidden_size)

            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            predicted_tokens = output.argmax(axis=1)  # -> (batch_size)
            input_batch = target[t] if teacher_force else predicted_tokens

        return outputs
