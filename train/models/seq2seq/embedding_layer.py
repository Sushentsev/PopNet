from torch import nn, Tensor


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def forward(self, seqs: Tensor) -> Tensor:
        """
        Layer gets batch of sequences and returns batch of embedded sequences.
        :param seqs: input sequences, shape (seq_len, batch_size)
        :return: embedded sequences, shape (seq_len, batch_size, embedding_dim)
        """

        return self.embedding(seqs)
