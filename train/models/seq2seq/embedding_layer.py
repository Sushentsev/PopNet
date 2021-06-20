from typing import Optional

from torch import nn, Tensor


class EmbeddingLayer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 padding_idx: Optional[int] = None,
                 embedding_weight: Optional[Tensor] = None,
                 update_embedding: bool = True):
        super().__init__()
        self.__embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

        if embedding_weight is not None:
            self.__embedding.weight = embedding_weight

        self.__embedding.weight.requires_grad = update_embedding

    def forward(self, seqs: Tensor) -> Tensor:
        """
        Gets batch of sequences and returns batch of embedded sequences.
        :param seqs: input sequences, shape (batch_size, seq_len)
        :return: embedded sequences, shape (batch_size, seq_len, embedding_dim)
        """

        return self.__embedding(seqs)
