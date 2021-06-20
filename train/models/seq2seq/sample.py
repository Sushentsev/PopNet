from typing import Tuple

import torch
from torch import Tensor, nn

from train.models.seq2seq.decoder import Decoder
from train.models.seq2seq.embedding_layer import EmbeddingLayer
from train.models.seq2seq.encoder import Encoder
from train.models.seq2seq.seq2seq_model import Seq2Seq

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 5
HIDDEN_SIZE = 7
MAX_SRC_SEQ_LEN = 20
MAX_TRG_SEQ_LEN = 100
INPUT_SIZE = EMBEDDING_DIM = 10
VOCAB_SIZE = 50
PADDING_IDX = 0
SOS_TOKEN_IDX = 1
EOS_TOKEN_IDX = 2


def gen_data(max_len: int) -> Tuple[Tensor, Tensor]:
    lens = torch.randint(1, max_len, (BATCH_SIZE,))  # -> (batch_size)
    seqs = torch.zeros(BATCH_SIZE, max_len, dtype=torch.long)  # -> (batch_size, max_len)

    for i in range(BATCH_SIZE):
        seq_len = lens[i]
        seqs[i, :seq_len] = torch.randint(1, VOCAB_SIZE, (seq_len,))

    return seqs, lens


def main():
    input_seqs, input_lens = gen_data(MAX_SRC_SEQ_LEN)
    target_seqs, target_lens = gen_data(MAX_TRG_SEQ_LEN)

    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM, INPUT_SIZE, HIDDEN_SIZE, DEVICE)
    decoder = Decoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE * 2,
                      SOS_TOKEN_IDX, EOS_TOKEN_IDX, DEVICE, MAX_TRG_SEQ_LEN)
    # HIDDEN_SIZE * 2 because of bidirectional encoder!

    seq2seq = Seq2Seq(encoder, decoder, DEVICE)

    outputs = seq2seq.forward(input_seqs, input_lens, target_seqs, target_lens, teacher_forcing_ratio=0.1)
    print(f"Teacher forcing example")
    print(f"Outputs shape (max_len, batch_size, vocab_size): {tuple(outputs.shape)}")
    print(f"Expected shape: ({target_seqs.shape[1]}, {BATCH_SIZE}, {VOCAB_SIZE})")
    print()

    outputs = seq2seq.forward(input_seqs, input_lens, teacher_forcing_ratio=0.)
    print(f"Without teacher forcing example")
    print(f"Outputs shape (max_len, batch_size, vocab_size): {tuple(outputs.shape)}")
    print(f"Expected shape: ({MAX_TRG_SEQ_LEN}, {BATCH_SIZE}, {VOCAB_SIZE})")
    print()


if __name__ == "__main__":
    main()
