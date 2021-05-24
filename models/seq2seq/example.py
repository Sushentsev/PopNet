from typing import Tuple

import torch
from torch import Tensor, nn

from models.seq2seq.decoder import Decoder
from models.seq2seq.embedding_layer import EmbeddingLayer
from models.seq2seq.encoder import Encoder
from models.seq2seq.seq2seq_model import Seq2Seq

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 5
HIDDEN_SIZE = 7
INPUT_SEQ_LEN = 20
TARGET_SEQ_LEN = 100
INPUT_SIZE = EMBEDDING_SIZE = 10
VOCAB_SIZE = 50
PADDING_IDX = 0


def gen_data(max_len: int) -> Tuple[Tensor, Tensor]:
    lens = torch.randint(1, max_len, (BATCH_SIZE,))  # -> (batch_size)
    seqs = torch.zeros(max_len, BATCH_SIZE, dtype=torch.long)  # -> (max_len, batch_size)

    for i in range(BATCH_SIZE):
        seq_len = lens[i]
        seqs[:seq_len, i] = torch.randint(1, VOCAB_SIZE, (seq_len,))

    return seqs, lens


def get_embedding_layer() -> nn.Module:
    return EmbeddingLayer(VOCAB_SIZE, EMBEDDING_SIZE, PADDING_IDX)


def get_encoder() -> nn.Module:
    return Encoder(INPUT_SIZE, HIDDEN_SIZE, dropout=0.3)


def get_decoder() -> nn.Module:
    # Note: decoder_hidden_size = 2 * encoder_hidden_size because encoder is bidirectional.
    # In bidirectional encoder we concatenate hidden states and cell states from every direction.
    # So, h_n and c_n in encoder have size (batch_size, 2 * hidden_size).
    return Decoder(INPUT_SIZE, 2 * HIDDEN_SIZE, VOCAB_SIZE, dropout=0.1)


def main():
    input_seqs, input_lens = gen_data(INPUT_SEQ_LEN)
    target_seqs, target_lens = gen_data(TARGET_SEQ_LEN)
    model = Seq2Seq(get_embedding_layer(), get_encoder(), get_decoder(), DEVICE)

    # Note: outputs[0] are all zeros. See notebook in seq2seq model.
    outputs = model(input_seqs, input_lens, target_seqs, teacher_forcing_ratio=0.3)

    print(f"Device: {DEVICE}")
    print(f"Expected outputs shape: (target_len={TARGET_SEQ_LEN}, batch_size={BATCH_SIZE}, vocab_size={VOCAB_SIZE})")
    print(f"Outputs shape: {tuple(outputs.shape)}")


if __name__ == "__main__":
    main()
