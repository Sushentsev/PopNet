from typing import Tuple, List
import numpy as np
import torch
from torch import Tensor, optim

from train.dataset.seq2seq_dataset import Seq2SeqDataset
from train.loss.cross_entropy import CrossEntropyLoss
from train.models.seq2seq.decoder import Decoder
from train.models.seq2seq.encoder import Encoder
from train.models.seq2seq.seq2seq_model import Seq2Seq
from train.trainer.seq2seq_trainer import Seq2SeqTrainer

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 50
HIDDEN_SIZE = 10

MAX_SRC_SEQ_LEN = 20
MAX_TRG_SEQ_LEN = 100

NUM_TRAIN_DATA = 200
NUM_DEV_DATA = 150

INPUT_SIZE = EMBEDDING_DIM = 10
VOCAB_SIZE = 50

PAD_TOKEN_IDX = 0
SOS_TOKEN_IDX = 1
EOS_TOKEN_IDX = 2


# Returns data and seq lens
def gen_data(n_examples: int, max_len: int) -> List[List[int]]:
    lens = list(np.random.randint(low=1, high=max_len - 1, size=n_examples))
    data = []

    for current_len in lens:
        seq = list(np.random.randint(low=1, high=VOCAB_SIZE - 1, size=current_len))
        data.append(seq)

    return data


def load_model() -> Seq2Seq:
    encoder = Encoder(VOCAB_SIZE, EMBEDDING_DIM,
                      HIDDEN_SIZE, DEVICE, 0.1,
                      PAD_TOKEN_IDX)
    decoder = Decoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_SIZE * 2,
                      SOS_TOKEN_IDX, EOS_TOKEN_IDX,
                      DEVICE, MAX_TRG_SEQ_LEN, 0.2, PAD_TOKEN_IDX)

    seq2seq = Seq2Seq(encoder, decoder, DEVICE)
    return seq2seq


# This is train example! Not real data!
def train_example():
    train_src = gen_data(NUM_TRAIN_DATA, MAX_SRC_SEQ_LEN)
    train_trg = gen_data(NUM_TRAIN_DATA, MAX_TRG_SEQ_LEN)
    train_dataset = Seq2SeqDataset(train_src, train_trg)

    dev_src = gen_data(NUM_DEV_DATA, MAX_SRC_SEQ_LEN)
    dev_trg = gen_data(NUM_DEV_DATA, MAX_TRG_SEQ_LEN)
    dev_dataset = Seq2SeqDataset(dev_src, dev_trg)

    seq2seq = load_model()
    loss = CrossEntropyLoss(ignore_index=PAD_TOKEN_IDX)
    optimizer = optim.Adam(seq2seq.parameters())

    trainer = Seq2SeqTrainer(loss, BATCH_SIZE, DEVICE)
    trainer.train(seq2seq, train_dataset, 100, dev_dataset, optimizer, teacher_forcing_ratio=0.1)


if __name__ == '__main__':
    train_example()
