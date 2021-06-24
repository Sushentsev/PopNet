import numpy as np
import torch
import wandb
import pandas as pd
import logging

from argparse import ArgumentParser
from omegaconf import OmegaConf
from typing import List
from torch import optim

from train.dataset.seq2seq_dataset import Seq2SeqDataset
from train.loss.cross_entropy import CrossEntropyLoss
from train.models.seq2seq.decoder import Decoder
from train.models.seq2seq.encoder import Encoder
from train.models.seq2seq.seq2seq_model import Seq2Seq
from train.preprocess.seq2seq.tokenizs import SpacyRuTokenizer, SberRuGPTTokenizer
from train.trainer.seq2seq_trainer import Seq2SeqTrainer
from utils import PAD, EOS, SOS

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str)
    return arg_parser


def load_data(train_path: str, dev_path: str):
    train_data = pd.read_csv(train_path, sep="\t")
    dev_data = pd.read_csv(dev_path, sep="\t")

    train_src = train_data["Title"].astype(str).tolist()
    train_trg = train_data["Lyrics"].astype(str).tolist()
    dev_src = dev_data["Title"].astype(str).tolist()
    dev_trg = dev_data["Lyrics"].astype(str).tolist()

    return train_src, train_trg, dev_src, dev_trg


def remove_empty(src: List[List[int]], trg: List[List[int]]):
    src_filtered, trg_filtered = [], []

    for src_, trg_ in zip(src, trg):
        if len(src_) > 0 and len(trg_) > 0:
            src_filtered.append(src_)
            trg_filtered.append(trg_)

    return src_filtered, trg_filtered


def encode(tokenizer,
           train_src: List[str], train_trg: List[str],
           dev_src: List[str], dev_trg: List[str]):
    print(f"Encoding.")
    train_src_encoded = tokenizer.encode(train_src)
    train_trg_encoded = tokenizer.encode(train_trg, target=True)
    train_src_encoded, train_trg_encoded = remove_empty(train_src_encoded, train_trg_encoded)

    dev_src_encoded = tokenizer.encode(dev_src)
    dev_trg_encoded = tokenizer.encode(dev_trg, target=True)
    dev_src_encoded, dev_trg_encoded = remove_empty(dev_src_encoded, dev_trg_encoded)
    return train_src_encoded, train_trg_encoded, dev_src_encoded, dev_trg_encoded


def train(config_path: str):
    logging.basicConfig(level=logging.INFO)

    config = OmegaConf.load(config_path)

    wandb.login()
    wandb.init(project="hse_dl_2021",
               notes="Seq2Seq architecture with encoder-decoder on LSTM",
               tags=["seq2seq", "lstm"],
               config=config)

    train_src, train_trg, dev_src, dev_trg = load_data(config.train_data, config.dev_data)

    tokenizer = SberRuGPTTokenizer(PAD, SOS, EOS)
    train_src, train_trg, dev_src, dev_trg = encode(tokenizer, train_src, train_trg, dev_src, dev_trg)

    max_song_len = max(max(map(len, train_trg)), max(map(len, dev_trg)))

    train_dataset = Seq2SeqDataset(train_src, train_trg)
    dev_dataset = Seq2SeqDataset(dev_src, dev_trg)

    print(f"Train dataset len: {len(train_dataset)}")
    print(f"Dev dataset len: {len(dev_dataset)}")
    print(f"Max lyrics len: {max_song_len}")
    print(f"Vocab size: {tokenizer.vocab_size}")

    encoder = Encoder(vocab_size=tokenizer.vocab_size,
                      device=DEVICE, **config.model.encoder)

    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      sos_id=tokenizer.sos_index, eos_id=tokenizer.eos_index, padding_idx=0,
                      max_len=max_song_len, device=DEVICE, **config.model.decoder)

    seq2seq = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    loss = CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(seq2seq.parameters())

    trainer = Seq2SeqTrainer(loss=loss, device=DEVICE, **config.trainer)
    trainer.train(model=seq2seq,
                  train_data=train_dataset, dev_data=dev_dataset,
                  optimizer=optimizer, **config.train)


if __name__ == "__main__":
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    train(__args.config)
