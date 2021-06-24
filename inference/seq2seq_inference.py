import pickle
import torch
import numpy as np

from typing import Optional
from omegaconf import OmegaConf
from train.models.seq2seq.decoder import Decoder
from train.models.seq2seq.encoder import Encoder
from train.models.seq2seq.seq2seq_model import Seq2Seq
from train.preprocess.seq2seq.tokenizs import SpacyRuTokenizer


class Seq2SeqInference:
    def __init__(self, device: str = "cpu"):
        self.__device = device
        self.__model: Optional[Seq2Seq] = None
        self.__tokenizer: Optional[SpacyRuTokenizer] = None

    def predict(self, song_name: str) -> str:
        encoded = self.__tokenizer.encode([song_name])
        src = torch.LongTensor(encoded).to(self.__device)
        src_len = torch.LongTensor(len(encoded[0])).to(self.__device)

        with torch.no_grad():
            outputs = self.__model.forward(src, src_len).unsqueeze(1)  # -> (trg_len, vocab_size)

        outputs = outputs.cpu().detach().numpy()
        ids = np.argmax(outputs, axis=1)
        predicted = self.__tokenizer.decode([list(ids)])[0]
        return predicted

    def load_params(self, config_path: str, checkpoint_path: str, tokenizer_path: str):
        with open(tokenizer_path, "rb") as file:
            self.__tokenizer = pickle.load(file)

        config = OmegaConf.load(config_path)

        encoder = Encoder(vocab_size=self.__tokenizer.vocab_size, **config.model.encoder)
        decoder = Decoder(vocab_size=self.__tokenizer.vocab_size, **config.model.decoder)

        self.__model = Seq2Seq(encoder, decoder, self.__device)
        self.__model.load_state_dict(torch.load(checkpoint_path))
        self.__model = self.__model.eval().to(self.__device)
