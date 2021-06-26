import pickle
import torch
import numpy as np

from typing import Optional
from omegaconf import OmegaConf
from train.models.seq2seq.decoder import Decoder
from train.models.seq2seq.encoder import Encoder
from train.models.seq2seq.seq2seq_model import Seq2Seq
from train.preprocess.seq2seq.tokenizs import SberRuGPTTokenizer
from utils import PAD, SOS, EOS


class Seq2SeqInference:
    def __init__(self, device: str = "cpu"):
        self.__device = device
        self.__model: Optional[Seq2Seq] = None
        self.__tokenizer = SberRuGPTTokenizer(PAD, SOS, EOS)

    def predict(self, song_name: str) -> str:
        encoded = self.__tokenizer.encode([song_name])
        src = torch.LongTensor(encoded).to(self.__device)
        src_len = torch.LongTensor([len(encoded[0])]).to(self.__device)

        with torch.no_grad():
            outputs = self.__model.forward(src, src_len).squeeze(1)  # -> (trg_len, vocab_size)

        outputs = outputs.cpu().detach().numpy()
        ids = np.argmax(outputs, axis=1)
        ids = [id for id in ids if id != 203]
        predicted = self.__tokenizer.decode([list(ids)])[0]

        print(self.__tokenizer.decode([[203]]))
        print(ids)

        return predicted

    def load_params(self, config_path: str, checkpoint_path: str):
        config = OmegaConf.load(config_path)

        encoder = Encoder(vocab_size=self.__tokenizer.vocab_size,
                          device=self.__device,
                          **config.model.encoder)
        decoder = Decoder(vocab_size=self.__tokenizer.vocab_size,
                          device=self.__device,
                          sos_id=self.__tokenizer.sos_index,
                          eos_id=self.__tokenizer.eos_index,
                          max_len=1_000,
                          **config.model.decoder)

        self.__model = Seq2Seq(encoder, decoder, self.__device)
        self.__model.load_state_dict(torch.load(checkpoint_path, map_location=self.__device))
        self.__model = self.__model.eval().to(self.__device)


if __name__ == '__main__':
    seq2seq = Seq2SeqInference()
    seq2seq.load_params("/home/denis/Study/PopNet/train/configs/seq2seq_config.yaml",
                        "/home/denis/Study/PopNet/train/trainer/weight/seq2seq_epoch1_step448.pth")
    predicted = seq2seq.predict("Дом")
    print(predicted)
