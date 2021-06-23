import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
from omegaconf import OmegaConf
from argparse import ArgumentParser

CONFIG_PATH = 'train/configs/gpt_config.yaml'

np.random.seed(42)
torch.manual_seed(42)


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--song_name", type=str, help="name of the song")
    arg_parser.add_argument("-c", "--config", help="Path to YAML configuration file", type=str, default=CONFIG_PATH)
    return arg_parser


class GPTInference:
    def __init__(self, model_path: str, device):
        self.__model_path = model_path
        self.__device = torch.device(device)

        self.__tok = GPT2Tokenizer.from_pretrained(self.__model_path)
        self.__model = GPT2LMHeadModel.from_pretrained(self.__model_path).to(self.__device)

    def predict(self, song_name: str) -> str:
        text = f"<startsong>\n<songname> {song_name}\n<songlyrics>\n"
        inpt = self.__tok.encode(text, return_tensors="pt")

        out = self.__model.generate(inpt.to(self.__device), max_length=500, repetition_penalty=5.0,
                                    do_sample=True, top_k=5, top_p=0.95, temperature=1)

        return self.__tok.decode(out[0])


if __name__ == '__main__':
    __arg_parser = configure_arg_parser()
    __args = __arg_parser.parse_args()
    config = OmegaConf.load(__args.config)

    inferencer = GPTInference(config.model.path, config.device)
    print(inferencer.predict(__args.song_name))
