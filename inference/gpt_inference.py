import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse

np.random.seed(42)
torch.manual_seed(42)

MODEL_PATH = 'train/models/gpt'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GPTInference:
    def __init__(self, model_path: str = MODEL_PATH, device: str = 'cpu'):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--song_name", type=str, help="name of the song", default='Дом (Home)')
    parser.add_argument("--model_path", type=str, help="path to model", default=MODEL_PATH)
    parser.add_argument("--device", type=str, help="device - cuda / cpu", default=DEVICE)
    args = parser.parse_args()

    inferencer = GPTInference(args.model_path, args.device)
    print(inferencer.predict(args.song_name))
