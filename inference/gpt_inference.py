import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
        inpt = tok.encode(text, return_tensors="pt")

        out = model.generate(inpt.to(self.__device), max_length=500, repetition_penalty=5.0,
                             do_sample=True, top_k=5, top_p=0.95, temperature=1)

        return tok.decode(out[0])


if __name__ == '__main__':
    inferencer = (MODEL_PATH, DEVICE)
    print(inferencer.predict('Дом (Home)'))
