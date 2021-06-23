import spacy
from typing import List
from spacy.tokens import Token
from train.preprocess.seq2seq.lang import Lang
from tqdm import tqdm



class SpacyRuTokenizer:
    def __init__(self, pad_token: str, sos_token: str, eos_token: str):
        self.__pad_token = pad_token
        self.__sos_token = sos_token
        self.__eos_token = eos_token

        self.__spacy_ru = spacy.load("ru_core_news_md")
        self.__lang = Lang(pad_token, sos_token, eos_token)

    def __tokenize(self, text: str) -> List[Token]:
        return [token for token in self.__spacy_ru(text)]

    def __preprocess(self, tokenized: List[Token]) -> List[str]:
        return [token.text.lower().strip() for token in tokenized if token.is_alpha]

    def build_vocab(self, corpus: List[str]):
        for text in tqdm(corpus):
            tokenized = self.__tokenize(text)
            preprocessed = self.__preprocess(tokenized)
            self.__lang.addSentence(preprocessed)

    def encode(self, corpus: List[str], target: bool = False) -> List[List[int]]:
        out = []
        for text in tqdm(corpus):
            coded = self.__lang.encodes(self.__preprocess(self.__tokenize(text)))
            if target:
                coded = [self.sos_index] + coded + [self.eos_index]
            out.append(coded)

        return out

    def decode(self, corpus: List[List[int]]) -> List[str]:
        return [" ".join(self.__lang.decodes(text)) for text in corpus]

    @property
    def vocab_size(self) -> int:
        return self.__lang.vocab_size

    @property
    def pad_index(self):
        return self.__lang.encode(self.__pad_token)

    @property
    def sos_index(self):
        return self.__lang.encode(self.__sos_token)

    @property
    def eos_index(self):
        return self.__lang.encode(self.__eos_token)


if __name__ == '__main__':
    from utils import EOS, SOS, PAD
    tokenizer = SpacyRuTokenizer(PAD, SOS, EOS)
    corpus = [
        "привет, что делаешь; я делаю пляски",
        "мы любим, ! ;; f пить"
    ]

    tokenizer.build_vocab(corpus)

    import pickle

    with open("tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    print(tokenizer.encode(["5,"]))
    print(tokenizer.decode([[1, 10, 11, 12, 13, 2]]))
