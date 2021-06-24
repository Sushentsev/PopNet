import spacy
from transformers import GPT2Tokenizer
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
    def pad_index(self) -> int:
        return self.__lang.encode(self.__pad_token)

    @property
    def sos_index(self) -> int:
        return self.__lang.encode(self.__sos_token)

    @property
    def eos_index(self) -> int:
        return self.__lang.encode(self.__eos_token)


class SberRuGPTTokenizer:
    def __init__(self, pad_token: str, sos_token: str, eos_token: str):
        self.__pad_token = pad_token
        self.__sos_token = sos_token
        self.__eos_token = eos_token

        self.__tokenizer = GPT2Tokenizer.from_pretrained("sberbank-ai/rugpt3large_based_on_gpt2",
                                                         pad_token=self.__pad_token,
                                                         bos_token=self.__sos_token,
                                                         eos_token=self.__eos_token)

    def encode(self, corpus: List[str], target: bool = False) -> List[List[int]]:
        out = []
        for text in tqdm(corpus):
            coded = self.__tokenizer.encode(text)
            if target:
                coded = [self.sos_index] + coded + [self.eos_index]
            out.append(coded)

            return out

    def decode(self, corpus: List[List[int]]) -> List[str]:
        return [self.__tokenizer.decode(text) for text in corpus]

    @property
    def vocab_size(self) -> int:
        return self.__tokenizer.vocab_size

    @property
    def pad_index(self) -> int:
        return self.__tokenizer.pad_token_id

    @property
    def sos_index(self) -> int:
        return self.__tokenizer.bos_token_id

    @property
    def eos_index(self) -> int:
        return self.__tokenizer.eos_token_id
