from typing import List, Optional


class Lang:
    def __init__(self, pad_token: str, sos_token: str, eos_token: str):
        self.__pad_token = pad_token
        self.__sos_token = sos_token
        self.__eos_token = eos_token
        self.__oov_token = "OOV"

        self.__word2index = {pad_token: 0, sos_token: 1, eos_token: 2, self.__oov_token: 3}
        self.__index2word = {index: word for word, index in self.__word2index.items()}
        self.__word2count = {}
        self.__n_words = len(self.__word2index)

    def addWord(self, word: str):
        if word not in self.__word2index:
            self.__word2index[word] = self.__n_words
            self.__word2count[word] = 1
            self.__index2word[self.__n_words] = word
            self.__n_words += 1
        else:
            self.__word2count[word] += 1

    def addSentence(self, sentence: List[str]):
        for word in sentence:
            self.addWord(word)

    def encode(self, word: str) -> int:
        if word in self.__word2index:
            return self.__word2index[word]
        else:
            return self.__word2index[self.__oov_token]

    def encodes(self, text: List[str]) -> List[int]:
        return [self.encode(word) for word in text]

    def decode(self, id: int) -> str:
        if id in self.__index2word:
            return self.__index2word[id]
        else:
            raise ValueError("Index not in lang.")

    def decodes(self, ids: List[int]) -> List[str]:
        return [self.decode(id) for id in ids]

    @property
    def vocab_size(self) -> int:
        return self.__n_words
