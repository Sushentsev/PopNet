from typing import List

PAD_TOKEN_IDX = 0
SOS_TOKEN_IDX = 1
EOS_TOKEN_IDX = 2
OOV_TOKEN_IDX = 3


class Lang:
    def __init__(self):
        self.__word2index = {}
        self.__index2word = {PAD_TOKEN_IDX: "<pad>",
                             SOS_TOKEN_IDX: "<sos>",
                             EOS_TOKEN_IDX: "<eos>",
                             OOV_TOKEN_IDX: "<oov>"}
        self.__word2count = {}
        self.__n_words = 4

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
            return OOV_TOKEN_IDX

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
