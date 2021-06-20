from typing import List, Tuple
from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    def __init__(self, src: List[List[int]], trg: List[List[int]]):
        self.__src = src
        self.__trg = trg

    def __getitem__(self, item) -> Tuple[List[int], List[int]]:
        return self.__src[item], self.__trg[item]

    def __len__(self) -> int:
        return len(self.__src)
