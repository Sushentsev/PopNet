from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from train.dataset.seq2seq_dataset import Seq2SeqDataset


def _collate_fn(batch: List[Tuple[List[int], List[int]]]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    src_tensors = [torch.LongTensor(src) for src, _ in batch]
    trg_tensors = [torch.LongTensor(trg) for _, trg in batch]

    src_lens = torch.LongTensor([len(src) for src in src_tensors])
    trg_lens = torch.LongTensor([len(trg) for trg in trg_tensors])

    src_padded = pad_sequence(src_tensors, batch_first=True)
    trg_padded = pad_sequence(trg_tensors, batch_first=True)

    return src_padded, src_lens, trg_padded, trg_lens


def get_dataloader(dataset: Seq2SeqDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size, shuffle, collate_fn=_collate_fn)
