import torch
from src.model import VOCAB


class FyodorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        all_books: str,
        block_size: int = 8,
        batch_size: int = 32,
        length: int = 1000,
    ):
        self.stoi = {c: i for i, c in enumerate(VOCAB)}
        self.itos = {i: c for i, c in enumerate(VOCAB)}
        self.data = torch.tensor(self.encode(all_books), dtype=torch.long)
        self.block_size = block_size
        self.batch_size = batch_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        ix = torch.randint(len(self.data) - self.block_size, (1,))
        x = torch.stack([self.data[i : i + self.block_size] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
        return x[0], y[0]

    def encode(self, x):
        return [self.stoi[c] for c in x]

    def decode(self, x):
        return "".join([self.itos[c] for c in x])


def __getitem__(self, idx):
    ix = torch.randint(len(self.data) - self.block_size, (1,))
    x = torch.stack([self.data[i : i + self.block_size] for i in ix])
    y = torch.stack([self.data[i + 1 : i + self.block_size + 1] for i in ix])
    return x[0], y[0]


def encode(self, x):
    return [self.stoi[c] for c in x]


def decode(self, x):
    return "".join([self.itos[c] for c in x])
