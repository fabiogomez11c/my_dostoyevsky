import torch
from model import VOCAB


class FyodorDataset(torch.utils.data.Dataset):
    def __init__(self, all_books: str):
        self.stoi = {c: i for i, c in enumerate(VOCAB)}
        self.itos = {i: c for i, c in enumerate(VOCAB)}
        self.data = torch.tensor(self.encode(all_books), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx >= len(self.data) - 1:
            return self.data[idx - 2], self.data[idx - 1]
        else:
            return self.data[idx], self.data[idx + 1]

    def encode(self, x):
        return [self.stoi[c] for c in x]

    def decode(self, x):
        return "".join([self.itos[c] for c in x])
