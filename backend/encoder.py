import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.model import VOCAB
from src.data import FyodorDataset


class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tensor([0.0], dtype=torch.float32)


class CustomFyodorDataset(FyodorDataset):
    def __init__(
        self,
        all_books: str,
        block_size: int = 8,
        batch_size: int = 32,
        length: int = 1000,
    ):
        super().__init__(all_books, block_size, batch_size, length)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, x):
        x = torch.tensor([self.data[x : x + self.block_size]])
        y = torch.tensor([self.data[x + self.block_size]])
        return x[0], y[0]


if __name__ == "__main__":
    from src.utils import get_files_from_folder, open_txt

    batch_size = 32
    block_size = 8

    books = get_files_from_folder("books")
    books_string = [open_txt(f"books/{i}") for i in books]
    books = "\n".join(books_string)
    train_dataset = CustomFyodorDataset(
        books[: int(len(books) * 0.8)],
        length=batch_size * 100,
        block_size=block_size,
        batch_size=batch_size,
    )
