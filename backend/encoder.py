import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from src.model import VOCAB
from src.data import FyodorDataset

torch.set_printoptions(linewidth=220)


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
        x = self.data[x : x + self.block_size]
        y = self.data[x + self.block_size]
        return x, y


class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), 4)
        self.query = nn.Linear(4, 8, bias=False)
        self.key = nn.Linear(4, 8, bias=False)
        self.value = nn.Linear(4, 8, bias=False)

    def forward(self, x):
        # attention
        x = self.embedding(x)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        kdim = k.shape[-1]

        wei = q @ k.transpose(-1, -2) / (kdim**0.5)
        wei = F.softmax(wei, dim=-1)

        out = wei @ v
        return out


if __name__ == "__main__":
    from src.utils import get_files_from_folder, open_txt

    # Hyperparameters
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

    x, y = train_dataset[0]

    model = EncoderModel()
    y_hat = model(x)

    print("Done")