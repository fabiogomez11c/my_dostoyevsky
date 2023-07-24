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


class Head(nn.Module):
    def __init__(self, embed_dim: int = 4, head_size: int = 8):
        super().__init__()
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

    def forward(self, hidden):
        q = self.query(hidden)
        k = self.key(hidden)
        v = self.value(hidden)
        kdim = k.shape[-1]

        wei = q @ k.transpose(-1, -2) / (kdim**0.5)
        wei = F.softmax(wei, dim=-1)

        out = wei @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, num_heads: int = 8, embed_dim: int = 4):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [Head(embed_dim=embed_dim, head_size=head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden):
        out = torch.cat([h(hidden) for h in self.heads], dim=-1)
        out = self.output_linear(out)
        return out


class EncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), 4)
        self.head = Head()

    def forward(self, x):
        # attention
        x = self.embedding(x)
        out = self.head(x)
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
