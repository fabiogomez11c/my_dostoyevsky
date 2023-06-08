import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lightning as pl

VOCAB = [
    "\n",
    " ",
    "!",
    '"',
    "'",
    "(",
    ")",
    "*",
    ",",
    "-",
    ".",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    ":",
    ";",
    "?",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "[",
    "]",
    "_",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "À",
    "Æ",
    "É",
    "à",
    "â",
    "ä",
    "æ",
    "ç",
    "è",
    "é",
    "ê",
    "ë",
    "î",
    "ï",
    "ô",
    "ö",
    "ü",
    "Œ",
    "œ",
    "‐",
    "—",
    "‘",
    "’",
    "“",
    "”",
]


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stoi = {c: i for i, c in enumerate(VOCAB)}
        self.itos = {i: c for i, c in enumerate(VOCAB)}
        self.token_embedding_table = nn.Embedding(len(VOCAB), len(VOCAB))

    def forward(self, idx, targets=None):
        # idx dimensions: (B, T)
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def encode(self, x):
        return [self.stoi[c] for c in x]

    def decode(self, x):
        return "".join([self.itos[c] for c in x])

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step, this is important!
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


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


class LitBigramLanguageModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.stoi = {c: i for i, c in enumerate(VOCAB)}
        self.itos = {i: c for i, c in enumerate(VOCAB)}
        self.token_embedding_table = nn.Embedding(len(VOCAB), len(VOCAB))

    def forward(self, idx, targets=None):
        # idx dimensions: (B, T)
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            if len(logits.shape) == 3:
                B, T, C = logits.shape
                logits = logits.view(B * T, C)
                targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        logits, loss = self(x, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    from utils import get_files_from_folder, open_txt

    torch.set_float32_matmul_precision("medium")

    books = get_files_from_folder("books")
    books_string = [open_txt(f"books/{i}") for i in books]
    books = "\n".join(books_string)

    train_dataset = FyodorDataset(books)
    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=24
    )

    trainer = pl.Trainer(
        fast_dev_run=False,
        max_epochs=10,
        log_every_n_steps=1,
        accelerator="gpu",
        # devices=1,
    )
    model = LitBigramLanguageModel()
    trainer.fit(model, train_dataloaders=train_dataloader)
