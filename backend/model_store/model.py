import torch
import torch.nn as nn
import torch.nn.functional as F

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
