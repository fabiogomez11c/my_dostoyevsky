import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from constant import VOCAB


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
            if len(logits.shape) == 3:
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
            model_result = self(idx)
            if type(model_result) == tuple:
                logits, loss = model_result
            else:
                logits = self(idx)
            # focus only on the last time step, this is important!
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


class BigramLanguageModelV2(BigramLanguageModel):
    def __init__(self):
        super().__init__()

    def forward(self, idx):
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if len(logits.shape) == 3:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
        return logits
