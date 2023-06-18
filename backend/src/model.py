import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from constant import VOCAB


class Head(nn.Module):
    """one head attention"""

    def __init__(self, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.block_size = block_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)  # (B, T, C)
        # compute attention scores ("affinities")
        # remember the scaling by dimension to avoid bias in softmax - scale attention
        wei = q @ k.transpose(-1, -2) * C**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v  # (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple head of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                Head(
                    head_size=head_size,
                    n_embd=n_embd,
                    block_size=block_size,
                    dropout=dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size,
            n_embd=n_embd,
            block_size=block_size,
            dropout=dropout,
        )
        self.ffwd = FeedForward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln1(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(
        self, n_embd=32, block_size=8, n_head=4, n_layer=3, dropout=0.1, device="cpu"
    ):
        super().__init__()
        self.stoi = {c: i for i, c in enumerate(VOCAB)}
        self.itos = {i: c for i, c in enumerate(VOCAB)}
        self.token_embedding_table = nn.Embedding(len(VOCAB), n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head=n_head, block_size=block_size, dropout=dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, len(VOCAB))
        self.block_size = block_size

        self.device = device

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx dimensions: (B, T)
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

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
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size :]  # (B, T)
            # get the predictions
            model_result = self(idx_cond)
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
