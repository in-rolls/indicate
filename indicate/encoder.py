from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """LSTM encoder: embedding -> LSTM.

    Returns the full output sequence (for attention) and the final
    ``(hidden, cell)`` state used to initialise the decoder.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, enc_units: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, enc_units, batch_first=True)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)
        outputs, (h, c) = self.lstm(emb)
        return outputs, (h, c)
