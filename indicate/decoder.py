from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - torch convention


class Decoder(nn.Module):
    """LSTM decoder with Luong (dot-product) attention.

    Mirrors the original Keras model: the attention query is a linear
    projection of the target embedding (not the recurrent state), attention is
    unscaled dot-product over the encoder outputs (Keras ``Attention`` with
    ``use_scale=False``), and the attention context is concatenated with the
    embedding before the LSTM.

    ``forward`` works for both the full target sequence (training, teacher
    forcing) and a single step (autoregressive inference) by carrying ``state``
    across calls.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, dec_units: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.query_layer = nn.Linear(embedding_dim, dec_units)
        self.lstm = nn.LSTM(dec_units + embedding_dim, dec_units, batch_first=True)
        self.fc = nn.Linear(dec_units, vocab_size)

    def forward(
        self,
        inputs: torch.Tensor,
        encoder_outputs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
        src_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.embedding(inputs)  # [B, T, E]
        query = self.query_layer(x)  # [B, T, U]

        # Unscaled dot-product (Luong) attention over the encoder outputs.
        scores = torch.bmm(query, encoder_outputs.transpose(1, 2))  # [B, T, S]
        if src_mask is not None:
            # Mask padded encoder positions (src_mask: True = real token).
            scores = scores.masked_fill(~src_mask.unsqueeze(1), float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoder_outputs)  # [B, T, U]

        lstm_in = torch.cat([context, x], dim=-1)  # [B, T, U + E]
        outputs, new_state = self.lstm(lstm_in, state)
        logits = self.fc(outputs)  # [B, T, V]
        return logits, new_state
