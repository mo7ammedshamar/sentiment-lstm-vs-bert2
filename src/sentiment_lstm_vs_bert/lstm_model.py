from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class LSTMConfig:
    """Configuration for the from-scratch SentimentLSTM classifier."""

    vocab_size: int = 25_000
    embed_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    num_classes: int = 2
    pad_idx: int = 0
    pooling: Literal["last", "mean"] = "last"


class SentimentLSTM(nn.Module):
    """Bidirectional/unidirectional LSTM sentiment classifier built from scratch.

    Expects token ID tensors of shape ``(batch_size, seq_len)`` and optional
    sequence lengths of shape ``(batch_size,)``. Returns raw logits of shape
    ``(batch_size, num_classes)`` (or ``(batch_size,)`` when ``num_classes == 1``).
    """

    def __init__(self, config: LSTMConfig | None = None, **kwargs) -> None:
        super().__init__()
        if config is None:
            config = LSTMConfig(**kwargs)
        elif kwargs:
            raise ValueError("Pass either `config` or keyword arguments, not both.")

        if config.pooling not in ("last", "mean"):
            raise ValueError(f"Unsupported pooling='{config.pooling}'. Use 'last' or 'mean'.")

        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embed_dim,
            padding_idx=config.pad_idx,
        )
        self.embed_dropout = nn.Dropout(config.dropout)

        lstm_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=config.bidirectional,
            batch_first=True,
            dropout=lstm_dropout,
        )

        num_directions = 2 if config.bidirectional else 1
        feature_dim = config.hidden_dim * num_directions
        self.head_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(feature_dim, config.num_classes)

    def _infer_lengths(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute non-pad token counts clamped to at least 1 for packing safety."""
        lengths = (input_ids != self.config.pad_idx).sum(dim=1)
        return lengths.clamp(min=1)

    def _pool_features(
        self,
        packed_output: nn.utils.rnn.PackedSequence,
        h_n: torch.Tensor,
        lengths: torch.Tensor,
        max_len: int,
    ) -> torch.Tensor:
        """Extract sequence-level representation via 'last' hidden state or masked 'mean'."""
        if self.config.pooling == "last":
            if self.config.bidirectional:
                return torch.cat((h_n[-2], h_n[-1]), dim=-1)
            return h_n[-1]

        unpacked, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=max_len
        )
        mask = (
            torch.arange(max_len, device=unpacked.device)
            .unsqueeze(0)
            .lt(lengths.unsqueeze(1))
            .unsqueeze(-1)
        )
        summed = (unpacked * mask).sum(dim=1)
        return summed / lengths.unsqueeze(1).clamp(min=1).to(unpacked.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        if lengths is None:
            lengths = self._infer_lengths(input_ids)
        else:
            lengths = lengths.clamp(min=1, max=input_ids.size(1))

        embedded = self.embed_dropout(self.embedding(input_ids))
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (h_n, _) = self.lstm(packed)

        features = self._pool_features(
            packed_output=packed_output,
            h_n=h_n,
            lengths=lengths,
            max_len=input_ids.size(1),
        )
        logits = self.classifier(self.head_dropout(features))
        if self.config.num_classes == 1:
            return logits.squeeze(-1)
        return logits

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Return total number of (trainable) parameters in the model."""
        params = (
            p for p in self.parameters() if not trainable_only or p.requires_grad
        )
        return sum(p.numel() for p in params)
