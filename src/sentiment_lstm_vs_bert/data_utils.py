from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

import torch
from datasets import DatasetDict, load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

TOKEN_RE = re.compile(r"[a-z0-9']+")
DEFAULT_DATA_DIR = Path("data/imdb_prepared")


# ---------------------------------------------------------------------------
# 1. Load prepared dataset & LSTM vocabulary
# ---------------------------------------------------------------------------

def load_prepared_dataset(data_dir: str | Path = DEFAULT_DATA_DIR) -> DatasetDict:
    """Load the cleaned and split IMDB DatasetDict from disk."""
    return load_from_disk(str(data_dir))


def load_lstm_vocab(
    vocab_path: str | Path = DEFAULT_DATA_DIR / "lstm_vocab.json",
) -> dict:
    """Load the saved LSTM vocabulary and metadata JSON."""
    return json.loads(Path(vocab_path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 2. LSTM tokenization, Dataset & DataLoaders
# ---------------------------------------------------------------------------

def lstm_tokenize(text: str) -> list[str]:
    """Lowercase and extract alphanumeric/apostrophe tokens."""
    return TOKEN_RE.findall(text.lower())


def encode_text(
    text: str,
    token_to_idx: dict[str, int],
    unk_id: int = 1,
    max_len: int = 256,
) -> list[int]:
    """Convert raw text to truncated token IDs (at least 1 token for packing)."""
    tokens = lstm_tokenize(text)[:max_len]
    if not tokens:
        return [unk_id]
    return [token_to_idx.get(tok, unk_id) for tok in tokens]


class LSTMDataset(Dataset):
    """PyTorch Dataset that pre-encodes texts into integer token ID tensors."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        token_to_idx: dict[str, int],
        unk_id: int = 1,
        max_len: int = 256,
    ) -> None:
        self.encoded = [
            torch.tensor(
                encode_text(text, token_to_idx, unk_id=unk_id, max_len=max_len),
                dtype=torch.long,
            )
            for text in texts
        ]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoded[idx], self.labels[idx]


def make_lstm_collate_fn(pad_id: int = 0) -> Callable:
    """Create a collate_fn that dynamically pads each batch to its max length."""

    def collate_fn(
        batch: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        sequences, labels = zip(*batch)
        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
        input_ids = pad_sequence(sequences, batch_first=True, padding_value=pad_id)
        return {
            "input_ids": input_ids,
            "lengths": lengths,
            "labels": torch.stack(labels),
        }

    return collate_fn


def build_lstm_dataloaders(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    vocab_path: str | Path | None = None,
    max_len: int | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Build train, validation, and test DataLoaders for SentimentLSTM."""
    data_dir = Path(data_dir)
    vocab_payload = load_lstm_vocab(vocab_path or (data_dir / "lstm_vocab.json"))
    ds = load_prepared_dataset(data_dir)

    token_to_idx = vocab_payload["token_to_idx"]
    pad_id = vocab_payload["pad_id"]
    unk_id = vocab_payload["unk_id"]
    seq_len = max_len or vocab_payload.get("recommended_max_len", 256)

    collate_fn = make_lstm_collate_fn(pad_id=pad_id)
    loaders = []
    for split in ("train", "validation", "test"):
        split_ds = LSTMDataset(
            texts=ds[split]["text"],
            labels=ds[split]["label"],
            token_to_idx=token_to_idx,
            unk_id=unk_id,
            max_len=seq_len,
        )
        loaders.append(
            DataLoader(
                split_ds,
                batch_size=batch_size,
                shuffle=(split == "train"),
                collate_fn=collate_fn,
                num_workers=num_workers,
            )
        )

    return loaders[0], loaders[1], loaders[2], vocab_payload


# ---------------------------------------------------------------------------
# 3. BERT Dataset & DataLoaders
# ---------------------------------------------------------------------------

def build_bert_dataloaders(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    model_name: str = "bert-base-uncased",
    max_len: int = 256,
    batch_size: int = 16,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, PreTrainedTokenizerBase]:
    """Build dynamically padded train, validation, and test DataLoaders for BERT."""
    ds = load_prepared_dataset(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(batch: dict) -> dict:
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    tokenized = ds.map(tokenize_batch, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")

    keep_cols = [c for c in ("input_ids", "attention_mask", "token_type_ids", "labels") if c in tokenized["train"].column_names]
    tokenized.set_format(type="torch", columns=keep_cols)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    loaders = [
        DataLoader(
            tokenized[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collator,
            num_workers=num_workers,
        )
        for split in ("train", "validation", "test")
    ]
    return loaders[0], loaders[1], loaders[2], tokenizer
