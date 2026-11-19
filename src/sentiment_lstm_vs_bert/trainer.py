from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup


@dataclass
class TrainConfig:
    """Shared training configuration for both LSTM and BERT models."""

    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    save_path: str | Path | None = None
    device: str | None = None


def resolve_device(device: str | None = None) -> torch.device:
    """Return the requested torch.device or auto-select CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _forward_logits(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Run a unified forward pass for either SentimentLSTM or HuggingFace BERT."""
    if "lengths" in batch:
        return model(batch["input_ids"], batch["lengths"])

    model_inputs = {
        k: batch[k]
        for k in ("input_ids", "attention_mask", "token_type_ids")
        if k in batch
    }
    outputs = model(**model_inputs)
    return outputs.logits if hasattr(outputs, "logits") else outputs


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device | None = None,
    return_predictions: bool = False,
) -> dict:
    """Evaluate a classifier on a DataLoader and return standard metrics + latency."""
    dev = resolve_device(str(device) if isinstance(device, torch.device) else device)
    model.to(dev)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_labels: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    start_time = time.perf_counter()
    for batch in dataloader:
        batch = {k: v.to(dev) for k, v in batch.items()}
        labels = batch["labels"]
        logits = _forward_logits(model, batch)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = logits.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    elapsed_s = time.perf_counter() - start_time
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    n_samples = len(y_true)

    metrics: dict = {
        "loss": total_loss / n_samples,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "eval_time_s": round(elapsed_s, 3),
        "latency_ms_per_sample": round((elapsed_s / n_samples) * 1000, 4),
    }
    if return_predictions:
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        metrics["y_prob"] = y_prob
    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig | None = None,
    **kwargs,
) -> dict:
    """Train a model with AdamW, linear warmup/decay, gradient clipping, and best-F1 checkpointing."""
    if config is None:
        config = TrainConfig(**kwargs)
    elif kwargs:
        raise ValueError("Pass either `config` or keyword arguments, not both.")

    dev = resolve_device(config.device)
    model.to(dev)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    save_path = Path(config.save_path) if config.save_path else None
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    total_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        seen = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}", leave=False)
        for batch in progress:
            batch = {k: v.to(dev) for k, v in batch.items()}
            labels = batch["labels"]

            optimizer.zero_grad(set_to_none=True)
            logits = _forward_logits(model, batch)
            loss = criterion(logits, labels)
            loss.backward()

            if config.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()
            scheduler.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{running_loss / seen:.4f}")

        train_loss = running_loss / seen
        val_metrics = evaluate(model, val_loader, device=dev)
        epoch_time = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_f1": round(val_metrics["f1"], 4),
            "val_roc_auc": round(val_metrics["roc_auc"], 4),
            "epoch_time_s": round(epoch_time, 2),
        }
        history.append(record)

        print(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={record['train_loss']:.4f} | "
            f"val_loss={record['val_loss']:.4f} | "
            f"val_acc={record['val_acc']:.4f} | "
            f"val_f1={record['val_f1']:.4f} | "
            f"{record['epoch_time_s']:.1f}s"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_path is not None:
                torch.save(best_state, save_path)

    if best_state is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_state.items()})

    return {
        "history": history,
        "best_val_f1": round(best_val_f1, 4),
        "best_epoch": best_epoch,
        "total_train_time_s": round(time.perf_counter() - total_start, 2),
    }
