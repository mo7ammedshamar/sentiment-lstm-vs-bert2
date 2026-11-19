from sentiment_lstm_vs_bert.data_utils import (
    build_bert_dataloaders,
    build_lstm_dataloaders,
    load_lstm_vocab,
    load_prepared_dataset,
)
from sentiment_lstm_vs_bert.lstm_model import LSTMConfig, SentimentLSTM
from sentiment_lstm_vs_bert.trainer import TrainConfig, evaluate, train_model

__all__ = [
    "LSTMConfig",
    "SentimentLSTM",
    "TrainConfig",
    "build_bert_dataloaders",
    "build_lstm_dataloaders",
    "evaluate",
    "load_lstm_vocab",
    "load_prepared_dataset",
    "main",
    "train_model",
]


def main() -> None:
    print("Hello from sentiment-lstm-vs-bert!")
