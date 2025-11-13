from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer_utils import encode_texts, pad_sequences


@dataclass
class DatasetSplit:
    inputs: torch.Tensor
    targets: torch.Tensor


class SummarizationDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[idx],
            "labels": self.targets[idx],
        }


def load_dataset(
    csv_path: str,
    text_column: str = "transcription",
    summary_column: str = "description",
) -> Tuple[Iterable[str], Iterable[str]]:
    df = pd.read_csv(csv_path)
    df = df[[text_column, summary_column]].dropna()
    df = df.rename(columns={text_column: "text", summary_column: "summary"})
    return df["text"].tolist(), df["summary"].tolist()


def create_dataloaders(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int = 4,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    dataset_size = len(inputs)
    train_size = int((1 - val_split) * dataset_size)
    train_inputs, val_inputs = inputs[:train_size], inputs[train_size:]
    train_targets, val_targets = targets[:train_size], targets[train_size:]

    train_dataset = SummarizationDataset(train_inputs, train_targets)
    val_dataset = SummarizationDataset(val_inputs, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader


def prepare_tensors(
    input_ids, target_ids, pad_token_id: int, max_input_length: int, max_target_length: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_encodings = pad_sequences(input_ids, max_input_length, pad_token_id)
    target_encodings = pad_sequences(target_ids, max_target_length, pad_token_id)
    input_tensors = torch.tensor(input_encodings, dtype=torch.long)
    target_tensors = torch.tensor(target_encodings, dtype=torch.long)
    return input_tensors, target_tensors

